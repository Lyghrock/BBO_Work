"""
WandbSync — BBO 结果同步到 Weights & Biases

支持两种模式：
1. sync_realtime  : 运行中实时流式记录（配合 run_bbo_mujoco.py / run_bbo_benchmark.py）
2. sync_existing  : 将已有的 JSON 结果文件批量同步到 wandb

metric 约定：
- MuJoCo:  metric = "reward"  = -best_fx（越大越好）
- CEC:      metric = "min_fx"  =  best_fx（越小越好）

wandb 图表统一以 budget（函数评估次数）作为 x 轴。
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

# ── wandb 导入（可选依赖）─────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False
    print("[WandbSync] wandb 未安装，跳过安装检查")


# ══════════════════════════════════════════════════════════════════════════════
# 统一配置
# ══════════════════════════════════════════════════════════════════════════════

class WandbConfig:
    """所有 benchmark 共享的 wandb 配置"""

    def __init__(
        self,
        entity: str = None,
        project: str = None,
        group: str = None,
        tags: list = None,
        notes: str = None,
        resume: str = "allow",
    ):
        self.entity = entity
        self.project = project
        self.group = group or datetime.now().strftime("%Y-%m-%d")
        self.tags = tags or []
        self.notes = notes or ""
        self.resume = resume  # "allow" | "never" | "must"

    def to_wandb_init(self) -> dict:
        kwargs = {
            "project": self.project,
            "group": self.group,
            "tags": self.tags,
            "notes": self.notes,
            "resume": self.resume,
        }
        if self.entity:
            kwargs["entity"] = self.entity
        return kwargs


# ══════════════════════════════════════════════════════════════════════════════
# 实时 WandbLogger（嵌入 benchmark 主循环）
# ══════════════════════════════════════════════════════════════════════════════

# 进程级别计数器文件，用于所有 WandbLogger 实例间的 FIFO 顺序保证
_COUNTER_DIR = Path("/tmp/wandb-counters")
_COUNTER_DIR.mkdir(parents=True, exist_ok=True)


def _next_counter(origin: str) -> int:
    """原子递增计数器（文件锁保护），返回该 origin 的下一个全局序号。"""
    counter_file = _COUNTER_DIR / f"counter_{origin}.txt"
    lock_file = _COUNTER_DIR / f"counter_{origin}.lock"
    import fcntl
    try:
        with open(lock_file, "w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                if counter_file.exists():
                    count = int(counter_file.read_text().strip())
                else:
                    count = 0
                count += 1
                counter_file.write_text(str(count))
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
        return count
    except Exception:
        # 回退：使用时间戳微秒（不完美但不会阻塞）
        import time
        return int(time.time() * 1e6)


class WandbLogger:
    """
    将 benchmark 的实时进度记录到 wandb。

    wandb config 固定写入三个维度字段：
      algorithm — 算法名（如 bo, turbo, cmaes）
      task_name — 任务名（如 hopper, ackley_50d）
      seed      — 随机种子

    metric_name 约定：
      "reward"  — MuJoCo，越大越好（=-best_fx）
      "min_fx"  — CEC 函数，越小越好（=best_fx）

    错误处理：认证失败时自动降级，benchmark 继续运行，不影响结果保存。

    离线恢复机制：
      - 每个 sub-task 有独立的 pending 文件（路径 /tmp/wandb-pending/pending_{origin}.jsonl）
      - 断网时 log_step 将数据追加写入本地缓冲，并分配全局 FIFO 序号
      - flush_pending（sub-task 开始时）：FIFO 顺序尝试补传，遇到失败立即停止，
        不阻塞训练；成功的条目移除，失败的留在文件中等待下一次 flush
      - ensure_upload（sub-task 结束时）：忽略失败，将所有剩余条目全部上传
      - 新 sub-task 独立重置，已失败 run 的数据留在各自 pending 文件中不互相干扰

    用法：
        logger = WandbLogger(config, task_name="swimmer", algo="bo", seed=42, metric_name="reward")
        logger.flush_pending()          # sub-task 开始时调用
        logger.log_step(budget=100, reward=42.5)   # mujoco
        logger.log_step(budget=100, min_fx=0.001)  # cec
        logger.finish()
        logger.ensure_upload()          # sub-task 结束时调用（兜底）
    """

    _PENDING_DIR = Path("/tmp/wandb-pending")

    def __init__(
        self,
        config: WandbConfig,
        task_name: str,
        algorithm: str,
        metric_name: str = "reward",
        step_metric: str = "budget",
        seed: int = None,
    ):
        self._task_name = task_name
        self._algorithm = algorithm
        self._metric_name = metric_name
        self._step_metric = step_metric
        self._seed = seed
        self._run = None
        self._config = config
        self._disabled = False
        # seed 加入 origin，确保不同 seed 的 run 有独立的 pending 文件和 wandb run
        self._origin = f"{task_name}_{algorithm}_seed{seed}" if seed is not None else f"{task_name}_{algorithm}"

        if not _WANDB_AVAILABLE:
            self._disabled = True
            return

        self._PENDING_DIR.mkdir(parents=True, exist_ok=True)
        self._pending_file = self._PENDING_DIR / f"pending_{self._origin}.jsonl"

    # ── 内部工具 ───────────────────────────────────────────────────────────────

    def _pending_file_for_origin(self, origin: str) -> Path:
        return self._PENDING_DIR / f"pending_{origin}.jsonl"

    def _append_pending(self, entry: dict):
        """追加一条待上传记录到本地缓冲文件（追加写，原子安全）。"""
        try:
            with open(self._pending_file, "a") as f:
                f.write(json.dumps(entry) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except Exception:
            pass

    def _try_upload(self, metrics: dict) -> bool:
        """单次上传尝试。成功返回 True，失败返回 False（不抛异常）。"""
        try:
            self._run.log(metrics, commit=True)
            return True
        except Exception:
            return False

    def _flush_fifo(self, stop_on_fail: bool = True) -> tuple[int, int]:
        """
        FIFO 顺序消费 pending 文件中的条目。

        Args:
            stop_on_fail: True  = 遇到第一次上传失败立即停止（不阻塞训练）
                          False = 忽略失败，全部条目都尝试上传（ensure_upload 用）

        Returns:
            (上传成功数, 剩余未上传数)
        """
        pf = self._pending_file
        if not pf.exists():
            return 0, 0

        pending = []
        for line in open(pf):
            line = line.strip()
            if not line:
                continue
            try:
                pending.append(json.loads(line))
            except Exception:
                continue

        if not pending:
            return 0, 0

        # 按 FIFO 序号排序
        pending.sort(key=lambda e: e.get("seq", 0))

        uploaded, failed = 0, 0
        for entry in pending:
            metrics = entry.get("metrics", {})
            ok = self._try_upload(metrics)
            if ok:
                uploaded += 1
            else:
                failed += 1
                if stop_on_fail:
                    # 第一次失败就停止，剩余未处理的条目保持在文件中
                    break

        # 写回未上传的条目（保持原 FIFO 顺序）
        if pending:
            with open(pf, "w") as f:
                for entry in pending[uploaded:]:
                    f.write(json.dumps(entry) + "\n")
            # 如果全部上传成功则删除文件
            if uploaded == len(pending):
                pf.unlink(missing_ok=True)

        return uploaded, failed

    # ── 公开 API ───────────────────────────────────────────────────────────────

    def flush_pending(self):
        """
        在 sub-task 开始时调用（FIFO 顺序）：
          - 尝试清空残留的旧 pending 数据
          - 遇到上传失败立即停止，不阻塞训练
          - 失败的条目留在 pending 文件中，等待下次 flush 或 ensure_upload
        """
        if self._disabled:
            return
        try:
            self._ensure_run()
        except Exception:
            pass
        if self._run is None:
            return

        ok, rest = self._flush_fifo(stop_on_fail=True)
        if ok > 0:
            print(f"[WandbLogger] flush_pending: 上传了 {ok} 条，剩余 {rest} 条待上传")
        if rest > 0:
            print(f"[WandbLogger] flush_pending: 遇到阻碍，{rest} 条留在缓冲区，"
                  "将在下次 flush 或 ensure_upload 时继续")

    def ensure_upload(self):
        """
        在 sub-task 结束时调用（兜底）：
          - 忽略上传失败，尝试将所有剩余 pending 条目全部上传
          - 适合在每个 run_bbo_*.py 脚本末尾调用，确保不遗漏任何数据
        """
        if self._disabled:
            # wandb 一直不可用，pending 数据只能保留在文件中（下次运行时 flush）
            if self._pending_file.exists():
                print(f"[WandbLogger] ensure_upload: wandb 不可用，"
                      f"{self._pending_file} 中的 {self._count_pending()} 条数据保留在本地，"
                      "下次 flush_pending 时继续尝试上传")
            return

        try:
            self._ensure_run()
        except Exception:
            pass
        if self._run is None:
            if self._pending_file.exists():
                print(f"[WandbLogger] ensure_upload: 无法初始化 wandb，"
                      f"{self._count_pending()} 条数据保留在本地")
            return

        ok, rest = self._flush_fifo(stop_on_fail=False)
        if ok > 0:
            print(f"[WandbLogger] ensure_upload: 上传了 {ok} 条", end="")
            if rest > 0:
                print(f"，仍有 {rest} 条上传失败（网络中断）", end="")
            print()
        elif not self._pending_file.exists():
            print("[WandbLogger] ensure_upload: 无待上传数据")

    def _count_pending(self) -> int:
        """返回 pending 文件中的条目数。"""
        if not self._pending_file.exists():
            return 0
        return sum(1 for line in open(self._pending_file) if line.strip())

    def _ensure_run(self):
        if self._run is not None or self._disabled:
            return
        init_kwargs = self._config.to_wandb_init()
        if not init_kwargs.get("project"):
            self._disabled = True
            return
        os.environ.setdefault("WANDB_DISABLE_CODE_USAGE", "1")
        os.environ.setdefault("WANDB_SYNC_DIR", "/tmp/wandb-sync")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        seed_suffix = f"_seed{self._seed}" if self._seed is not None else ""
        run_name = f"{self._task_name}_{self._algorithm}{seed_suffix}_{ts}"

        # config 写入三个关键维度，方便 wandb UI 按 algo / function / seed 过滤和分组
        config_extra = {
            "algorithm": self._algorithm,
            "task_name": self._task_name,
            "seed": self._seed,
        }
        existing_config = init_kwargs.pop("config", {}) or {}
        init_kwargs["config"] = {**existing_config, **config_extra}

        try:
            self._run = wandb.init(name=run_name, **init_kwargs)
            self._run.define_metric(self._step_metric, step_sync=True)
            self._run.define_metric(self._metric_name, step_metric=self._step_metric)
        except Exception as e:
            print(f"[WandbLogger] wandb 初始化失败: {e}，跳过 wandb 记录")
            self._disabled = True
            self._run = None

    def log_step(self, budget: int, **metrics):
        """
        记录单个 step。

        - 网络正常：直接上传，并尝试 FIFO 补传缓冲区中的旧数据（stop_on_fail=True）
        - 网络中断：将当前数据追加到本地 pending 文件后标记 _disabled=True，
          后续 log_step 会尝试重新连接
        """
        if self._disabled:
            # 尝试重新初始化连接
            self._disabled = False
            self._run = None
            try:
                self._ensure_run()
            except Exception:
                pass
            if self._disabled:
                # wandb 仍不可用，写入本地缓冲
                entry = {
                    "seq": _next_counter(self._origin),
                    "origin": self._origin,
                    "metrics": {self._step_metric: budget, **metrics},
                }
                self._append_pending(entry)
                return

        try:
            self._ensure_run()
            if self._disabled:
                entry = {
                    "seq": _next_counter(self._origin),
                    "origin": self._origin,
                    "metrics": {self._step_metric: budget, **metrics},
                }
                self._append_pending(entry)
                return
            metrics[self._step_metric] = budget
            ok = self._try_upload(metrics)
            if not ok:
                raise RuntimeError("upload failed")

            # 网络正常，FIFO 补传旧数据（stop_on_fail=True，遇到失败立即停止）
            up_ok, up_rest = self._flush_fifo(stop_on_fail=True)
            if up_ok > 0:
                print(f"[WandbLogger] 自动补传了 {up_ok} 条 pending 数据", end="")
                if up_rest > 0:
                    print(f"，{up_rest} 条遇到阻碍暂停", end="")
                print()
        except Exception as e:
            print(f"[WandbLogger] log_step 失败: {e}，写入本地缓冲")
            entry = {
                "seq": _next_counter(self._origin),
                "origin": self._origin,
                "metrics": {self._step_metric: budget, **metrics},
            }
            self._append_pending(entry)
            self._disabled = True

    def log_summary(self, **metrics):
        """记录最终汇总指标"""
        if self._disabled:
            self._disabled = False
            self._run = None
            try:
                self._ensure_run()
            except Exception:
                pass
            if self._disabled:
                return

        try:
            self._ensure_run()
            if self._disabled:
                return
            self._run.log(metrics, commit=True)
        except Exception as e:
            print(f"[WandbLogger] log_summary 失败: {e}，跳过")
            self._disabled = True

    def finish(self):
        if self._run is not None and not self._disabled:
            try:
                self._run.finish(exit_code=0)
            except Exception:
                pass
            self._run = None


# ══════════════════════════════════════════════════════════════════════════════
# 历史 JSON 批量同步
# ══════════════════════════════════════════════════════════════════════════════

def _build_progress_table(
    results: list,
    objective_mode: str = "auto",
) -> list:
    """
    将 progress.json 的 best_fx 列表转换为 wandb 步进记录。

    objective_mode:
      "mujoco"   → metric = "reward" = -best_fx（越大越好）
      "cec"      → metric = "min_fx"  =  best_fx（越小越好）
      "auto"     → 根据目录名自动判断
    """
    rows = []
    for r in results:
        step = r["step"]
        best_fx = r["best_fx"]
        if objective_mode == "mujoco":
            value = -best_fx
            metric_name = "reward"
        else:
            value = best_fx
            metric_name = "min_fx"
        rows.append({
            "budget": step,
            metric_name: value,
            "elapsed_time": r.get("elapsed_time", 0),
        })
    return rows


def _upload_run(
    wandb_api,
    entity: str,
    project: str,
    run_name: str,
    table_rows: list,
    config: dict,
    tags: list,
    notes: str,
    group: str,
):
    """将单个实验的进度数据上传到 wandb"""
    import wandb

    wandb_kwargs = {
        "project": project,
        "name": run_name,
        "config": config,
        "tags": tags,
        "notes": notes,
        "group": group,
        "dir": "/tmp/wandb",
    }
    if entity:
        wandb_kwargs["entity"] = entity

    try:
        run = wandb.init(**wandb_kwargs)
    except Exception as e:
        print(f"[WandbSync] wandb init 失败: {e}，跳过 {run_name}")
        return

    if table_rows:
        metric_name = [k for k in table_rows[0] if k not in ("budget", "elapsed_time")][0]
        for row in table_rows:
            run.log(row)
        run.log({"_final": True})

    # 记录 summary
    if metric_name == "reward":
        values = [r["reward"] for r in table_rows]
        run.summary["final_reward"] = max(values)
        run.summary["best_budget"] = table_rows[np.argmax(values)]["budget"]
    else:  # min_fx（越小越好）
        values = [r["min_fx"] for r in table_rows]
        run.summary["final_min_fx"] = min(values)
        run.summary["best_budget"] = table_rows[np.argmin(values)]["budget"]

    try:
        run.finish()
    except Exception:
        pass


def sync_existing_results(
    result_dir: str,
    config: WandbConfig,
    objective_mode: str = "auto",
    dry_run: bool = False,
):
    """
    扫描 result_dir，将所有 progress.json 批量同步到 wandb。

    objective_mode:
      "mujoco"  — reward = -best_fx
      "cec"     — min_fx = best_fx
      "auto"    — 根据目录名自动判断
    """
    if not _WANDB_AVAILABLE:
        raise RuntimeError("wandb 未安装")

    import wandb
    wandb.login()

    result_path = Path(result_dir)
    if not result_path.exists():
        print(f"[WandbSync] 目录不存在: {result_dir}")
        return

    entity = config.entity or ""
    project = config.project

    runs_uploaded = []
    runs_skipped = []

    for task_dir in sorted(result_path.iterdir()):
        if not task_dir.is_dir():
            continue

        if objective_mode == "auto":
            if "mujoco" in str(result_path):
                mode = "mujoco"
            else:
                mode = "cec"
        else:
            mode = objective_mode

        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue

            progress_file = algo_dir / "progress.json"
            if not progress_file.exists():
                continue

            run_name = f"{task_dir.name}_{algo_dir.name}"

            if dry_run:
                print(f"  [dry-run] 会上传: {run_name}")
                runs_skipped.append(run_name)
                continue

            with open(progress_file) as f:
                data = json.load(f)

            results = data.get("results", [])
            if not results:
                runs_skipped.append(run_name)
                continue

            table_rows = _build_progress_table(results, mode)

            algo_config = {
                "task_name": task_dir.name,
                "algorithm": algo_dir.name,
                "mode": mode,
                "total_budget": results[-1]["step"] if results else 0,
            }

            try:
                _upload_run(
                    wandb_api=None,
                    entity=entity,
                    project=project,
                    run_name=run_name,
                    table_rows=table_rows,
                    config=algo_config,
                    tags=[task_dir.name, algo_dir.name, mode],
                    notes=f"Synced from {progress_file}",
                    group=config.group,
                )
                runs_uploaded.append(run_name)
                print(f"  ✓ {run_name} ({len(table_rows)} steps)")
            except Exception as e:
                print(f"  ✗ {run_name}: {e}")
                runs_skipped.append(run_name)

    print(f"\n完成: 上传 {len(runs_uploaded)} 个, 跳过 {len(runs_skipped)} 个")

    if runs_uploaded:
        print("上传的 run 列表：")
        for r in runs_uploaded:
            print(f"  - {r}")

    return runs_uploaded, runs_skipped


# ══════════════════════════════════════════════════════════════════════════════
# 主动查询 wandb run 历史记录（可选）
# ══════════════════════════════════════════════════════════════════════════════

def fetch_wandb_runs(
    entity: str,
    project: str,
    filters: dict = None,
) -> list:
    """查询 wandb 上的已有 run。"""
    if not _WANDB_AVAILABLE:
        return []

    import wandb
    api = wandb.Api()

    try:
        runs = api.runs(f"{entity}/{project}" if entity else project, filters=filters)
        return runs
    except Exception as e:
        print(f"[WandbSync] 查询失败: {e}")
        return []


# ══════════════════════════════════════════════════════════════════════════════
# 便捷 CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BBO Benchmark → Weights & Biases 同步工具"
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # ── sync-exists ──────────────────────────────────────────────────────────
    p_sync = subparsers.add_parser("sync-exists", help="同步已存在的 JSON 结果")
    p_sync.add_argument("--result-dir", "-d", required=True,
                        help="JSON 结果所在目录（mujoco/results 或 cec_functions/results）")
    p_sync.add_argument("--project", "-p", required=True,
                        help="wandb project 名称")
    p_sync.add_argument("--entity", "-e", default=None,
                        help="wandb entity/username（可选）")
    p_sync.add_argument("--group", "-g", default=None,
                        help="wandb run group（默认当天日期）")
    p_sync.add_argument("--objective-mode", "-m", default="auto",
                        choices=["mujoco", "cec", "auto"],
                        help="目标转换模式，默认 auto 自动判断")
    p_sync.add_argument("--dry-run", action="store_true",
                        help="只打印不上传")

    # ── flush-pending ─────────────────────────────────────────────────────────
    p_flush = subparsers.add_parser("flush-pending",
                                     help="尝试将所有 pending 文件中的数据上传到 wandb")
    p_flush.add_argument("--origin", "-o", default=None,
                         help="只 flush 指定 origin（如 ackley_50d_bo），不指定则 flush 所有")
    p_flush.add_argument("--project", "-p", required=True,
                         help="wandb project 名称")
    p_flush.add_argument("--entity", "-e", default=None,
                         help="wandb entity（可选）")
    p_flush.add_argument("--dry-run", action="store_true",
                         help="只打印不上传")

    # ── fetch ────────────────────────────────────────────────────────────────
    p_fetch = subparsers.add_parser("fetch", help="查询 wandb 上的已有 runs")
    p_fetch.add_argument("--project", "-p", required=True)
    p_fetch.add_argument("--entity", "-e", default=None)

    args = parser.parse_args()

    if args.command == "sync-exists":
        config = WandbConfig(
            entity=args.entity,
            project=args.project,
            group=args.group,
        )
        sync_existing_results(
            result_dir=args.result_dir,
            config=config,
            objective_mode=args.objective_mode,
            dry_run=args.dry_run,
        )

    elif args.command == "flush-pending":
        if not _WANDB_AVAILABLE:
            print("wandb 未安装，无法 flush pending 数据")
            return
        wandb.login()
        pending_dir = Path("/tmp/wandb-pending")
        if not pending_dir.exists():
            print("无 pending 数据")
            return
        patterns = [f"pending_{args.origin}.jsonl"] if args.origin else ["pending_*.jsonl"]
        import glob
        matched = []
        for pat in patterns:
            matched.extend(glob.glob(str(pending_dir / pat)))
        matched = sorted(set(matched))
        if not matched:
            print(f"未找到匹配的 pending 文件: {patterns}")
            return
        for pf_str in matched:
            pf = Path(pf_str)
            origin = pf.stem  # "pending_xxx" → "xxx"
            print(f"\n处理 pending 文件: {pf.name}")
            pending = []
            for line in open(pf):
                line = line.strip()
                if not line:
                    continue
                try:
                    pending.append(json.loads(line))
                except Exception:
                    continue
            if not pending:
                print("  空文件，删除")
                pf.unlink(missing_ok=True)
                continue
            pending.sort(key=lambda e: e.get("seq", 0))
            if args.dry_run:
                print(f"  [dry-run] 会上传 {len(pending)} 条:")
                for e in pending:
                    m = e.get("metrics", {})
                    print(f"    seq={e.get('seq')}  budget={m.get('budget')}")
                continue
            config = WandbConfig(
                entity=args.entity,
                project=args.project,
                group=origin,
            )
            import wandb as _wandb
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            try:
                run = _wandb.init(name=f"flush_{origin}_{ts}", **config.to_wandb_init())
            except Exception as e:
                print(f"  wandb init 失败: {e}，跳过")
                continue
            uploaded, failed = 0, 0
            for entry in pending:
                metrics = entry.get("metrics", {})
                try:
                    run.log(metrics, commit=True)
                    uploaded += 1
                except Exception:
                    failed += 1
            try:
                run.finish()
            except Exception:
                pass
            print(f"  上传 {uploaded}/{len(pending)} 条", end="")
            if failed:
                print(f"，失败 {failed} 条")
            else:
                print()
            if failed == 0:
                pf.unlink(missing_ok=True)

    elif args.command == "fetch":
        runs = fetch_wandb_runs(
            entity=args.entity or "",
            project=args.project,
        )
        print(f"找到 {len(runs)} 个 runs：")
        for r in runs:
            print(f"  - {r.name} | state={r.state} | {r.created_at}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
