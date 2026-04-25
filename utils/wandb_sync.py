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

class WandbLogger:
    """
    将 benchmark 的实时进度记录到 wandb。

    metric_name 约定：
      "reward"  — MuJoCo，越大越好（=-best_fx）
      "min_fx"  — CEC 函数，越小越好（=best_fx）

    错误处理：认证失败时自动降级，benchmark 继续运行，不影响结果保存。

    用法：
        logger = WandbLogger(config, task_name="swimmer", algo="bo", metric_name="reward")
        logger.log_step(budget=100, reward=42.5)   # mujoco
        logger.log_step(budget=100, min_fx=0.001)  # cec
        logger.finish()
    """

    def __init__(
        self,
        config: WandbConfig,
        task_name: str,
        algorithm: str,
        metric_name: str = "reward",
        step_metric: str = "budget",
    ):
        if not _WANDB_AVAILABLE:
            self._disabled = True
            self._run = None
            return

        self.task_name = task_name
        self.algorithm = algorithm
        self.metric_name = metric_name
        self.step_metric = step_metric
        self._run = None
        self._config = config
        self._disabled = False

    def _ensure_run(self):
        if self._run is not None or self._disabled:
            return
        init_kwargs = self._config.to_wandb_init()
        if not init_kwargs.get("project"):
            self._disabled = True
            return
        # 强制 wandb 同步写入，避免异步 buffer 在进程异常退出时丢失数据
        os.environ.setdefault("WANDB_DISABLE_CODE_USAGE", "1")
        os.environ.setdefault("WANDB_SYNC_DIR", "/tmp/wandb-sync")
        ts = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{self.task_name}_{self.algorithm}_{ts}"
        try:
            self._run = wandb.init(name=run_name, **init_kwargs)
            self._run.define_metric(self.step_metric, step_sync=True)
            self._run.define_metric(self.metric_name, step_metric=self.step_metric)
        except Exception as e:
            print(f"[WandbLogger] wandb 初始化失败: {e}，跳过 wandb 记录")
            self._disabled = True
            self._run = None

    def log_step(self, budget: int, **metrics):
        """记录单个 step。认证失败时静默跳过。"""
        if self._disabled:
            return
        try:
            self._ensure_run()
            if self._disabled:
                return
            metrics[self.step_metric] = budget
            self._run.log(metrics, commit=True)
        except Exception as e:
            print(f"[WandbLogger] log_step 失败: {e}，跳过")
            self._disabled = True

    def log_summary(self, **metrics):
        """记录最终汇总指标"""
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
