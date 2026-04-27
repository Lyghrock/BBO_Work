#!/usr/bin/env python3
"""
wandb_aggregate.py — 将 wandb 上的多-seed runs 聚合为 mean±std summary runs

用法:
  # 查看帮助
  python wandb_aggregate.py --help

  # 预览（dry-run，不上传）
  python wandb_aggregate.py \
      --entity hongweijun_jack-peking-university \
      --project bbo \
      --runs-dir mujoco/results \
      --dry-run

  # 正式上传聚合结果
  python wandb_aggregate.py \
      --entity hongweijun_jack-peking-university \
      --project bbo \
      --runs-dir mujoco/results

说明:
  - 从 --runs-dir 读取已有的本地 JSON 结果文件（和 wandb 同步脚本配合）
  - 也可以从 wandb API 直接拉数据（加 --from-api）
  - 每个 (task, algorithm) 组合生成 1 条 summary run
  - wandb 端点会得到名为 "{task}_{algorithm}_mean±std" 的 run，
    可以直接用 wandb panels 做对比图
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np

# ── wandb 导入（可选）───────────────────────────────────────────────────────
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# 数据聚合逻辑
# ══════════════════════════════════════════════════════════════════════════════

def load_progress_files(runs_dir: str, objective_mode: str = "mujoco") -> dict:
    """
    扫描 runs_dir，读取所有 progress.json，按 (task, algorithm) 分组。

    Returns:
        grouped: {
            (task_name, algo_name): [
                {"seed": 123, "steps": [0, 100, ...], "values": [v0, v1, ...]},
                ...
            ]
        }
    """
    runs_path = Path(runs_dir)
    grouped = defaultdict(list)

    for task_dir in sorted(runs_path.iterdir()):
        if not task_dir.is_dir():
            continue

        for algo_dir in sorted(task_dir.iterdir()):
            if not algo_dir.is_dir():
                continue

            progress_file = algo_dir / "progress.json"
            if not progress_file.exists():
                continue

            with open(progress_file) as f:
                data = json.load(f)

            results = data.get("results", [])
            if not results:
                continue

            # 读取 seed（如果 progress.json 里有的话）
            seed = data.get("seed", 0)

            # 提取 steps 和 values
            steps = [r["step"] for r in results]
            best_fxs = [r["best_fx"] for r in results]

            if objective_mode == "mujoco":
                # reward = -best_fx（越大越好）
                values = [-fx for fx in best_fxs]
                metric_name = "reward"
            else:
                # min_fx = best_fx（越小越好）
                values = best_fxs
                metric_name = "min_fx"

            grouped[(task_dir.name, algo_dir.name)].append({
                "seed": seed,
                "steps": steps,
                "values": values,
            })

    return dict(grouped)


def fetch_wandb_runs_by_config(entity: str, project: str) -> dict:
    """
    从 wandb API 拉取所有 runs，按 (task_name, algorithm) 分组。
    返回格式同 load_progress_files。
    """
    if not _WANDB_AVAILABLE:
        raise RuntimeError("wandb 未安装")

    api = wandb.Api()
    entity_prefix = f"{entity}/" if entity else ""
    runs = api.runs(f"{entity_prefix}{project}")

    grouped = defaultdict(list)

    for run in runs:
        if run.state == "finished":
            config = run.config or {}
            task_name = config.get("task_name", run.name.split("_")[0])
            algorithm = config.get("algorithm", "")

            if not task_name or not algorithm:
                continue

            history = run.history(samples=500)
            if history is None or history.empty:
                continue

            step_col = "_step" if "_step" in history.columns else "budget"
            metric_col = "reward" if "reward" in history.columns else "min_fx"

            if step_col not in history.columns or metric_col not in history.columns:
                continue

            # 对齐到统一 budget 网格
            steps = history[step_col].dropna().values.tolist()
            values = history[metric_col].dropna().values.tolist()

            if len(steps) < 2:
                continue

            grouped[(task_name, algorithm)].append({
                "seed": run.name.split("_")[-1] if "_" in run.name else 0,
                "steps": steps,
                "values": values,
            })

    return dict(grouped)


def aggregate_runs(grouped: dict) -> dict:
    """
    对每个 (task, algorithm) 组合的多条 runs 做 mean ± std 聚合。

    对齐策略：
      - 以第一条 run 的 budget 网格为基准
      - 其他 run 通过线性插值对齐到该网格
      - 最终统计 mean 和 std

    Returns:
        {
            (task, algo): {
                "steps":     [...aligned budget grid...],
                "mean":      [...mean values...],
                "std":       [...std values...],
                "raw_runs":  n,   # 参与聚合的 run 数量
            }
        }
    """
    aggregated = {}

    for key, runs in grouped.items():
        if len(runs) < 1:
            continue

        # 以第一条 run 为基准网格
        ref_steps = np.array(runs[0]["steps"])
        n_grid = len(ref_steps)

        aligned_values = []
        for run in runs:
            rv = np.array(run["values"])
            if len(rv) == n_grid:
                # 长度完全一致，直接用
                aligned_values.append(rv)
            else:
                # 对齐到 ref_steps（线性插值）
                rs = np.array(run["steps"])
                aligned = np.interp(ref_steps, rs[::-1], rv[::-1])
                aligned_values.append(aligned)

        aligned_values = np.array(aligned_values)  # shape: (n_runs, n_grid)

        aggregated[key] = {
            "steps": ref_steps.tolist(),
            "mean": np.mean(aligned_values, axis=0).tolist(),
            "std": np.std(aligned_values, axis=0).tolist(),
            "raw_runs": len(runs),
        }

    return aggregated


def upload_summary_runs(
    aggregated: dict,
    entity: str,
    project: str,
    base_group: str,
    objective_mode: str,
    dry_run: bool = False,
):
    """
    将聚合结果以专门的 summary run 上传到 wandb。

    每个 (task, algorithm) 上传一条 run：
      - name: {task}_{algorithm}_mean±std
      - config: task_name, algorithm, n_runs, objective_mode
      - metrics: 每一步的 mean, std 上传为独立的 key（如 reward_mean, reward_upper, reward_lower）
      - tags: [task, algorithm, "aggregated", "mean±std"]
    """
    if not _WANDB_AVAILABLE:
        raise RuntimeError("wandb 未安装")

    wandb.login(force=True)

    metric_name = "reward" if objective_mode == "mujoco" else "min_fx"
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    uploaded = []

    for (task_name, algo_name), agg in aggregated.items():
        n_runs = agg["raw_runs"]
        if n_runs < 2:
            print(f"  [skip] {task_name}/{algo_name}: 只有 {n_runs} 条 run，不足以聚合")
            continue

        run_name = f"{task_name}_{algo_name}_mean±std_{ts}"
        group_name = f"{base_group}/aggregated/{task_name}"

        config = {
            "task_name": task_name,
            "algorithm": algo_name,
            "n_runs": n_runs,
            "objective_mode": objective_mode,
            "aggregated_at": ts,
        }

        tags = [task_name, algo_name, "aggregated", "mean±std"]

        if dry_run:
            print(f"  [dry-run] 会创建 run: {run_name}")
            print(f"    group={group_name}, tags={tags}")
            print(f"    steps={len(agg['steps'])} 个点, runs={n_runs}")
            uploaded.append(run_name)
            continue

        wandb_kwargs = {
            "project": project,
            "name": run_name,
            "config": config,
            "tags": tags,
            "notes": f"Mean ± std aggregation of {n_runs} runs",
            "group": group_name,
            "dir": "/tmp/wandb",
        }
        if entity:
            wandb_kwargs["entity"] = entity

        run = wandb.init(**wandb_kwargs)

        steps = agg["steps"]
        mean_vals = agg["mean"]
        std_vals = agg["std"]

        # 上传每一步的 mean ± std
        for i, step in enumerate(steps):
            run.log({
                "budget": step,
                f"{metric_name}_mean": mean_vals[i],
                f"{metric_name}_upper": mean_vals[i] + std_vals[i],
                f"{metric_name}_lower": mean_vals[i] - std_vals[i],
            })

        # summary metrics
        run.summary[f"final_{metric_name}_mean"] = mean_vals[-1]
        run.summary[f"final_{metric_name}_std"] = std_vals[-1]
        run.summary["n_runs"] = n_runs

        run.finish()
        print(f"  ✓ {run_name} (n_runs={n_runs})")
        uploaded.append(run_name)

    return uploaded


# ══════════════════════════════════════════════════════════════════════════════
# CLI 入口
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="将多-seed runs 聚合为 mean±std summary runs 并上传 wandb"
    )
    parser.add_argument("--entity", "-e", default=None,
                        help="wandb entity（如 hongweijun_jack-peking-university）")
    parser.add_argument("--project", "-p", required=True,
                        help="wandb project 名称")
    parser.add_argument("--runs-dir", "-d", default=None,
                        help="本地 results 目录（如 mujoco/results）")
    parser.add_argument("--base-group", "-g", default="bbo-main",
                        help="wandb group 前缀（默认: bbo-main）")
    parser.add_argument("--mode", "-m", default="mujoco",
                        choices=["mujoco", "cec"],
                        help="mujoco: reward=-best_fx; cec: min_fx=best_fx")
    parser.add_argument("--dry-run", action="store_true",
                        help="只打印，不上传")
    parser.add_argument("--from-api", action="store_true",
                        help="从 wandb API 拉数据（需要 --entity），优先于 --runs-dir")

    args = parser.parse_args()

    if not args.runs_dir and not args.from_api:
        parser.error("请指定 --runs-dir 或使用 --from-api 从 wandb API 拉取")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    if args.from_api:
        if not args.entity:
            parser.error("--from-api 需要指定 --entity")
        print(f"从 wandb API 拉取 runs: {args.entity}/{args.project} ...")
        grouped = fetch_wandb_runs_by_config(args.entity, args.project)
    else:
        print(f"从本地文件加载: {args.runs_dir}")
        grouped = load_progress_files(args.runs_dir, args.mode)

    if not grouped:
        print("没有找到任何 runs，退出")
        return

    print(f"共找到 {len(grouped)} 个 (task, algorithm) 组合：")
    for key in sorted(grouped.keys()):
        print(f"  - {key[0]} / {key[1]}: {len(grouped[key])} runs")

    # ── 聚合 ─────────────────────────────────────────────────────────────────
    print("\n聚合中 ...")
    aggregated = aggregate_runs(grouped)

    # ── 上传 ─────────────────────────────────────────────────────────────────
    print(f"\n上传到 wandb ({args.project}) ...")
    uploaded = upload_summary_runs(
        aggregated=aggregated,
        entity=args.entity or "",
        project=args.project,
        base_group=args.base_group,
        objective_mode=args.mode,
        dry_run=args.dry_run,
    )

    print(f"\n完成，共 {'上传' if not args.dry_run else '预览'} {len(uploaded)} 个聚合 run")


if __name__ == "__main__":
    main()
