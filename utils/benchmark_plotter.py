"""
统一绘图模块 - 同时支持 CEC Functions 和 MuJoCo 环境的 benchmark 结果可视化。

两种数据格式自动识别：
  - CEC Functions: {"func_name", "dims", "algorithm", "results":[{step,best_fx,elapsed_time}]}
  - MuJoCo:         {"env_name",  "algorithm", "results":[{step,best_fx,elapsed_time}]}

图片默认保存在对应 JSON 同目录。

用法（作为模块调用）:
    from utils.benchmark_plotter import auto_plot_all

    # CEC 函数结果
    auto_plot_all("cec_functions/results/ackley_100d/")

    # 遍历所有 CEC 结果（自动找到每个 {algo}/progress.json）
    auto_plot_all("cec_functions/results/")

    # MuJoCo 结果
    auto_plot_all("mujoco/results/swimmer/")

    # 遍历整个 mujoco/results
    auto_plot_all("mujoco/results/")
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Literal, Tuple

import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# 内部常量
# ─────────────────────────────────────────────────────────────────────────────

_COLOR_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]
_LINESTYLE_PALETTE = ["-", "--", "-.", ":", "-", "--"]

# ─────────────────────────────────────────────────────────────────────────────
# 核心：自动检测数据格式
# ─────────────────────────────────────────────────────────────────────────────

def _detect_format(data: dict) -> Literal["cec", "mujoco"]:
    if "func_name" in data:
        return "cec"
    elif "env_name" in data:
        return "mujoco"
    raise ValueError(
        f"Unknown JSON format: neither 'func_name' nor 'env_name' found. "
        f"Keys present: {list(data.keys())}"
    )


def _resolve_name(data: dict, fmt: str) -> str:
    if fmt == "cec":
        return f"{data['func_name']}_{data['dims']}d"
    return data["env_name"]


# ─────────────────────────────────────────────────────────────────────────────
# 数据加载
# ─────────────────────────────────────────────────────────────────────────────

def load_progress_json(json_path: str) -> Tuple[dict, str, str]:
    """
    加载单个 progress.json，返回 (data, problem_name, format_str)。

    Returns:
        data: 原始 JSON dict
        problem_name: "func_dim" (cec) 或 "env_name" (mujoco)
        fmt: "cec" | "mujoco"
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    fmt = _detect_format(data)
    name = _resolve_name(data, fmt)
    return data, name, fmt


def load_final_result_json(json_path: str) -> dict:
    """加载单个 final_result.json，返回标准化字段字典。"""
    with open(json_path, "r") as f:
        data = json.load(f)
    fmt = _detect_format(data)

    return {
        "problem_name": _resolve_name(data, fmt),
        "algorithm":    data["algorithm"],
        "best_fx":      data.get("best_fx"),
        "total_time":   data.get("total_time"),
        "total_calls":  data.get("total_calls") or data.get("total_evaluations"),
        "best_x":       data.get("best_x"),
        "format":       fmt,
    }


def collect_algo_results(folder: str) -> Dict[str, List[float]]:
    """
    扫描 folder（支持 {algo}/progress.json 结构），返回
    {algo_name: best_fx_list}。

    适用于同一个 problem（func+dim 或 env）下多个算法的对比。
    """
    results: Dict[str, List[float]] = {}
    folder_path = Path(folder)

    if not folder_path.exists():
        return results

    for algo_dir in sorted(folder_path.iterdir()):
        if not algo_dir.is_dir():
            continue
        progress_file = algo_dir / "progress.json"
        if not progress_file.exists():
            continue
        try:
            with open(progress_file, "r") as f:
                data = json.load(f)
            best_fx_list = [r["best_fx"] for r in data.get("results", [])]
            if best_fx_list:
                results[algo_dir.name] = best_fx_list
        except Exception:
            continue

    return results


def collect_all_experiments(root: str) -> Dict[str, Dict[str, List[float]]]:
    """
    从根目录向下扫描所有 {problem}/{algo}/progress.json，
    返回 {problem_name: {algo_name: best_fx_list}}。

    支持多级目录：
      cec_functions/results/{func}_{dim}d/{algo}/progress.json
      mujoco/results/{env_name}/{algo}/progress.json
    """
    all_results: Dict[str, Dict[str, List[float]]] = {}
    root_path = Path(root)

    if not root_path.exists():
        return all_results

    for problem_dir in sorted(root_path.iterdir()):
        if not problem_dir.is_dir():
            continue

        # 跳过 summary.json 等非 problem 文件夹
        if problem_dir.name in ("__pycache__", ".git", "summary.json"):
            continue

        # 尝试加载 summary.json：如果 problem_dir 本身就是一个汇总文件则跳过
        summary = problem_dir / "summary.json"
        if summary.exists() and not (problem_dir / (list(os.listdir(str(problem_dir)))[0] if os.listdir(str(problem_dir)) else "").isdir()):
            # 如果有 summary.json 但里面没有任何子目录，可能是顶层汇总，跳过
            try:
                with open(summary) as f:
                    sdata = json.load(f)
                # 如果 summary.json 的顶层 key 是算法名而不是 problem 名，说明是顶层汇总，跳过
                if any(k in ("bo", "cmaes", "turbo", "lamcts", "nevergrad") for k in sdata.keys()):
                    continue
            except Exception:
                pass

        algo_results = collect_algo_results(str(problem_dir))
        if algo_results:
            all_results[problem_dir.name] = algo_results

    return all_results


# ─────────────────────────────────────────────────────────────────────────────
# 绘图
# ─────────────────────────────────────────────────────────────────────────────

def _best_fx_smooth(values: List[float], window: int = 1) -> List[float]:
    """返回每个位置的 running-min（单调递减曲线）。"""
    result = []
    cur = float("inf")
    for v in values:
        cur = min(cur, v)
        result.append(cur)
    return result


def plot_convergence_curve(
    results_dict: Dict[str, List[float]],
    save_path: Optional[str] = None,
    title: str = "Convergence Curve",
    xlabel: str = "Function Evaluations",
    ylabel: str = "Best Fitness Value",
    use_running_min: bool = True,
) -> Optional[str]:
    """
    绘制单图多算法收敛曲线。

    Args:
        results_dict: {algo_name: [best_fx_1, best_fx_2, ...]}
        save_path:    图片保存路径，为 None 则只显示
        title/xlabel/ylabel: 图表标签
        use_running_min: 是否对每条曲线做 running-min（显示单调递减）
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, (algo_name, values) in enumerate(results_dict.items()):
        # 至少 1 个点
        x_vals = list(range(1, len(values) + 1))

        y_vals = _best_fx_smooth(values) if use_running_min else values

        color = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
        ls    = _LINESTYLE_PALETTE[i % len(_LINESTYLE_PALETTE)]

        ax.plot(x_vals, y_vals, label=algo_name,
                color=color, linestyle=ls, linewidth=2.0, marker="", alpha=0.85)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    plt.close(fig)
    return save_path


def plot_multi_problem_grid(
    problem_results: Dict[str, Dict[str, List[float]]],
    save_path: Optional[str] = None,
    title: str = "Algorithm Comparison Across Problems",
    use_running_min: bool = True,
) -> Optional[str]:
    """
    绘制多 problem 子图网格，每个子图对应一个 problem，
    子图内有多条算法曲线。

    Args:
        problem_results: {problem_name: {algo_name: best_fx_list}}
        save_path:       图片保存路径
        title:           总标题
        use_running_min: running-min 平滑
    """
    problems = sorted(problem_results.keys())
    n = len(problems)
    if n == 0:
        return None

    # 自适应列数：最多 4 列
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(6 * ncols, 5 * nrows), squeeze=False
    )

    for idx, prob_name in enumerate(problems):
        ax = axes[idx // ncols][idx % ncols]
        algo_dict = problem_results[prob_name]

        for i, (algo_name, values) in enumerate(algo_dict.items()):
            x_vals = list(range(1, len(values) + 1))
            y_vals = _best_fx_smooth(values) if use_running_min else values
            color  = _COLOR_PALETTE[i % len(_COLOR_PALETTE)]
            ls     = _LINESTYLE_PALETTE[i % len(_LINESTYLE_PALETTE)]
            ax.plot(x_vals, y_vals, label=algo_name,
                    color=color, linestyle=ls, linewidth=1.8, alpha=0.85)

        ax.set_xlabel("Eval", fontsize=9)
        ax.set_ylabel("Best FX", fontsize=9)
        ax.set_title(prob_name, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.25, linestyle="--")

    # 隐藏多余的空白子图
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  [saved] {save_path}")

    plt.close(fig)
    return save_path


# ─────────────────────────────────────────────────────────────────────────────
# 智能绘图入口
# ─────────────────────────────────────────────────────────────────────────────

def auto_plot_single_folder(folder: str) -> List[str]:
    """
    对单个 problem 文件夹（如 ackley_100d/ 或 swimmer/）绘图，
    图片保存在 {folder}/{algo}/convergence.png。

    Returns: 保存的图片路径列表
    """
    saved = []
    folder_path = Path(folder)
    if not folder_path.exists():
        return saved

    # ── 1. 绘制每算法单独曲线 ────────────────────────────────────────────────
    for algo_dir in sorted(folder_path.iterdir()):
        if not algo_dir.is_dir():
            continue
        progress_file = algo_dir / "progress.json"
        if not progress_file.exists():
            continue
        try:
            with open(progress_file) as f:
                data = json.load(f)
            fmt = _detect_format(data)
            name = _resolve_name(data, fmt)
            best_fx_list = [r["best_fx"] for r in data.get("results", [])]
            if not best_fx_list:
                continue

            save_path = algo_dir / "convergence.png"
            plot_convergence_curve(
                {algo_dir.name: best_fx_list},
                save_path=str(save_path),
                title=f"{algo_dir.name} on {name} — Convergence",
            )
            saved.append(str(save_path))
        except Exception as e:
            print(f"  [skip] {progress_file}: {e}")
            continue

    # ── 2. 绘制多算法对比曲线 ────────────────────────────────────────────────
    all_algos = collect_algo_results(str(folder_path))
    if len(all_algos) > 1:
        try:
            # 从第一个 progress.json 获取 problem 信息
            first_progress = next(
                (p for p in folder_path.rglob("progress.json")), None
            )
            if first_progress:
                with open(first_progress) as f:
                    meta = json.load(f)
                fmt  = _detect_format(meta)
                prob = _resolve_name(meta, fmt)
            else:
                prob = folder_path.name

            save_path = folder_path / "comparison.png"
            plot_convergence_curve(
                all_algos,
                save_path=str(save_path),
                title=f"Algorithm Comparison on {prob}",
            )
            saved.append(str(save_path))
        except Exception as e:
            print(f"  [skip comparison] {folder_path}: {e}")

    return saved


def auto_plot_all(root: str) -> Dict[str, List[str]]:
    """
    遍历 root 目录，智能为每个 problem 生成图片。
    图片保存位置与对应 progress.json 同目录。

    支持三种调用粒度：
      - 顶层（results/）         → 每个子 problem 文件夹分别绘图
      - problem 级（ackley_100d/）→ 该 problem 下的所有算法绘图 + 对比图
      - algo 级（ackley_100d/bo/）→ 仅该算法的收敛曲线

    Returns:
        {folder_or_problem: [saved_image_paths]}
    """
    root_path = Path(root)
    summary: Dict[str, List[str]] = {}

    if not root_path.exists():
        print(f"[benchmark_plotter] root does not exist: {root}")
        return summary

    # ── 情况 A：传入的是 {problem}/{algo}/ 层级（直接找到 progress.json）────
    if (root_path / "progress.json").exists():
        return {root: auto_plot_single_folder(str(root_path.parent))}

    # ── 情况 B：传入的是 {problem}/ 层级 ───────────────────────────────────
    subdirs = [d for d in root_path.iterdir() if d.is_dir()]
    if subdirs:
        # 检查是否是 problem 层级（包含 algo 子目录）
        has_algo_dirs = any(
            (d / "progress.json").exists() for d in subdirs
        )
        if has_algo_dirs:
            saved = auto_plot_single_folder(root)
            summary[root] = saved
            return summary

    # ── 情况 C：传入的是 results/ 顶层（包含多个 problem 子目录）───────────
    problem_dirs = []
    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in ("__pycache__", ".git"):
            continue
        # 检查是否是 problem 目录（含 algo 子目录）
        has_algo = any(
            (entry / sd.name / "progress.json").exists()
            for sd in entry.iterdir()
            if sd.is_dir()
        )
        if has_algo:
            problem_dirs.append(entry)

    # 如果上面的启发式方法没找到，再兜底扫描
    if not problem_dirs:
        for entry in sorted(root_path.iterdir()):
            if not entry.is_dir():
                continue
            if any((entry / d.name / "progress.json").exists() for d in entry.iterdir() if d.is_dir()):
                problem_dirs.append(entry)

    if not problem_dirs:
        print(f"[benchmark_plotter] no problem directories found under {root}")
        return summary

    for prob_dir in problem_dirs:
        saved = auto_plot_single_folder(str(prob_dir))
        summary[str(prob_dir)] = saved

    # ── 情况 D：整个 root 是 benchmark 顶层 → 额外生成一张汇总大图 ───────
    all_experiments = collect_all_experiments(root)
    if len(all_experiments) > 1:
        grid_save = Path(root) / "_benchmark_overview.png"
        plot_multi_problem_grid(
            all_experiments,
            save_path=str(grid_save),
            title=f"Benchmark Overview — {root_path.name}",
        )
        summary["__overview__"] = [str(grid_save)]

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Benchmark plotter — auto-detects CEC and MuJoCo JSON formats"
    )
    parser.add_argument(
        "--root", "-r", type=str, required=True,
        help="Root directory containing results "
             "(e.g. cec_functions/results/ or mujoco/results/)"
    )
    parser.add_argument(
        "--single", "-s", type=str, default=None,
        help="Plot a single problem folder (e.g. ackley_100d/)"
    )
    parser.add_argument(
        "--dpi", type=int, default=150,
        help="Figure DPI (default: 150)"
    )
    parser.add_argument(
        "--no-smooth", action="store_true",
        help="Disable running-min smoothing (show raw best_fx)"
    )
    args = parser.parse_args()

    import matplotlib
    matplotlib.rcParams["figure.dpi"] = args.dpi

    use_smooth = not args.no_smooth

    if args.single:
        saved = auto_plot_single_folder(args.single)
    else:
        saved_map = auto_plot_all(args.root)
        saved = [p for paths in saved_map.values() for p in paths]

    print(f"\n[done] {len(saved)} figure(s) saved.")
