#!/usr/bin/env python3
"""
BBO Benchmark Script - 测试各种BBO算法在测试函数上的表现
支持: BO, TuRBO, CMAES, HeSBO, BAxUS, SAASBO, Scalpel
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions.test_functions import (
    Ackley, Rastrigin, Rosenbrock, Griewank, Michalewicz, Schwefel, Levy
)
from utils.wandb_sync import WandbConfig, WandbLogger
from baselines import (
    create_optimizer,
    BaseOptimizer,
    StatsFuncWrapper,
    BayesianOptimizer,
    TurboOptimizer,
    CMAESOptimizer,
    HeSBOOptimizer,
    BAxUSOptimizer,
    SAASBOOptimizer,
    ScalpelOptimizer,
)


TEST_FUNCTIONS = {
    'ackley': Ackley,
    'rastrigin': Rastrigin,
    'rosenbrock': Rosenbrock,
    'griewank': Griewank,
    'michalewicz': Michalewicz,
    'schwefel': Schwefel,
    'levy': Levy,
}

ALGORITHMS = ['bo', 'turbo', 'cmaes', 'hesbo', 'baxus', 'saasbo', 'scalpel']


def _is_gpu_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keys = [
        'no available gpu',
        'gpu scheduler failed',
        'all gpus are occupied',
        'cuda out of memory',
    ]
    return any(k in msg for k in keys)


def get_optimizer_overrides(algorithm: str, dims: int, budget: int) -> Dict:
    """按问题规模返回算法超参数覆盖项。"""
    algo = algorithm.lower()
    cfg: Dict = {}

    if algo == 'bo':
        cfg.update({
            'num_cands': 3000 if dims <= 30 else (5000 if dims <= 100 else 8000),
            'n_initial_points': min(max(dims + 8, 16), 80),
            'max_train_size': min(800, max(200, dims * 4)),
            'training_epochs': 60 if dims <= 100 else 40,
        })
    elif algo == 'turbo':
        cfg.update({
            'n_trusts': 1,
            'max_evals': budget,
        })
    elif algo == 'cmaes':
        popsize = max(8, min(4 + int(3 * np.log(max(dims, 2))), 96))
        restarts = 4 if dims <= 50 else (6 if dims <= 200 else 8)
        cfg.update({
            'popsize': popsize,
            'restarts': restarts,
        })
    elif algo == 'hesbo':
        eff_dim = max(5, int(np.sqrt(dims))) if dims <= 1000 else 20
        cfg.update({
            'eff_dim': eff_dim,
            'n_cands': 2000 if dims <= 100 else 4000,
            'training_epochs': 40,
        })
    elif algo == 'baxus':
        target_dim = max(5, min(20, int(np.sqrt(dims))))
        cfg.update({
            'target_dim': target_dim,
            'n_initial_points': max(target_dim + 5, 20),
            'n_candidates': 5000 if dims <= 100 else 8000,
        })
    elif algo == 'saasbo':
        if dims <= 50:
            warmup_steps, num_samples = 64, 32
        elif dims <= 200:
            warmup_steps, num_samples = 96, 48
        elif dims <= 500:
            warmup_steps, num_samples = 128, 64
        else:
            warmup_steps, num_samples = 64, 32

        cfg.update({
            'warmup_steps': warmup_steps,
            'num_samples': num_samples,
            'n_candidates': 256 if dims <= 100 else 384,
            'n_init': min(max(8, int(0.1 * dims)), 40),
        })
    elif algo == 'scalpel':
        pass

    return cfg


def _combine_run_results(algo_dir: Path, n_runs: int,
                         problem_name: str, algorithm: str):
    """读取所有 run_i/progress.json，对齐 step，计算 mean±std，保存 combined_progress.json。

    同时读取所有 run_i/final_result.json，聚合 best_fx / elapsed_time 的统计量，
    保存为 combined_final_results.json。
    """
    # ── 1. 收集所有 run 的 progress 数据 ──────────────────────────────────────
    all_runs_trajectories = []   # list of list of {step, best_fx, elapsed_time}
    all_runs_final = []           # list of dicts from final_result.json

    for run_idx in range(n_runs):
        run_dir = algo_dir / f"run_{run_idx}"
        prog_file = run_dir / "progress.json"
        final_file = run_dir / "final_result.json"

        if prog_file.exists():
            with open(prog_file) as f:
                data = json.load(f)
            all_runs_trajectories.append(data.get('results', []))
        else:
            all_runs_trajectories.append([])

        if final_file.exists():
            with open(final_file) as f:
                all_runs_final.append(json.load(f))
        else:
            all_runs_final.append({})

    if not all_runs_trajectories or all(traj == [] for traj in all_runs_trajectories):
        return

    # ── 2. 对齐所有 step，取 union ─────────────────────────────────────────────
    all_steps = set()
    for traj in all_runs_trajectories:
        for entry in traj:
            all_steps.add(entry.get('step', 0))
    sorted_steps = sorted(all_steps)

    # 对每个 step，从每个 run 中找最接近的（不超过该 step 的）best_fx
    def _get_value_at_step(traj, step):
        """找 traj 中 step <= target 的最近记录。"""
        last = None
        for entry in traj:
            if entry.get('step', 0) <= step:
                last = entry
            else:
                break
        return last

    # ── 3. 计算 mean ± std ────────────────────────────────────────────────────
    combined_results = []
    for step in sorted_steps:
        fx_values = []
        time_values = []
        for traj in all_runs_trajectories:
            entry = _get_value_at_step(traj, step)
            if entry is not None:
                fx_values.append(entry.get('best_fx', np.nan))
                time_values.append(entry.get('elapsed_time', np.nan))

        if not fx_values:
            continue

        fx_arr = np.array(fx_values, dtype=float)
        combined_results.append({
            'step': step,
            'best_fx_mean': float(np.nanmean(fx_arr)),
            'best_fx_std': float(np.nanstd(fx_arr)),
        })

    # ── 4. 保存 combined_progress.json ─────────────────────────────────────────
    combined_prog = {
        'problem': problem_name,
        'algorithm': algorithm,
        'n_runs': n_runs,
        'results': combined_results,
    }
    with open(algo_dir / 'combined_progress.json', 'w') as f:
        json.dump(combined_prog, f, indent=2)

    # ── 5. 聚合 final results ───────────────────────────────────────────────────
    final_fx = []
    final_time = []
    for fr in all_runs_final:
        if 'best_fx' in fr:
            final_fx.append(fr['best_fx'])
        if 'total_time' in fr:
            final_time.append(fr['total_time'])

    agg = {
        'problem': problem_name,
        'algorithm': algorithm,
        'n_runs': n_runs,
    }
    if final_fx:
        fx_arr = np.array(final_fx, dtype=float)
        agg['best_fx_mean'] = float(np.nanmean(fx_arr))
        agg['best_fx_std'] = float(np.nanstd(fx_arr))
        agg['best_fx_min'] = float(np.nanmin(fx_arr))
        agg['best_fx_max'] = float(np.nanmax(fx_arr))
        agg['best_fx_runs'] = final_fx
    if final_time:
        t_arr = np.array(final_time, dtype=float)
        agg['total_time_mean'] = float(np.nanmean(t_arr))
        agg['total_time_std'] = float(np.nanstd(t_arr))

    with open(algo_dir / 'combined_final_results.json', 'w') as f:
        json.dump(agg, f, indent=2)


class BenchmarkResult:

    def __init__(self, result_dir: str, func_name: str, dims: int,
                 algorithm: str, run_index: int = None):
        """
        Args:
            result_dir: 结果根目录
            func_name: 函数名称
            dims: 问题维度
            algorithm: 算法名称
            run_index: 本次运行的索引（用于多 run 模式，None 表示单次运行）
        """
        self.result_dir = Path(result_dir)
        self.func_name = func_name
        self.dims = dims
        self.algorithm = algorithm
        self.run_index = run_index

        # 多 run 模式下：result/{func}_{dims}d/{algorithm}/run_{i}/
        # 单次运行：result/{func}_{dims}d/{algorithm}/
        if run_index is not None:
            self.algo_dir = self.result_dir / f"{func_name}_{dims}d" / algorithm / f"run_{run_index}"
        else:
            self.algo_dir = self.result_dir / f"{func_name}_{dims}d" / algorithm
        self.algo_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.start_time = None
        self.last_save_step = 0
        self.total_budget = 1000
        self.last_log_step = 0

    def add_result(self, step: int, best_fx: float, elapsed_time: float) -> bool:
        """添加结果"""
        self.results.append({
            'step': step,
            'best_fx': float(best_fx),
            'elapsed_time': float(elapsed_time),
        })

        save_interval = max(self.total_budget // 1000, 1)
        need_save = step - self.last_save_step >= save_interval or step >= self.total_budget

        if need_save:
            self.save()
            self.last_save_step = step

        log_interval = max(self.total_budget // 1000, 1)
        need_log = step - self.last_log_step >= log_interval or step >= self.total_budget

        if need_log:
            self.last_log_step = step

        return need_log

    def _get_total_budget(self) -> int:
        """获取总预算"""
        return self.total_budget

    def set_total_budget(self, budget: int):
        """设置总预算"""
        self.total_budget = budget

    def save(self):
        """保存结果到 JSON（覆盖式）"""
        if not self.results:
            return

        # 多 run 模式下保存为 progress_{run}.json，单次运行为 progress.json
        if self.run_index is not None:
            result_file = self.algo_dir / f"progress_{self.run_index}.json"
        else:
            result_file = self.algo_dir / "progress.json"

        with open(result_file, 'w') as f:
            json.dump({
                'func_name': self.func_name,
                'dims': self.dims,
                'algorithm': self.algorithm,
                'run_index': self.run_index,
                'results': self.results,
                'total_results': len(self.results),
                'last_step': self.results[-1]['step'] if self.results else 0,
                'last_best_fx': self.results[-1]['best_fx'] if self.results else None,
            }, f, indent=2)

    def save_final(self, best_x: np.ndarray, best_fx: float, total_time: float,
                   total_evaluations: int = None):
        """保存最终结果"""
        if self.run_index is not None:
            final_file = self.algo_dir / f"final_result_{self.run_index}.json"
        else:
            final_file = self.algo_dir / "final_result.json"

        if total_evaluations is None:
            total_evaluations = self._get_total_budget()

        with open(final_file, 'w') as f:
            json.dump({
                'func_name': self.func_name,
                'dims': self.dims,
                'algorithm': self.algorithm,
                'run_index': self.run_index,
                'best_x': best_x.tolist() if best_x is not None else None,
                'best_fx': float(best_fx),
                'total_time': float(total_time),
                'total_evaluations': total_evaluations,
                'progress_history': self.results,
            }, f, indent=2)


class FuncWrapper:
    """函数包装器"""

    def __init__(self, func, is_minimizing: bool = True):
        self.func = func
        self.lb = func.lb
        self.ub = func.ub
        self.dims = func.dims
        self.is_minimizing = is_minimizing
        self.sign = 1.0 if is_minimizing else -1.0
        self.call_count = 0

    def __call__(self, x):
        """评估函数"""
        self.call_count += 1
        result = self.func(x)
        if isinstance(result, tuple):
            fx = result[0]
        else:
            fx = result
        return self.sign * fx

    def gen_random_inputs(self, n: int) -> np.ndarray:
        """生成随机输入"""
        return np.random.uniform(self.lb, self.ub, size=(n, self.dims))


def run_single_algorithm(
    algorithm: str,
    func_name: str,
    dims: int,
    budget: int,
    batch_size: int = 1,
    use_gpu: bool = True,
    device: torch.device = None,
    verbose: bool = True,
    result_manager: Optional[BenchmarkResult] = None,
    use_continuous: bool = False,
    wandb_config: Optional[WandbConfig] = None,
    seed: int = None,
    allow_occupied: bool = False,
) -> Tuple[Optional[np.ndarray], float, float, int]:
    """
    运行单个算法的基准测试
    """
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if func_name not in TEST_FUNCTIONS:
        raise ValueError(f"Unknown function: {func_name}. "
                         f"Available: {list(TEST_FUNCTIONS.keys())}")

    test_func = TEST_FUNCTIONS[func_name](dims=dims)
    func_wrapper = FuncWrapper(test_func, is_minimizing=True)

    if result_manager:
        result_manager.set_total_budget(budget)

    scalpel_kwargs = {}
    if algorithm == 'scalpel':
        scalpel_kwargs['func_name'] = func_name
        scalpel_kwargs['use_continuous'] = use_continuous

    optimizer_kwargs = {
        'batch_size': batch_size,
        'use_gpu': use_gpu,
        'device': device,
        'allow_occupied': allow_occupied,
        **get_optimizer_overrides(algorithm, dims, budget),
        **scalpel_kwargs,
    }

    optimizer = create_optimizer(
        algorithm,
        func_wrapper,
        **optimizer_kwargs,
    )

    start_time = time.time()
    total_calls = 0
    best_x = None
    best_fx = float('inf')
    last_log_percent = 0

    wandb_logger = None
    if wandb_config is not None:
        wandb_logger = WandbLogger(
            config=wandb_config,
            task_name=f"{func_name}_{dims}d",
            algorithm=algorithm,
            metric_name="min_fx",
            step_metric="budget",
        )

    try:
        if optimizer is not None:
            _opt_start = time.time()
            with tqdm(total=budget, desc=f"{algorithm}/{func_name}",
                      disable=not verbose, position=0) as pbar:
                while total_calls < budget:
                    n_suggest = min(batch_size, budget - total_calls)
                    x_suggest = optimizer.suggest(n_suggest)
                    fx_suggest = np.array([func_wrapper(xi) for xi in x_suggest])
                    optimizer.observe(x_suggest, fx_suggest)
                    total_calls += n_suggest

                    if func_wrapper.is_minimizing:
                        idx = np.argmin(fx_suggest)
                    else:
                        idx = np.argmax(fx_suggest)

                    if fx_suggest[idx] < best_fx:
                        best_fx = fx_suggest[idx]
                        best_x = x_suggest[idx].copy()

                    pbar.update(n_suggest)

                    if result_manager:
                        should_log = result_manager.add_result(
                            total_calls,
                            best_fx,
                            time.time() - start_time
                        )
                        if wandb_logger is not None:
                            wandb_logger.log_step(budget=total_calls,
                                                  min_fx=float(best_fx))
                        current_percent = int(total_calls / budget * 100)
                        if should_log and current_percent > last_log_percent:
                            elapsed = time.time() - start_time
                            print(f"  [{current_percent:3d}%] step={total_calls}/{budget} | "
                                  f"best_fx={best_fx:.6f} | elapsed={elapsed:.1f}s")
                            last_log_percent = current_percent

            elapsed_time = time.time() - _opt_start
            if wandb_logger is not None:
                wandb_logger.log_summary(final_min_fx=best_fx,
                                         total_time_s=elapsed_time)
        else:
            with tqdm(total=budget, desc=f"random/{func_name}",
                      disable=not verbose, position=0) as pbar:
                while total_calls < budget:
                    n_suggest = min(batch_size, budget - total_calls)
                    x_suggest = func_wrapper.gen_random_inputs(n_suggest)
                    fx_suggest = np.array([func_wrapper(x) for x in x_suggest])
                    total_calls += n_suggest

                    if func_wrapper.is_minimizing:
                        idx = np.argmin(fx_suggest)
                    else:
                        idx = np.argmax(fx_suggest)

                    if fx_suggest[idx] < best_fx:
                        best_fx = fx_suggest[idx]
                        best_x = x_suggest[idx].copy()

                    pbar.update(n_suggest)

                    if result_manager:
                        should_log = result_manager.add_result(
                            total_calls,
                            best_fx,
                            time.time() - start_time
                        )
                        if wandb_logger is not None:
                            wandb_logger.log_step(budget=total_calls,
                                                  min_fx=float(best_fx))
                        current_percent = int(total_calls / budget * 100)
                        if should_log and current_percent > last_log_percent:
                            elapsed = time.time() - start_time
                            print(f"  [{current_percent:3d}%] step={total_calls}/{budget} | "
                                  f"best_fx={best_fx:.6f} | elapsed={elapsed:.1f}s")
                            last_log_percent = current_percent

        elapsed_time = time.time() - start_time

        if wandb_logger is not None:
            wandb_logger.log_summary(
                final_min_fx=best_fx,
                total_time_s=elapsed_time,
            )
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()

    if result_manager:
        result_manager.save_final(best_x, best_fx, elapsed_time,
                                  total_evaluations=total_calls)

    if verbose:
        print(f"{algorithm}/{func_name}({dims}d): "
              f"best_fx={best_fx:.6f}, time={elapsed_time:.2f}s, calls={total_calls}")

    return best_x, best_fx, elapsed_time, total_calls


def run_benchmark(
    algorithms: List[str],
    functions: List[str],
    dims: int = 100,
    budget: int = 10000,
    batch_size: int = 1,
    result_dir: str = "cec_functions/results",
    use_gpu: bool = True,
    verbose: bool = True,
    single_algorithm: Optional[str] = None,
    use_continuous: bool = False,
    wandb_config: Optional[WandbConfig] = None,
    seed: int = None,
    run_times: int = 1,
    allow_occupied: bool = False,
):
    """运行完整的基准测试

    Args:
        run_times: 重复运行次数（每次使用不同的 seed：0, 1, ..., run_times-1）
    """
    device = None

    if single_algorithm:
        algorithms = [single_algorithm]

    if not functions:
        functions = list(TEST_FUNCTIONS.keys())

    # 多 run 模式下，所有 run 使用相同的 problem×algorithm 配置
    seeds = list(range(run_times)) if run_times > 1 else ([seed] if seed is not None else [None])
    n_runs = run_times

    print(f"\n{'='*60}")
    print(f"BBO Benchmark Configuration")
    print(f"{'='*60}")
    print(f"Algorithms: {algorithms}")
    print(f"Functions: {functions}")
    print(f"Dimensions: {dims}")
    print(f"Budget: {budget}")
    print(f"Batch size: {batch_size}")
    print(f"GPU enabled: {use_gpu}")
    print(f"Result directory: {result_dir}")
    print(f"Run times: {n_runs}  (seeds: {seeds})")
    print(f"{'='*60}\n")

    for algorithm in algorithms:
        print(f"\n{'='*40}")
        print(f"Testing algorithm: {algorithm}")
        print(f"{'='*40}")

        for func_name in functions:
            print(f"\n--- {func_name}({dims}d) ---")

            # 多 run 模式：外层循环遍历每个 seed
            for run_idx, actual_seed in enumerate(seeds):
                run_label = f"run_{run_idx}" if n_runs > 1 else "run_0"
                seed_str = str(actual_seed) if actual_seed is not None else "random"
                print(f"  [{run_label}] seed={seed_str}")

                result_manager = BenchmarkResult(
                    result_dir=result_dir,
                    func_name=func_name,
                    dims=dims,
                    algorithm=algorithm,
                    run_index=run_idx if n_runs > 1 else None,
                )

                try:
                    best_x, best_fx, elapsed_time, total_calls = run_single_algorithm(
                        algorithm=algorithm,
                        func_name=func_name,
                        dims=dims,
                        budget=budget,
                        batch_size=batch_size,
                        use_gpu=use_gpu,
                        device=device,
                        verbose=verbose,
                        result_manager=result_manager,
                        use_continuous=use_continuous,
                        wandb_config=wandb_config,
                        seed=actual_seed,
                        allow_occupied=allow_occupied,
                    )
                except Exception as e:
                    print(f"Error running {algorithm}/{func_name} "
                          f"[{run_label}] seed={seed_str}: {e}")
                    import traceback
                    traceback.print_exc()
                    if _is_gpu_unavailable_error(e):
                        raise RuntimeError(
                            f"Stop benchmark: GPU unavailable for "
                            f"{algorithm}/{func_name}."
                        ) from e
                    continue

            # 所有 run 跑完后，生成 combined 文件
            if n_runs > 1:
                algo_dir = Path(result_dir) / f"{func_name}_{dims}d" / algorithm
                problem_name = f"{func_name}_{dims}d"
                _combine_run_results(algo_dir, n_runs, problem_name, algorithm)
                print(f"  -> combined progress saved to {algo_dir}/combined_progress.json")

    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"Results saved to: {result_dir}")
    print(f"{'='*60}")


def plot_results(result_dir: str = "cec_functions/results"):
    """绘制结果图表（支持单次和多 run combined 结果）"""
    try:
        import matplotlib.pyplot as plt

        result_path = Path(result_dir)
        summary_file = result_path / "summary.json"

        func_dirs = [d for d in result_path.iterdir() if d.is_dir()]

        for func_dir in func_dirs:
            plt.figure(figsize=(10, 6))
            has_legend = False

            for algo_dir in func_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                # 优先使用 combined_progress.json
                combined_file = algo_dir / "combined_progress.json"
                if combined_file.exists():
                    with open(combined_file) as f:
                        data = json.load(f)
                    steps = [r['step'] for r in data['results']]
                    means = [r['best_fx_mean'] for r in data['results']]
                    stds  = [r['best_fx_std'] for r in data['results']]
                    plt.plot(steps, means, label=algo_dir.name, alpha=0.8)
                    plt.fill_between(steps,
                                     np.array(means) - np.array(stds),
                                     np.array(means) + np.array(stds),
                                     alpha=0.15)
                    has_legend = True
                    continue

                # 单次运行：直接读 progress.json
                prog_file = algo_dir / "progress.json"
                if not prog_file.exists():
                    continue
                with open(prog_file) as f:
                    data = json.load(f)
                steps  = [r['step'] for r in data['results']]
                values = [r['best_fx'] for r in data['results']]
                plt.plot(steps, values, label=algo_dir.name, alpha=0.7)
                has_legend = True

            if has_legend:
                plt.xlabel('Samples')
                plt.ylabel('Best Function Value')
                plt.title(f'BBO Benchmark: {func_dir.name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(func_dir / "performance.png",
                            dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Plots saved to {result_dir}")

    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")


def main():
    parser = argparse.ArgumentParser(description='BBO Benchmark')

    parser.add_argument('--algorithms', '-a', nargs='+',
                        default=['bo', 'turbo', 'cmaes', 'hesbo', 'baxus', 'saasbo', 'scalpel'],
                        help='Algorithms to test')
    parser.add_argument('--functions', '-f', nargs='+',
                        default=['ackley', 'rastrigin', 'rosenbrock', 'griewank',
                                'michalewicz', 'schwefel', 'levy'],
                        help='Test functions')
    parser.add_argument('--dims', '-d', type=int, default=100,
                        help='Problem dimensions')
    parser.add_argument('--budget', '-b', type=int, default=10000,
                        help='Total evaluation budget')
    parser.add_argument('--batch', '-bt', type=int, default=10,
                        help='Batch size for suggestions')
    parser.add_argument('--continuous', action='store_true',
                        help='Use continuous mode for Scalpel')

    parser.add_argument('--result-dir', '-r', default='cec_functions/results',
                        help='Result directory')
    parser.add_argument('--single', '-s', type=str, default=None,
                        help='Test single algorithm')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU')
    parser.add_argument('--use-GPU-occupied', action='store_true',
                        help='Allow using GPUs that are occupied by other processes but have free memory')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Plot results after benchmark')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--wandb-project', '-wp', default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', '-we', default=None,
                        help='Weights & Biases entity')
    parser.add_argument('--wandb-group', '-wg', default=None,
                        help='Weights & Biases run group')
    parser.add_argument('--wandb-tags', nargs='+', default=[],
                        help='Weights & Biases tags')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for single run (ignored when --run-times > 1)')
    parser.add_argument('--run-times', type=int, default=1,
                        help='Number of repeated runs (default: 1). '
                             'Each run uses a different seed 0..N-1.')

    args = parser.parse_args()

    wandb_config = None
    if args.wandb_project:
        wandb_config = WandbConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            tags=args.wandb_tags,
        )

    results = run_benchmark(
        algorithms=args.algorithms,
        functions=args.functions,
        dims=args.dims,
        budget=args.budget,
        batch_size=args.batch,
        result_dir=args.result_dir,
        use_gpu=not args.no_gpu,
        verbose=args.verbose,
        single_algorithm=args.single,
        use_continuous=args.continuous,
        wandb_config=wandb_config,
        seed=args.seed,
        run_times=args.run_times,
        allow_occupied=args.use_GPU_occupied,
    )

    if args.plot:
        plot_results(args.result_dir)


if __name__ == "__main__":
    main()
