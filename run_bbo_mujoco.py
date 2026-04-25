#!/usr/bin/env python3
"""
BBO Benchmark Script for MuJoCo Environments
评测各种BBO算法在MuJoCo强化学习任务上的表现
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
from joblib import Parallel, delayed

import multiprocessing
N_CPU = multiprocessing.cpu_count()
N_PARALLEL_JOBS = max(1, N_CPU // 2)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions.mujoco_functions import Swimmer, Hopper, HalfCheetah, Ant, Walker2d, Humanoid
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


def _is_gpu_unavailable_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keys = [
        'no available gpu',
        'gpu scheduler failed',
        'all gpus are occupied',
        'cuda out of memory',
    ]
    return any(k in msg for k in keys)


MUJOCO_ENVIRONMENTS = {
    'swimmer': Swimmer,
    'hopper': Hopper,
    'halfcheetah': HalfCheetah,
    'ant': Ant,
    'walker2d': Walker2d,
    'humanoid': Humanoid,
}

ALGORITHMS = ['bo', 'turbo', 'cmaes', 'hesbo', 'baxus', 'saasbo', 'scalpel']

DEFAULT_BUDGET = 5000
DEFAULT_RESULT_DIR = 'mujoco/results'

ENV_BUDGETS = {
    'swimmer': 2000,
    'hopper': 5000,
    'halfcheetah': 40000,
    'walker2d': 40000,
    'ant': 40000,
    'humanoid': 40000,
}


def get_optimizer_overrides(algorithm: str, dims: int, budget: int) -> Dict:
    algo = algorithm.lower()
    cfg: Dict = {}

    if algo == 'bo':
        cfg.update({
            'num_cands': 4000 if dims <= 100 else 7000,
            'n_initial_points': min(max(dims // 2, 20), 100),
            'max_train_size': min(1000, max(250, dims * 3)),
            'training_epochs': 50 if dims <= 200 else 35,
        })
    elif algo == 'turbo':
        cfg.update({'n_trusts': 1, 'max_evals': budget})
    elif algo == 'cmaes':
        popsize = max(12, min(4 + int(3 * np.log(max(dims, 2))), 128))
        restarts = 5 if dims <= 100 else (7 if dims <= 1000 else 9)
        cfg.update({'popsize': popsize, 'restarts': restarts})
    elif algo == 'hesbo':
        eff_dim = max(10, int(np.sqrt(dims))) if dims <= 1000 else 20
        cfg.update({
            'eff_dim': eff_dim,
            'n_cands': 3000 if dims <= 300 else 5000,
            'training_epochs': 35,
        })
    elif algo == 'baxus':
        target_dim = max(8, min(24, int(np.sqrt(dims))))
        cfg.update({
            'target_dim': target_dim,
            'n_initial_points': max(target_dim + 6, 24),
            'n_candidates': 6000 if dims <= 300 else 9000,
        })
    elif algo == 'saasbo':
        if dims <= 100:
            warmup_steps, num_samples = 64, 32
        elif dims <= 500:
            warmup_steps, num_samples = 96, 48
        else:
            warmup_steps, num_samples = 48, 24
        cfg.update({
            'warmup_steps': warmup_steps,
            'num_samples': num_samples,
            'n_candidates': 256 if dims <= 300 else 192,
            'n_init': min(max(10, int(0.08 * dims)), 40),
        })

    return cfg


def _combine_run_results(algo_dir: Path, n_runs: int,
                         problem_name: str, algorithm: str):
    """读取所有 run_i/progress.json，对齐 step，计算 reward mean±std，
    保存 combined_progress.json + combined_final_results.json。
    """
    all_trajs = []
    all_finals = []

    for run_idx in range(n_runs):
        run_dir = algo_dir / f"run_{run_idx}"
        prog_file = run_dir / f"progress_{run_idx}.json"
        final_file = run_dir / f"final_result_{run_idx}.json"

        if prog_file.exists():
            with open(prog_file) as f:
                all_trajs.append(json.load(f).get('results', []))
        else:
            all_trajs.append([])

        if final_file.exists():
            with open(final_file) as f:
                all_finals.append(json.load(f))
        else:
            all_finals.append({})

    if all(traj == [] for traj in all_trajs):
        return

    # union of all step values
    all_steps = set()
    for traj in all_trajs:
        for entry in traj:
            all_steps.add(entry.get('step', 0))
    sorted_steps = sorted(all_steps)

    def _value_at(traj, step):
        last = None
        for entry in traj:
            if entry.get('step', 0) <= step:
                last = entry
            else:
                break
        return last

    combined_results = []
    for step in sorted_steps:
        reward_vals = []
        for traj in all_trajs:
            entry = _value_at(traj, step)
            if entry is not None:
                reward_vals.append(entry.get('best_reward', np.nan))

        if not reward_vals:
            continue

        r_arr = np.array(reward_vals, dtype=float)
        combined_results.append({
            'step': step,
            'best_reward_mean': float(np.nanmean(r_arr)),
            'best_reward_std': float(np.nanstd(r_arr)),
        })

    with open(algo_dir / 'combined_progress.json', 'w') as f:
        json.dump({
            'problem': problem_name,
            'algorithm': algorithm,
            'n_runs': n_runs,
            'results': combined_results,
        }, f, indent=2)

    # final aggregation
    final_rewards = []
    final_times = []
    for fr in all_finals:
        if 'best_reward' in fr:
            final_rewards.append(fr['best_reward'])
        if 'total_time' in fr:
            final_times.append(fr['total_time'])

    agg = {
        'problem': problem_name,
        'algorithm': algorithm,
        'n_runs': n_runs,
    }
    if final_rewards:
        r_arr = np.array(final_rewards, dtype=float)
        agg['best_reward_mean'] = float(np.nanmean(r_arr))
        agg['best_reward_std'] = float(np.nanstd(r_arr))
        agg['best_reward_min'] = float(np.nanmin(r_arr))
        agg['best_reward_max'] = float(np.nanmax(r_arr))
        agg['best_reward_runs'] = final_rewards
    if final_times:
        t_arr = np.array(final_times, dtype=float)
        agg['total_time_mean'] = float(np.nanmean(t_arr))
        agg['total_time_std'] = float(np.nanstd(t_arr))

    with open(algo_dir / 'combined_final_results.json', 'w') as f:
        json.dump(agg, f, indent=2)


class MuJoCoBenchmarkResult:

    def __init__(self, result_dir: str, env_name: str, algorithm: str,
                 run_index: int = None):
        self.result_dir = Path(result_dir)
        self.env_name = env_name
        self.algorithm = algorithm
        self.run_index = run_index

        if run_index is not None:
            self.algo_dir = self.result_dir / env_name / algorithm / f"run_{run_index}"
        else:
            self.algo_dir = self.result_dir / env_name / algorithm
        self.algo_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.last_save_step = 0
        self.total_budget = DEFAULT_BUDGET
        self.last_log_step = 0

    def add_result(self, step: int, best_fx: float, elapsed_time: float) -> bool:
        """best_fx 是 loss（负 reward）"""
        best_reward = -float(best_fx)
        self.results.append({
            'step': step,
            'best_reward': best_reward,
            'best_loss': float(best_fx),
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
        return self.total_budget

    def set_total_budget(self, budget: int):
        self.total_budget = budget

    def save(self):
        if not self.results:
            return

        if self.run_index is not None:
            result_file = self.algo_dir / f"progress_{self.run_index}.json"
        else:
            result_file = self.algo_dir / "progress.json"

        with open(result_file, 'w') as f:
            json.dump({
                'env_name': self.env_name,
                'algorithm': self.algorithm,
                'run_index': self.run_index,
                'results': self.results,
                'total_results': len(self.results),
                'last_step': self.results[-1]['step'] if self.results else 0,
                'last_best_reward': self.results[-1]['best_reward'] if self.results else None,
            }, f, indent=2)

    def save_final(self, best_x: np.ndarray, best_fx: float, total_time: float):
        if self.run_index is not None:
            final_file = self.algo_dir / f"final_result_{self.run_index}.json"
        else:
            final_file = self.algo_dir / "final_result.json"

        best_reward = -float(best_fx)
        with open(final_file, 'w') as f:
            json.dump({
                'env_name': self.env_name,
                'algorithm': self.algorithm,
                'run_index': self.run_index,
                'best_x': best_x.tolist() if best_x is not None else None,
                'best_reward': best_reward,
                'best_loss': float(best_fx),
                'total_time': float(total_time),
                'total_evaluations': self._get_total_budget(),
                'progress_history': self.results,
            }, f, indent=2)


class MuJoCoFuncWrapper:
    """MuJoCo 环境函数包装器 - 统一接口"""

    def __init__(self, env, is_minimizing: bool = True):
        self.env = env
        self.lb = env.lb
        self.ub = env.ub
        self.dims = env.dims
        self.is_minimizing = is_minimizing
        self.sign = 1.0 if is_minimizing else -1.0
        self.call_count = 0
        self.func = lambda x: self.env(x)

    def __call__(self, x):
        self.call_count += 1
        result = self.env(x)
        return self.sign * result

    def gen_random_inputs(self, n: int) -> np.ndarray:
        return np.random.uniform(self.lb, self.ub, size=(n, self.dims))


def run_single_algorithm(
    algorithm: str,
    env_name: str,
    budget: int,
    batch_size: int = 1,
    use_gpu: bool = True,
    device: torch.device = None,
    verbose: bool = True,
    result_manager: Optional[MuJoCoBenchmarkResult] = None,
    seed: int = None,
    use_continuous: bool = True,
    wandb_config: Optional[WandbConfig] = None,
    allow_occupied: bool = False,
) -> Tuple[Optional[np.ndarray], float, float, int]:
    """运行单个算法的基准测试"""
    if env_name not in MUJOCO_ENVIRONMENTS:
        raise ValueError(f"Unknown environment: {env_name}. "
                         f"Available: {list(MUJOCO_ENVIRONMENTS.keys())}")

    np.random.seed(seed)
    torch.manual_seed(seed)

    mujoco_env = MUJOCO_ENVIRONMENTS[env_name]()
    func_wrapper = MuJoCoFuncWrapper(mujoco_env, is_minimizing=True)

    if result_manager:
        result_manager.set_total_budget(budget)

    optimizer_kwargs = {
        'batch_size': batch_size,
        'use_gpu': use_gpu,
        'device': device,
        'allow_occupied': allow_occupied,
        'max_evals': budget,
        **get_optimizer_overrides(algorithm, mujoco_env.dims, budget),
    }

    if algorithm == 'scalpel':
        optimizer_kwargs['func_name'] = env_name
        optimizer_kwargs['use_continuous'] = use_continuous

    optimizer = create_optimizer(
        algorithm,
        func_wrapper,
        **optimizer_kwargs,
    )

    wandb_logger = None
    if wandb_config is not None:
        wandb_logger = WandbLogger(
            config=wandb_config,
            task_name=env_name,
            algorithm=algorithm,
            metric_name="reward",
            step_metric="budget",
        )

    start_time = time.time()
    total_calls = 0
    best_x = None
    best_fx = float('inf')
    last_log_percent = 0

    try:
        if optimizer is not None and hasattr(optimizer, 'optimize'):
            with tqdm(total=budget, desc=f"{algorithm}/{env_name}",
                      disable=not verbose, position=0) as pbar:
                while total_calls < budget:
                    n_suggest = min(batch_size, budget - total_calls)
                    x_suggest = optimizer.suggest(n_suggest)
                    fx_suggest = np.array(
                        Parallel(n_jobs=N_PARALLEL_JOBS, prefer="processes")(
                            delayed(func_wrapper)(x) for x in x_suggest
                        )
                    )
                    total_calls += len(x_suggest)
                    optimizer.observe(x_suggest, fx_suggest)

                    if func_wrapper.is_minimizing:
                        idx = np.argmin(fx_suggest)
                    else:
                        idx = np.argmax(fx_suggest)

                    if fx_suggest[idx] < best_fx:
                        best_fx = fx_suggest[idx]
                        best_x = x_suggest[idx].copy()

                    pbar.update(len(x_suggest))

                    if result_manager:
                        should_log = result_manager.add_result(
                            total_calls, best_fx, time.time() - start_time)
                        if wandb_logger is not None:
                            wandb_logger.log_step(budget=total_calls,
                                                reward=-best_fx)
                        current_percent = int(total_calls / budget * 100)
                        if should_log and current_percent > last_log_percent:
                            elapsed = time.time() - start_time
                            print(f"\n  [{current_percent:3d}%] step={total_calls}/{budget} | "
                                  f"reward={-best_fx:.4f} | elapsed={elapsed:.1f}s")
                            last_log_percent = current_percent
        else:
            with tqdm(total=budget, desc=f"random/{env_name}",
                      disable=not verbose, position=0) as pbar:
                while total_calls < budget:
                    n_suggest = min(batch_size, budget - total_calls)
                    x_suggest = func_wrapper.gen_random_inputs(n_suggest)
                    fx_suggest = np.array(
                        Parallel(n_jobs=N_PARALLEL_JOBS, prefer="processes")(
                            delayed(func_wrapper)(x) for x in x_suggest
                        )
                    )
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
                            total_calls, best_fx, time.time() - start_time)
                        if wandb_logger is not None:
                            wandb_logger.log_step(budget=total_calls,
                                                reward=-best_fx)
                        current_percent = int(total_calls / budget * 100)
                        if should_log and current_percent > last_log_percent:
                            elapsed = time.time() - start_time
                            print(f"\n  [{current_percent:3d}%] step={total_calls}/{budget} | "
                                  f"reward={-best_fx:.4f} | elapsed={elapsed:.1f}s")
                            last_log_percent = current_percent

        elapsed_time = time.time() - start_time

        if wandb_logger is not None:
            wandb_logger.log_summary(final_reward=-best_fx,
                                    total_time=elapsed_time)
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()

    if result_manager:
        result_manager.save_final(best_x, best_fx, elapsed_time)

    if verbose:
        print(f"{algorithm}/{env_name}: reward={-best_fx:.4f}, "
              f"time={elapsed_time:.2f}s, calls={total_calls}")

    return best_x, best_fx, elapsed_time, total_calls


def run_benchmark(
    algorithms: List[str],
    environments: List[str],
    budget: int = DEFAULT_BUDGET,
    batch_size: int = 1,
    result_dir: str = DEFAULT_RESULT_DIR,
    use_gpu: bool = True,
    verbose: bool = True,
    single_algorithm: Optional[str] = None,
    seed: int = None,
    use_continuous: bool = True,
    wandb_config: Optional[WandbConfig] = None,
    use_per_env_budget: bool = True,
    run_times: int = 1,
    allow_occupied: bool = False,
):
    """运行完整的 MuJoCo 基准测试

    Args:
        run_times: 重复运行次数（每次使用不同的 seed：0, 1, ..., run_times-1）
    """
    device = None

    if single_algorithm:
        algorithms = [single_algorithm]

    if not environments:
        environments = list(MUJOCO_ENVIRONMENTS.keys())

    seeds = list(range(run_times)) if run_times > 1 else ([seed] if seed is not None else [None])
    n_runs = run_times

    print(f"\n{'='*60}")
    print(f"MuJoCo BBO Benchmark Configuration")
    print(f"{'='*60}")
    print(f"Algorithms: {algorithms}")
    print(f"Environments: {environments}")
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

        for env_name in environments:
            print(f"\n--- {env_name} ---")

            effective_budget = budget
            if use_per_env_budget and env_name in ENV_BUDGETS:
                effective_budget = ENV_BUDGETS[env_name]

            for run_idx, actual_seed in enumerate(seeds):
                run_label = f"run_{run_idx}" if n_runs > 1 else "run_0"
                seed_str = str(actual_seed) if actual_seed is not None else "random"
                print(f"  [{run_label}] seed={seed_str}")

                result_manager = MuJoCoBenchmarkResult(
                    result_dir=result_dir,
                    env_name=env_name,
                    algorithm=algorithm,
                    run_index=run_idx if n_runs > 1 else None,
                )

                try:
                    best_x, best_fx, elapsed_time, total_calls = run_single_algorithm(
                        algorithm=algorithm,
                        env_name=env_name,
                        budget=effective_budget,
                        batch_size=batch_size,
                        use_gpu=use_gpu,
                        device=device,
                        verbose=verbose,
                        result_manager=result_manager,
                        seed=actual_seed,
                        use_continuous=use_continuous,
                        wandb_config=wandb_config,
                        allow_occupied=allow_occupied,
                    )
                except Exception as e:
                    print(f"Error running {algorithm}/{env_name} "
                          f"[{run_label}] seed={seed_str}: {e}")
                    import traceback
                    traceback.print_exc()
                    if _is_gpu_unavailable_error(e):
                        raise RuntimeError(
                            f"Stop benchmark: GPU unavailable for "
                            f"{algorithm}/{env_name}."
                        ) from e
                    continue

            if n_runs > 1:
                algo_dir = Path(result_dir) / env_name / algorithm
                _combine_run_results(algo_dir, n_runs, env_name, algorithm)
                print(f"  -> combined progress saved to {algo_dir}/combined_progress.json")

    print(f"\n{'='*60}")
    print("MuJoCo Benchmark completed!")
    print(f"Results saved to: {result_dir}")
    print(f"{'='*60}")


def plot_results(result_dir: str = DEFAULT_RESULT_DIR):
    """绘制结果图表（支持单次和多 run combined 结果）"""
    try:
        import matplotlib.pyplot as plt

        result_path = Path(result_dir)
        env_dirs = [d for d in result_path.iterdir() if d.is_dir()]

        for env_dir in env_dirs:
            plt.figure(figsize=(10, 6))
            has_legend = False

            for algo_dir in env_dir.iterdir():
                if not algo_dir.is_dir():
                    continue

                combined_file = algo_dir / "combined_progress.json"
                if combined_file.exists():
                    with open(combined_file) as f:
                        data = json.load(f)
                    steps = [r['step'] for r in data['results']]
                    means = [r['best_reward_mean'] for r in data['results']]
                    stds  = [r['best_reward_std'] for r in data['results']]
                    plt.plot(steps, means, label=algo_dir.name, alpha=0.8)
                    plt.fill_between(steps,
                                     np.array(means) - np.array(stds),
                                     np.array(means) + np.array(stds),
                                     alpha=0.15)
                    has_legend = True
                    continue

                prog_file = algo_dir / "progress.json"
                if not prog_file.exists():
                    continue
                with open(prog_file) as f:
                    data = json.load(f)
                steps  = [r['step'] for r in data['results']]
                values = [r.get('best_reward', -r.get('best_fx', 0.0))
                          for r in data['results']]
                plt.plot(steps, values, label=algo_dir.name, alpha=0.7)
                has_legend = True

            if has_legend:
                plt.xlabel('Samples')
                plt.ylabel('Best Reward')
                plt.title(f'MuJoCo BBO Benchmark: {env_dir.name}')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(env_dir / "performance.png",
                            dpi=150, bbox_inches='tight')
            plt.close()

        print(f"Plots saved to {result_dir}")

    except ImportError:
        print("matplotlib not installed. Skipping plot generation.")


def main():
    parser = argparse.ArgumentParser(description='MuJoCo BBO Benchmark')

    parser.add_argument('--algorithms', '-a', nargs='+',
                        default=['bo', 'turbo', 'cmaes', 'hesbo', 'baxus', 'saasbo', 'scalpel'],
                        help='Algorithms to test')
    parser.add_argument('--environments', '-e', nargs='+',
                        default=['swimmer', 'hopper'],
                        help='MuJoCo environments')
    parser.add_argument('--budget', '-b', type=int, default=DEFAULT_BUDGET,
                        help='Total evaluation budget')
    parser.add_argument('--batch', '-bt', type=int, default=1,
                        help='Batch size for suggestions')
    parser.add_argument('--seed', '-s', type=int, default=None,
                        help='Random seed for single run (ignored when --run-times > 1)')
    parser.add_argument('--continuous', action='store_true', default=True,
                        help='Use continuous mode for Scalpel (default: True)')
    parser.add_argument('--discrete', dest='continuous', action='store_false',
                        help='Use discrete mode for Scalpel')

    parser.add_argument('--result-dir', '-r', default=DEFAULT_RESULT_DIR,
                        help='Result directory')
    parser.add_argument('--single', '-si', type=str, default=None,
                        help='Test single algorithm')
    parser.add_argument('--env-single', '-es', type=str, default=None,
                        help='Test single environment')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU')
    parser.add_argument('--use-GPU-occupied', action='store_true',
                        help='Allow using GPUs that are occupied by other processes but have free memory')
    parser.add_argument('--plot', '-p', action='store_true',
                        help='Plot results after benchmark')
    parser.add_argument('--verbose', '-v', action='store_true', default=True,
                        help='Verbose output')
    parser.add_argument('--render', '-rnd', action='store_true', default=False,
                        help='Render animations after benchmark')
    parser.add_argument('--wandb-project', '-wp', default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', '-we', default=None,
                        help='Weights & Biases entity')
    parser.add_argument('--wandb-group', '-wg', default=None,
                        help='Weights & Biases run group')
    parser.add_argument('--wandb-tags', nargs='+', default=[],
                        help='Weights & Biases tags')
    parser.add_argument('--no-per-env-budget', action='store_true',
                        help='Disable per-environment budget')
    parser.add_argument('--run-times', type=int, default=1,
                        help='Number of repeated runs (default: 1). '
                             'Each run uses a different seed 0..N-1.')

    args = parser.parse_args()

    environments = [args.env_single] if args.env_single else args.environments

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
        environments=environments,
        budget=args.budget,
        batch_size=args.batch,
        result_dir=args.result_dir,
        use_gpu=not args.no_gpu,
        verbose=args.verbose,
        single_algorithm=args.single,
        seed=args.seed,
        use_continuous=args.continuous,
        wandb_config=wandb_config,
        use_per_env_budget=not args.no_per_env_budget,
        run_times=args.run_times,
        allow_occupied=args.use_GPU_occupied,
    )

    if args.plot:
        plot_results(args.result_dir)

    if args.render:
        from utils.render_mujoco import render_all_results
        print("\n" + "="*60)
        print("Rendering animations...")
        print("="*60 + "\n")
        render_all_results(args.result_dir)


if __name__ == "__main__":
    main()
