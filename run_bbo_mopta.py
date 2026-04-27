#!/usr/bin/env python3
"""
Mopta08 Benchmark Script
高维约束黑盒优化问题的 BBO 算法评测
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from baselines import create_optimizer
from functions.mopta08_wrapper import create_mopta08_benchmark, resolve_mopta08_executable
from utils.wandb_sync import WandbConfig, WandbLogger


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
    algo = algorithm.lower()
    cfg: Dict = {}

    if algo == 'bo':
        cfg.update({
            'num_cands': 6000,
            'n_initial_points': min(max(dims // 2, 24), 96),
            'max_train_size': min(1000, max(250, dims * 4)),
            'training_epochs': 50,
        })
    elif algo == 'turbo':
        cfg.update({'n_trusts': 1, 'max_evals': budget})
    elif algo == 'cmaes':
        popsize = max(12, min(4 + int(3 * np.log(max(dims, 2))), 128))
        cfg.update({'popsize': popsize, 'restarts': 6})
    elif algo == 'hesbo':
        cfg.update({'eff_dim': 12, 'n_cands': 4000, 'training_epochs': 35})
    elif algo == 'baxus':
        cfg.update({'target_dim': 12, 'n_initial_points': 24, 'n_candidates': 7000})
    elif algo == 'saasbo':
        cfg.update({
            'warmup_steps': 96,
            'num_samples': 48,
            'n_candidates': 256,
            'n_init': 28,
        })
    elif algo == 'scalpel':
        pass

    return cfg


def _combine_run_results(algo_dir: Path, seeds: List[int],
                         problem_name: str, algorithm: str):
    """读取所有 seed 的 progress_seed_{N}.json，计算 mean±std，保存 combined 文件。"""
    all_trajs = []
    all_finals = []

    for seed in seeds:
        prog_file = algo_dir / f"progress_seed_{seed}.json"
        final_file = algo_dir / f"final_result_seed_{seed}.json"

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
        fx_vals = []
        for traj in all_trajs:
            entry = _value_at(traj, step)
            if entry is not None:
                fx_vals.append(entry.get('best_fx', np.nan))

        if not fx_vals:
            continue

        fx_arr = np.array(fx_vals, dtype=float)
        combined_results.append({
            'step': step,
            'best_fx_mean': float(np.nanmean(fx_arr)),
            'best_fx_std': float(np.nanstd(fx_arr)),
        })

    with open(algo_dir / 'combined_progress.json', 'w') as f:
        json.dump({
            'problem': problem_name,
            'algorithm': algorithm,
            'n_runs': len(seeds),
            'seeds': seeds,
            'results': combined_results,
        }, f, indent=2)

    final_fx = []
    final_time = []
    for fr in all_finals:
        if 'best_fx' in fr:
            final_fx.append(fr['best_fx'])
        if 'time' in fr:
            final_time.append(fr['time'])

    agg = {
        'problem': problem_name,
        'algorithm': algorithm,
        'n_runs': len(seeds),
        'seeds': seeds,
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
        agg['time_mean'] = float(np.nanmean(t_arr))
        agg['time_std'] = float(np.nanstd(t_arr))

    with open(algo_dir / 'combined_final_results.json', 'w') as f:
        json.dump(agg, f, indent=2)


class BenchmarkResult:
    """Mopta08 结果管理器"""

    def __init__(self, result_dir: str, algorithm: str, actual_seed: int = None):
        self.result_dir = Path(result_dir)
        self.algorithm = algorithm
        self.actual_seed = actual_seed

        # 多 run 模式：result/mopta08/{algo}/progress_seed_{N}.json
        if actual_seed is not None:
            self.algo_dir = self.result_dir / 'mopta08' / algorithm
        else:
            self.algo_dir = self.result_dir / 'mopta08' / algorithm
        self.algo_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.total_budget = 1000
        self.last_save_step = 0

    def set_total_budget(self, budget: int):
        self.total_budget = int(budget)

    def add_result(
        self,
        step: int,
        best_fx: float,
        elapsed: float,
        best_objective: Optional[float] = None,
        best_violations: Optional[int] = None,
        best_max_violation: Optional[float] = None,
    ):
        row = {
            'step': int(step),
            'best_fx': float(best_fx),
            'elapsed': float(elapsed),
        }
        if best_objective is not None:
            row['best_objective'] = float(best_objective)
        if best_violations is not None:
            row['num_violations'] = int(best_violations)
        if best_max_violation is not None:
            row['max_violation'] = float(best_max_violation)

        self.results.append(row)

        interval = max(self.total_budget // 1000, 1)
        if step - self.last_save_step >= interval or step >= self.total_budget:
            self.save()
            self.last_save_step = step

    def save(self):
        if not self.results:
            return

        if self.actual_seed is not None:
            result_file = self.algo_dir / f"progress_seed_{self.actual_seed}.json"
        else:
            result_file = self.algo_dir / "progress.json"

        with open(result_file, 'w') as f:
            json.dump(
                {
                    'benchmark': 'mopta08',
                    'algorithm': self.algorithm,
                    'seed': self.actual_seed,
                    'results': self.results,
                },
                f,
                indent=2,
            )

    def save_final(
        self,
        best_x: Optional[np.ndarray],
        best_fx: float,
        total_time: float,
        best_objective: Optional[float] = None,
        best_violations: Optional[int] = None,
        best_max_violation: Optional[float] = None,
    ):
        if self.actual_seed is not None:
            final_file = self.algo_dir / f"final_result_seed_{self.actual_seed}.json"
        else:
            final_file = self.algo_dir / "final_result.json"

        payload = {
            'benchmark': 'mopta08',
            'algorithm': self.algorithm,
            'seed': self.actual_seed,
            'best_x': best_x.tolist() if best_x is not None else None,
            'best_fx': float(best_fx),
            'time': float(total_time),
            'budget': self.total_budget,
            'progress': self.results,
        }
        if best_objective is not None:
            payload['best_objective'] = float(best_objective)
        if best_violations is not None:
            payload['num_violations'] = int(best_violations)
        if best_max_violation is not None:
            payload['max_violation'] = float(best_max_violation)

        with open(final_file, 'w') as f:
            json.dump(payload, f, indent=2)


def run_single(
    algorithm: str,
    budget: int,
    batch_size: int,
    use_gpu: bool,
    gpu_id: Optional[int],
    verbose: bool,
    result_manager: Optional[BenchmarkResult],
    wandb_config: Optional[WandbConfig],
    mopta_executable: Optional[str],
    constraint_penalty: float,
    seed: int = None,
    allow_occupied: bool = False,
) -> Tuple[float, Optional[float], Optional[int], Optional[float]]:
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    func_wrapper = create_mopta08_benchmark(
        executable_path=mopta_executable,
        constraint_penalty=constraint_penalty,
    )
    if result_manager:
        result_manager.set_total_budget(budget)

    optimizer = None
    if algorithm != 'random':
        scalpel_kwargs = {}
        if algorithm == 'scalpel':
            scalpel_kwargs = {
                'func_name': 'mopta08',
                'use_continuous': False,
            }
        optimizer = create_optimizer(
            algorithm,
            func_wrapper,
            batch_size=batch_size,
            use_gpu=use_gpu,
            gpu_id=gpu_id,
            allow_occupied=allow_occupied,
            **get_optimizer_overrides(algorithm, func_wrapper.dims, budget),
            **scalpel_kwargs,
        )

    start_time = time.time()
    total_calls = 0

    best_fx = float('inf')
    best_x = None
    best_objective = None
    best_violations = None
    best_max_violation = None
    current_violations = None
    current_max_violation = None

    wandb_logger = None
    if wandb_config is not None:
        wandb_logger = WandbLogger(
            config=wandb_config,
            task_name='mopta08',
            algorithm=algorithm,
            metric_name='min_fx',
            step_metric='budget',
            seed=seed,
        )
        wandb_logger.flush_pending()

    try:
        with tqdm(total=budget, desc=f"{algorithm}/mopta08", disable=not verbose) as pbar:
            while total_calls < budget:
                n_suggest = min(batch_size, budget - total_calls)

                if optimizer is not None:
                    x_suggest = optimizer.suggest(n_suggest)
                else:
                    x_suggest = func_wrapper.gen_random_inputs(n_suggest)

                fx_suggest = np.array([func_wrapper(x) for x in x_suggest],
                                      dtype=float)
                total_calls += len(x_suggest)

                if optimizer is not None:
                    optimizer.observe(x_suggest, fx_suggest)

                idx = int(np.argmin(fx_suggest))
                current_best_in_batch_fx = float(fx_suggest[idx])
                current_metrics = None
                current_violations = None
                current_max_violation = None
                try:
                    current_metrics = func_wrapper.evaluate_with_metrics(x_suggest[idx])
                    current_violations = current_metrics.get('num_violations')
                    current_max_violation = current_metrics.get('max_violation')
                except Exception:
                    pass

                if current_best_in_batch_fx < best_fx:
                    best_fx = current_best_in_batch_fx
                    best_x = x_suggest[idx].copy()
                    best_objective = current_metrics.get('objective') if current_metrics else None
                    best_violations = current_violations
                    best_max_violation = current_max_violation

                pbar.update(len(x_suggest))

                if result_manager is not None:
                    result_manager.add_result(
                        total_calls,
                        best_fx,
                        time.time() - start_time,
                        best_objective,
                        best_violations,
                        best_max_violation,
                    )

                if wandb_logger is not None:
                    payload = {
                        'budget': total_calls,
                        'min_fx': float(best_fx),
                    }
                    if current_violations is not None:
                        payload['num_violations'] = int(current_violations)
                    if current_max_violation is not None:
                        payload['max_violation'] = float(current_max_violation)
                    if best_objective is not None:
                        payload['objective'] = float(best_objective)
                    wandb_logger.log_step(**payload)

        elapsed = time.time() - start_time

        if wandb_logger is not None:
            summary_payload = {
                'final_min_fx': float(best_fx),
                'total_time_s': float(elapsed),
            }
            if best_objective is not None:
                summary_payload['best_objective'] = float(best_objective)
            if best_violations is not None:
                summary_payload['num_violations'] = int(best_violations)
            if best_max_violation is not None:
                summary_payload['max_violation'] = float(best_max_violation)
            wandb_logger.log_summary(**summary_payload)
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()
            wandb_logger.ensure_upload()

    if result_manager is not None:
        result_manager.save_final(
            best_x,
            best_fx,
            elapsed,
            best_objective,
            best_violations,
            best_max_violation,
        )

    if verbose:
        obj_str = f"{best_objective:.6f}" if best_objective is not None else 'nan'
        vio_str = str(best_violations) if best_violations is not None else 'nan'
        print(
            f"{algorithm}/mopta08: best_fx={best_fx:.6f}, "
            f"objective={obj_str}, violations={vio_str}, time={elapsed:.1f}s"
        )

    return best_fx, best_objective, best_violations, best_max_violation


def run_benchmark(
    algorithms: List[str],
    budget: int,
    batch_size: int,
    result_dir: str,
    use_gpu: bool,
    verbose: bool,
    gpu_id: Optional[int],
    wandb_config: Optional[WandbConfig],
    mopta_executable: Optional[str],
    constraint_penalty: float,
    seeds: List[int] = None,
    allow_occupied: bool = False,
):
    """运行 Mopta08 基准测试

    Args:
        seeds: 空格分隔的 seed 列表，如 [14, 888, 92, 473, 546]。
               每个 seed 独立跑一次，结果存为 progress_seed_{N}.json。
               未传入时，随机生成 5 个 seed。
        allow_occupied: 是否允许使用有空闲显存但已被占用的 GPU。
    """
    all_results = {}

    # seeds 逻辑：传入 seeds > 0 → 每个 seed 独立跑一次；未传入 / [] → 随机生成 1 个
    if seeds is not None and len(seeds) > 0:
        actual_seeds = seeds
    else:
        actual_seeds = [random.randint(0, 2**31 - 1)]
    n_runs = len(actual_seeds)

    print(f"\n{'=' * 50}")
    print('Mopta08 Benchmark')
    print(f"{'=' * 50}")
    print(f"Algorithms: {algorithms}")
    print(f"Budget: {budget}")
    print(f"Constraint penalty: {constraint_penalty}")
    print(f"Run times: {n_runs}  (seeds: {actual_seeds})")
    print(f"{'=' * 50}\n")

    for algo in algorithms:
        print(f"\n--- {algo} ---")

        for actual_seed in actual_seeds:
            print(f"  [seed={actual_seed}]")

            result_mgr = BenchmarkResult(
                result_dir=result_dir,
                algorithm=algo,
                actual_seed=actual_seed,
            )

            try:
                best_fx, best_objective, best_violations, best_max_violation = run_single(
                    algorithm=algo,
                    budget=budget,
                    batch_size=batch_size,
                    use_gpu=use_gpu,
                    gpu_id=gpu_id,
                    verbose=verbose,
                    result_manager=result_mgr,
                    wandb_config=wandb_config,
                    mopta_executable=mopta_executable,
                    constraint_penalty=constraint_penalty,
                    seed=actual_seed,
                    allow_occupied=allow_occupied,
                )
                if n_runs == 1:
                    all_results[algo] = {
                        'mopta08': {
                            'best_fx': best_fx,
                            'objective': best_objective,
                            'num_violations': best_violations,
                            'max_violation': best_max_violation,
                        }
                    }
            except Exception as e:
                print(f"Error: {e}")
                if _is_gpu_unavailable_error(e):
                    raise RuntimeError(
                        f"Stop benchmark: GPU unavailable for {algo}/mopta08."
                    ) from e
                if n_runs == 1:
                    all_results[algo] = {'mopta08': {'error': str(e)}}

        # generate combined files
        if n_runs > 1:
            algo_dir = Path(result_dir) / 'mopta08' / algo
            _combine_run_results(algo_dir, actual_seeds, 'mopta08', algo)
            print(f"  -> combined progress saved to {algo_dir}/combined_progress.json")

    if n_runs == 1:
        summary_file = Path(result_dir) / 'summary.json'
        existing = {}
        if summary_file.exists():
            try:
                existing = json.loads(summary_file.read_text())
            except Exception:
                pass

        for algo, result in all_results.items():
            if algo not in existing:
                existing[algo] = {}
            existing[algo].update(result)

        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(existing, indent=2))

    print(f"\n{'=' * 50}")
    print(f"Done! Results: {result_dir}")
    print(f"{'=' * 50}")


def list_benchmarks(mopta_executable: Optional[str]):
    print('=' * 50)
    print('Mopta Benchmarks')
    print('=' * 50)
    print('  mopta08      | 124D | constrained real-world engineering')

    try:
        resolved = resolve_mopta08_executable(mopta_executable)
        print(f"Executable: {resolved}")
    except Exception as e:
        print(f"Executable: NOT FOUND ({e})")


def main():
    parser = argparse.ArgumentParser(description='Mopta08 Benchmark')
    parser.add_argument('-a', '--algorithms', nargs='+', default=ALGORITHMS,
                       help='Algorithms')
    parser.add_argument('--budget', type=int, default=1000, help='Evaluation budget')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--result-dir', default='mopta/results')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=None)
    parser.add_argument('--use-GPU-occupied', action='store_true',
                        help='Allow using GPUs that are occupied by other processes but have free memory')
    parser.add_argument('--mopta-executable', default=None,
                        help='Path to Mopta08 executable')
    parser.add_argument('--constraint-penalty', type=float, default=10.0,
                        help='Penalty coefficient for positive constraints')
    parser.add_argument('--wandb-project', '-wp', default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', '-we', default=None,
                        help='Weights & Biases entity')
    parser.add_argument('--wandb-group', '-wg', default=None,
                        help='Weights & Biases run group')
    parser.add_argument('--wandb-tags', nargs='+', default=[],
                        help='Weights & Biases tags')
    parser.add_argument('--list', action='store_true',
                        help='List benchmark and executable status')
    parser.add_argument('--seed', type=int, nargs='+', default=None,
                        help='Space-separated seeds, e.g. "14 888 92 473 546".')
    parser.add_argument('--run-times', type=int, default=5,
                        help='[Deprecated] Number of runs is now determined by --seed count or default 5.')
    args = parser.parse_args()

    if args.list:
        list_benchmarks(args.mopta_executable)
        return

    wandb_config = None
    if args.wandb_project:
        wandb_config = WandbConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            tags=args.wandb_tags,
        )

    seeds = args.seed

    run_benchmark(
        algorithms=args.algorithms,
        budget=args.budget,
        batch_size=args.batch,
        result_dir=args.result_dir,
        use_gpu=not args.no_gpu,
        verbose=True,
        gpu_id=args.gpu_id,
        wandb_config=wandb_config,
        mopta_executable=args.mopta_executable,
        constraint_penalty=args.constraint_penalty,
        seeds=seeds,
        allow_occupied=args.use_GPU_occupied,
    )


if __name__ == '__main__':
    main()
