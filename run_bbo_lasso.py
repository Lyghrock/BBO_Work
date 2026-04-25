#!/usr/bin/env python3
"""
Lasso-Bench Benchmark Script
高维稀疏优化问题的 BBO 算法评测

包含两个真实任务:
- dna: 180D 真实数据集
- rcv1: 19959D 真实数据集
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions.lasso_wrapper import (
    create_lasso_benchmark,
    get_benchmark_dims,
    get_adaptive_budget,
    LASSO_BENCHMARKS,
)
from baselines import create_optimizer
from utils.wandb_sync import WandbConfig, WandbLogger


def get_optimizer_overrides(algorithm: str, dims: int, budget: int) -> dict:
    algo = algorithm.lower()
    cfg = {}

    if algo == 'bo':
        cfg.update({
            'num_cands': 5000 if dims <= 300 else 9000,
            'n_initial_points': min(max(dims // 3, 20), 120),
            'max_train_size': min(1200, max(300, dims * 2)),
            'training_epochs': 45 if dims <= 300 else 30,
        })
    elif algo == 'turbo':
        cfg.update({'n_trusts': 1, 'max_evals': budget})
    elif algo == 'cmaes':
        popsize = max(16, min(4 + int(3 * np.log(max(dims, 2))), 160))
        restarts = 6 if dims <= 500 else 9
        cfg.update({'popsize': popsize, 'restarts': restarts})
    elif algo == 'hesbo':
        eff_dim = max(10, int(np.sqrt(dims))) if dims <= 1000 else 20
        cfg.update({'eff_dim': eff_dim, 'n_cands': 4000, 'training_epochs': 30})
    elif algo == 'baxus':
        target_dim = max(10, min(28, int(np.sqrt(dims))))
        cfg.update({'target_dim': target_dim,
                    'n_initial_points': max(target_dim + 8, 30),
                    'n_candidates': 9000})
    elif algo == 'saasbo':
        if dims <= 300:
            warmup_steps, num_samples = 80, 40
        else:
            warmup_steps, num_samples = 48, 24
        cfg.update({
            'warmup_steps': warmup_steps,
            'num_samples': num_samples,
            'n_candidates': 192,
            'n_init': min(max(10, int(0.08 * dims)), 48),
        })

    return cfg


def _combine_run_results(algo_dir: Path, n_runs: int,
                         problem_name: str, algorithm: str):
    """读取所有 run_i/progress.json，对齐 step，计算 best_fx mean±std，
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
            'n_runs': n_runs,
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
        agg['time_mean'] = float(np.nanmean(t_arr))
        agg['time_std'] = float(np.nanstd(t_arr))

    with open(algo_dir / 'combined_final_results.json', 'w') as f:
        json.dump(agg, f, indent=2)


class BenchmarkResult:
    """结果管理器"""

    def __init__(self, result_dir: str, bench_name: str, algorithm: str,
                 run_index: int = None):
        self.result_dir = Path(result_dir)
        self.bench_name = bench_name
        self.algorithm = algorithm
        self.run_index = run_index

        if run_index is not None:
            self.algo_dir = self.result_dir / bench_name / algorithm / f"run_{run_index}"
        else:
            self.algo_dir = self.result_dir / bench_name / algorithm
        self.algo_dir.mkdir(parents=True, exist_ok=True)

        self.results = []
        self.total_budget = 1000
        self.last_save_step = 0
        self.last_log_step = 0

    def add_result(self, step: int, best_fx: float, elapsed: float,
                   mspe: float = None, fscore: float = None) -> bool:
        record = {'step': step, 'best_fx': float(best_fx), 'elapsed': float(elapsed)}
        if mspe is not None:
            record['mspe'] = float(mspe)
        if fscore is not None:
            record['fscore'] = float(fscore)
        self.results.append(record)

        interval = max(self.total_budget // 1000, 1)
        need_save = step - self.last_save_step >= interval or step >= self.total_budget
        if need_save:
            self.save()
            self.last_save_step = step
        return need_save

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
                'bench': self.bench_name,
                'algorithm': self.algorithm,
                'run_index': self.run_index,
                'results': self.results,
            }, f, indent=2)

    def save_final(self, best_x: np.ndarray, best_fx: float, total_time: float,
                   mspe: float = None, fscore: float = None):
        if self.run_index is not None:
            final_file = self.algo_dir / f"final_result_{self.run_index}.json"
        else:
            final_file = self.algo_dir / "final_result.json"

        data = {
            'bench': self.bench_name,
            'algorithm': self.algorithm,
            'run_index': self.run_index,
            'best_x': best_x.tolist() if best_x is not None else None,
            'best_fx': float(best_fx),
            'time': float(total_time),
            'budget': self.total_budget,
            'progress': self.results,
        }
        if mspe is not None:
            data['mspe'] = float(mspe)
        if fscore is not None:
            data['fscore'] = float(fscore)
        with open(final_file, 'w') as f:
            json.dump(data, f, indent=2)


def run_single(
    algorithm: str,
    bench_name: str,
    budget: int,
    batch_size: int = 10,
    use_gpu: bool = True,
    gpu_id: int = None,
    verbose: bool = True,
    result_manager: BenchmarkResult = None,
    wandb_config: Optional[WandbConfig] = None,
    seed: int = None,
    allow_occupied: bool = False,
) -> Tuple[float, float, float]:
    """运行单个测试"""
    if seed is not None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    func_wrapper = create_lasso_benchmark(bench_name)
    if result_manager:
        result_manager.set_total_budget(budget)

    optimizer = None
    if algorithm != 'random':
        scalpel_kwargs = {}
        if algorithm == 'scalpel':
            scalpel_kwargs = {
                'func_name': bench_name,
                'use_continuous': True,
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
    best_fx = float('inf')
    best_x = None
    best_mspe = None
    best_fscore = None
    total_calls = 0

    wandb_logger = None
    if wandb_config is not None:
        wandb_logger = WandbLogger(
            config=wandb_config,
            task_name=bench_name,
            algorithm=algorithm,
            metric_name="min_fx",
            step_metric="budget",
        )

    try:
        with tqdm(total=budget, desc=f"{algorithm}/{bench_name}",
                  disable=not verbose) as pbar:
            while total_calls < budget:
                n_suggest = min(batch_size, budget - total_calls)

                if optimizer:
                    x_suggest = optimizer.suggest(n_suggest)
                else:
                    x_suggest = func_wrapper.gen_random_inputs(n_suggest)

                fx_suggest = np.array([func_wrapper(x) for x in x_suggest])
                total_calls += len(x_suggest)

                if optimizer:
                    optimizer.observe(x_suggest, fx_suggest)

                idx = np.argmin(fx_suggest)
                if fx_suggest[idx] < best_fx:
                    best_fx = fx_suggest[idx]
                    best_x = x_suggest[idx].copy()
                    try:
                        metrics = func_wrapper.evaluate_with_metrics(best_x)
                        best_mspe = metrics.get('mspe')
                        best_fscore = metrics.get('fscore')
                    except Exception:
                        pass

                pbar.update(len(x_suggest))

                if result_manager:
                    result_manager.add_result(total_calls, best_fx,
                                             time.time() - start_time,
                                             best_mspe, best_fscore)

                if wandb_logger is not None:
                    payload = {
                        'budget': total_calls,
                        'min_fx': float(best_fx),
                    }
                    if best_mspe is not None:
                        payload['mspe'] = float(best_mspe)
                    if best_fscore is not None:
                        payload['fscore'] = float(best_fscore)
                    wandb_logger.log_step(**payload)

        elapsed = time.time() - start_time

        if wandb_logger is not None:
            summary_payload = {
                'final_min_fx': float(best_fx),
                'total_time_s': float(elapsed),
            }
            if best_mspe is not None:
                summary_payload['final_mspe'] = float(best_mspe)
            if best_fscore is not None:
                summary_payload['final_fscore'] = float(best_fscore)
            wandb_logger.log_summary(**summary_payload)
    finally:
        if wandb_logger is not None:
            wandb_logger.finish()

    if result_manager:
        result_manager.save_final(best_x, best_fx, elapsed,
                                 best_mspe, best_fscore)

    if verbose:
        mspe_str = f"{best_mspe:.4f}" if best_mspe is not None else "nan"
        fscore_str = f"{best_fscore:.4f}" if best_fscore is not None else "nan"
        print(f"{algorithm}/{bench_name}: best_fx={best_fx:.6f}, "
              f"mspe={mspe_str}, fscore={fscore_str}, time={elapsed:.1f}s")

    return best_fx, best_mspe, best_fscore


def run_benchmark(
    algorithms: List[str],
    benchmarks: List[str],
    budget: int = None,
    batch_size: int = 10,
    result_dir: str = "lasso_bench/results",
    use_gpu: bool = True,
    verbose: bool = True,
    gpu_id: int = None,
    wandb_config: Optional[WandbConfig] = None,
    seed: int = None,
    run_times: int = 1,
    allow_occupied: bool = False,
):
    """运行完整 benchmark

    Args:
        run_times: 重复运行次数（每次使用不同的 seed：0, 1, ..., run_times-1）
    """
    all_results = {}

    seeds = list(range(run_times)) if run_times > 1 else ([seed] if seed is not None else [None])
    n_runs = run_times

    print(f"\n{'='*50}")
    print("Lasso-Bench Benchmark")
    print(f"{'='*50}")
    print(f"Algorithms: {algorithms}")
    print(f"Benchmarks: {benchmarks}")
    print(f"Run times: {n_runs}  (seeds: {seeds})")
    print(f"{'='*50}\n")

    def _is_gpu_unavailable_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        keys = [
            'no available gpu',
            'gpu scheduler failed',
            'all gpus are occupied',
            'cuda out of memory',
        ]
        return any(k in msg for k in keys)

    for algo in algorithms:
        print(f"\n--- {algo} ---")
        algo_results = {}

        for bench in benchmarks:
            dims = get_benchmark_dims(bench)
            actual_budget = budget if budget else get_adaptive_budget(dims)
            print(f"\n{bench} ({dims}D) budget={actual_budget}")

            for run_idx, actual_seed in enumerate(seeds):
                run_label = f"run_{run_idx}" if n_runs > 1 else "run_0"
                seed_str = str(actual_seed) if actual_seed is not None else "random"
                print(f"  [{run_label}] seed={seed_str}")

                result_mgr = BenchmarkResult(
                    result_dir=result_dir,
                    bench_name=bench,
                    algorithm=algo,
                    run_index=run_idx if n_runs > 1 else None,
                )

                try:
                    best_fx, mspe, fscore = run_single(
                        algorithm=algo,
                        bench_name=bench,
                        budget=actual_budget,
                        batch_size=batch_size,
                        use_gpu=use_gpu,
                        gpu_id=gpu_id,
                        verbose=verbose,
                        result_manager=result_mgr,
                        wandb_config=wandb_config,
                        seed=actual_seed,
                        allow_occupied=allow_occupied,
                    )
                    # store per-run result (only for n_runs==1 summary compat)
                    if n_runs == 1:
                        algo_results[bench] = {
                            'best_fx': best_fx,
                            'mspe': mspe,
                            'fscore': fscore,
                        }
                except Exception as e:
                    print(f"Error: {e}")
                    if _is_gpu_unavailable_error(e):
                        raise RuntimeError(
                            f"Stop benchmark: GPU unavailable for {algo}/{bench}."
                        ) from e
                    if n_runs == 1:
                        algo_results[bench] = {'error': str(e)}

            # generate combined files after all runs for this (algo, bench)
            if n_runs > 1:
                algo_dir = Path(result_dir) / bench / algo
                _combine_run_results(algo_dir, n_runs, bench, algo)
                print(f"  -> combined progress saved to {algo_dir}/combined_progress.json")

        all_results[algo] = algo_results

    # save summary (only for n_runs==1; for multi-run combined files replace it)
    summary_file = Path(result_dir) / "summary.json"
    existing = {}
    if summary_file.exists() and n_runs == 1:
        try:
            existing = json.loads(summary_file.read_text())
        except Exception:
            pass

    if n_runs == 1:
        for algo, results in all_results.items():
            if algo not in existing:
                existing[algo] = {}
            existing[algo].update(results)

        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(existing, indent=2))

    print(f"\n{'='*50}")
    print(f"Done! Results: {result_dir}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description='Lasso-Bench Benchmark')
    parser.add_argument('-a', '--algorithms', nargs='+',
                        default=['bo', 'turbo', 'cmaes', 'hesbo', 'baxus', 'saasbo', 'scalpel'],
                        help='Algorithms')
    parser.add_argument('-b', '--benchmarks', nargs='+',
                        default=['dna', 'rcv1'],
                        help='Benchmarks')
    parser.add_argument('--budget', type=int, default=None, help='Evaluation budget')
    parser.add_argument('--batch', type=int, default=10, help='Batch size')
    parser.add_argument('--result-dir', default='lasso_bench/results')
    parser.add_argument('--no-gpu', action='store_true')
    parser.add_argument('--gpu-id', type=int, default=None)
    parser.add_argument('--use-GPU-occupied', action='store_true',
                        help='Allow using GPUs that are occupied by other processes but have free memory')
    parser.add_argument('--wandb-project', '-wp', default=None,
                        help='Weights & Biases project name')
    parser.add_argument('--wandb-entity', '-we', default=None,
                        help='Weights & Biases entity')
    parser.add_argument('--wandb-group', '-wg', default=None,
                        help='Weights & Biases run group')
    parser.add_argument('--wandb-tags', nargs='+', default=[],
                        help='Weights & Biases tags')
    parser.add_argument('--list', action='store_true', help='List available benchmarks')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for single run (ignored when --run-times > 1)')
    parser.add_argument('--run-times', type=int, default=1,
                        help='Number of repeated runs (default: 1). '
                             'Each run uses a different seed 0..N-1.')
    args = parser.parse_args()

    if args.list:
        from functions.lasso_wrapper import list_benchmarks
        list_benchmarks()
        return

    wandb_config = None
    if args.wandb_project:
        wandb_config = WandbConfig(
            entity=args.wandb_entity,
            project=args.wandb_project,
            group=args.wandb_group,
            tags=args.wandb_tags,
        )

    run_benchmark(
        algorithms=args.algorithms,
        benchmarks=args.benchmarks,
        budget=args.budget,
        batch_size=args.batch,
        result_dir=args.result_dir,
        use_gpu=not args.no_gpu,
        gpu_id=args.gpu_id,
        wandb_config=wandb_config,
        seed=args.seed,
        run_times=args.run_times,
        allow_occupied=args.use_GPU_occupied,
    )


if __name__ == "__main__":
    main()
