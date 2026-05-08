#!/usr/bin/env python3
"""
Scalpel 时间对比测试脚本

对比两种实现：
1. 原版 fluxer + main.py (scalpel/original/)
2. 包装版 scalpel_core + scalpel_opt.py (scalpel/)

使用方法:
    python test_scalpel_timing.py --func ackley --dims 10 --budget 200 --seed 42
"""
import argparse
import os
import sys
import time
import random
import numpy as np
import torch

# 添加路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scalpel'))

# 禁用 wandb 和其他花里胡哨的东西
os.environ['WANDB_MODE'] = 'disabled'
os.environ['NO_WANDB'] = '1'

# ==================== 原版 fluxer 包装器 ====================

def run_original_fluxer(func_name, dims, budget, seed, use_continuous=True):
    """运行原版 fluxer + main.py"""
    from scalpel.original.main import (
        Ackley, Rastrigin, Rosenbrock, Griewank, Michalewicz, Schwefel, Levy,
        Surrogate, Scalpel, ModelTrainer, tracker
    )

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Original] Using device: {device}", flush=True)

    # 初始化函数
    if func_name == 'ackley':
        f = Ackley(dims=dims)
    elif func_name == 'rastrigin':
        f = Rastrigin(dims=dims)
    elif func_name == 'rosenbrock':
        f = Rosenbrock(dims=dims)
    elif func_name == 'griewank':
        f = Griewank(dims=dims)
    elif func_name == 'michalewicz':
        f = Michalewicz(dims=dims)
    elif func_name == 'schwefel':
        f = Schwefel(dims=dims)
    elif func_name == 'levy':
        f = Levy(dims=dims)
    else:
        raise ValueError(f"Unknown function: {func_name}")

    # 初始化 tracker
    method = 'Continuous-Scalpel' if use_continuous else 'Discrete-Scalpel'
    fx = Surrogate(dims=dims, name=method + '-' + func_name, f=f, iters=budget)

    # 初始采样
    input_X = np.random.uniform(f.lb[0], f.ub[0], size=(200, dims))
    input_y2 = np.array([fx(i)[1] for i in input_X])
    input_X = np.array(input_X)
    print(f"[Original] Initial data collection: {len(input_X)} points", flush=True)

    # 主循环
    rollout_round = 200 if func_name in ['ackley', 'rastrigin'] else 100
    ratio = 0.1 if func_name == 'rosenbrock' else 0.02

    start_time = time.time()
    rounds = budget // 20

    for i in range(rounds):
        # 训练模型
        trainer = ModelTrainer(func_name, dims)
        trainer.device = device
        model = trainer.train(input_X, input_y2)

        # MCTS rollout
        optimizer = Scalpel(f=f, model=model, name=func_name, use_continuous=use_continuous)
        top_X = optimizer.rollout(input_X, input_y2, rollout_round, ratio, i)

        if len(top_X) == 0:
            print(f"[Original] Round {i}: No new points found.", flush=True)
            continue

        # 评估
        top_y = np.array([fx(xx)[1] for xx in top_X])
        input_X = np.concatenate((input_X, top_X), axis=0)
        input_y2 = np.concatenate((input_y2, top_y))

        best_idx = np.argmax(input_y2)
        raw_val = f(input_X[best_idx])[0]
        elapsed = time.time() - start_time
        print(f"[Original] Round {i}/{rounds}: best_f={raw_val:.6f}, time={elapsed:.2f}s, total={len(input_X)} evals", flush=True)

    elapsed_time = time.time() - start_time
    return elapsed_time, fx.tracker.curt_best, fx.tracker.counter


# ==================== 包装版 scalpel 测试 ====================

def run_wrapped_scalpel(func_name, dims, budget, seed, use_continuous=True):
    """运行包装版 scalpel_core + scalpel_opt.py"""
    from functions.test_functions import (
        Ackley, Rastrigin, Rosenbrock, Griewank, Michalewicz, Schwefel, Levy
    )
    from scalpel.scalpel_opt import ScalpelOptimizer

    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Wrapped] Using device: {device}", flush=True)

    # 初始化函数
    if func_name == 'ackley':
        test_func = Ackley(dims=dims)
    elif func_name == 'rastrigin':
        test_func = Rastrigin(dims=dims)
    elif func_name == 'rosenbrock':
        test_func = Rosenbrock(dims=dims)
    elif func_name == 'griewank':
        test_func = Griewank(dims=dims)
    elif func_name == 'michalewicz':
        test_func = Michalewicz(dims=dims)
    elif func_name == 'schwefel':
        test_func = Schwefel(dims=dims)
    elif func_name == 'levy':
        test_func = Levy(dims=dims)
    else:
        raise ValueError(f"Unknown function: {func_name}")

    # 创建 wrapper（简化版）
    class SimpleFuncWrapper:
        def __init__(self, func):
            self.func = func
            self.lb = func.lb
            self.ub = func.ub
            self.dims = func.dims
            self.is_minimizing = True

        def __call__(self, x):
            result = self.func(x)
            if isinstance(result, tuple):
                return result[0]  # 返回 f
            return result

        def get_raw_result(self, x):
            result = self.func(x)
            if isinstance(result, tuple):
                return result[0], result[1]
            return result, result

    func_wrapper = SimpleFuncWrapper(test_func)

    # 创建优化器
    optimizer = ScalpelOptimizer(
        func_wrapper=func_wrapper,
        func_name=func_name,
        use_continuous=use_continuous,
        device=device,
    )

    # 运行优化（自包含模式）
    start_time = time.time()

    # 手动执行与原版相同的流程
    from baselines.lamcts.utils import latin_hypercube, from_unit_cube

    # 初始 200 个点
    n_init = 200
    init_X = latin_hypercube(n_init, dims)
    init_X = from_unit_cube(init_X, test_func.lb, test_func.ub)

    raw_scores = []
    for xi in init_X:
        f_val, score = func_wrapper.get_raw_result(xi)
        raw_scores.append(score)
        optimizer.history_x.append(xi.copy())
        optimizer.history_fx.append(f_val)
        if f_val < optimizer.best_fx:
            optimizer.best_fx = f_val
            optimizer.best_x = xi.copy()

    optimizer._X_all = init_X.astype(np.float32)
    optimizer._y_score_all = np.array(raw_scores, dtype=np.float32)
    optimizer.call_count = n_init

    rounds = budget // 20

    for i in range(rounds):
        # 训练模型
        optimizer._ensure_scalpel()
        optimizer._scalpel.update(optimizer._X_all, optimizer._y_score_all)
        iteration = optimizer._round
        optimizer._round += 1

        # MCTS rollout
        new_X = optimizer._scalpel.rollout(
            optimizer._X_all, optimizer._y_score_all, iteration=iteration
        )

        if len(new_X) == 0:
            new_X = np.random.uniform(test_func.lb, test_func.ub, size=(20, dims))

        new_X = np.atleast_2d(new_X)

        # 评估
        raw_scores = []
        for xi in new_X:
            f_val, score = func_wrapper.get_raw_result(xi)
            raw_scores.append(score)
            optimizer.history_x.append(xi.copy())
            optimizer.history_fx.append(f_val)
            optimizer.call_count += 1
            if f_val < optimizer.best_fx:
                optimizer.best_fx = f_val
                optimizer.best_x = xi.copy()

        optimizer._X_all = np.concatenate([optimizer._X_all, new_X.astype(np.float32)], axis=0)
        optimizer._y_score_all = np.concatenate([optimizer._y_score_all, np.array(raw_scores, dtype=np.float32)], axis=0)

        elapsed = time.time() - start_time
        print(f"[Wrapped] Round {i}/{rounds}: best_f={optimizer.best_fx:.6f}, time={elapsed:.2f}s, total={optimizer.call_count} evals", flush=True)

    elapsed_time = time.time() - start_time
    return elapsed_time, optimizer.best_fx, optimizer.call_count


# ==================== 主测试函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Scalpel 时间对比测试')
    parser.add_argument('--func', '-f', type=str, default='ackley',
                       choices=['ackley', 'rastrigin', 'rosenbrock', 'griewank',
                               'michalewicz', 'schwefel', 'levy'],
                       help='测试函数')
    parser.add_argument('--dims', '-d', type=int, default=10,
                       help='问题维度')
    parser.add_argument('--budget', '-b', type=int, default=200,
                       help='总评估预算')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='随机种子')
    parser.add_argument('--continuous', '-c', action='store_true',
                       help='使用连续模式')
    parser.add_argument('--mode', '-m', type=str, default='both',
                       choices=['original', 'wrapped', 'both'],
                       help='测试模式')

    args = parser.parse_args()

    print("=" * 60)
    print("Scalpel 时间对比测试")
    print("=" * 60)
    print(f"Function: {args.func}")
    print(f"Dims: {args.dims}")
    print(f"Budget: {args.budget}")
    print(f"Seed: {args.seed}")
    print(f"Continuous: {args.continuous}")
    print(f"Mode: {args.mode}")
    print("=" * 60, flush=True)

    results = {}

    if args.mode in ['original', 'both']:
        print("\n" + "=" * 40)
        print("测试原版 fluxer + main.py")
        print("=" * 40, flush=True)
        try:
            time_original, best_original, evals_original = run_original_fluxer(
                args.func, args.dims, args.budget, args.seed, args.continuous
            )
            results['original'] = {
                'time': time_original,
                'best_fx': best_original,
                'evals': evals_original,
            }
        except Exception as e:
            print(f"[ERROR] Original failed: {e}")
            import traceback
            traceback.print_exc()
            results['original'] = {'time': None, 'error': str(e)}

    if args.mode in ['wrapped', 'both']:
        print("\n" + "=" * 40)
        print("测试包装版 scalpel_core + scalpel_opt.py")
        print("=" * 40, flush=True)
        try:
            time_wrapped, best_wrapped, evals_wrapped = run_wrapped_scalpel(
                args.func, args.dims, args.budget, args.seed, args.continuous
            )
            results['wrapped'] = {
                'time': time_wrapped,
                'best_fx': best_wrapped,
                'evals': evals_wrapped,
            }
        except Exception as e:
            print(f"[ERROR] Wrapped failed: {e}")
            import traceback
            traceback.print_exc()
            results['wrapped'] = {'time': None, 'error': str(e)}

    # 输出对比结果
    print("\n" + "=" * 60)
    print("时间对比结果")
    print("=" * 60)

    for mode, res in results.items():
        print(f"\n{mode.upper()}:")
        if 'error' in res:
            print(f"  ERROR: {res['error']}")
        else:
            print(f"  Time: {res['time']:.4f}s")
            print(f"  Best f(x): {res['best_fx']:.6f}")
            print(f"  Total evals: {res['evals']}")

    if 'original' in results and 'wrapped' in results:
        orig = results['original']
        wrap = results['wrapped']
        if orig.get('time') and wrap.get('time'):
            speedup = orig['time'] / wrap['time']
            print(f"\n速度比 (original/wrapped): {speedup:.4f}x")
            if speedup > 1:
                print("结论: 包装版更快")
            else:
                print("结论: 原版更快")


if __name__ == '__main__':
    main()
