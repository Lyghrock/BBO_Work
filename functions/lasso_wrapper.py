"""
Lasso-Bench 函数包装器
当前仅保留真实任务：DNA, RCV1
"""

import numpy as np


class LassoFuncWrapper:
    """
    LassoBench 函数包装器 - 统一接口

    搜索空间: [-1, 1]^n_features
    目标: 最小化 CV 损失
    """

    def __init__(self, bench, is_minimizing: bool = True):
        """
        Args:
            bench: RealBenchmark 实例
            is_minimizing: 是否是最小化问题（默认 True）
        """
        self.bench = bench
        self.lb = -1.0 * np.ones(bench.n_features)
        self.ub = 1.0 * np.ones(bench.n_features)
        self.dims = bench.n_features
        self.is_minimizing = is_minimizing
        self.sign = 1.0 if is_minimizing else -1.0
        self.call_count = 0
        self.func = lambda x: self.bench.evaluate(np.asarray(x).flatten())

    def __call__(self, x):
        """评估配置"""
        self.call_count += 1
        x = np.asarray(x).flatten()

        if np.any(x < -1) or np.any(x > 1):
            raise ValueError(f"配置超出范围 [-1, 1]: min={x.min():.4f}, max={x.max():.4f}")

        loss = self.bench.evaluate(x)
        return self.sign * loss

    def evaluate_with_metrics(self, x):
        """评估并返回完整指标"""
        x = np.asarray(x).flatten()
        loss = self.bench.evaluate(x)
        test_metrics = self.bench.test(x)
        return {
            'val_loss': loss,
            'mspe': test_metrics.get('mspe'),
            'fscore': test_metrics.get('fscore'),
        }

    def gen_random_inputs(self, n: int) -> np.ndarray:
        """生成随机配置"""
        return np.random.uniform(self.lb, self.ub, size=(n, self.dims))


# Lasso-Bench 任务配置（仅真实数据）
LASSO_BENCHMARKS = {
    'dna': {
        'type': 'real',
        'n_features': 180,
    },
    'rcv1': {
        'type': 'real',
        'n_features': 19959,
    },
}


def create_lasso_benchmark(bench_name: str, **kwargs):
    """
    创建 Lasso Benchmark 实例

    Args:
        bench_name: 'dna', 'rcv1'

    Returns:
        LassoFuncWrapper 实例
    """
    import LassoBench

    if bench_name not in LASSO_BENCHMARKS:
        raise ValueError(f"Unknown benchmark: {bench_name}. Available: {list(LASSO_BENCHMARKS.keys())}")

    bench = LassoBench.RealBenchmark(pick_data=bench_name, **kwargs)

    return LassoFuncWrapper(bench)


def get_benchmark_dims(bench_name: str) -> int:
    """获取 benchmark 维度"""
    return LASSO_BENCHMARKS[bench_name]['n_features']


def get_adaptive_budget(dims: int) -> int:
    """根据维度自适应预算"""
    if dims < 200:
        return 1000
    elif dims < 500:
        return 1500
    else:
        return 2000


def list_benchmarks():
    """列出所有 benchmark"""
    print("=" * 50)
    print("Lasso-Bench Benchmarks")
    print("=" * 50)
    for name, info in LASSO_BENCHMARKS.items():
        print(f"  {name:12s} | {info['n_features']:5d}D | {info['type']}")


if __name__ == "__main__":
    list_benchmarks()

    print("\n" + "=" * 50)
    print("Quick Test")
    print("=" * 50)

    for bench_name in LASSO_BENCHMARKS.keys():
        wrapper = create_lasso_benchmark(bench_name)
        x = wrapper.gen_random_inputs(1)[0]
        loss = wrapper(x)
        print(f"{bench_name}: dims={wrapper.dims}, loss={loss:.6f}")
