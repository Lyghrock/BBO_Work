import numpy as np

# ==================== Benchmark Functions ====================
# 与 scalpel/original/main.py 保持一致，返回 (f, score) tuple
# score 用于 MCTS 训练，始终越大越好


class Ackley:
    def __init__(self, dims=3):
        self.dims = dims
        self.lb, self.ub = -5 * np.ones(dims), 5 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        x = np.array(x)
        assert len(x) == self.dims and x.ndim == 1
        result = (-20 * np.exp(-0.2 * np.sqrt(np.inner(x, x) / x.size)) - np.exp(
            np.cos(2 * np.pi * x).sum() / x.size) + 20 + np.e)
        return result, 100 / (result + 0.01)


class Rastrigin:
    def __init__(self, dims=3):
        self.dims = dims
        self.lb, self.ub = -5 * np.ones(dims), 5 * np.ones(dims)
        self.counter = 0

    def __call__(self, x, A=10):
        self.counter += 1
        x = np.array(x)
        assert len(x) == self.dims and x.ndim == 1
        n = len(x)
        result = A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))
        return result, -result


class Rosenbrock:
    def __init__(self, dims=3):
        self.dims = dims
        self.lb, self.ub = -5 * np.ones(dims), 5 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        x = np.array(x)
        assert len(x) == self.dims and x.ndim == 1
        result = np.sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)
        return result, 100 / (result / (self.dims * 100) + 0.01)


class Griewank:
    def __init__(self, dims=3):
        self.dims = dims
        self.lb, self.ub = -600 * np.ones(dims), 600 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        x = np.array(x)
        assert len(x) == self.dims and x.ndim == 1
        sum_term = np.sum(x ** 2)
        prod_term = np.prod(np.cos(x / np.sqrt(np.arange(1, len(x) + 1))))
        result = 1 + sum_term / 4000 - prod_term
        return result, 10 / (result / self.dims + 0.001)


class Michalewicz:
    def __init__(self, dims=3):
        self.dims = dims
        self.lb, self.ub = 0 * np.ones(dims), np.pi * np.ones(dims)
        self.counter = 0

    def __call__(self, x, m=10):
        self.counter += 1
        x = np.array(x)
        assert len(x) == self.dims and x.ndim == 1
        d = len(x)
        total = 0
        for i in range(d):
            total += np.sin(x[i]) * np.sin((i + 1) * x[i]**2 / np.pi)**(2 * m)
        return -total, total


class Schwefel:
    def __init__(self, dims=3):
        self.dims = dims
        self.lb, self.ub = -500 * np.ones(dims), 500 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        x = np.array(x)
        assert len(x) == self.dims and x.ndim == 1
        dimension = len(x)
        sum_part = np.sum(-x * np.sin(np.sqrt(np.abs(x))))
        result = 418.9829 * dimension + sum_part
        return result, -result / 100


class Levy:
    def __init__(self, dims=1):
        self.dims = dims
        self.lb, self.ub = -10 * np.ones(dims), 10 * np.ones(dims)
        self.counter = 0

    def __call__(self, x):
        self.counter += 1
        x = np.array(x)
        assert len(x) == self.dims and x.ndim == 1
        w = 1 + (x - 1) / 4
        term1 = (np.sin(np.pi * w[0]))**2
        term3 = (w[-1] - 1)**2 * (1 + (np.sin(2 * np.pi * w[-1]))**2)
        term2 = np.sum((w[:-1] - 1)**2 * (1 + 10 * (np.sin(np.pi * w[:-1] + 1))**2))
        result = term1 + term2 + term3
        return result, -result
