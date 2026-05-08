# Scalpel Optimizer — BaseOptimizer wrapper for ScalpelCore
import numpy as np
import torch
from baselines.base import BaseOptimizer


class ScalpelOptimizer(BaseOptimizer):
    """
    Scalpel optimizer wrapped as a BaseOptimizer.
    Wraps scalpel.scalpel_core.ScalpelCore.

    设计原则（和 original/main.py 完全对齐）：
    1. ScalpelCore.rollout(X, y) 用 np.argmax(y) → y 越大越好
    2. func_wrapper(x) 返回 sign * f(x)（minimize: sign=1, maximize: sign=-1）
    3. test_functions 返回 (f, score)，score 始终越大越好
       → raw_score = score（从 benchmark 传入，不重新计算）
    4. benchmark 报告 best_fx = sign * f(x)
    """

    POINTS_PER_ROUND = 20  # 与原版 main.py 一致，每轮 20 点

    def __init__(self, func_wrapper,
                 func_name='ackley',
                 use_continuous=True,
                 gpu_id=None,
                 **kwargs):
        super().__init__(func_wrapper, **kwargs)

        self.func_name = func_name
        self.use_continuous = use_continuous
        self.gpu_id = gpu_id

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        # 数据存储（与原版一致的批量处理方式）
        self._X_all = np.zeros((0, self.dims), dtype=np.float32)
        self._y_score_all = np.zeros((0,), dtype=np.float32)

        # Warm-start pool（latin hypercube，lazy 生成）
        self._warm_start_pool = None
        self._warm_start_pos = 0

        # ScalpelCore（lazy 创建）
        self._scalpel = None
        self._round = 0

    def _ensure_scalpel(self):
        if self._scalpel is not None:
            return
        from scalpel.scalpel_core import ScalpelCore
        self._scalpel = ScalpelCore(
            func=self.func_wrapper,
            func_name=self.func_name,
            dims=self.dims,
            use_continuous=self.use_continuous,
            device=str(self.device),
        )

    # ── BaseOptimizer interface ───────────────────────────────────────────────

    def suggest(self, n_suggestions=1):
        """
        返回候选点（不自评，由 benchmark 评估后调用 observe()）。

        流程：
          1. 生成 warm-start pool（200 个 latin hypercube 点，lazy）
          2. 先从 pool 消耗（直到 200 个全被 observe 回来）
          3. pool 耗尽后：每 20 点训练模型 → MCTS rollout → 返回候选点
        """
        # 生成 warm-start pool
        if self._warm_start_pool is None:
            from baselines.lamcts.utils import latin_hypercube, from_unit_cube
            n_init = max(200, self.POINTS_PER_ROUND)
            init_X = latin_hypercube(n_init, self.dims)
            init_X = from_unit_cube(init_X, self.lb, self.ub)
            self._warm_start_pool = init_X

        # 从 warm-start pool 取点
        if self._warm_start_pos < len(self._warm_start_pool):
            end_pos = min(self._warm_start_pos + n_suggestions,
                          len(self._warm_start_pool))
            batch = self._warm_start_pool[self._warm_start_pos:end_pos]
            self._warm_start_pos = end_pos
            return np.atleast_2d(batch)

        # pool 耗尽 → MCTS 模式
        # 检查是否需要训练（每 POINTS_PER_ROUND 点训练一次）
        if len(self._X_all) > 0 and len(self._X_all) % self.POINTS_PER_ROUND == 0:
            self._ensure_scalpel()
            self._scalpel.update(self._X_all, self._y_score_all)
            self._round += 1

        if self._scalpel is None:
            return np.random.uniform(self.lb, self.ub, size=(n_suggestions, self.dims))

        # MCTS rollout
        new_X = self._scalpel.rollout(
            self._X_all, self._y_score_all, iteration=self._round
        )

        if len(new_X) == 0:
            return np.random.uniform(self.lb, self.ub, size=(n_suggestions, self.dims))

        new_X = np.atleast_2d(new_X)
        return np.clip(new_X[:n_suggestions], self.lb, self.ub)

    def observe(self, x, fx=None, scores=None):
        """
        记录评估结果。

        Args:
            x: 评估的点 (n x dims)
            fx: 评估的函数值 (n,) = sign * f(x)
            scores: 原始 score (n,) = 原始 score（从 benchmark 传入，用于 MCTS 训练）
        """
        x = np.atleast_2d(x)

        if fx is None:
            # 内部路径（自评）：调用 func_wrapper
            fx = np.array([self.func_wrapper(xi) for xi in x])

        fx = np.atleast_1d(np.asarray(fx, dtype=np.float32))
        if x.shape[0] != len(fx):
            fx = np.full(x.shape[0], fx[0])

        # 处理 scores
        if scores is not None:
            scores = np.atleast_1d(np.asarray(scores, dtype=np.float32))
            if len(scores) != len(x):
                scores = np.full(len(x), scores[0])
        else:
            # Fallback：如果没有传入 scores，从 func_wrapper 推断
            scores = np.zeros(len(x), dtype=np.float32)
            for i, xi in enumerate(x):
                result = self.func_wrapper.func(xi)
                if isinstance(result, tuple):
                    scores[i] = result[1]
                else:
                    # 无法推断，使用默认值
                    scores[i] = -result if self.func_wrapper.is_minimizing else result

        # 更新历史记录
        for xi, fi, si in zip(x, fx, scores):
            self.history_x.append(xi.copy())
            self.history_fx.append(float(fi))
            self.call_count += 1
            if fi < self.best_fx:
                self.best_fx = float(fi)
                self.best_x = xi.copy()

        # 追加到训练数据（批量处理）
        x_batch = np.atleast_2d(x).astype(np.float32)
        self._X_all = np.concatenate([self._X_all, x_batch], axis=0)
        self._y_score_all = np.concatenate([self._y_score_all, scores], axis=0)

    def optimize(self, call_budget, batch_size=20):
        """
        自包含的 Scalpel 循环（与原版 main.py 完全对齐）。

        流程：
        1. 初始化 200 个 latin hypercube 点
        2. 每 20 点：训练模型 → MCTS rollout → 评估 → 追加数据
        """
        if len(self._X_all) == 0:
            from baselines.lamcts.utils import latin_hypercube, from_unit_cube
            # 初始 200 个点（与原版一致）
            n_init = 200
            init_X = latin_hypercube(n_init, self.dims)
            init_X = from_unit_cube(init_X, self.lb, self.ub)

            # 批量评估
            raw_scores = []
            for xi in init_X:
                result = self.func_wrapper(xi)
                if isinstance(result, tuple):
                    raw_scores.append(result[1])
                else:
                    # fallback
                    raw_scores.append(-result if self.func_wrapper.is_minimizing else result)
                self.history_x.append(xi.copy())
                if isinstance(result, tuple):
                    self.history_fx.append(float(result[0]))
                else:
                    self.history_fx.append(float(result))
                if isinstance(result, tuple):
                    if result[0] < self.best_fx:
                        self.best_fx = float(result[0])
                        self.best_x = xi.copy()
                else:
                    if result < self.best_fx:
                        self.best_fx = float(result)
                        self.best_x = xi.copy()

            self._X_all = init_X.astype(np.float32)
            self._y_score_all = np.array(raw_scores, dtype=np.float32)
            self.call_count = n_init

        # 主循环：每 POINTS_PER_ROUND 点训练一次
        while self.call_count < call_budget:
            # 训练模型
            self._ensure_scalpel()
            self._scalpel.update(self._X_all, self._y_score_all)
            iteration = self._round
            self._round += 1

            # MCTS rollout
            new_X = self._scalpel.rollout(
                self._X_all, self._y_score_all, iteration=iteration
            )

            if len(new_X) == 0:
                new_X = np.random.uniform(
                    self.lb, self.ub,
                    size=(self.POINTS_PER_ROUND, self.dims)
                )
            new_X = np.atleast_2d(new_X)

            # 批量评估（减少接口开销）
            raw_scores = []
            for xi in new_X:
                result = self.func_wrapper(xi)
                if isinstance(result, tuple):
                    raw_scores.append(result[1])
                    fx_val = result[0]
                else:
                    raw_scores.append(-result if self.func_wrapper.is_minimizing else result)
                    fx_val = result
                self.history_x.append(xi.copy())
                self.history_fx.append(float(fx_val))
                self.call_count += 1
                if fx_val < self.best_fx:
                    self.best_fx = float(fx_val)
                    self.best_x = xi.copy()

            # 追加到训练数据
            self._X_all = np.concatenate([self._X_all, new_X.astype(np.float32)], axis=0)
            self._y_score_all = np.concatenate([self._y_score_all, np.array(raw_scores, dtype=np.float32)], axis=0)

        return self.best_x, self.best_fx, self.call_count

    def reset(self):
        super().reset()
        self._X_all = np.zeros((0, self.dims), dtype=np.float32)
        self._y_score_all = np.zeros((0,), dtype=np.float32)
        self._warm_start_pool = None
        self._warm_start_pos = 0
        self._scalpel = None
        self._round = 0
