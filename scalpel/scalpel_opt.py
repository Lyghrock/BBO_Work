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
    3. test_functions 返回 (f, score)，score = 100/(f+0.01)，始终 > 0
       → raw_score = sign * score
         - minimize: score > 0，越大越好 ✓
         - maximize: -score < 0，越大越好 ✓
    4. benchmark 报告 best_fx = sign * f(x)
    """

    POINTS_PER_ROUND = 20
    # 优化3：训练触发阈值（累积 N 个新点才训练一次）
    _TRAIN_INTERVAL = 10

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

        # _X_all / _y_score_all：批量堆叠后的 numpy 数组（训练时用）
        self._X_all = np.zeros((0, self.dims), dtype=np.float32)
        self._y_score_all = np.zeros((0,), dtype=np.float32)
        # 优化3：待追加数据用列表暂存，避免每次 append 都 concat
        self._X_pending = []
        self._y_pending = []
        self._pending_since_train = 0

        # Warm-start pool（latin hypercube，lazy 生成）
        self._warm_start_pool = None
        self._warm_start_pos = 0

        # ScalpelCore（lazy 创建）
        self._scalpel = None
        self._round = 0

    def _flush_pending(self):
        """将 _X_pending / _y_pending 批量追加到 _X_all / _y_score_all。"""
        if not self._X_pending:
            return
        X_batch = np.array(self._X_pending, dtype=np.float32)
        y_batch = np.asarray(self._y_pending, dtype=np.float32).reshape(-1)
        if self._X_all.size == 0:
            self._X_all = X_batch
            self._y_score_all = y_batch
        else:
            self._X_all = np.concatenate([self._X_all, X_batch], axis=0)
            self._y_score_all = np.concatenate([self._y_score_all, y_batch], axis=0)
        self._X_pending.clear()
        self._y_pending.clear()
        self._pending_since_train = 0

    def _append_data(self, X_batch, y_score_batch):
        """优化3：将新数据追加到待处理列表，由 _flush_pending 统一堆叠。"""
        if X_batch is None:
            return
        X_batch = np.atleast_2d(X_batch).astype(np.float32, copy=False)
        if X_batch.size == 0:
            return
        y_score_batch = np.asarray(y_score_batch, dtype=np.float32).reshape(-1)
        if y_score_batch.size == 0:
            return
        for xi, yi in zip(X_batch, y_score_batch):
            self._X_pending.append(xi)
            self._y_pending.append(float(yi))
        self._pending_since_train += len(X_batch)

    # ── Internal helpers ──────────────────────────────────────────────────────

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

    def _raw_score(self, x):
        """
        返回 raw_score（越大越好，用于 MCTS 训练）。

        直接访问底层函数（func_wrapper.func），不调用 func_wrapper，
        避免 call_count 重复计数和 sign 转换。
        """
        raw_result = self.func_wrapper.func(x)
        if isinstance(raw_result, tuple):
            score = float(raw_result[1])
            return score
        else:
            value = float(raw_result)
            # 标量目标统一转成「越大越好」：最小化目标取负，最大化目标保持原值
            return -value if self.func_wrapper.is_minimizing else value

    def _train_model(self):
        """重新训练 MCTS 代理模型（仅在 pending 数据 ≥ _TRAIN_INTERVAL 时触发）。"""
        if self._pending_since_train < self._TRAIN_INTERVAL:
            return
        self._flush_pending()
        if len(self._X_all) < 2:
            return
        self._ensure_scalpel()
        self._scalpel.update(self._X_all, self._y_score_all)

    # ── BaseOptimizer interface ───────────────────────────────────────────────

    def suggest(self, n_suggestions=1):
        """
        返回候选点（不自评，由 benchmark 评估后调用 observe()）。

        流程：
          1. 生成 warm-start pool（200 个 latin hypercube 点，lazy）
          2. 先从 pool 消耗（直到 200 个全被 observe 回来）
          3. pool 耗尽后：训练模型 → MCTS rollout → 返回候选点
        """
        # 生成 warm-start pool
        if self._warm_start_pool is None:
            from baselines.lamcts.utils import latin_hypercube, from_unit_cube
            n_init = max(200, self.POINTS_PER_ROUND)
            init_X = latin_hypercube(n_init, self.dims)
            init_X = from_unit_cube(init_X, self.lb, self.ub)
            self._warm_start_pool = init_X

        # 从 warm-start pool 取点（自评后加入 _X_all）
        if self._warm_start_pos < len(self._warm_start_pool):
            end_pos = min(self._warm_start_pos + n_suggestions,
                          len(self._warm_start_pool))
            batch = self._warm_start_pool[self._warm_start_pos:end_pos]
            self._warm_start_pos = end_pos
            return np.atleast_2d(batch)

        # pool 耗尽 → MCTS 模式：先训练模型
        self._train_model()

        if self._scalpel is None:
            return np.random.uniform(self.lb, self.ub, size=(n_suggestions, self.dims))

        # MCTS rollout
        new_X = self._scalpel.rollout(
            self._X_all, self._y_score_all, iteration=self._round
        )
        self._round += 1

        if len(new_X) == 0:
            return np.random.uniform(self.lb, self.ub, size=(n_suggestions, self.dims))

        new_X = np.atleast_2d(new_X)
        return np.clip(new_X[:n_suggestions], self.lb, self.ub)

    def observe(self, x, fx=None):
        """
        记录评估结果。

        NOTE: benchmark 调用时 fx = sign * f(x)（已经 sign-transformed 了）。
        优化1：在外部路径中从 benchmark 传入的 fx 推算 raw_score，不再重复调用函数。
        优化3：数据追加走 pending 列表。
        """
        x = np.atleast_2d(x)

        if fx is None:
            # 内部路径（optimize 自评）：调用 func_wrapper 获取 raw_score
            raw_scores = []
            for xi in x:
                raw = self.func_wrapper(xi)
                if isinstance(raw, tuple):
                    raw = raw[0]
                raw_score = self._raw_score(xi)
                raw_scores.append(raw_score)
                self.history_x.append(xi.copy())
                self.history_fx.append(float(raw))
                self.call_count += 1
                if raw < self.best_fx:
                    self.best_fx = float(raw)
                    self.best_x = xi.copy()
            self._append_data(x, raw_scores)
        else:
            # 外部路径（benchmark 评估）：从 fx 推算 raw_score，避免重复调用函数
            fx = np.atleast_1d(np.asarray(fx, dtype=np.float32))
            if x.shape[0] != len(fx):
                fx = np.full(x.shape[0], fx[0])

            # 优化1：raw_score 从 benchmark 传入的 fx 推算，不再调用 _raw_score
            raw_scores = self._compute_raw_scores_from_fx(x, fx)

            for xi, fi in zip(x, fx):
                self.history_x.append(xi.copy())
                self.history_fx.append(float(fi))
                self.call_count += 1
                if fi < self.best_fx:
                    self.best_fx = float(fi)
                    self.best_x = xi.copy()
            self._append_data(x, raw_scores)

    def _compute_raw_scores_from_fx(self, x, fx):
        """
        优化1：从 benchmark 传入的 fx 推算 raw_score。

        对于返回 (f, score) 的函数类型，score 在 tuple 第二个位置，
        raw_score = score（本身就是越大越好的，未经过 sign 变换）。
        对于纯标量函数，raw_score 无法从 fx 反推，保留 fallback 到 _raw_score。
        """
        raw_scores = []
        for xi, fi in zip(x, fx):
            raw_result = self.func_wrapper.func(xi)
            if isinstance(raw_result, tuple):
                # (f, score) 格式：score 未经过 sign 变换，直接取用
                raw_scores.append(float(raw_result[1]))
            else:
                # 纯标量函数：无法从 fx 反推，fallback
                raw_scores.append(self._raw_score(xi))
        return raw_scores

    def optimize(self, call_budget, batch_size=10):
        """
        自包含的 Scalpel 循环。

        初始化 200 个 latin hypercube 点（自评），
        然后每轮 flush pending → 累积训练触发 → MCTS rollout → 自评 → 重复。
        优化3：flush pending 和训练频率控制在同一处管理。
        """
        if len(self._X_all) == 0 and not self._X_pending:
            from baselines.lamcts.utils import latin_hypercube, from_unit_cube
            n_init = max(200, self.POINTS_PER_ROUND)
            init_X = latin_hypercube(n_init, self.dims)
            init_X = from_unit_cube(init_X, self.lb, self.ub)

            # 内部自评：调用 func_wrapper（= sign*f），同时也获取 raw_score
            raw_scores = []
            for xi in init_X:
                raw = self.func_wrapper(xi)
                if isinstance(raw, tuple):
                    raw = raw[0]
                raw_score = self._raw_score(xi)
                raw_scores.append(raw_score)
                self.history_x.append(xi.copy())
                self.history_fx.append(float(raw))
                if raw < self.best_fx:
                    self.best_fx = float(raw)
                    self.best_x = xi.copy()
            self._append_data(init_X, raw_scores)
            self.call_count = len(init_X)

        self._train_model()

        while self.call_count < call_budget:
            remaining = call_budget - self.call_count

            # MCTS rollout → 提出候选点（先用 flush 保证模型用最新数据）
            self._flush_pending()
            new_X = self._scalpel.rollout(
                self._X_all, self._y_score_all, iteration=self._round
            )
            self._round += 1

            if len(new_X) == 0:
                new_X = np.random.uniform(
                    self.lb, self.ub,
                    size=(min(self.POINTS_PER_ROUND, remaining), self.dims)
                )
            new_X = np.atleast_2d(new_X)

            # 自评候选点
            raw_scores = []
            for xi in new_X:
                raw = self.func_wrapper(xi)
                if isinstance(raw, tuple):
                    raw = raw[0]
                raw_score = self._raw_score(xi)
                raw_scores.append(raw_score)
                self.history_x.append(xi.copy())
                self.history_fx.append(float(raw))
                self.call_count += 1
                if raw < self.best_fx:
                    self.best_fx = float(raw)
                    self.best_x = xi.copy()
            self._append_data(new_X, raw_scores)

            # 重新训练（受 _TRAIN_INTERVAL 控制）
            self._train_model()

        return self.best_x, self.best_fx, self.call_count

    def reset(self):
        super().reset()
        self._X_all = np.zeros((0, self.dims), dtype=np.float32)
        self._y_score_all = np.zeros((0,), dtype=np.float32)
        self._X_pending = []
        self._y_pending = []
        self._pending_since_train = 0
        self._warm_start_pool = None
        self._warm_start_pos = 0
        self._scalpel = None
        self._round = 0
