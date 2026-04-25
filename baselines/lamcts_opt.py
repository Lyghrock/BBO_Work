# LAMCTS Optimizer Wrapper
import numpy as np
import torch
from typing import Optional, Tuple
from baselines.base import BaseOptimizer


class LAMCTSOptimizer(BaseOptimizer):
    """LAMCTS优化器包装器 - 兼容BaseOptimizer接口"""
    
    def __init__(self, func_wrapper,
                 cp: float = None,
                 leaf_size: int = None,
                 ninits: int = None,
                 solver_type: str = 'turbo',
                 gpu_id: int = None,
                 kernel_type: str = None,
                 gamma_type: str = None,
                 **kwargs):
        """
        初始化LAMCTS优化器
        
        Args:
            func_wrapper: 函数包装器
            cp: UCT探索参数
            leaf_size: 叶子节点最小样本数
            ninits: 初始样本数（全局初始化样本）
            solver_type: 求解器类型 ('bo' 或 'turbo')
            gpu_id: GPU设备ID
            kernel_type: SVM核类型（'rbf', 'linear', 'poly' 等）
            gamma_type: SVM核的gamma类型
        """
        super().__init__(func_wrapper, **kwargs)

        self.solver_type = solver_type
        self.gpu_id = gpu_id

        # 设置GPU设备（如果需要）
        if gpu_id is not None and torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)

        # 根据底层环境/函数自动推断超参数
        # 对于 MuJoCo 环境，从 env 中读取；对于合成函数，使用论文中的默认值
        base_obj = getattr(func_wrapper, "env", None)
        if base_obj is None:
            base_obj = getattr(func_wrapper, "func", None)
        if base_obj is None:
            base_obj = func_wrapper

        # 1) 如果底层环境显式提供了超参数（MuJoCo 环境）
        has_env_hparams = any(
            hasattr(base_obj, attr)
            for attr in ("Cp", "leaf_size", "ninits", "kernel_type", "gamma_type")
        )

        if has_env_hparams:
            # MuJoCo / 复杂任务：直接尊重环境里的配置，除非用户显式覆盖
            self.cp = cp if cp is not None else getattr(base_obj, "Cp", 1.414)
            self.leaf_size = leaf_size if leaf_size is not None else getattr(base_obj, "leaf_size", 100)
            self.ninits = ninits if ninits is not None else getattr(base_obj, "ninits", 30)
            self.kernel_type = kernel_type if kernel_type is not None else getattr(base_obj, "kernel_type", "rbf")
            self.gamma_type = gamma_type if gamma_type is not None else getattr(base_obj, "gamma_type", "auto")
        else:
            # 合成数学测试函数使用论文中的默认值：
            # ninits = 30, leaf_size = 20, Cp = 1
            self.cp = cp if cp is not None else 1.0
            default_leaf = 20
            self.leaf_size = leaf_size if leaf_size is not None else default_leaf
            self.ninits = ninits if ninits is not None else 30
            if self.ninits <= self.leaf_size:
                self.ninits = self.leaf_size + 10
            self.kernel_type = kernel_type if kernel_type is not None else "rbf"
            self.gamma_type = gamma_type if gamma_type is not None else "auto"

        # MCTS实例
        self.mcts = None
        self._last_selected_leaf = None  # 上一次 suggest() 选中的 leaf，供 observe() 回溯用
        
    def _create_mcts(self):
        """创建MCTS实例"""
        from baselines.lamcts.MCTS import MCTS
        
        self.mcts = MCTS(
            lb=self.lb,
            ub=self.ub,
            dims=self.dims,
            ninits=self.ninits,
            func=self.func_wrapper,
            Cp=self.cp,
            leaf_size=self.leaf_size,
            kernel_type=self.kernel_type,
            gamma_type=self.gamma_type,
            gpu_id=self.gpu_id
        )
        
        # 设置求解器类型
        self.mcts.solver_type = self.solver_type
        
    def suggest(self, n_suggestions: int = 1):
        """
        Propose new sampling points.

        This mirrors the original LaMCTS search() loop:
          1. Rebuild the tree from all accumulated samples (dynamic_treeify)
          2. Select a leaf via UCT (select)
          3. Propose samples from that leaf (via BO or TuRBO sampler)

        For the benchmark path, function evaluation is done externally;
        observe() receives fx and performs backprop.

        Args:
            n_suggestions: Number of points to propose

        Returns:
            (n_suggestions x dims) ndarray of proposed points
        """
        if self.mcts is None:
            self._create_mcts()

        # First call: return Latin hypercube initial points (no MCTS needed yet)
        if len(self.mcts.samples) == 0:
            from baselines.lamcts.utils import latin_hypercube, from_unit_cube
            init_points = latin_hypercube(self.ninits, self.dims)
            init_points = from_unit_cube(init_points, self.lb, self.ub)
            return init_points[:n_suggestions]

        # Step 1: Rebuild tree + adaptive UCT selection
        self.mcts.dynamic_treeify()
        adaptive_Cp = self.mcts._compute_adaptive_Cp()
        leaf, path = self.mcts.select(adaptive_Cp=adaptive_Cp)

        # Step 2: Propose from leaf (NO evaluation here — benchmark does it externally)
        self._last_selected_leaf = leaf
        self._last_selected_path = path

        if self.solver_type == 'bo':
            samples = leaf.propose_samples_bo(
                n_suggestions, path, self.lb, self.ub, self.mcts.samples
            )
        else:
            # TuRBO: collect=False (benchmark will evaluate externally)
            # DO NOT use collect=True here — it would pre-evaluate func internally
            # and then the benchmark would evaluate again externally, causing
            # inconsistency. The benchmark path uses suggest()+observe() separately.
            samples = leaf.propose_samples_turbo(
                n_suggestions, path, self.func_wrapper, collect=False
            )

        return np.atleast_2d(samples)[:n_suggestions]

    def observe(self, x: np.ndarray, fx: Optional[np.ndarray] = None):
        """
        Receive externally-evaluated results.

        Per the original LaMCTS search() loop, backprop must use the ACTUAL
        value of each specific sample. The key difference between modes:
          - BO: backprop with batch_best (all BO samples evaluated as a batch)
          - TuRBO: backprop with each individual sample's value
        """
        if fx is None:
            return

        x = np.atleast_2d(x)
        fx = np.atleast_1d(np.asarray(fx, dtype=np.float32))
        if x.shape[0] != len(fx):
            fx = np.full(x.shape[0], fx[0])

        leaf = self._last_selected_leaf
        self._last_selected_leaf = None

        if self.solver_type == 'bo':
            # BO: convert fx to maximization format for MCTS tree
            # MCTS uses maximization convention (curt_best_value starts at -inf, larger is better)
            # For minimization: negate so that smaller original fx → larger negated value → MCTS prefers it
            values = np.array([-fi for fi in fx])
            for xi, vi in zip(x, values):
                self.mcts.samples.append((xi, vi))
                if vi > self.mcts.curt_best_value:
                    self.mcts.curt_best_value = float(vi)
                    self.mcts.curt_best_sample = xi
                self.mcts.sample_counter += 1
            batch_best_val = float(np.max(values))
            if leaf is not None:
                self.mcts.backpropogate(leaf, batch_best_val)

        else:
            # TuRBO: convert externally-evaluated fx to maximization format for MCTS tree
            # Negation flips the ordering so that minimization objective is preferred by MCTS
            values = np.array([-fi for fi in fx])

            for xi, vi in zip(x, values):
                self.mcts.samples.append((xi, float(vi)))
                if vi > self.mcts.curt_best_value:
                    self.mcts.curt_best_value = float(vi)
                    self.mcts.curt_best_sample = xi
                self.mcts.sample_counter += 1

            # Backprop with each individual sample's value (original LaMCTS behavior)
            if leaf is not None:
                for vi in values:
                    self.mcts.backpropogate(leaf, float(vi))

        # BaseOptimizer best tracking
        for xi, fi in zip(x, fx):
            self.history_x.append(xi)
            self.history_fx.append(fi)
            self.call_count += 1
            if self.is_minimizing:
                if fi < self.best_fx:
                    self.best_fx = float(fi)
                    self.best_x = xi
            else:
                if fi > self.best_fx:
                    self.best_fx = float(fi)
                    self.best_x = xi
    
    def optimize(self, call_budget: int, batch_size: int = 1):
        """
        Execute LAMCTS optimization until budget is exhausted.

        Every iteration follows the corrected LaMCTS search loop:
          1. dynamic_treeify()  — rebuild tree from all accumulated samples
          2. adaptive Cp         — recompute based on tree imbalance
          3. select()            — UCT leaf selection
          4. propose()           — generate candidates from leaf
          5. collect_samples()  — evaluate (value already in maximization format)
          6. backpropogate()    — update x_bar and n along path
        """
        if self.mcts is None:
            self._create_mcts()

        if self.mcts.sample_counter >= call_budget:
            self.best_x = self.mcts.curt_best_sample
            # MCTS stores in maximization format: value = -func(x)
            # Return original function value (minimization format)
            self.best_fx = -self.mcts.curt_best_value
            self.call_count = self.mcts.sample_counter
            return self.best_x, self.best_fx, self.call_count

        extra_budget = call_budget - self.mcts.sample_counter

        for _ in range(extra_budget):
            if self.mcts.sample_counter >= call_budget:
                break

            # Every iteration: rebuild tree + adaptive UCT select
            self.mcts.dynamic_treeify()
            adaptive_Cp = self.mcts._compute_adaptive_Cp()
            leaf, path = self.mcts.select(adaptive_Cp=adaptive_Cp)

            if self.solver_type == 'bo':
                samples = leaf.propose_samples_bo(
                    1, path, self.lb, self.ub, self.mcts.samples
                )
                for s in samples:
                    if self.mcts.sample_counter >= call_budget:
                        break
                    # collect_samples handles sign internally:
                    #   value = func(s) * -1  (maximization format)
                    value = self.mcts.collect_samples(s)
                    self.mcts.backpropogate(leaf, value)

            else:
                # CRITICAL FIX: use collect=True so propose_samples_turbo returns
                # (proposed_X, fX) where fX = -func(x) already negated.
                # Without collect=True, fX was just the sample array (BUG).
                proposed_X, fX = leaf.propose_samples_turbo(
                    10, path, self.func_wrapper, collect=True
                )
                for j, s in enumerate(proposed_X):
                    if self.mcts.sample_counter >= call_budget:
                        break
                    # fX[j] is already negated: no additional negation needed
                    value = self.mcts.collect_samples(s, value=fX[j])
                    self.mcts.backpropogate(leaf, value)

            # Update best tracking (convert back to minimization format for user)
            self.best_x = self.mcts.curt_best_sample
            self.best_fx = -self.mcts.curt_best_value

        self.call_count = self.mcts.sample_counter
        return self.best_x, self.best_fx, self.call_count
    
    def reset(self):
        """重置优化器"""
        super().reset()
        self.mcts = None
