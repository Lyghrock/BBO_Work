# SAASBO Optimizer - 修复版本
"""
SAASBO: High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces
Reference: Eriksson & Jankowiak; UAI 2021

修复内容:
1. suggest() 正确处理批量请求，逐个返回
2. best_f 正确处理 sign
3. GPU 做 GP 推断，CPU 做函数评估
4. 参考 turbo 的简洁架构

NUTS 参数: 适合 budget 2000 的中等规模优化
"""
import numpy as np
import torch
import gc
import importlib
from typing import Optional
from baselines.base import BaseOptimizer
from torch.quasirandom import SobolEngine


def _cleanup_gpu_memory(sync: bool = False):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if sync:
            torch.cuda.synchronize()
    gc.collect()


def _build_saas_acquisition(model, best_f):
    """跨版本构建采集函数。

    优先使用 qLogExpectedImprovement；若当前 BoTorch 版本不支持，则回退 qExpectedImprovement。
    """
    candidates = [
        ("botorch.acquisition.logei", "qLogExpectedImprovement"),
        ("botorch.acquisition", "qLogExpectedImprovement"),
        ("botorch.acquisition.monte_carlo", "qExpectedImprovement"),
        ("botorch.acquisition", "qExpectedImprovement"),
    ]

    for module_name, symbol in candidates:
        try:
            mod = importlib.import_module(module_name)
            cls = getattr(mod, symbol)
            return cls(model=model, best_f=best_f)
        except Exception:
            continue

    raise ImportError(
        "No compatible q(Log)ExpectedImprovement found in current botorch version."
    )


class SAASBOOptimizer(BaseOptimizer):
    """SAASBO: 高维稀疏子空间贝叶斯优化器

    修复: suggest() 逐个返回，best_f sign 正确
    """

    def __init__(self, func_wrapper,
                 batch_size: int = 10,
                 use_gpu: bool = True,
                 device: torch.device = None,
                 warmup_steps: int = 128,
                 num_samples: int = 64,
                 thinning: int = 16,
                 noise_var: float = 1e-6,
                 n_candidates: int = 512,
                 num_restarts: int = 6,
                 raw_samples: int = 128,
                 training_epochs: int = 50,
                 n_init: int = None,
                 **kwargs):
        super().__init__(func_wrapper, **kwargs)

        self.batch_size = batch_size
        self.warmup_steps = warmup_steps
        self.num_samples = num_samples
        self.thinning = thinning
        self.noise_var = noise_var
        self.n_candidates = n_candidates
        self.num_restarts = num_restarts
        self.raw_samples = raw_samples
        self.training_epochs = training_epochs

        # 初始采样配置
        self.n_init = n_init if n_init is not None else max(5, min(20, self.dims))

        # GPU 设置
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if device is not None:
            self.device = device
            self.use_gpu = (use_gpu and device.type == 'cuda')
        else:
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        if self.use_gpu:
            gpu_id = self.device.index if self.device.index is not None else 0
            torch.cuda.set_device(gpu_id)
            print(f"SAASBO using GPU: {torch.cuda.get_device_name(gpu_id)}")

        self.dtype = torch.float64

        self.X_train = []
        self.y_train = []

        self.model = None
        self._best_f_model = None

        # 初始化阶段: 用 Sobol 序列填充
        self._sobol_engine = SobolEngine(dimension=self.dims, scramble=True,
                                         seed=np.random.randint(1e6))
        self._pending_sobol_points = []

        # 先预生成足够的 Sobol 点
        sobol_all = self._sobol_engine.draw(self.n_init + self.n_candidates).cpu().numpy()
        lb, ub = self.lb, self.ub
        sobol_scaled = lb + (ub - lb) * sobol_all
        self._pending_sobol_points = list(sobol_scaled)
        self._sobol_exhausted = False

    def _reset_sobol(self):
        """重新生成 Sobol 序列"""
        self._sobol_engine = SobolEngine(dimension=self.dims, scramble=True,
                                         seed=np.random.randint(1e6))
        sobol_all = self._sobol_engine.draw(self.n_init + self.n_candidates).cpu().numpy()
        lb, ub = self.lb, self.ub
        sobol_scaled = lb + (ub - lb) * sobol_all
        self._pending_sobol_points = list(sobol_scaled)
        self._sobol_exhausted = False

    def _to_tensor(self, x):
        """转换为 torch tensor，确保在正确设备上"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.dtype)
        if x.device != self.device:
            x = x.to(self.device)
        return x

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """建议新的候选点。

        按官方实践：一次模型拟合 + 批量(q)采集，而不是每个点重复拟合。
        """
        n_suggestions = max(1, int(n_suggestions))
        suggestions = []

        # 阶段1: 初始 Sobol 采样
        while (
            len(suggestions) < n_suggestions
            and not self._sobol_exhausted
            and self._pending_sobol_points
            and len(self.X_train) < self.n_init
        ):
            suggestions.append(self._pending_sobol_points.pop(0))

        if len(self.X_train) >= self.n_init:
            self._sobol_exhausted = True

        remaining = n_suggestions - len(suggestions)
        if remaining <= 0:
            return np.array(suggestions)

        # 阶段2: 基于 SAAS-GP + LogEI 的 BO
        if not self._build_model():
            for _ in range(remaining):
                suggestions.append(np.random.uniform(self.lb, self.ub, size=(self.dims,)))
            return np.array(suggestions)

        try:
            bounds = torch.cat([
                self._to_tensor(self.lb).reshape(1, -1),
                self._to_tensor(self.ub).reshape(1, -1),
            ])

            acqf = _build_saas_acquisition(self.model, self._best_f_model)

            with torch.no_grad():
                candidates, _ = optimize_acqf(
                    acqf,
                    bounds=bounds,
                    q=remaining,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                )

            pts = candidates.detach().cpu().numpy()
            suggestions.extend([p.copy() for p in pts])
        except Exception:
            for _ in range(remaining):
                suggestions.append(np.random.uniform(self.lb, self.ub, size=(self.dims,)))

        return np.array(suggestions)

    def _build_model(self) -> bool:
        """构建 SAAS-GP 模型。

        对最小化问题，按官方示例翻转符号（拟合 -Y，再做最大化采集）。
        """
        if len(self.X_train) < 2:
            return False

        try:
            from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
            from botorch.models.transforms import Standardize
            from botorch import fit_fully_bayesian_model_nuts
            pass
        except ImportError as e:
            print(f"Import error: {e}")
            return False

        # 使用全量历史样本训练，不做下采样
        train_X = np.array(self.X_train, dtype=np.float64)
        train_Y_raw = np.array(self.y_train, dtype=np.float64).reshape(-1, 1)
        train_Y_model = -train_Y_raw if self.is_minimizing else train_Y_raw

        train_X_t = self._to_tensor(train_X)
        train_Y_t = self._to_tensor(train_Y_model)

        try:
            self.model = SaasFullyBayesianSingleTaskGP(
                train_X=train_X_t,
                train_Y=train_Y_t,
                train_Yvar=torch.full_like(train_Y_t, self.noise_var),
                outcome_transform=Standardize(m=1),
            )

            self.model = self.model.to(self.device)
            train_X_t = train_X_t.to(self.device)
            train_Y_t = train_Y_t.to(self.device)

            fit_fully_bayesian_model_nuts(
                self.model,
                warmup_steps=self.warmup_steps,
                num_samples=self.num_samples,
                thinning=self.thinning,
                disable_progbar=True,
            )

            self.model.eval()
            self._best_f_model = train_Y_t.max()
            return True

        except Exception as e:
            print(f"SAASBO model build failed: {e}")
            self.model = None
            return False

    def observe(self, x: np.ndarray, fx: Optional[np.ndarray] = None, scores: Optional[np.ndarray] = None):
        """记录评估结果"""
        if fx is None:
            fx = np.array([self.func_wrapper(xi) for xi in x])

        # 确保 fx 是一维数组
        fx_arr = np.atleast_1d(np.array(fx)).ravel()

        for xi, fi in zip(x, fx_arr):
            self.X_train.append(np.asarray(xi, dtype=np.float64).copy())
            self.y_train.append(fi)

        super().observe(x, fx_arr)

        if self.call_count % 10 == 0:
            _cleanup_gpu_memory(sync=False)
        if self.call_count % 200 == 0:
            _cleanup_gpu_memory(sync=True)

    def optimize(self, call_budget: int, batch_size: int = None):
        """执行优化"""
        if batch_size is None:
            batch_size = self.batch_size

        while self.call_count < call_budget:
            x_suggest = self.suggest(batch_size)
            fx_suggest = np.array([self.func_wrapper(xi) for xi in x_suggest])
            self.observe(x_suggest, fx_suggest)

        return self.best_x, self.best_fx, self.call_count

    def reset(self):
        """重置优化器"""
        super().reset()
        self.X_train = []
        self.y_train = []
        self.model = None
        self._best_f_model = None

        self._reset_sobol()
        _cleanup_gpu_memory(sync=True)


def optimize_acqf(acquisition_function, bounds, q, num_restarts, raw_samples):
    """optimize_acqf 的延迟导入包装器"""
    from botorch.optim import optimize_acqf as _optimize_acqf
    return _optimize_acqf(
        acquisition_function,
        bounds=bounds,
        q=q,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )
