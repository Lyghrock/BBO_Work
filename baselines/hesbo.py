# HeSBO Optimizer - 修复版本
"""
HeSBO: A Framework for Bayesian Optimization in Embedded Subspaces
Reference: Munteanu, Nayebi, Poloczek; ICML 2019

修复内容:
1. 修复 sign 未初始化 bug：移除 self.sign = None 的覆盖
2. GPU 做 GP 推断，CPU 做函数评估
3. 参考 turbo 的简洁架构

BBO参数:
- hyper_opt_interval=20: 原始仓库默认值，不过度优化超参数
- ARD=True: 各向异性核自动识别重要维度
- box_size=1.0: 固定搜索空间，不随维度膨胀
"""
import numpy as np
import torch
import gpytorch
import gc
from typing import Optional
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from scipy.stats import norm
from baselines.base import BaseOptimizer
from torch.quasirandom import SobolEngine


def _cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


class HeSBOGPyTorchGP(ExactGP):
    """HeSBO 的 GP 模型 - 运行在低维嵌入空间"""

    def __init__(self, train_x, train_y, likelihood, dims: int = 1, device=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=dims,
                lengthscale_constraint=Interval(1e-3, 1e4)
            ),
            outputscale_constraint=Interval(1e-3, 1e4)
        )
        if device is not None:
            self.to(device)
            likelihood.to(device)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class HeSBOOptimizer(BaseOptimizer):
    """HeSBO: 低维嵌入子空间贝叶斯优化器

    修复: sign 不再被 None 覆盖
    """

    def __init__(self, func_wrapper,
                 eff_dim: int = None,
                 n_doe: int = None,
                 n_cands: int = 2000,
                 ARD: bool = True,
                 batch_size: int = 1,
                 use_gpu: bool = True,
                 device: torch.device = None,
                 n_initial_points: int = None,
                 max_train_size: int = 500,
                 use_double: bool = True,
                 training_epochs: int = 50,
                 hyper_opt_interval: int = 20,
                 box_size: float = 1.0,
                 variance: float = 1.0,
                 noise_var: float = 0.0,
                 **kwargs):
        super().__init__(func_wrapper, **kwargs)

        # 低维嵌入维度（核心超参数）
        if eff_dim is None:
            D = self.dims
            if D <= 50:
                eff_dim = max(5, D)
            elif D <= 200:
                eff_dim = max(10, int(np.sqrt(D)))
            elif D <= 1000:
                eff_dim = max(10, int(np.sqrt(D)))
            else:
                eff_dim = 20
        self.eff_dim = eff_dim

        # 初始采样数量
        if n_doe is None:
            n_doe = eff_dim + 1
        self.n_doe = n_doe

        self.n_cands = n_cands
        self.ARD = ARD
        self.batch_size = batch_size
        self.hyper_opt_interval = hyper_opt_interval
        self.box_size = box_size
        self.variance = variance
        self.noise_var = noise_var

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
            print(f"HeSBO using GPU: {torch.cuda.get_device_name(gpu_id)}, eff_dim={self.eff_dim}, box_size={self.box_size}")

        self.use_double = use_double
        self.dtype = torch.float64 if use_double else torch.float32

        self.max_train_size = max_train_size
        self.training_epochs = training_epochs

        self.model = None
        self.likelihood = None
        self.mll = None
        self.is_fitted = False

        self.X_low = []
        self.y_low = []

        # 高维投影映射
        self.high_to_low = None
        self.proj_sign = None
        self._init_projection()

        self._doe_completed = False
        self._last_valid_model = None
        self._last_valid_likelihood = None
        self._last_valid_mll = None
        self._last_y_stats = None
        self._opt_iter = 0

    def _init_projection(self):
        """初始化高维 -> 低维的哈希投影映射"""
        self.high_to_low = np.random.randint(0, self.eff_dim, size=self.dims)
        self.proj_sign = np.random.choice([-1, 1], size=self.dims)

    def _to_tensor(self, x):
        """转换为 torch tensor"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.dtype)
        if x.device != self.device:
            x = x.to(self.device)
        return x

    def _forward_project(self, x_high: np.ndarray) -> np.ndarray:
        """高维 -> 低维嵌入（前向投影）"""
        if x_high.ndim == 1:
            x_high = x_high.reshape(1, -1)
        n = x_high.shape[0]
        low_obs = np.zeros((n, self.eff_dim))
        for i in range(self.dims):
            low_obs[:, self.high_to_low[i]] += self.proj_sign[i] * x_high[:, i]
        low_obs = np.clip(low_obs, -self.box_size, self.box_size)
        return low_obs

    def _back_project(self, x_low: np.ndarray) -> np.ndarray:
        """低维 -> 高维嵌入（反向投影）"""
        if x_low.ndim == 1:
            x_low = x_low.reshape(1, -1)
        n = x_low.shape[0]
        high_obs = np.zeros((n, self.dims))
        for i in range(self.dims):
            high_obs[:, i] = self.proj_sign[i] * x_low[:, self.high_to_low[i]]
        high_obs = np.clip(high_obs, self.lb, self.ub)
        return high_obs

    def _lhs_design(self, n: int) -> np.ndarray:
        """Latin Hypercube Design - 在低维空间生成初始样本"""
        try:
            sobol = SobolEngine(dimension=self.eff_dim, scramble=True,
                                seed=np.random.randint(1e6))
            points = sobol.draw(n).cpu().numpy()
            return points * 2 * self.box_size - self.box_size
        except ImportError:
            return np.random.uniform(-self.box_size, self.box_size, size=(n, self.eff_dim))

    def _build_model(self):
        """在低维嵌入空间构建 GP 模型"""
        if len(self.X_low) < 2:
            return False

        train_size = min(len(self.X_low), self.max_train_size)
        start_idx = len(self.X_low) - train_size
        X_train = np.array(self.X_low[start_idx:])
        y_train = np.array(self.y_low[start_idx:])

        train_x = self._to_tensor(X_train)
        train_y = self._to_tensor(y_train)

        y_mean = train_y.mean()
        y_std = train_y.std()
        if y_std < 1e-8:
            y_std = 1.0
        train_y_normalized = (train_y - y_mean) / y_std

        try:
            self.likelihood = GaussianLikelihood(
                noise_constraint=Interval(1e-6, 10.0),
                noise_prior=gpytorch.priors.LogNormalPrior(-2.0, 1.0)
            ).to(self.device).to(self.dtype)

            with torch.no_grad():
                self.likelihood.noise = max(self.noise_var, 1e-3)

            self.model = HeSBOGPyTorchGP(
                train_x, train_y_normalized, self.likelihood,
                dims=self.eff_dim, device=self.device
            )

            if self.variance != 1.0:
                with torch.no_grad():
                    self.model.covar_module.outputscale = self.variance

            self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)

            self.y_mean = y_mean
            self.y_std = y_std

            self.model.train()
            self.likelihood.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

            best_loss = float('inf')
            patience = 10
            no_improve = 0

            for epoch in range(self.training_epochs):
                optimizer.zero_grad()
                try:
                    output = self.model(train_x)
                    loss = -self.mll(output, train_y_normalized)

                    if not torch.isfinite(loss):
                        no_improve += 1
                        if no_improve >= patience:
                            break
                        continue

                    loss.backward()

                    has_nan = False
                    for param in self.model.parameters():
                        if param.grad is not None and not torch.isfinite(param.grad).all():
                            has_nan = True
                            break

                    if has_nan:
                        no_improve = patience
                        break

                    optimizer.step()

                    loss_val = loss.item()
                    if loss_val < best_loss:
                        best_loss = loss_val
                        no_improve = 0
                    else:
                        no_improve += 1

                    if loss_val < 0.1 or no_improve >= patience:
                        break

                except Exception:
                    no_improve += 1
                    if no_improve >= patience:
                        break

            if best_loss >= 1e6:
                return False

            self.model.eval()
            self.likelihood.eval()
            self.is_fitted = True

            self._last_valid_model = self.model
            self._last_valid_likelihood = self.likelihood
            self._last_valid_mll = self.mll
            self._last_y_stats = (y_mean, y_std)

            _cleanup_gpu_memory()
            return True

        except Exception as e:
            print(f"HeSBO GP build error: {e}")
            return False

    def _predict(self, X_low_cand: np.ndarray):
        """在低维空间进行预测"""
        if not self.is_fitted or self.model is None:
            return None, None

        X_cand = self._to_tensor(X_low_cand)
        if X_cand.ndim == 1:
            X_cand = X_cand.unsqueeze(0)

        with torch.no_grad():
            try:
                with gpytorch.settings.cholesky_jitter(1e-4):
                    pred = self.model(X_cand)
            except Exception:
                try:
                    with gpytorch.settings.cholesky_jitter(1e-2):
                        pred = self.model(X_cand)
                except Exception:
                    return None, None

            mean = pred.mean.cpu().numpy()
            std = pred.stddev.cpu().numpy()

            mean_val = self.y_mean.cpu().item() if isinstance(self.y_mean, torch.Tensor) else self.y_mean
            std_val = self.y_std.cpu().item() if isinstance(self.y_std, torch.Tensor) else self.y_std

            mean = mean * std_val + mean_val
            std = std * std_val

        _cleanup_gpu_memory()
        return mean, std

    def _expected_improvement(self, X_low_cand: np.ndarray, xi: float = 0.001):
        """计算 EI 采集函数"""
        mean, std = self._predict(X_low_cand)
        if mean is None or std is None:
            return np.zeros(len(X_low_cand))

        std = np.maximum(std, 1e-10)

        if self.is_minimizing:
            best_y = np.min(self.y_low)
            imp = best_y - mean - xi
        else:
            best_y = np.max(self.y_low)
            imp = mean - best_y - xi

        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0

        return ei

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """建议新的候选点"""
        suggestions = []

        for _ in range(n_suggestions):
            # 阶段1: 初始 DOE
            if not self._doe_completed:
                n_remaining = self.n_doe - len(self.X_low)
                if n_remaining > 0:
                    lhs_points = self._lhs_design(n_remaining)
                    x_high = self._back_project(lhs_points)
                    for pt in x_high:
                        suggestions.append(pt)
                    if len(suggestions) >= n_suggestions:
                        break
                    self._doe_completed = True
                    continue

                self._doe_completed = True

            # 阶段2: 基于 GP-EI 的 BO
            if not self._build_model():
                if self._last_valid_model is not None:
                    self.model = self._last_valid_model
                    self.likelihood = self._last_valid_likelihood
                    self.mll = self._last_valid_mll
                    self.y_mean, self.y_std = self._last_y_stats
                    self.model.eval()
                    self.likelihood.eval()
                    self.is_fitted = True
                else:
                    lhs_point = self._lhs_design(1)
                    x_high = self._back_project(lhs_point)
                    suggestions.append(x_high[0])
                    self._opt_iter += 1
                    continue

            X_cand_low = self._lhs_design(self.n_cands)
            ei_values = self._expected_improvement(X_cand_low)

            if ei_values is None or not np.isfinite(ei_values).any() or np.max(ei_values) <= 0:
                idx = np.random.randint(len(X_cand_low))
            else:
                idx = np.argmax(ei_values)

            x_low_best = X_cand_low[idx:idx+1]
            x_high_best = self._back_project(x_low_best)

            suggestions.append(x_high_best[0])
            self._opt_iter += 1

        return np.array(suggestions)

    def observe(self, x: np.ndarray, fx=None, scores: Optional[np.ndarray] = None):
        """记录评估结果"""
        if fx is None:
            fx = np.array([self.func_wrapper(xi) for xi in x])

        for xi, fi in zip(x, fx):
            x_low = self._forward_project(xi)
            self.X_low.append(x_low[0])
            self.y_low.append(fi)

        super().observe(x, fx)

        if self.call_count % 10 == 0:
            _cleanup_gpu_memory()

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
        self.X_low = []
        self.y_low = []
        self.model = None
        self.likelihood = None
        self.mll = None
        self.is_fitted = False
        self._doe_completed = False
        self._last_valid_model = None
        self._last_valid_likelihood = None
        self._last_valid_mll = None
        self._last_y_stats = None
        self._opt_iter = 0
        self._init_projection()
