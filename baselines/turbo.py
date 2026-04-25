# TuRBO Optimizer (Trust Region Bayesian Optimization) - Based on Official BoTorch Implementation
"""
关键点:
1. Trust Region: 初始0.8, length_min=0.5^7, length_max=1.6
2. success_tolerance=10 (需要10次成功才扩展)
3. failure_tolerance = max(4/batch_size, dim/batch_size)
4. 每次成功length *= 2.0, 每次失败length /= 2.0
5. GP训练前标准化Y数据
6. MaternKernel lengthscale约束: Interval(0.005, 4.0)
7. Likelihood noise约束: Interval(1e-8, 1e-3)
8. 定期清理GPU缓存防止OOM

GPU修复:
- 只在 TuRBO.__init__ 中调用一次 torch.cuda.set_device()
- GPyTorchGP 内部将 likelihood 和 model 都放到正确 device
- _train_gpr 不再重复调用 set_device()
"""
import numpy as np
import torch
import gpytorch
import gc
import math
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from scipy.stats import norm
from typing import Optional, Tuple
from baselines.base import BaseOptimizer


def _cleanup_gpu_memory(sync: bool = False):
    """清理GPU内存的辅助函数。默认不做强制同步以减少阻塞。"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if sync:
            torch.cuda.synchronize()
    gc.collect()


class GPyTorchGP(ExactGP):
    """GPyTorch Gaussian Process Model for TuRBO - 按照官方配置
    
    GPU修复: likelihood 和 model 都在 __init__ 中移动到 device
    """
    
    def __init__(self, train_x, train_y, likelihood, dims: int = 1, device=None):
        super(GPyTorchGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=dims,
                lengthscale_constraint=Interval(0.005, 4.0)
            ),
            outputscale_constraint=Interval(0.001, 100.0)
        )
        # 修复: 将模型和likelihood都显式移动到device
        if device is not None:
            self.to(device)
            likelihood.to(device)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class TurboState:
    """Turbo状态类 - 按照官方实现"""
    
    def __init__(self, dim: int, batch_size: int = 1):
        self.dim = dim
        self.batch_size = batch_size
        self.length = 0.8  # 初始长度
        self.length_min = 0.5 ** 7  # ≈ 0.0078125
        self.length_max = 1.6
        self.failure_counter = 0
        self.failure_tolerance = max(4.0 / batch_size, float(dim) / batch_size)
        self.success_counter = 0
        self.success_tolerance = 10  # 官方使用10
        self.best_value = -float('inf')
        self.restart_triggered = False
    
    def update(self, Y_next):
        """更新状态 - 官方实现逻辑"""
        if np.max(Y_next) > self.best_value + 1e-3 * abs(self.best_value):
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.success_counter = 0
            self.failure_counter += 1
        
        if self.success_counter == self.success_tolerance:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        
        elif self.failure_counter == self.failure_tolerance:
            self.length /= 2.0
            self.failure_counter = 0
        
        self.best_value = max(self.best_value, np.max(Y_next))
        
        if self.length < self.length_min:
            self.restart_triggered = True
        
        return self


class TuRBO:
    """TuRBO优化器 - 基于官方BoTorch实现
    
    GPU修复: 只在 __init__ 中调用一次 torch.cuda.set_device()
    """
    
    def __init__(self, func_wrapper,
                 n_trusts: int = 1,
                 dims: int = None,
                 max_evals: int = 1000,
                 batch_size: int = 4,
                 use_gpu: bool = True,
                 device: torch.device = None):
        self.func_wrapper = func_wrapper
        self.lb = func_wrapper.lb
        self.ub = func_wrapper.ub
        self.dims = dims if dims is not None else func_wrapper.dims
        self.is_minimizing = func_wrapper.is_minimizing
        self.sign = -1.0 if self.is_minimizing else 1.0

        self.n_trusts = n_trusts
        self.max_evals = max_evals
        self.batch_size = batch_size

        # GPU设置: 只在这里调用一次 set_device()
        if device is not None:
            self.device = device
            self.use_gpu = (use_gpu and device.type == 'cuda')
        else:
            self.use_gpu = use_gpu and torch.cuda.is_available()
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        if self.use_gpu:
            gpu_id = self.device.index if self.device.index is not None else 0
            torch.cuda.set_device(gpu_id)
            # 允许 TF32 提升 Ampere+ GPU 上的矩阵计算吞吐
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            print(f"TuRBO using GPU: {torch.cuda.get_device_name(gpu_id)}")
        
        # 初始化trust region states
        self.states = [TurboState(self.dims, batch_size) for _ in range(n_trusts)]
        
        self.X = []
        self.y = []
        self.call_count = 0
        
        self.best_x = None
        self.best_fx = float('inf') if self.is_minimizing else float('-inf')
        self._last_good_x = None  # 用于极端PSD情况下的备选中心点

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if x.device != self.device:
            x = x.to(self.device)
        return x

    def _generate_candidates(self, lb, ub, n: int):
        """生成候选点"""
        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=self.dims, scramble=True, seed=np.random.randint(1e6))
            cands = sobol.draw(n).cpu().numpy()
            return (ub - lb) * cands + lb
        except ImportError:
            return np.random.uniform(lb, ub, size=(n, self.dims))
    
    def _train_gpr(self, X, y):
        """训练GPR模型 - 关键：标准化Y数据
        
        GPU修复: 不再重复调用 set_device()，设备已在 TuRBO.__init__ 设置
        """
        if len(X) < 2:
            return None, None, None, None
        
        X_np = np.array(X)
        y_np = np.array(y)

        # 过滤非有限样本，避免 NaN/Inf 传播到核矩阵分解
        if X_np.ndim == 1:
            finite_mask = np.isfinite(X_np) & np.isfinite(y_np)
        else:
            finite_mask = np.isfinite(X_np).all(axis=1) & np.isfinite(y_np)
        X_np = X_np[finite_mask]
        y_np = y_np[finite_mask]
        if len(X_np) < 2:
            return None, None, None, None

        train_x = self._to_tensor(X_np).unsqueeze(-1) if X_np.ndim == 1 else self._to_tensor(X_np)
        train_y_original = y_np
        
        y_mean = train_y_original.mean()
        y_std = train_y_original.std()
        if y_std < 1e-8:
            y_std = 1.0
        train_y = self._to_tensor((train_y_original - y_mean) / y_std).squeeze(-1)
        
        if train_x.ndim == 1:
            train_x = train_x.unsqueeze(-1)
        
        try:
            likelihood = GaussianLikelihood(
                noise_constraint=Interval(1e-8, 1e-3)
            ).to(self.device)
            
            # GPU修复: GPyTorchGP.__init__ 内部会将 likelihood 和 model 移动到 device
            model = GPyTorchGP(train_x, train_y, likelihood, dims=self.dims, device=self.device)
            mll = ExactMarginalLogLikelihood(likelihood, model)
            
            model.train()
            likelihood.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            
            for _ in range(50):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()
                if loss.item() < 0.01:
                    break
            
            model.eval()
            likelihood.eval()
            
            _cleanup_gpu_memory(sync=False)
            
            return model, likelihood, y_mean, y_std
        except Exception as e:
            print(f"GPR fit error: {e}")
            return None, None, None, None
    
    def _get_best_candidate(self, state, n_cands: int = 1500):
        """获取最优候选点 - 使用Thompson Sampling"""
        if len(self.X) < 2:
            return np.random.uniform(self.lb, self.ub)

        gp_result = self._train_gpr(np.array(self.X), np.array(self.y))
        if gp_result[0] is None:
            return np.random.uniform(self.lb, self.ub)

        model, likelihood, y_mean, y_std = gp_result

        if self.is_minimizing:
            best_idx = np.argmin(self.y)
        else:
            best_idx = np.argmax(self.y)
        x_center = self.X[best_idx]

        tr_length = state.length * (self.ub - self.lb).max()
        lb = np.clip(x_center - tr_length / 2, self.lb, self.ub)
        ub = np.clip(x_center + tr_length / 2, self.lb, self.ub)

        X_cand = self._generate_candidates(lb, ub, n_cands)

        X_cand_tensor = self._to_tensor(X_cand)
        if X_cand_tensor.ndim == 1:
            X_cand_tensor = X_cand_tensor.unsqueeze(0)

        # 优先尝试 Thompson Sampling；若模型接口/数值不稳定则降级
        try:
            if hasattr(model, "posterior"):
                from botorch.generation import MaxPosteriorSampling
                sampler = MaxPosteriorSampling(model=model, replacement=False)
                with torch.no_grad():
                    X_next = sampler(X_cand_tensor, num_samples=self.batch_size)
                _cleanup_gpu_memory(sync=False)
                return X_next.cpu().numpy()[0] if self.batch_size == 1 else X_next.cpu().numpy()
        except Exception as e:
            print(f"TS sampling fallback due to: {e}")

        # 降级1：用 GP 均值/方差计算 EI；若仍失败则再降级到随机
        try:
            with torch.no_grad():
                pred = model(X_cand_tensor)
                mean = pred.mean.cpu().numpy()
                std = pred.stddev.cpu().numpy()

            _cleanup_gpu_memory(sync=False)

            mean = mean * y_std + y_mean

            if self.is_minimizing:
                best_y = np.min(self.y)
                imp = best_y - mean
            else:
                best_y = np.max(self.y)
                imp = mean - best_y

            std = np.maximum(std, 1e-10)
            Z = imp / std
            ei = imp * norm.cdf(Z) + std * norm.pdf(Z)

            idx = np.argmax(ei)
            return X_cand[idx]
        except Exception as e:
            print(f"GP predict fallback due to: {e}")

        # 降级2（极端PSD）：用最近的最优点附近重新采样，不重新训练GP
        _cleanup_gpu_memory(sync=True)
        try:
            if hasattr(self, "_last_good_x") and self._last_good_x is not None:
                center = self._last_good_x
            else:
                center = x_center
            tr_len = max(state.length * (self.ub - self.lb).max(), 1e-6)
            lb_safe = np.clip(center - tr_len / 2, self.lb, self.ub)
            ub_safe = np.clip(center + tr_len / 2, self.lb, self.ub)
            safe_cand = np.random.uniform(lb_safe, ub_safe, size=(max(n_cands // 4, 100), self.dims))
            idx = np.random.randint(len(safe_cand))
            return safe_cand[idx]
        except Exception:
            return np.random.uniform(self.lb, self.ub, size=(self.dims,))
    
    def suggest(self, n_suggestions: int = 1):
        """建议新的采样点"""
        suggestions = []
        
        for i in range(n_suggestions):
            state = self.states[self.call_count % len(self.states)]
            x_new = self._get_best_candidate(state)
            if x_new.ndim > 1:
                x_new = x_new[0]
            suggestions.append(x_new)
        
        return np.array(suggestions)
    
    def observe(self, x, fx):
        """观察评估结果"""
        fx_internal = fx * self.sign
        
        for state in self.states:
            state.update(fx_internal)
        
        for state in self.states:
            if state.restart_triggered:
                state.length = 0.8
                state.failure_counter = 0
                state.success_counter = 0
                state.restart_triggered = False
                state.best_value = -float('inf')
        
        for xi, fi in zip(x, fx):
            self.X.append(xi)
            self.y.append(fi)
            self.call_count += 1

            if self.is_minimizing:
                if fi < self.best_fx:
                    self.best_fx = fi
                    self.best_x = xi.copy()
                    self._last_good_x = xi.copy()
            else:
                if fi > self.best_fx:
                    self.best_fx = fi
                    self.best_x = xi.copy()
                    self._last_good_x = xi.copy()

        if self.call_count % 10 == 0:
            _cleanup_gpu_memory(sync=False)
        if self.call_count % 200 == 0:
            _cleanup_gpu_memory(sync=True)
    
    def optimize(self, call_budget: int = None, batch_size: int = None):
        """执行优化"""
        if call_budget is None:
            call_budget = self.max_evals
        if batch_size is None:
            batch_size = self.batch_size
        
        while self.call_count < call_budget:
            x_suggest = self.suggest(batch_size)
            fx_suggest = np.array([self.func_wrapper(xi) for xi in x_suggest])
            self.observe(x_suggest, fx_suggest)
        
        return self.best_x, self.best_fx, self.call_count
    
    def reset(self):
        """重置优化器"""
        self.states = [TurboState(self.dims, self.batch_size) for _ in range(self.n_trusts)]
        self.X = []
        self.y = []
        self.call_count = 0
        self.best_x = None
        self.best_fx = float('inf') if self.is_minimizing else float('-inf')
        self._last_good_x = None


class TurboOptimizer(BaseOptimizer):
    """TuRBO优化器包装器 - 兼容BaseOptimizer接口

    GPU修复:
    - 不再重复调用 torch.cuda.set_device()（只由 TuRBO.__init__ 调用一次）
    - 不再重复 pop device from kwargs（create_optimizer 已经处理）
    """

    def __init__(self, func_wrapper,
                 n_trusts: int = 1,
                 max_evals: int = 1000,
                 batch_size: int = 4,
                 use_gpu: bool = True,
                 device: torch.device = None,
                 **kwargs):
        super().__init__(func_wrapper, **kwargs)

        self.turbo = TuRBO(
            func_wrapper=func_wrapper,
            n_trusts=n_trusts,
            dims=self.dims,
            max_evals=max_evals,
            batch_size=batch_size,
            use_gpu=use_gpu,
            device=device,
        )

        self.device = self.turbo.device
        self.use_gpu = self.turbo.use_gpu
    
    def suggest(self, n_suggestions: int = 1):
        return self.turbo.suggest(n_suggestions)
    
    def observe(self, x, fx=None):
        if fx is None:
            fx = np.array([self.func_wrapper(xi) for xi in x])
        
        self.turbo.observe(x, fx)
        
        self.best_x = self.turbo.best_x
        self.best_fx = self.turbo.best_fx
        self.call_count = self.turbo.call_count
    
    def optimize(self, call_budget: int, batch_size: int = None):
        best_x, best_fx, calls = self.turbo.optimize(call_budget, batch_size)
        self.best_x = best_x
        self.best_fx = best_fx
        self.call_count = calls
        return best_x, best_fx, calls
    
    def reset(self):
        super().reset()
        self.turbo.reset()
