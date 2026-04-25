# BAxUS Optimizer - 重构版本
"""
BAxUS: Bayesian Optimization with Adaptively Expanding Subspaces
Reference: Papenmeier, Nardi, Poloczek; NeurIPS 2022

重构原则（参考 turbo 架构）:
1. GPU 做 GP 推断，CPU 做函数评估（通过 func_wrapper）
2. 不依赖 baxus 库的私有 API
3. 简洁的 suggest/observe 循环
4. Trust Region 机制：当 TR 收缩到最小值时重启
"""
import numpy as np
import torch
import gpytorch
import gc
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from scipy.stats import norm
from baselines.base import BaseOptimizer


def _cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


class BAXUSGP(ExactGP):
    """BAxUS 的 GP 模型 - 运行在低维嵌入空间"""

    def __init__(self, train_x, train_y, likelihood, dims: int = 1, device=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(
            MaternKernel(
                nu=2.5,
                ard_num_dims=dims,
                lengthscale_constraint=Interval(0.005, 4.0)
            ),
            outputscale_constraint=Interval(0.001, 100.0)
        )
        if device is not None:
            self.to(device)
            likelihood.to(device)

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class BAxUSOptimizer(BaseOptimizer):
    """BAxUS: 自适应扩展子空间贝叶斯优化器

    重构核心:
    - 不再使用 baxus 库的 EmbeddedTuRBO（依赖私有 API）
    - 改用简洁的嵌入 + GP 架构（参考 turbo）
    - Trust Region 机制：当 length <= length_min 时重启
    - GPU 做 GP 推断，CPU 做函数评估
    """

    def __init__(self, func_wrapper,
                 batch_size: int = 1,
                 use_gpu: bool = True,
                 device: torch.device = None,
                 target_dim: int = None,
                 n_initial_points: int = None,
                 training_epochs: int = 100,
                 n_candidates: int = 5000,
                 **kwargs):
        super().__init__(func_wrapper, **kwargs)

        self.batch_size = batch_size
        self.training_epochs = training_epochs
        self.n_candidates = n_candidates

        # sign 初始化（修复：必须初始化才能用于 TR 更新）
        self.sign = -1.0 if self.is_minimizing else 1.0

        # 低维嵌入维度（核心超参数）- 改为更激进的设置
        if target_dim is None:
            # 使用与 HeSBO 相似的策略
            D = self.dims
            if D <= 50:
                target_dim = max(5, D)
            elif D <= 200:
                target_dim = max(10, int(np.sqrt(D)))
            elif D <= 1000:
                target_dim = max(10, int(np.sqrt(D)))
            else:
                target_dim = 20
        self.target_dim = target_dim

        # 初始采样数量
        if n_initial_points is None:
            self.n_initial_points = max(10, target_dim + 1)
        else:
            self.n_initial_points = n_initial_points

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
            print(f"BAxUS using GPU: {torch.cuda.get_device_name(gpu_id)}, target_dim={self.target_dim}")

        self.dtype = torch.float64

        # Trust Region 状态
        self.tr_length = 0.8
        self.tr_length_init = 0.8
        self.tr_length_min = 0.05
        self.tr_length_max = 1.6
        self.succtol = 5
        self.failtol = 5
        self.succcount = 0
        self.failcount = 0
        self.success_decision_factor = 0.001

        # 随机嵌入投影（修复：与 baxus 库不同的简洁实现）
        self._init_embedding()

        # 数据存储
        self.X_low = []  # 低维空间数据
        self.y_low = []  # 原始函数值

        # GP 模型
        self.model = None
        self.likelihood = None
        self.y_mean = 0.0
        self.y_std = 1.0

        self._doe_completed = False
        self._initialized = False

    def _init_embedding(self):
        """初始化随机嵌入投影"""
        # 使用随机投影从高维映射到低维
        self.proj_matrix = np.random.randn(self.dims, self.target_dim)
        # 归一化每一列
        norms = np.linalg.norm(self.proj_matrix, axis=0, keepdims=True)
        self.proj_matrix = self.proj_matrix / (norms + 1e-8)

        # TR 中心点（初始化为搜索空间中心）
        self.tr_center = np.zeros(self.dims)

    def _to_tensor(self, x):
        """转换为 torch tensor"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        x = x.to(self.dtype)
        if x.device != self.device:
            x = x.to(self.device)
        return x

    def _project_high_to_low(self, x_high: np.ndarray) -> np.ndarray:
        """高维 -> 低维投影"""
        if x_high.ndim == 1:
            x_high = x_high.reshape(1, -1)
        x_low = x_high @ self.proj_matrix  # (n, dims) @ (dims, target_dim) -> (n, target_dim)
        return x_low

    def _project_low_to_high(self, x_low: np.ndarray) -> np.ndarray:
        """低维 -> 高维投影"""
        if x_low.ndim == 1:
            x_low = x_low.reshape(1, -1)
        x_high = x_low @ self.proj_matrix.T  # (n, target_dim) @ (target_dim, dims) -> (n, dims)
        return x_high

    def _lhs_design(self, n: int) -> np.ndarray:
        """Latin Hypercube Design - 在低维空间生成样本"""
        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=self.target_dim, scramble=True,
                                seed=np.random.randint(1e6))
            points = sobol.draw(n).cpu().numpy()
            return points * 2 - 1  # 映射到 [-1, 1]
        except ImportError:
            return np.random.uniform(-1, 1, size=(n, self.target_dim))

    def _sample_within_tr(self, n: int) -> np.ndarray:
        """在当前 trust region 内生成样本（低维空间）"""
        # TR 是以 best 点为中心，length 为半径的超立方体
        # 在低维空间采样
        lb_low = -self.tr_length
        ub_low = self.tr_length

        points = np.random.uniform(lb_low, ub_low, size=(n, self.target_dim))
        return points

    def _build_model(self) -> bool:
        """构建 GP 模型"""
        if len(self.X_low) < 2:
            return False

        train_x = np.array(self.X_low[-500:])
        train_y = np.array(self.y_low[-500:])

        train_x_t = self._to_tensor(train_x)
        train_y_original = train_y.copy()

        self.y_mean = train_y.mean()
        self.y_std = train_y.std()
        if self.y_std < 1e-8:
            self.y_std = 1.0
        train_y_t = self._to_tensor((train_y - self.y_mean) / self.y_std).squeeze(-1)

        try:
            self.likelihood = GaussianLikelihood(
                noise_constraint=Interval(1e-8, 1e-3)
            ).to(self.device).to(self.dtype)

            self.model = BAXUSGP(
                train_x_t, train_y_t, self.likelihood,
                dims=self.target_dim, device=self.device
            )

            self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
            self.model.train()
            self.likelihood.train()

            optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            for _ in range(self.training_epochs):
                optimizer.zero_grad()
                output = self.model(train_x_t)
                loss = -self.mll(output, train_y_t)
                if not torch.isfinite(loss):
                    break
                loss.backward()
                optimizer.step()
                if loss.item() < 0.01:
                    break

            self.model.eval()
            self.likelihood.eval()
            _cleanup_gpu_memory()
            return True

        except Exception as e:
            print(f"BAxUS GP build error: {e}")
            return False

    def _predict(self, X_low: np.ndarray):
        """预测"""
        if self.model is None:
            return None, None

        X_t = self._to_tensor(X_low)
        if X_t.ndim == 1:
            X_t = X_t.unsqueeze(0)

        with torch.no_grad(), gpytorch.settings.cholesky_jitter(1e-4):
            try:
                pred = self.model(X_t)
                mean = pred.mean.cpu().numpy()
                std = pred.stddev.cpu().numpy()
                mean = mean * self.y_std + self.y_mean
                std = std * self.y_std
                return mean, std
            except Exception:
                return None, None

    def _thompson_sampling_candidate(self) -> np.ndarray:
        """用 Thompson Sampling 生成候选点"""
        if self.model is None:
            return self._lhs_design(1)[0]

        # 找到 best 观测点的低维坐标
        if self.is_minimizing:
            best_idx = np.argmin(self.y_low)
        else:
            best_idx = np.argmax(self.y_low)
        x_best_low = np.array(self.X_low[best_idx])

        # 在 best 点附近生成候选点（在 TR 内）
        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=self.target_dim, scramble=True,
                                seed=np.random.randint(1e6))
            sobol_cands = sobol.draw(self.n_candidates).cpu().numpy()
            # 以 best 点为中心，在 TR 范围内采样
            X_cand = x_best_low + (sobol_cands * 2 - 1) * self.tr_length
        except ImportError:
            X_cand = np.random.uniform(
                x_best_low - self.tr_length,
                x_best_low + self.tr_length,
                size=(self.n_candidates, self.target_dim)
            )

        # GP 预测
        mean, std = self._predict(X_cand)
        if mean is None:
            idx = np.random.randint(len(X_cand))
            return X_cand[idx]

        # 修复: 正确的 Thompson Sampling - 从预测分布中采样
        # 对于最小化：取最低的采样值
        # 对于最大化：取最高的采样值
        samples = mean + std * np.random.randn(len(mean))
        if self.is_minimizing:
            idx = np.argmin(samples)
        else:
            idx = np.argmax(samples)
        return X_cand[idx]

    def _update_tr(self, fx_new: float):
        """更新 Trust Region"""
        if len(self.y_low) > 0:
            y_best = np.min(self.y_low) if self.is_minimizing else np.max(self.y_low)
        else:
            y_best = float('inf') if self.is_minimizing else float('-inf')

        margin = self.success_decision_factor * abs(y_best) if np.isfinite(y_best) else 0.0
        if self.is_minimizing:
            improved = fx_new < (y_best - margin)
        else:
            improved = fx_new > (y_best + margin)

        if improved:
            self.succcount += 1
            self.failcount = 0
        else:
            self.succcount = 0
            self.failcount += 1

        # 调整 TR 长度
        if self.succcount == self.succtol:
            self.tr_length = min(2.0 * self.tr_length, self.tr_length_max)
            self.succcount = 0
        elif self.failcount == self.failtol:
            self.tr_length /= 2.0
            self.failcount = 0

        # 检查是否需要重启
        if self.tr_length <= self.tr_length_min:
            self._restart()
            return True  # 表示触发了重启
        return False

    def _restart(self):
        """重启 TR"""
        self.tr_length = self.tr_length_init
        self.succcount = 0
        self.failcount = 0
        self.X_low = []
        self.y_low = []
        self._doe_completed = False

    def suggest(self, n_suggestions: int = 1):
        """建议新的候选点"""
        suggestions = []

        for _ in range(n_suggestions):
            # 阶段1: 初始 LHS 采样
            if not self._doe_completed:
                n_remaining = self.n_initial_points - len(self.X_low)
                if n_remaining > 0:
                    # 一次性生成所有剩余的 LHS 点
                    lhs_points = self._lhs_design(n_remaining)
                    x_high = self._project_low_to_high(lhs_points)
                    x_high = np.clip(x_high, self.lb, self.ub)
                    # 将所有剩余点追加到 suggestions（逐个追加，保持接口一致）
                    for pt in x_high:
                        suggestions.append(pt)
                    # 如果追加的点足够，跳出循环
                    if len(suggestions) >= n_suggestions:
                        break
                    # 否则继续循环，触发 _doe_completed 标志
                    self._doe_completed = True
                    continue

                self._doe_completed = True

            # 阶段2: GP 引导搜索
            if not self._build_model():
                # 如果 GP 构建失败，用随机点
                x_low = self._lhs_design(1)[0]
                x_high = self._project_low_to_high(x_low.reshape(1, -1))
                x_high = np.clip(x_high, self.lb, self.ub)
                suggestions.append(x_high[0])
                continue

            # 生成候选点
            x_low = self._thompson_sampling_candidate()
            x_high = self._project_low_to_high(x_low.reshape(1, -1))
            x_high = np.clip(x_high, self.lb, self.ub)
            suggestions.append(x_high[0])

        return np.array(suggestions)

    def observe(self, x: np.ndarray, fx=None):
        """记录评估结果"""
        if fx is None:
            fx = np.array([self.func_wrapper(xi) for xi in x])

        for xi, fi in zip(x, fx):
            # 转换到低维空间
            x_low = self._project_high_to_low(xi.reshape(1, -1))[0]
            self.X_low.append(x_low)
            self.y_low.append(fi)

            # 更新 TR
            self._update_tr(fi)

        super().observe(x, fx)

        if self.call_count % 10 == 0:
            _cleanup_gpu_memory()

    def optimize(self, call_budget: int, batch_size: int = None):
        """执行优化（参考 turbo 的模式）"""
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
        self.tr_length = self.tr_length_init
        self.succcount = 0
        self.failcount = 0
        self.X_low = []
        self.y_low = []
        self.model = None
        self.likelihood = None
        self._doe_completed = False
        self._init_embedding()
        _cleanup_gpu_memory()
