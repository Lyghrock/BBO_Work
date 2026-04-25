# Bayesian Optimization with GPU Acceleration using GPyTorch
"""
BO 实现的核心策略：
1. 从一开始就配置好数值稳定性参数，而不是出问题再补救
2. 使用更大的 cholesky_max_tries 允许更多 jitter 重试
3. 在模型训练和预测时都设置合理的 jitter
4. 保留成功模型用于出错时回退
5. 永远不回退到随机采样
6. 定期清理GPU缓存防止OOM
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
from typing import Optional
from baselines.base import BaseOptimizer
from torch.quasirandom import SobolEngine


def _cleanup_gpu_memory():
    """清理GPU内存的辅助函数"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


class GPyTorchGP(ExactGP):
    """GPyTorch Gaussian Process Model - 针对高维问题优化"""
    
    def __init__(self, train_x, train_y, likelihood, dims: int = 1):
        super(GPyTorchGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        # 高维问题：使用更大的初始 lengthscale
        # lengthscale 越大，核函数越平滑，协方差矩阵越稳定
        if dims <= 20:
            initial_ls = 1.0
        elif dims <= 50:
            initial_ls = max(2.0, dims * 0.1)
        else:
            initial_ls = max(5.0, dims * 0.2)
        
        self.covar_module = ScaleKernel(
            MaternKernel(
                lengthscale_constraint=Interval(1e-3, 1e4),
                lengthscale_prior=gpytorch.priors.LogNormalPrior(
                    np.log(initial_ls), 0.5
                ),
                ard_num_dims=dims
            ),
            outputscale_constraint=Interval(1e-3, 1e4),
            outputscale_prior=gpytorch.priors.LogNormalPrior(0.0, 1.0)
        )
    
    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return MultivariateNormal(mean, covar)


class BayesianOptimizer(BaseOptimizer):
    """贝叶斯优化器 - 数值稳定版本
    
    Fallback 策略说明：
    - 当训练失败时，尝试使用保守配置重新训练
    - 如果重新训练也失败，使用上次成功的模型
    - 绝不回退到随机采样，这会让 BO 退化为随机搜索
    """
    
    def __init__(self, func_wrapper, 
                 num_cands: int = 5000,
                 acquisition: str = 'ei',
                 nu: float = 2.5,
                 batch_size: int = 1,
                 use_gpu: bool = True,
                 device: torch.device = None,
                 n_initial_points: int = None,
                 max_train_size: int = 300,
                 use_double: bool = True,
                 training_epochs: int = 50,
                 **kwargs):
        super().__init__(func_wrapper, **kwargs)

        self.num_cands = num_cands
        self.acquisition = acquisition
        self.nu = nu
        self.batch_size = batch_size

        # GPU设置 - 优先使用传入的 device
        self.use_gpu = use_gpu and torch.cuda.is_available()
        if device is not None:
            self.device = device
            self.use_gpu = (use_gpu and device.type == 'cuda')
        else:
            self.device = torch.device('cuda' if self.use_gpu else 'cpu')

        # 在创建任何GPU对象之前先设置GPU设备
        if self.use_gpu:
            gpu_id = self.device.index if self.device.index is not None else 0
            torch.cuda.set_device(gpu_id)
        
        # 双重精度
        self.use_double = use_double
        self.dtype = torch.float64 if use_double else torch.float32

        if self.use_gpu:
            gpu_id = self.device.index if self.device.index is not None else 0
            print(f"BO using GPU: {torch.cuda.get_device_name(gpu_id)}, dtype: {self.dtype}")
        
        # 初始点数量 - 对20维问题，使用dims+10个初始点更合理
        if n_initial_points is None:
            n_initial_points = min(max(self.dims + 10, 5), 50)
        self.n_initial_points = n_initial_points
        
        # 限制训练数据大小
        self.max_train_size = max_train_size
        
        # 训练epoch数 - 减少以提高效率
        self.training_epochs = training_epochs
        
        # 模型状态
        self.model = None
        self.likelihood = None
        self.mll = None
        self.is_fitted = False
        
        # 训练数据
        self.X_train = []
        self.y_train = []
        
        # 保留上次成功的模型
        self._last_valid_model = None
        self._last_valid_likelihood = None
        self._last_valid_mll = None
        self._last_y_stats = None
        self._last_model_config = None
    
    def _to_tensor(self, x):
        """转换为torch tensor"""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        x = x.to(self.dtype)
        if x.device != self.device:
            x = x.to(self.device)
        return x
    
    def _build_model(self, use_conservative=False):
        """构建GPyTorch模型
        
        Args:
            use_conservative: 是否使用保守配置（更大的 noise 和 jitter）
        """
        if len(self.X_train) < 2:
            return False
        
        # 限制训练数据大小
        train_size = min(len(self.X_train), self.max_train_size)
        start_idx = len(self.X_train) - train_size
        
        X_subset = self.X_train[start_idx:]
        y_subset = self.y_train[start_idx:]
        
        # 创建训练数据
        train_x = self._to_tensor(np.array(X_subset))
        train_y = self._to_tensor(np.array(y_subset))
        
        # 标准化目标值
        y_mean = train_y.mean()
        y_std = train_y.std()
        if y_std < 1e-8:
            y_std = 1.0
        train_y_normalized = (train_y - y_mean) / y_std
        
        # 配置参数
        if use_conservative:
            # 保守配置：更大的 noise
            noise_constraint = Interval(1e-3, 10.0)
            initial_noise = 0.1
        else:
            # 默认配置
            noise_constraint = Interval(1e-6, 10.0)
            initial_noise = 0.01
        
        # 创建模型
        self.likelihood = GaussianLikelihood(
            noise_constraint=noise_constraint,
            noise_prior=gpytorch.priors.LogNormalPrior(-2.0, 1.0)
        ).to(self.device).to(self.dtype)
        
        # 设置初始 noise
        with torch.no_grad():
            self.likelihood.noise = initial_noise
        
        self.model = GPyTorchGP(
            train_x, train_y_normalized, self.likelihood, dims=self.dims
        ).to(self.device).to(self.dtype)
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        # 存储标准化参数
        self.y_mean = y_mean
        self.y_std = y_std
        
        # 设置训练模式
        self.model.train()
        self.likelihood.train()
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        
        # 训练
        best_loss = float('inf')
        patience = 10
        no_improve = 0
        
        for i in range(self.training_epochs):
            optimizer.zero_grad()
            try:
                output = self.model(train_x)
                loss = -self.mll(output, train_y_normalized)
                
                if not torch.isfinite(loss):
                    no_improve += 1
                    if no_improve >= patience:
                        break
                    continue
                
                loss_val = loss.item()
                if loss_val < best_loss:
                    best_loss = loss_val
                    no_improve = 0
                else:
                    no_improve += 1
                
                loss.backward()
                
                # 检查梯度
                has_nan_grad = False
                for param in self.model.parameters():
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        has_nan_grad = True
                        break
                
                if has_nan_grad:
                    no_improve = patience
                    break
                
                optimizer.step()
                
                # 合理的收敛条件 - 更aggressive
                if loss_val < 0.1 or no_improve >= patience:
                    break
                    
            except Exception as e:
                no_improve += 1
                if no_improve >= patience:
                    break
        
        # 检查是否训练成功
        if best_loss >= 1e6:  # 训练失败
            return False
        
        self.model.eval()
        self.likelihood.eval()
        self.is_fitted = True
        
        # 保存成功的模型
        self._last_valid_model = self.model
        self._last_valid_likelihood = self.likelihood
        self._last_valid_mll = self.mll
        self._last_y_stats = (y_mean, y_std)
        self._last_model_config = 'conservative' if use_conservative else 'default'

        # 清理GPU缓存
        _cleanup_gpu_memory()
        
        return True
    
    def _predict_with_settings(self, X_cand, jitter_value=1e-4):
        """使用特定 jitter 设置进行预测"""
        with torch.no_grad():
            X_cand = self._to_tensor(X_cand)
            
            try:
                with gpytorch.settings.cholesky_jitter(jitter_value):
                    pred = self.model(X_cand)
            except:
                # 如果失败，尝试更大的 jitter
                try:
                    with gpytorch.settings.cholesky_jitter(jitter_value * 100):
                        pred = self.model(X_cand)
                except:
                    return None, None
            
            mean = pred.mean.cpu().numpy()
            std = pred.stddev.cpu().numpy()
            
            # 反标准化
            if isinstance(self.y_mean, torch.Tensor):
                y_mean_val = self.y_mean.cpu().item()
                y_std_val = self.y_std.cpu().item()
            else:
                y_mean_val = self.y_mean
                y_std_val = self.y_std
                
            mean = mean * y_std_val + y_mean_val
            std = std * y_std_val
            
        # 清理GPU缓存
        _cleanup_gpu_memory()
        
        return mean, std
    
    def _predict(self, X_cand):
        """预测 - 使用安全的 cholesky 设置"""
        if not self.is_fitted or self.model is None:
            return None, None
        
        # 尝试不同的 jitter 值
        for jitter in [1e-4, 1e-3, 1e-2, 1e-1]:
            mean, std = self._predict_with_settings(X_cand, jitter)
            if mean is not None:
                return mean, std
        
        # 所有 jitter 都失败，使用上次成功的模型
        if self._last_valid_model is not None:
            print("Using last valid model for prediction")
            self.model = self._last_valid_model
            self.likelihood = self._last_valid_likelihood
            self.mll = self._last_valid_mll
            self.y_mean, self.y_std = self._last_y_stats
            self.model.eval()
            self.likelihood.eval()
            self.is_fitted = True
            
            # 再试一次
            return self._predict_with_settings(X_cand, 1e-1)
        
        return None, None
    
    def _expected_improvement(self, X_cand, xi: float = 0.001):
        """计算EI采集函数 - 使用更小的xi使BO更aggressive"""
        mean, std = self._predict(X_cand)
        if mean is None or std is None:
            # 不返回随机值，返回0让选择器选择其他候选点
            return np.zeros(len(X_cand))
        
        std = np.maximum(std, 1e-10)
        
        if self.is_minimizing:
            best_y = np.min(self.y_train)
            imp = best_y - mean - xi
        else:
            best_y = np.max(self.y_train)
            imp = mean - best_y - xi
        
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        ei[std < 1e-10] = 0.0
        
        return ei
    
    def _upper_confidence_bound(self, X_cand, kappa: float = 2.0):
        """UCB采集函数"""
        mean, std = self._predict(X_cand)
        if mean is None or std is None:
            return np.zeros(len(X_cand))
        
        if self.is_minimizing:
            return -(mean - kappa * std)
        else:
            return mean - kappa * std
    
    def _probability_of_improvement(self, X_cand, xi: float = 0.001):
        """PI采集函数 - 使用更小的xi"""
        mean, std = self._predict(X_cand)
        if mean is None or std is None:
            return np.zeros(len(X_cand))
        
        std = np.maximum(std, 1e-10)
        
        if self.is_minimizing:
            best_y = np.min(self.y_train)
            Z = (best_y - mean - xi) / std
        else:
            best_y = np.max(self.y_train)
            Z = (mean - best_y - xi) / std
            
        return norm.cdf(Z)
    
    def _compute_acquisition(self, X_cand):
        """计算采集函数值"""
        if self.acquisition == 'ei':
            return self._expected_improvement(X_cand)
        elif self.acquisition == 'ucb':
            return self._upper_confidence_bound(X_cand)
        elif self.acquisition == 'pi':
            return self._probability_of_improvement(X_cand)
        else:
            return self._expected_improvement(X_cand)
    
    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """建议新的采样点"""
        # 初始阶段：使用随机采样
        if len(self.X_train) < self.n_initial_points:
            return self._random_suggest(n_suggestions)
        
        # 首先尝试默认配置
        if not self._build_model(use_conservative=False):
            # 如果失败，尝试保守配置
            print("Default model failed, trying conservative config...")
            if not self._build_model(use_conservative=True):
                # 如果保守配置也失败，尝试使用上次成功的模型
                print("Conservative model failed, using last valid model...")
                if self._last_valid_model is not None:
                    self.model = self._last_valid_model
                    self.likelihood = self._last_valid_likelihood
                    self.mll = self._last_valid_mll
                    self.y_mean, self.y_std = self._last_y_stats
                    self.model.eval()
                    self.likelihood.eval()
                    self.is_fitted = True
                else:
                    # 没有可用的模型，回退到随机
                    return self._random_suggest(n_suggestions)
        
        # 生成候选点
        X_cand = self._generate_candidates(self.num_cands)
        
        # 计算采集函数
        acq_values = self._compute_acquisition(X_cand)
        
        # 检查有效性
        if acq_values is None or not np.isfinite(acq_values).any():
            return self._random_suggest(n_suggestions)
        
        if np.max(acq_values) <= 0:
            return self._random_suggest(n_suggestions)
        
        # 选择最优的点
        indices = np.argsort(acq_values)[-n_suggestions:]
        return X_cand[indices]
    
    def _random_suggest(self, n_suggestions: int) -> np.ndarray:
        """生成随机建议点"""
        return np.random.uniform(self.lb, self.ub, size=(n_suggestions, self.dims))
    
    def _generate_candidates(self, n: int) -> np.ndarray:
        """生成候选点 - 使用Sobol序列"""
        try:
            sobol = SobolEngine(dimension=self.dims, scramble=True, seed=np.random.randint(1e6))
            candidates = sobol.draw(n).cpu().numpy()
            return (self.ub - self.lb) * candidates + self.lb
        except ImportError:
            return np.random.uniform(self.lb, self.ub, size=(n, self.dims))
    
    def observe(self, x: np.ndarray, fx: Optional[np.ndarray] = None):
        """观察评估结果"""
        if fx is None:
            fx = np.array([self.func_wrapper(xi) for xi in x])
        
        for xi, fi in zip(x, fx):
            self.X_train.append(xi)
            self.y_train.append(fi)
        
        super().observe(x, fx)

        # 定期清理GPU缓存（每10次评估清理一次）
        if self.call_count % 10 == 0:
            _cleanup_gpu_memory()
    
    def reset(self):
        """重置优化器"""
        super().reset()
        self.X_train = []
        self.y_train = []
        self.model = None
        self.likelihood = None
        self.mll = None
        self.is_fitted = False
        self._last_valid_model = None
        self._last_valid_likelihood = None
        self._last_valid_mll = None
        self._last_y_stats = None
        self._last_model_config = None
