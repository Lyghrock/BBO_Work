# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import torch
import json
import numpy as np
import gc

from sklearn.cluster import KMeans
from scipy.stats import norm
import copy as cp
from sklearn.svm import SVC

from torch.quasirandom import SobolEngine
from mpl_toolkits.mplot3d import axes3d, Axes3D
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.means import ConstantMean
from gpytorch.constraints import Interval

import matplotlib.pyplot as plt
from matplotlib import cm


def _cleanup_gpu_memory():
    """清理GPU内存的辅助函数"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

# the input will be samples!
class GPyTorchGP(ExactGP):
    def __init__(self, train_x, train_y, likelihood, dims: int = None):
        super(GPyTorchGP, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        ard_dims = dims if dims is not None else (train_x.shape[-1] if train_x.ndim > 1 else 1)
        self.covar_module = ScaleKernel(
            MaternKernel(lengthscale=1.0, nu=2.5, ard_num_dims=ard_dims)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def _get_gpu_device(gpu_id=None):
    """获取指定的GPU设备"""
    if gpu_id is not None and torch.cuda.is_available():
        if gpu_id < torch.cuda.device_count():
            return torch.device(f'cuda:{gpu_id}')
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    return torch.device('cpu')


class Classifier():
    def __init__(self, samples, dims, kernel_type, gamma_type = "auto", gpu_id = None, lb = None, ub = None):
        self.training_counter = 0
        assert dims >= 1
        assert type(samples)  ==  type([])
        self.dims    =   dims
        self.gpu_id  =   gpu_id
        self.lb = lb   # 搜索空间下界，由 MCTS 创建时传入
        self.ub = ub   # 搜索空间上界，由 MCTS 创建时传入

        # 使用指定的GPU设备
        self.device = _get_gpu_device(gpu_id)

        # 显式设置当前GPU设备，避免GPyTorch使用默认GPU 0
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)

        # 创建GPyTorch模型和likelihood
        self.likelihood = GaussianLikelihood().to(self.device)
        self.model = None
        self.mll = None

        # 缓存
        self._cached_X_hash = None
        self._cached_fX_hash = None
        self._is_model_trained = False
        self._cached_samples_len = 0

        # ── GP 持久化缓存（关键：避免每次 propose 都重新训练 GP）──
        self._gp_cache_len = 0
        self._gp_cache_hash = None
        self._turbo_gp_model = None
        self._turbo_gp_likelihood = None

        # 旧版sklearn GPR保留用于兼容
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import ConstantKernel, Matern
        noise = 0.1
        m52 = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
        self.gpr = GaussianProcessRegressor(kernel=m52, alpha=noise**2)

        self.kmean   =   KMeans(n_clusters=2)
        #learned boundary
        self.svm     =   SVC(kernel = kernel_type, gamma=gamma_type)
        #data structures to store
        self.samples = []
        self.X       = np.array([])
        self.fX      = np.array([])

        #good region is labeled as zero
        #bad  region is labeled as one
        self.good_label_mean  = -1
        self.bad_label_mean   = -1

        self.update_samples(samples)
    
    def is_splittable_svm(self):
        plabel = self.learn_clusters()
        self.learn_boundary(plabel)
        svm_label = self.svm.predict( self.X )
        if len( np.unique(svm_label) ) == 1:
            return False
        else:
            return True
        
    def get_max(self):
        return np.max(self.fX)
        
    def plot_samples_and_boundary(self, func, name):
        assert func.dims == 2
        
        plabels   = self.svm.predict( self.X )
        good_counts = len( self.X[np.where( plabels == 0 )] )
        bad_counts  = len( self.X[np.where( plabels == 1 )] )
        good_mean = np.mean( self.fX[ np.where( plabels == 0 ) ] )
        bad_mean  = np.mean( self.fX[ np.where( plabels == 1 ) ] )
        
        if np.isnan(good_mean) == False and np.isnan(bad_mean) == False:
            assert good_mean > bad_mean

        lb = func.lb
        ub = func.ub
        x = np.linspace(lb[0], ub[0], 100)
        y = np.linspace(lb[1], ub[1], 100)
        xv, yv = np.meshgrid(x, y)
        true_y = []
        for row in range(0, xv.shape[0]):
            for col in range(0, xv.shape[1]):
                x = xv[row][col]
                y = yv[row][col]
                true_y.append( func( np.array( [x, y] ) ) )
        true_y = np.array( true_y )
        pred_labels = self.svm.predict( np.c_[xv.ravel(), yv.ravel()] )
        pred_labels = pred_labels.reshape( xv.shape )
        
        fig, ax = plt.subplots()
        ax.contour(xv, yv, true_y.reshape(xv.shape), cmap=cm.coolwarm)
        ax.contourf(xv, yv, pred_labels, alpha=0.4)
        
        ax.scatter(self.X[ np.where(plabels == 0) , 0 ], self.X[ np.where(plabels == 0) , 1 ], marker='x', label="good-"+str(np.round(good_mean, 2))+"-"+str(good_counts) )
        ax.scatter(self.X[ np.where(plabels == 1) , 0 ], self.X[ np.where(plabels == 1) , 1 ], marker='x', label="bad-"+str(np.round(bad_mean, 2))+"-"+str(bad_counts)    )
        ax.legend(loc="best")
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig(name)
        plt.close()
    
    def get_mean(self):
        return np.mean(self.fX)
        
    def update_samples(self, latest_samples):
        assert type(latest_samples) == type([])
        X  = []
        fX  = []
        for sample in latest_samples:
            X.append(  sample[0] )
            fX.append( sample[1] )
        
        self.X          = np.asarray(X, dtype=np.float32).reshape(-1, self.dims)
        self.fX         = np.asarray(fX,  dtype=np.float32).reshape(-1)
        self.samples    = latest_samples

        # 计算数据哈希用于缓存判断
        self._cached_X_hash = hash(self.X.tobytes())
        self._cached_fX_hash = hash(self.fX.tobytes())
        self._cached_samples_len = len(self.samples)

        # 数据变化了，重置模型训练状态
        self._is_model_trained = False
        self.model = None

    def _check_cache_valid(self, samples):
        """检查缓存是否有效"""
        if len(samples) != self._cached_samples_len:
            return False
        X = np.array([s[0] for s in samples], dtype=np.float32).reshape(-1, self.dims)
        fX = np.array([s[1] for s in samples], dtype=np.float32).reshape(-1)
        if hash(X.tobytes()) != self._cached_X_hash:
            return False
        if hash(fX.tobytes()) != self._cached_fX_hash:
            return False
        return True
        
    def train_gpr(self, samples):
        """使用GPU加速的GPyTorch训练GPR，带缓存"""
        # 检查缓存是否有效
        if self._is_model_trained and len(samples) == self._cached_samples_len:
            X_check = np.array([s[0] for s in samples], dtype=np.float32).reshape(-1, self.dims)
            fX_check = np.array([s[1] for s in samples], dtype=np.float32).reshape(-1)
            if hash(X_check.tobytes()) == self._cached_X_hash and hash(fX_check.tobytes()) == self._cached_fX_hash:
                return  # 数据没变，使用缓存

        # 清理之前的GPU模型以释放内存
        if self.model is not None:
            del self.model
        if self.likelihood is not None:
            del self.likelihood
        if self.mll is not None:
            del self.mll
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        X  = []
        fX  = []
        for sample in samples:
            X.append(  sample[0] )
            fX.append( sample[1] )
        X  = np.asarray(X).reshape(-1, self.dims)
        fX = np.asarray(fX).reshape(-1)

        # 转换为GPU tensor
        train_x = torch.from_numpy(X).float().to(self.device)
        train_y = torch.from_numpy(fX).float().to(self.device)

        # 创建新模型 - 确保在正确的设备上
        self.likelihood = GaussianLikelihood(
            noise_constraint=Interval(1e-4, 1e1)
        ).to(self.device)
        
        # 设置数值稳定性参数
        with gpytorch.settings.cholesky_jitter(1e-3):
            self.model = GPyTorchGP(train_x, train_y, self.likelihood).to(self.device)
            self.model.train()
            self.likelihood.train()

        # 设置mll - 确保在正确的设备上
        self.mll = ExactMarginalLogLikelihood(self.likelihood, self.model).to(self.device)

        # 训练
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

        # 更新缓存
        self._cached_X_hash = hash(X.tobytes())
        self._cached_fX_hash = hash(fX.tobytes())
        self._cached_samples_len = len(samples)

        # 训练迭代
        for i in range(10):
            optimizer.zero_grad(set_to_none=True)
            output = self.model(train_x)
            loss = -self.mll(output, train_y)
            loss.backward()
            optimizer.step()

        self._is_model_trained = True

        # 清理GPU缓存
        _cleanup_gpu_memory()
    
    ###########################
    # BO sampling with EI
    ###########################


    def expected_improvement(self, X, xi=0.0001, use_ei = True):
        ''' Computes the EI at points X based on existing samples using GPU-accelerated GPyTorch.
        Returns: Expected improvements at points X. '''
        X_sample = self.X
        Y_sample = self.fX.reshape((-1, 1))

        # 如果模型未训练，先训练
        if not self._is_model_trained:
            self.train_gpr(self.samples)

        # 转换X到GPU
        X_tensor = torch.from_numpy(X).float().to(self.device)

        # GPU预测
        self.model.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
            pred_dist = self.model(X_tensor)
            mu = pred_dist.mean.cpu().numpy()
            sigma = pred_dist.stddev.cpu().numpy()

        if not use_ei:
            return mu
        else:
            # calculate EI - 使用所有训练样本
            if len(self.X) == 0:
                # 如果没有样本，返回随机值
                return np.random.rand(len(X)).reshape(-1)
            
            X_sample_tensor = torch.from_numpy(self.X).float().to(self.device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
                mu_sample_pred = self.model(X_sample_tensor)
                mu_sample = mu_sample_pred.mean.cpu().numpy()
            mu_sample_opt = np.max(mu_sample)
            sigma = sigma.reshape(-1, 1)
            with np.errstate(divide='warn'):
                imp = mu - mu_sample_opt - xi
                imp = imp.reshape((-1, 1))
                Z = imp / sigma
                ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
                ei[sigma == 0.0] = 0.0

            # 清理GPU缓存
            _cleanup_gpu_memory()

            return ei
            
    def plot_boundary(self, X):
        if X.shape[1] > 2:
            return
        fig, ax = plt.subplots()
        ax.scatter( X[ :, 0 ], X[ :, 1 ] , marker='.')
        ax.scatter(self.X[ : , 0 ], self.X[ : , 1 ], marker='x')
        ax.set_xlim([-10, 10])
        ax.set_ylim([-10, 10])
        plt.savefig("boundary.pdf")
        plt.close()
    
    def get_sample_ratio_in_region(self, cands, path):
        """优化的区域采样比率计算 - 批量预测"""
        total = len(cands)
        if total == 0:
            return 0, np.array([])

        # 批量预测所有节点
        for node in path:
            boundary = node[0].classifier.svm
            if len(cands) == 0:
                return 0, np.array([])
            # 使用numpy向量化操作替代逐个预测
            predictions = boundary.predict(cands)
            cands = cands[predictions == node[1]]
            # node[1] store the direction to go

        ratio = len(cands) / total
        return ratio, cands

    def propose_rand_samples_probe(self, nums_samples, path, lb, ub):

        seed   = np.random.randint(int(1e6))
        sobol  = SobolEngine(dimension = self.dims, scramble=True, seed=seed)

        center = np.mean(self.X, axis = 0)
        #check if the center located in the region
        ratio, tmp = self.get_sample_ratio_in_region( np.reshape(center, (1, len(center) ) ), path )
        if ratio == 0:
            print("==>center not in the region, using random samples")
            return self.propose_rand_samples(nums_samples, lb, ub)
        # it is possible that the selected region has no points,
        # so we need check here

        axes    = len( center )
        
        final_L = []
        for axis in range(0, axes):
            L       = np.zeros( center.shape )
            L[axis] = 0.01
            ratio   = 1
            
            while ratio >= 0.9:
                L[axis] = L[axis]*2
                if L[axis] >= (ub[axis] - lb[axis]):
                    break
                lb_     = np.clip( center - L/2, lb, ub )
                ub_     = np.clip( center + L/2, lb, ub )
                cands_  = sobol.draw(10000).to(dtype=torch.float64).cpu().detach().numpy()
                cands_  = (ub_ - lb_)*cands_ + lb_
                ratio, tmp = self.get_sample_ratio_in_region(cands_, path )
            final_L.append( L[axis] )

        final_L   = np.array( final_L )
        lb_       = np.clip( center - final_L/2, lb, ub )
        ub_       = np.clip( center + final_L/2, lb, ub )
        print("center:", center)
        print("final lb:", lb_)
        print("final ub:", ub_)
    
        count         = 0
        cands         = np.array([])
        while len(cands) < 10000:
            count    += 10000
            cands     = sobol.draw(count).to(dtype=torch.float64).cpu().detach().numpy()
        
            cands     = (ub_ - lb_)*cands + lb_
            ratio, cands = self.get_sample_ratio_in_region(cands, path)
            samples_count = len( cands )
        
        #extract candidates 
        
        return cands
            
    def propose_rand_samples_sobol(self, nums_samples, path, lb, ub):
        """优化的候选点采样 - 使用向量化操作"""
        # 获取区域内的样本中心
        ratio_check, centers = self.get_sample_ratio_in_region(self.X, path)
        if ratio_check == 0 or len(centers) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)

        # 直接使用整个空间采样，然后用SVM批量过滤
        seed = np.random.randint(int(1e6))
        sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)

        # 一次性生成足够多的候选点
        batch_size = 10000
        max_batches = 5
        selected_cands = []

        for _ in range(max_batches):
            cands = sobol.draw(batch_size).to(dtype=torch.float64).cpu().detach().numpy()
            cands = (ub - lb) * cands + lb

            # 使用SVM批量预测
            predictions = self.svm.predict(cands)

            # 逐层过滤
            filtered = cands
            for node in path:
                boundary = node[0].classifier.svm
                if len(filtered) == 0:
                    break
                preds = boundary.predict(filtered)
                filtered = filtered[preds == node[1]]

            selected_cands.extend(filtered.tolist())

            if len(selected_cands) >= nums_samples:
                break

        selected_cands = np.array(selected_cands)

        if len(selected_cands) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)

        # 如果候选点过多，随机选择
        if len(selected_cands) > nums_samples:
            indices = np.random.choice(len(selected_cands), nums_samples, replace=False)
            return selected_cands[indices]
        else:
            return selected_cands
        
    def propose_samples_bo( self, nums_samples = 10, path = None, lb = None, ub = None, samples = None):
        ''' Proposes the next sampling point by optimizing the acquisition function. 
        Args: acquisition: Acquisition function. X_sample: Sample locations (n x d). 
        Y_sample: Sample values (n x 1). gpr: A GaussianProcessRegressor fitted to samples. 
        Returns: Location of the acquisition function maximum. '''
        assert path is not None and len(path) >= 0
        assert lb is not None and ub is not None
        assert samples is not None and len(samples) > 0
        
        self.train_gpr( samples ) # learn in unit cube
        
        dim  = self.dims
        nums_rand_samples = 5000
        if len(path) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)
        
        X    = self.propose_rand_samples_sobol(nums_rand_samples, path, lb, ub)
        # print("samples in the region:", len(X) )
        # self.plot_boundary(X)
        if len(X) == 0:
            return self.propose_rand_samples(nums_samples, lb, ub)
        
        X_ei = self.expected_improvement(X, xi=0.01, use_ei = True)
        row, col = X.shape
    
        X_ei = X_ei.reshape(len(X))
        n = nums_samples
        if X_ei.shape[0] < n:
            n = X_ei.shape[0]
        indices = np.argsort(X_ei)[-n:]
        proposed_X = X[indices]
        return proposed_X
        
    ###########################
    # sampling with turbo - TuRBO-1 style local optimization
    ###########################
    
    def propose_samples_turbo(self, num_samples, path, func, collect=False):
        """
        GPU-accelerated TuRBO-1 style candidate sampling.

        Returns:
            - When collect=True (internal use in search()):
              (proposed_X, fX) where fX = -func(x) already negated (maximization format)
            - When collect=False:
              proposed_X ndarray (only samples, no evaluation)

        The return signature mirrors the original LaMCTS:
            proposed_X, fX = leaf.propose_samples_turbo(10000, path, self.func)
            for s, v in zip(proposed_X, fX):
                value = self.collect_samples(s, value=v)  # v already negated
        """
        n_cands = 5000

        # ── Step 1: Train (or reuse) GP ────────────────────────────────────────
        if len(self.samples) < 2:
            samples = self.propose_rand_samples(num_samples, self.lb, self.ub)
            if collect:
                values = np.array([func(s) * -1 for s in samples])  # negate for maximization
                return samples, values
            return samples

        X_all = np.array([s[0] for s in self.samples], dtype=np.float32)
        y_all = np.array([s[1] for s in self.samples], dtype=np.float32)  # already negated

        cur_hash = hash(X_all.tobytes()) ^ hash(y_all.tobytes())
        need_train = (
            self._turbo_gp_model is None
            or self._turbo_gp_cache_len != len(self.samples)
            or self._gp_cache_hash != cur_hash
        )

        if need_train:
            if self._turbo_gp_model is not None:
                del self._turbo_gp_model
            if self._turbo_gp_likelihood is not None:
                del self._turbo_gp_likelihood
            torch.cuda.empty_cache()

            try:
                train_x = torch.from_numpy(X_all).float().to(self.device)
                train_y = torch.from_numpy(y_all).float().to(self.device)
                likelihood = GaussianLikelihood(
                    noise_constraint=Interval(1e-6, 1e-2)
                ).to(self.device)
                with gpytorch.settings.cholesky_jitter(1e-3):
                    model = GPyTorchGP(train_x, train_y, likelihood).to(self.device)
                model.train()
                likelihood.train()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
                for _ in range(10):
                    optimizer.zero_grad(set_to_none=True)
                    output = model(train_x)
                    loss = -ExactMarginalLogLikelihood(likelihood, model)(output, train_y)
                    loss.backward()
                    optimizer.step()
                model.eval()
                likelihood.eval()

                self._turbo_gp_model = model
                self._turbo_gp_likelihood = likelihood
                self._turbo_gp_cache_len = len(self.samples)
                self._gp_cache_hash = cur_hash
            except Exception:
                self._turbo_gp_model = None
                self._turbo_gp_likelihood = None
                samples = self.propose_rand_samples(num_samples, self.lb, self.ub)
                if collect:
                    values = np.array([func(s) * -1 for s in samples])
                    return samples, values
                return samples
        else:
            model = self._turbo_gp_model
            likelihood = self._turbo_gp_likelihood

        # ── Step 2: Find GP mean best point as TR center ──────────────────────
        try:
            seed = np.random.randint(int(1e6))
            sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)
            grid = sobol.draw(5000).cpu().numpy()
            lb_full = np.min(X_all, axis=0)
            ub_full = np.max(X_all, axis=0)
            grid_scaled = (ub_full - lb_full) * grid + lb_full

            grid_t = torch.from_numpy(grid_scaled).float().to(self.device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = model(grid_t)
                mu_grid = pred.mean.cpu().numpy()

            best_local_idx = np.argmax(mu_grid)
            x_center = grid_scaled[best_local_idx]
        except Exception:
            x_center = np.mean(X_all, axis=0)

        # ── Step 3: Build Trust Region ─────────────────────────────────────────
        tr_length = 0.8 * np.mean(ub_full - lb_full)
        lb_tr = np.clip(x_center - tr_length / 2.0, lb_full, ub_full)
        ub_tr = np.clip(x_center + tr_length / 2.0, lb_full, ub_full)

        # ── Step 4: Sobol in TR + SVM path filter ──────────────────────────────
        max_attempts = 10
        all_cands = []
        target = num_samples * 20

        for _ in range(max_attempts):
            seed = np.random.randint(int(1e6))
            sobol = SobolEngine(dimension=self.dims, scramble=True, seed=seed)
            batch = sobol.draw(target).cpu().numpy()
            cands = (ub_tr - lb_tr) * batch + lb_tr

            for node, direction in path:
                svm = node.classifier.svm
                if len(cands) == 0:
                    break
                preds = svm.predict(cands)
                cands = cands[preds == direction]

            all_cands.extend(cands.tolist())
            if len(all_cands) >= target:
                break

        if len(all_cands) == 0:
            samples = self.propose_rand_samples(num_samples, self.lb, self.ub)
            if collect:
                values = np.array([func(s) * -1 for s in samples])
                return samples, values
            return samples

        all_cands = np.array(all_cands[:target])

        # ── Step 5: EI-based selection with diversity bonus ──────────────────
        try:
            cands_t = torch.from_numpy(all_cands).float().to(self.device)
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                pred = model(cands_t)
                mu = pred.mean.cpu().numpy()
                sigma = pred.stddev.cpu().numpy()

            # Proper Expected Improvement with exploration bonus
            # xi = 0.01: larger than the tiny 1e-4, meaningfully encourages exploration
            best_y = np.max(y_all)
            xi_ei = 0.01  # exploration bias — increased from 1e-4

            with np.errstate(divide='warn', invalid='warn'):
                imp = mu - best_y - xi_ei
                sigma_safe = np.where(sigma < 1e-10, 1e-10, sigma)
                Z = imp / sigma_safe
                ei_vals = imp * norm.cdf(Z) + sigma_safe * norm.pdf(Z)
                ei_vals = np.where(sigma < 1e-10, np.maximum(imp, 0), ei_vals)
                ei_vals = np.maximum(ei_vals, 0.0)

            # Diversity bonus: keep 30% of candidates as uniformly random from pool
            # This prevents all candidates clustering at the GP mode
            n_ei_selected = int(num_samples * 0.7)
            n_diversity = num_samples - n_ei_selected

            if n_ei_selected > 0 and len(ei_vals) >= n_ei_selected:
                idx_top = np.argsort(ei_vals)[-n_ei_selected:]
                selected_ei = all_cands[idx_top]
            else:
                selected_ei = all_cands
                n_diversity = 0

            if n_diversity > 0:
                remaining_mask = np.ones(len(all_cands), dtype=bool)
                if n_ei_selected > 0:
                    remaining_mask[idx_top] = False
                remaining_cands = all_cands[remaining_mask]
                if len(remaining_cands) >= n_diversity:
                    selected_random = remaining_cands[
                        np.random.choice(len(remaining_cands), n_diversity, replace=False)
                    ]
                elif len(remaining_cands) > 0:
                    selected_random = remaining_cands[
                        np.random.choice(len(remaining_cands),
                                        min(n_diversity, len(remaining_cands)), replace=False)
                    ]
                else:
                    selected_random = all_cands[
                        np.random.choice(len(all_cands), n_diversity, replace=False)
                    ]
                selected = np.vstack([selected_ei, selected_random])
            else:
                selected = selected_ei

            if len(selected) < num_samples:
                extra = num_samples - len(selected)
                selected = np.vstack([
                    selected,
                    self.propose_rand_samples(extra, self.lb, self.ub)
                ])

        except Exception:
            if len(all_cands) >= num_samples:
                selected = all_cands[np.random.choice(len(all_cands), num_samples, replace=False)]
            else:
                extra = num_samples - len(all_cands)
                fallback = np.vstack([
                    all_cands,
                    self.propose_rand_samples(extra, self.lb, self.ub)
                ])
                samples = fallback
                if collect:
                    values = np.array([func(s) * -1 for s in samples])
                    return samples, values
                return samples

        # ── Return with values (collect=True) or samples only (collect=False) ──
        _cleanup_gpu_memory()
        if collect:
            values = np.array([func(s) * -1 for s in selected])
            return selected, values
        return selected

    ###########################
    # random sampling
    ###########################
    
    def propose_rand_samples(self, nums_samples, lb, ub):
        x = np.random.uniform(lb, ub, size = (nums_samples, self.dims) )
        return x
        
        
    def propose_samples_rand( self, nums_samples = 10):
        return self.propose_rand_samples(nums_samples, self.lb, self.ub)
                
    ###########################
    # learning boundary
    ###########################
    
        
    def get_cluster_mean(self, plabel):
        assert plabel.shape[0] == self.fX.shape[0] 
        
        zero_label_fX = []
        one_label_fX  = []
        
        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                zero_label_fX.append( self.fX[idx]  )
            elif plabel[idx] == 1:
                one_label_fX.append( self.fX[idx] )
            else:
                print("kmean should only predict two clusters, Classifiers.py:line73")
                os._exit(1)
                
        good_label_mean = np.mean( np.array(zero_label_fX) )
        bad_label_mean  = np.mean( np.array(one_label_fX) )
        return good_label_mean, bad_label_mean
        
    def learn_boundary(self, plabel):
        assert len(plabel) == len(self.X)
        self.svm.fit(self.X, plabel)
        
    def learn_clusters(self):
        assert len(self.samples) >= 2, "samples must > 0"
        assert self.X.shape[0], "points must > 0"
        assert self.fX.shape[0], "fX must > 0"
        assert self.X.shape[0] == self.fX.shape[0]
        
        tmp = np.concatenate( (self.X, self.fX.reshape([-1, 1]) ), axis = 1 )
        assert tmp.shape[0] == self.fX.shape[0]
        
        self.kmean  = self.kmean.fit(tmp)
        plabel      = self.kmean.predict( tmp )
        
        # the 0-1 labels in kmean can be different from the actual
        # flip the label is not consistent
        # 0: good cluster, 1: bad cluster
        
        self.good_label_mean , self.bad_label_mean = self.get_cluster_mean(plabel)
        
        if self.bad_label_mean > self.good_label_mean:
            for idx in range(0, len(plabel)):
                if plabel[idx] == 0:
                    plabel[idx] = 1
                else:
                    plabel[idx] = 0
                    
        self.good_label_mean , self.bad_label_mean = self.get_cluster_mean(plabel)
        
        return plabel
        
    def split_data(self):
        good_samples = []
        bad_samples  = []
        train_good_samples = []
        train_bad_samples  = []
        if len( self.samples ) == 0:
            return good_samples, bad_samples
        
        plabel = self.learn_clusters( )
        self.learn_boundary( plabel )
        
        for idx in range(0, len(plabel)):
            if plabel[idx] == 0:
                diff = abs(self.samples[idx][-1] - self.fX[idx])
                assert diff <= 1e-5, f"MISMATCH idx={idx}: samples[idx][-1]={self.samples[idx][-1]}, fX[idx]={self.fX[idx]}, diff={diff}, len_samples={len(self.samples)}, len_fX={len(self.fX)}"
                good_samples.append(self.samples[idx])
                train_good_samples.append(self.X[idx])
            else:
                bad_samples.append(self.samples[idx])
                train_bad_samples.append(self.X[idx])
        
        train_good_samples = np.array(train_good_samples)
        train_bad_samples  = np.array(train_bad_samples)
                        
        assert len(good_samples) + len(bad_samples) == len(self.samples)
                
        return  good_samples, bad_samples



    
    
    

