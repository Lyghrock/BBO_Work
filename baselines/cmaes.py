# CMA-ES Optimizer - 基于官方pycma实现
"""
CMA-ES 优化器 - 按照官方文档正确配置

关键点：
1. 使用官方默认的popsize公式: 4 + 3*log(dims)
2. 当popsize=1时禁用mirroring (CMA_mirrors=0)
3. 添加IPOP重启策略 (restart)
4. 开启Active CMA加速收敛 (CMA_active=True)
5. 内部维护完整population，批量调用tell
"""
import numpy as np
import torch
import cma
import warnings
from typing import Optional, Tuple, List
from baselines.base import BaseOptimizer

# 预先过滤掉InjectionWarning - 这个警告在外部调用模式时是正常的
warnings.filterwarnings("ignore", category=cma.evolution_strategy.InjectionWarning)


class CMAESOptimizer(BaseOptimizer):
    """CMA-ES optimizer using official pycma with IPOP and Active CMA"""
    
    def __init__(self, func_wrapper, 
                 popsize: int = None,
                 sigma: float = None,
                 maxfevals: int = None,
                 restarts: int = 9,  # IPOP重启次数
                 verbose: bool = False,
                 device=None,
                 gpu_id=None,
                 **kwargs):
        """
        Initialize CMA-ES optimizer
        
        Args:
            func_wrapper: Function wrapper
            popsize: Population size, default auto (官方公式: 4 + 3*log(dims))
            sigma: Initial step size, default auto
            maxfevals: Max function evaluations
            restarts: IPOP重启次数 (0=禁用, 9=默认推荐)
            verbose: 是否打印详细信息
        """
        super().__init__(func_wrapper, **kwargs)
        
        # 保险措施：即使CPU算法也确保GPU设备状态正确
        if device is not None:
            import torch
            if torch.cuda.is_available():
                gpu_id_to_use = gpu_id if gpu_id is not None else (device.index if hasattr(device, 'index') and device.index is not None else 0)
                torch.cuda.set_device(gpu_id_to_use)
        
        # 使用官方默认公式计算popsize
        if popsize is None:
            popsize = 4 + int(3 * np.log(self.dims))
        self.popsize = popsize
        
        # Auto set sigma
        if sigma is None:
            sigma = (self.ub - self.lb).max() / 3.0
        self.sigma = sigma
        
        self.maxfevals = maxfevals
        self.restarts = restarts
        self.verbose = verbose
        
        # CMA-ES instance
        self.es = None
        
        # 关键：内部维护完整的population缓存
        self._internal_solutions: List[np.ndarray] = []  # 完整的popsize个解
        self._internal_fitness: List[float] = []          # 对应的fitness
        self._pending_x: List[np.ndarray] = []           # 外部请求的解（待评估）
    
    def _create_es(self):
        """Create CMA-ES instance - 使用官方推荐的选项"""
        x0 = (self.lb + self.ub) / 2.0
        
        cma_options = {
            'popsize': self.popsize,
            'maxfevals': self.maxfevals if self.maxfevals else float('inf'),
            'bounds': [list(self.lb), list(self.ub)],
            'seed': np.random.randint(1e6),
            'verbose': -1 if not self.verbose else 0,
            # 关键修复：CMA_mirrors是比例（0到0.5之间），不能超过0.5
            'CMA_mirrors': 0.0 if self.popsize < 5 else 0.15,  # 理论最优值约为0.159
            # 开启Active CMA加速收敛
            'CMA_active': True,
        }
        
        self.es = cma.CMAEvolutionStrategy(x0, self.sigma, cma_options)
        
        # 重置内部缓存
        self._internal_solutions = []
        self._internal_fitness = []
        self._pending_x = []
    
    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """
        Suggest new sampling points
        
        内部始终生成完整的popsize个点，但只返回外部请求的n_suggestions个
        """
        if self.es is None:
            self._create_es()
        
        # 如果内部缓存中有未返回的点，直接返回
        if len(self._pending_x) >= n_suggestions:
            result = np.array(self._pending_x[:n_suggestions])
            self._pending_x = self._pending_x[n_suggestions:]
            return result
        
        try:
            # 始终请求完整的popsize以确保CMA-ES正常工作
            solutions = self.es.ask(self.popsize)
            
            # 转换为numpy数组
            if isinstance(solutions, list):
                solutions = np.array(solutions)
            
            # 确保在边界内
            solutions = np.clip(solutions, self.lb, self.ub)
            
            # 保存完整的population到内部缓存
            self._internal_solutions = solutions.tolist()
            self._internal_fitness = [None] * len(solutions)  # 还未评估
            
            # 返回外部请求的n_suggestions个点
            self._pending_x = solutions[n_suggestions:].tolist() if n_suggestions < len(solutions) else []
            return solutions[:n_suggestions]
            
        except Exception as e:
            if self.verbose:
                print(f"CMA-ES ask error: {e}")
            return np.random.uniform(self.lb, self.ub, size=(n_suggestions, self.dims))
    
    def observe(self, x: np.ndarray, fx: Optional[np.ndarray] = None, scores: Optional[np.ndarray] = None):
        """Observe evaluation results
        
        当积累到完整的population后，调用CMA-ES的tell方法
        """
        if fx is None:
            fx = np.array([self.sign * self.func_wrapper(xi) for xi in x])
        
        # 将评估结果填入内部缓存
        # x是外部请求的点，fx是对应的fitness
        # 需要找到这些点在_internal_solutions中的位置并更新
        
        if len(self._internal_solutions) > 0 and len(self._internal_fitness) > 0:
            # 检查x是否在_internal_solutions中
            for xi, fi in zip(x, fx):
                # 尝试找到对应的solution
                found = False
                for i, sol in enumerate(self._internal_solutions):
                    if self._internal_fitness[i] is None and np.allclose(sol, xi):
                        self._internal_fitness[i] = float(fi)
                        found = True
                        break
                
                # 如果没找到（可能是随机生成的点），添加到末尾
                if not found:
                    self._internal_solutions.append(xi.tolist() if isinstance(xi, np.ndarray) else xi)
                    self._internal_fitness.append(float(fi))
        
        # 检查是否可以调用tell（所有fitness都已评估）
        if self.es is not None and None not in self._internal_fitness and len(self._internal_fitness) >= 2:
            try:
                self.es.tell(self._internal_solutions, self._internal_fitness)
            except Exception as e:
                if self.verbose:
                    print(f"CMA-ES tell error: {e}")
            
            # 清空缓存，开始新的迭代
            self._internal_solutions = []
            self._internal_fitness = []
        
        # 调用父类方法更新最佳解
        super().observe(x, fx)
    
    def optimize(self, call_budget: int, batch_size: int = None) -> Tuple[np.ndarray, float, int]:
        """
        Run CMA-ES optimization with IPOP restarts
        
        使用内部完整的popsize进行优化，忽略外部的batch_size
        """
        # 记录当前最佳结果用于重启后比较
        global_best_x = self.best_x.copy() if self.best_x is not None else None
        global_best_fx = self.best_fx
        total_calls = 0
        
        for restart in range(self.restarts + 1):
            # 创建新的CMA-ES实例
            if restart > 0:
                # IPOP策略：成倍增加popsize
                self.popsize = self.popsize * 2
                if self.verbose:
                    print(f"IPOP restart {restart}: popsize={self.popsize}")
            
            self._create_es()
            
            # 优化直到达到budget或收敛
            while self.call_count < call_budget and not self.es.stop():
                try:
                    # 始终使用完整的popsize
                    solutions = self.es.ask(self.popsize)
                    solutions = np.array(solutions)
                    solutions = np.clip(solutions, self.lb, self.ub)
                except Exception as e:
                    if self.verbose:
                        print(f"CMA-ES ask error in optimize: {e}")
                    break
                
                # 评估
                fx_values = []
                for xi in solutions:
                    if total_calls >= call_budget:
                        break

                    fi = self.sign * self.func_wrapper(xi)
                    fx_values.append(fi)
                    # self.call_count is NOT incremented here: func_wrapper(xi) already
                    # increments func_wrapper.call_count.  The benchmark's suggest/observe
                    # path only touches func_wrapper.call_count, so we keep the two
                    # counters in sync by only counting here (not both places).
                    total_calls += 1

                    # 更新最佳解
                    if self.is_minimizing:
                        if fi < self.best_fx:
                            self.best_fx = fi
                            self.best_x = xi.copy()
                    else:
                        if fi > self.best_fx:
                            self.best_fx = fi
                            self.best_x = xi.copy()
                
                # Tell (只有完整population才tell)
                if len(solutions) >= 2 and total_calls <= call_budget:
                    try:
                        self.es.tell(solutions.tolist(), fx_values)
                    except:
                        pass
                
                self.history_x.extend(solutions)
                self.history_fx.extend(fx_values)
                
                # 检查是否收敛
                if self.es.stop():
                    break
            
            # 更新全局最佳
            if self.best_fx < global_best_fx:
                global_best_fx = self.best_fx
                global_best_x = self.best_x.copy() if self.best_x is not None else None
            
            # 检查是否达到budget
            if total_calls >= call_budget:
                break
        
        # 设置最终结果
        self.best_fx = global_best_fx
        self.best_x = global_best_x
        
        return self.best_x, self.best_fx, self.call_count
    
    def reset(self):
        """Reset optimizer"""
        super().reset()
        self.es = None
        self._internal_solutions = []
        self._internal_fitness = []
        self._pending_x = []
