# Base Optimizer Interface
import numpy as np
import os
import time
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict, Any

import torch


class BaseOptimizer(ABC):
    """所有优化器的基类"""
    
    def __init__(self, func_wrapper, **kwargs):
        """
        初始化优化器
        
        Args:
            func_wrapper: 函数包装器，需要有以下属性和方法:
                - lb: 下界 (np.ndarray)
                - ub: 上界 (np.ndarray) 
                - dims: 维度 (int)
                - is_minimizing: 是否最小化 (bool)
                - __call__(x): 评估函数
                - gen_random_inputs(n): 生成随机输入
            **kwargs: 额外参数
        """
        self.func_wrapper = func_wrapper
        self.lb = func_wrapper.lb
        self.ub = func_wrapper.ub
        self.dims = func_wrapper.dims
        self.is_minimizing = func_wrapper.is_minimizing
        self.sign = 1.0 if self.is_minimizing else -1.0
        
        # 记录优化结果
        self.best_x = None
        self.best_fx = float('inf') if self.is_minimizing else float('-inf')
        self.call_count = 0
        self.history_x = []
        self.history_fx = []
        
    @abstractmethod
    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """
        建议新的采样点
        
        Args:
            n_suggestions: 建议的点数量
            
        Returns:
            建议的点 (n_suggestions x dims)
        """
        pass
    
    def observe(self, x: np.ndarray, fx: Optional[np.ndarray] = None, scores: Optional[np.ndarray] = None):
        """
        观察评估结果

        Args:
            x: 评估的点 (n x dims)
            fx: 评估的函数值 (n,), 如果为None则自动计算
            scores: 原始 score (n,) = 原始 score（用于 MCTS 训练，可选）
        """
        if fx is None:
            fx = np.array([self.sign * self.func_wrapper(xi) for xi in x])

        for xi, fi in zip(x, fx):
            self.history_x.append(xi)
            self.history_fx.append(fi)
            self.call_count += 1

            # 更新最优解
            if self.is_minimizing:
                if fi < self.best_fx:
                    self.best_fx = fi
                    self.best_x = xi
            else:
                if fi > self.best_fx:
                    self.best_fx = fi
                    self.best_x = xi
    
    def optimize(self, call_budget: int, batch_size: int = 10) -> Tuple[np.ndarray, float, int]:
        """
        执行优化
        
        Args:
            call_budget: 函数评估预算
            batch_size: 每轮批量大小
            
        Returns:
            (best_x, best_fx, total_calls)
        """
        while self.call_count < call_budget:
            # 计算本轮需要评估的数量
            n_suggestions = min(batch_size, call_budget - self.call_count)
            
            # 获取建议点
            x_suggest = self.suggest(n_suggestions)
            
            # 评估函数值
            fx_suggest = np.array([self.sign * self.func_wrapper(xi) for xi in x_suggest])
            
            # 观察结果
            self.observe(x_suggest, fx_suggest)
            
        return self.best_x, self.best_fx, self.call_count
    
    def reset(self):
        """重置优化器状态"""
        self.best_x = None
        self.best_fx = float('inf') if self.is_minimizing else float('-inf')
        self.call_count = 0
        self.history_x = []
        self.history_fx = []


class StatsFuncWrapper:
    """简化的函数包装器，兼容原有接口"""
    
    def __init__(self, func):
        self.func = func
        if hasattr(func, 'lb'):
            self.lb = func.lb
            self.ub = func.ub
            self.dims = func.dims
        if hasattr(func, 'is_minimizing'):
            self.is_minimizing = func.is_minimizing
        else:
            self.is_minimizing = True
            
        self.total_calls = 0
        self.call_history = []
        
    def __call__(self, x):
        self.total_calls += 1
        result = self.func(x)
        if isinstance(result, tuple):
            fx = result[0]
        else:
            fx = result
        self.call_history.append((x, fx))
        return fx
    
    def gen_random_inputs(self, n):
        """生成随机输入"""
        if hasattr(self.func, 'gen_random_inputs'):
            return self.func.gen_random_inputs(n)
        # 默认使用均匀分布
        return np.random.uniform(self.lb, self.ub, size=(n, self.dims))
    
    @property
    def stats(self):
        """返回统计信息"""
        class Stats:
            def __init__(s, wrapper):
                s.total_calls = wrapper.total_calls
                s.call_history = wrapper.call_history
        return Stats(self)


def resolve_device(
    device: Optional[torch.device] = None,
    use_gpu: bool = True,
    gpu_id: Optional[int] = None,
    job_key: Optional[str] = None,
    min_memory_mb: int = 3000,
    allow_occupied: bool = False,
) -> torch.device:
    """统一设备解析逻辑。

    优先级:
    1) 显式传入 device
    2) 显式传入 gpu_id
    3) 使用 utils.gpu_scheduler 根据 job_key 自动保留GPU
       - allow_occupied=False（默认）：只选完全空闲（memory_used==0）的 GPU
       - allow_occupied=True：允许选择已有其他进程但显存有空闲的 GPU
    4) 回退 CPU
    """
    if device is not None:
        return device

    if (not use_gpu) or (not torch.cuda.is_available()):
        return torch.device('cpu')

    if gpu_id is not None:
        torch.cuda.set_device(gpu_id)
        return torch.device(f'cuda:{gpu_id}')

    try:
        from utils.gpu_scheduler import get_gpu_scheduler
        scheduler = get_gpu_scheduler()
        key = job_key or f"bbo:{os.getpid()}"
        selected = scheduler.acquire_gpu(
            job_key=key,
            min_memory_mb=min_memory_mb,
            allow_occupied=allow_occupied,
        )

        # 可选等待：通过环境变量 BBO_GPU_WAIT_SECONDS 控制等待时长（秒）
        # 默认 0 秒，不等待。
        if selected is None:
            wait_seconds = float(os.environ.get("BBO_GPU_WAIT_SECONDS", "0"))
            if wait_seconds > 0:
                deadline = time.time() + wait_seconds
                while time.time() < deadline and selected is None:
                    time.sleep(5.0)
                    selected = scheduler.acquire_gpu(
                        job_key=key,
                        min_memory_mb=min_memory_mb,
                        allow_occupied=allow_occupied,
                    )

        if selected is not None:
            torch.cuda.set_device(selected)
            return torch.device(f'cuda:{selected}')
    except Exception as e:
        raise RuntimeError(
            f"GPU scheduler failed while acquiring device: {e}"
        ) from e

    # use_gpu=True 且 CUDA 可用，但没有空闲卡：直接失败（不回退 CPU）
    raise RuntimeError(
        "No available GPU: all GPUs are occupied/reserved or below free-memory threshold."
    )
