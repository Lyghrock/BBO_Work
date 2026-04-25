# NeverGrad Optimizer
"""
NeverGrad 优化器 - 按照官方文档实现

按照 NeverGrad 官方文档重写：
https://facebookresearch.github.io/nevergrad/optimization.html

关键点：
1. 使用 ask/tell 接口而非 minimize() 模式
2. 正确处理 Parameter 对象的 value 属性
3. 随机初始化避免默认值恰好命中全局最优

提供两种使用模式：
1. optimize(call_budget): 内部自我管理循环
2. suggest() + observe(): 外部调用接口
"""
import numpy as np
import torch
import nevergrad as ng
from typing import Optional, Tuple
from baselines.base import BaseOptimizer


class NeverGradOptimizer(BaseOptimizer):
    """NeverGrad优化器 - 基于官方 ask/tell 接口"""

    def __init__(self, func_wrapper,
                 optimizer: str = 'NGOpt',
                 budget: int = None,
                 batch_size: int = 1,
                 num_workers: int = 1,
                 device=None,
                 gpu_id=None,
                 **kwargs):
        super().__init__(func_wrapper, **kwargs)

        # 保险措施：即使CPU算法也确保GPU设备状态正确
        if device is not None:
            import torch
            if torch.cuda.is_available():
                gpu_id_to_use = gpu_id if gpu_id is not None else (device.index if hasattr(device, 'index') and device.index is not None else 0)
                torch.cuda.set_device(gpu_id_to_use)

        self.optimizer_name = optimizer
        self.budget = budget
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.ng_optimizer = None
        self.param = None
        # 保存最近 ask 返回的建议对象，用于后续 tell
        self._pending_suggestions = []

    def _create_optimizer(self):
        """创建NeverGrad优化器 - 按照官方文档"""
        # 获取边界值（处理 numpy 数组的情况）
        if isinstance(self.lb, np.ndarray):
            lb_scalar = float(self.lb.min())
            ub_scalar = float(self.ub.max())
        else:
            lb_scalar = float(self.lb)
            ub_scalar = float(self.ub)

        # 使用 Array 参数化，设置边界
        # nevergrad 1.0+ 使用 shape 参数，并用 .set_bounds() 设置边界
        self.param = ng.p.Array(
            shape=(self.dims,)
        ).set_bounds(lower=lb_scalar, upper=ub_scalar)

        budget = self.budget if self.budget else 10000

        opt_name = self.optimizer_name
        try:
            # 从注册表中获取优化器
            self.ng_optimizer = ng.optimizers.registry[opt_name](
                parametrization=self.param,
                budget=budget,
                num_workers=self.num_workers
            )
        except Exception as e:
            print(f"Failed to create {opt_name}, falling back to NGOpt: {e}")
            self.ng_optimizer = ng.optimizers.NGOpt(
                parametrization=self.param,
                budget=budget,
                num_workers=self.num_workers
            )

    def _ng_value_to_array(self, x_ng):
        """将 NeverGrad 的参数转换为 numpy 数组"""
        if hasattr(x_ng, 'value'):
            x = np.array(x_ng.value, dtype=np.float64)
        elif hasattr(x_ng, 'args'):
            x = np.array(x_ng.args, dtype=np.float64)
        else:
            x = np.array(x_ng, dtype=np.float64)

        # 确保维度正确
        if x.shape != (self.dims,):
            x = x.flatten()
            if x.shape[0] < self.dims:
                # 填充随机值
                x = np.concatenate([x, np.random.uniform(self.lb, self.ub, self.dims - x.shape[0])])
            elif x.shape[0] > self.dims:
                x = x[:self.dims]
        
        # 检查是否全0，如果是则用随机值替换
        if np.allclose(x, 0.0):
            x = np.random.uniform(self.lb, self.ub, size=(self.dims,))

        return x

    def suggest(self, n_suggestions: int = 1) -> np.ndarray:
        """
        建议新的采样点 - 使用官方 ask/tell 接口

        官方文档示例:
        for _ in range(optimizer.budget):
            x = optimizer.ask()
            loss = square(*x.args, **x.kwargs)  # 或者 square(x.value)
            optimizer.tell(x, loss)
        """
        if self.ng_optimizer is None:
            self._create_optimizer()

        # 清空之前保存的建议对象
        self._pending_suggestions = []

        try:
            suggestions = []
            for _ in range(n_suggestions):
                # 使用 NeverGrad 的 ask() 来获取建议点
                # NeverGrad 内部会根据算法特性自动探索搜索空间
                x_ng = self.ng_optimizer.ask()

                # 将 NeverGrad 的参数转换为 numpy 数组
                x = self._ng_value_to_array(x_ng)

                # 裁剪到边界
                x = np.clip(x, self.lb, self.ub)

                suggestions.append(x)
                # 保存建议对象，用于后续 tell
                self._pending_suggestions.append(x_ng)

            return np.array(suggestions)
        except Exception as e:
            print(f"NeverGrad ask error: {e}")
            import traceback
            traceback.print_exc()
            return np.random.uniform(self.lb, self.ub, size=(n_suggestions, self.dims))

    def observe(self, x: np.ndarray, fx: Optional[np.ndarray] = None):
        """
        观察评估结果，并将结果反馈给 NeverGrad 优化器

        使用 tell() 反馈评估结果，让 NeverGrad 能够学习并改进采样策略
        注意：这里使用 suggest() 时保存的建议对象来 tell
        """
        if fx is None:
            fx = np.array([self.func_wrapper(xi) for xi in x])

        # 将评估结果反馈给 NeverGrad
        if self.ng_optimizer is not None and len(self._pending_suggestions) > 0:
            try:
                for i, (x_ng, fi) in enumerate(zip(self._pending_suggestions, fx)):
                    if i < len(x):
                        # 确保建议对象的值与实际评估的点一致
                        xi = x[i]
                        if hasattr(x_ng, 'value'):
                            x_ng.value = xi.tolist() if isinstance(xi, np.ndarray) else xi
                        elif hasattr(x_ng, 'args'):
                            x_ng.args = tuple(xi.tolist() if isinstance(xi, np.ndarray) else xi)
                        # 告诉优化器这个点的函数值
                        self.ng_optimizer.tell(x_ng, fi)
            except Exception as e:
                # 如果 tell 失败，静默继续，不影响主流程
                pass

        # 清空待反馈的建议对象
        self._pending_suggestions = []

        super().observe(x, fx)

    def optimize(self, call_budget: int, batch_size: int = None) -> Tuple[np.ndarray, float, int]:
        """
        执行NeverGrad优化 - 使用官方的 ask-tell 接口

        严格按照 NeverGrad 官方文档模式：
        for _ in range(optimizer.budget):
            x = optimizer.ask()
            loss = f(x)
            optimizer.tell(x, loss)
        """
        if batch_size is None:
            batch_size = self.batch_size

        if self.ng_optimizer is None:
            self.budget = call_budget
            self._create_optimizer()

        # 使用 NeverGrad 的 ask-tell 循环
        try:
            for _ in range(call_budget):
                # ask: 获取建议点
                x_ng = self.ng_optimizer.ask()

                # 将 NeverGrad 的参数转换为 numpy 数组
                x_arr = self._ng_value_to_array(x_ng)

                # 裁剪到边界
                x_arr = np.clip(x_arr, self.lb, self.ub)

                # 评估函数
                fx_val = float(self.func_wrapper(x_arr))

                # 更新最佳解跟踪
                self.call_count += 1
                if self.is_minimizing:
                    if fx_val < self.best_fx:
                        self.best_fx = fx_val
                        self.best_x = x_arr.copy()
                else:
                    if fx_val > self.best_fx:
                        self.best_fx = fx_val
                        self.best_x = x_arr.copy()

                # tell: 反馈结果给 NeverGrad
                try:
                    self.ng_optimizer.tell(x_ng, fx_val)
                except Exception as e:
                    # 如果 tell 失败，尝试重新赋值后 tell
                    try:
                        x_arr_ng = self.ng_optimizer.ask()
                        if hasattr(x_arr_ng, 'value'):
                            x_arr_ng.value = x_arr.tolist()
                        elif hasattr(x_arr_ng, 'args'):
                            x_arr_ng.args = tuple(x_arr.tolist())
                        self.ng_optimizer.tell(x_arr_ng, fx_val)
                    except:
                        pass

                # 每 10% 打印进度
                if call_budget > 0 and self.call_count % (call_budget // 10) == 0:
                    print(f"  [{self.call_count // (call_budget // 10) * 10}%] step={self.call_count}/{call_budget} | best_fx={self.best_fx:.6f}")

        except Exception as e:
            print(f"NeverGrad optimize error: {e}")
            import traceback
            traceback.print_exc()

        return self.best_x, self.best_fx, self.call_count

    def reset(self):
        """重置优化器"""
        super().reset()
        self.ng_optimizer = None
        self.param = None
        self._pending_suggestions = []


# 导出别名
NGOpt = NeverGradOptimizer
