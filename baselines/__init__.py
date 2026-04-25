# Unified BBO Optimizer Interface
from .base import BaseOptimizer, StatsFuncWrapper, resolve_device
from .bo import BayesianOptimizer
from .cmaes import CMAESOptimizer
from .turbo import TurboOptimizer
from .hesbo import HeSBOOptimizer
from .baxus import BAxUSOptimizer
from .saasbo import SAASBOOptimizer
import os

try:
    from scalpel.scalpel_opt import ScalpelOptimizer
    HAS_SCALPEL = True
except ImportError:
    HAS_SCALPEL = False
    ScalpelOptimizer = None

__all__ = [
    'BaseOptimizer',
    'StatsFuncWrapper',
    'BayesianOptimizer',
    'CMAESOptimizer',
    'TurboOptimizer',
    'HeSBOOptimizer',
    'BAxUSOptimizer',
    'SAASBOOptimizer',
    'ScalpelOptimizer',
]

# 优化器工厂函数
def create_optimizer(name, func_wrapper, **kwargs):
    """Create optimizer instance.

    Args:
        name: Optimizer name ('bo', 'turbo', 'cmaes', 'hesbo', 'baxus', 'saasbo', 'scalpel', 'random')
        func_wrapper: Function wrapper
        **kwargs: Optimizer-specific params (gpu_id, device, etc.)

    Returns:
        Optimizer instance
    """
    _name = name.lower().strip()

    optimizers = {
        'bo': BayesianOptimizer,
        'cmaes': CMAESOptimizer,
        'turbo': TurboOptimizer,
        'hesbo': HeSBOOptimizer,
        'baxus': BAxUSOptimizer,
        'saasbo': SAASBOOptimizer,
        'random': None,
    }

    if _name == 'random':
        return None

    # Lazy import for optional optimizers
    if _name == 'scalpel':
        if not HAS_SCALPEL:
            raise ImportError(
                "ScalpelOptimizer not available. "
                "Ensure scalpel/scalpel_opt.py is importable."
            )
        optimizers['scalpel'] = ScalpelOptimizer

    # 明确禁用不再评测的算法，避免脚本误调用
    if _name in {'nevergrad', 'lamcts'}:
        raise ValueError(
            f"Optimizer '{_name}' is disabled for this benchmark suite. "
            f"Enabled optimizers: {list(optimizers.keys()) + ['scalpel']}"
        )

    if _name not in optimizers:
        raise ValueError(
            f"Unknown optimizer: {name}. Available: {list(optimizers.keys())}"
        )

    # 统一设备分配（由工厂负责）
    use_gpu = kwargs.get('use_gpu', True)
    explicit_device = kwargs.get('device', None)
    explicit_gpu_id = kwargs.get('gpu_id', None)
    min_gpu_memory_mb = int(kwargs.pop('min_gpu_memory_mb', 3000))
    allow_occupied = bool(kwargs.pop('allow_occupied', False))
    job_key = kwargs.get('job_key') or f"algo:{_name}:pid:{os.getpid()}"

    device = resolve_device(
        device=explicit_device,
        use_gpu=use_gpu,
        gpu_id=explicit_gpu_id,
        job_key=job_key,
        min_memory_mb=min_gpu_memory_mb,
        allow_occupied=allow_occupied,
    )

    kwargs['device'] = device
    kwargs['gpu_id'] = device.index if device.type == 'cuda' else None

    return optimizers[_name](func_wrapper, **kwargs)
