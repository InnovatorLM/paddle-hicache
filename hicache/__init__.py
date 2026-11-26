# -*- coding: utf-8 -*-
"""HiCache · PaddlePaddle 适配实现。

``hicache.reference`` 只依赖 NumPy，可在无 Paddle 环境下导入（用于离线对齐校验）；
其余模块需要安装 paddle。引用 Paddle 相关 API 时不可用会抛 ImportError。
"""

from .reference import (
    backward_differences as np_backward_differences,
    extrapolate as np_extrapolate,
    hermite_polynomial_exact,
    scaled_hermite_values as np_scaled_hermite_values,
)

__all__ = [
    'np_scaled_hermite_values',
    'np_backward_differences',
    'np_extrapolate',
    'hermite_polynomial_exact',
]

__version__ = '1.0.0'

try:  # pragma: no cover - 依赖运行环境
    from .predictor import (  # noqa: F401
        backward_differences,
        extrapolate,
        relative_l1,
        scaled_hermite_values,
    )
    from .cache import FeatureCache, HiCacheState  # noqa: F401
    from .flux_patch import disable_hicache, enable_hicache, hicache_state  # noqa: F401

    __all__ += [
        'scaled_hermite_values',
        'backward_differences',
        'extrapolate',
        'relative_l1',
        'FeatureCache',
        'HiCacheState',
        'enable_hicache',
        'disable_hicache',
        'hicache_state',
    ]
    PADDLE_AVAILABLE = True
except ImportError:  # 仅 NumPy 环境
    PADDLE_AVAILABLE = False