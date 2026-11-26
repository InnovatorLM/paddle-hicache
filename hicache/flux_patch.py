# -*- coding: utf-8 -*-
"""ppdiffusers FluxPipeline 的 HiCache 补丁（即插即用，不修改模型权重与结构）。

用法::

    from hicache.flux_patch import enable_hicache, disable_hicache
    enable_hicache(pipe, interval=7, order=2, sigma=0.5)
    image = pipe(prompt, num_inference_steps=50).images[0]
    print(pipe.transformer._hicache.state.speedup)

原理：包装 ``pipe.transformer.forward``。判定为"外推步"时跳过整个 Transformer 前向，
以上一步的历史输出（去噪隐状态/噪声预测）为特征序列，用缩放埃尔米特基外推当前步输出；
判定为"完整计算步"时正常前向并记录特征。

注意：本补丁作用于 pipeline 层（对 Transformer 输出做 cache-then-forecast），与论文中
"对 Transformer 内部 block 特征做缓存"等价于同一套外推数学，工程上更通用、对不同
ppdiffusers 版本更鲁棒；如需 block 级缓存，见 docs/适配说明.md 的"进阶"一节。
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import paddle

from .cache import FeatureCache, HiCacheState

__all__ = ['enable_hicache', 'disable_hicache', 'hicache_state']


def _is_tensor(x: Any) -> bool:
    return isinstance(x, paddle.Tensor)


def _tensor_of(output: Any) -> Optional[paddle.Tensor]:
    """从 Transformer 输出中取出用于缓存的张量。"""
    if _is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ('sample', 'hidden_states', 'last_hidden_state'):
            if _is_tensor(output.get(key)):
                return output[key]
    # ppdiffusers 的 Transformer2DModelOutput 为 dataclass / tuple 子类
    if hasattr(output, 'sample') and _is_tensor(getattr(output, 'sample')):
        return output.sample
    if isinstance(output, (tuple, list)) and output and _is_tensor(output[0]):
        return output[0]
    return None


def _rebuild(output: Any, tensor: paddle.Tensor) -> Any:
    """把预测出的张量放回与原始输出同类型的容器中。"""
    if _is_tensor(output):
        return tensor
    if isinstance(output, dict):
        new = dict(output)
        for key in ('sample', 'hidden_states', 'last_hidden_state'):
            if _is_tensor(new.get(key)):
                new[key] = tensor
                return new
        return new
    if hasattr(output, 'sample'):
        try:
            return type(output)(sample=tensor)
        except Exception:
            pass
    if isinstance(output, (tuple, list)):
        items = list(output)
        items[0] = tensor
        return type(output)(items) if isinstance(output, tuple) else items
    raise TypeError('无法重建 Transformer 输出类型：%r' % type(output))


def enable_hicache(
    pipe: Any,
    interval: int = 7,
    order: int = 2,
    sigma: float = 0.5,
    warmup: int = 2,
    mode: str = "anchored",
    verbose: bool = False,
) -> FeatureCache:
    """为 pipeline 启用 HiCache 加速。

    Args:
        pipe: ppdiffusers 的 ``FluxPipeline``（或任何内部持有 ``transformer`` 的 pipeline）。
        interval: 缓存间隔 N，每隔 N 步做一次完整前向（默认 7）。
        order: 外推阶数 Order（默认 2）。
        sigma: 收缩因子 σ（默认 0.5）。
        warmup: 预热步数，期间全部完整计算（默认 2）。
        verbose: 是否逐步打印"完整计算 / 外推"日志。

    Returns:
        运行期 :class:`FeatureCache` 对象，可通过 ``cache.state.speedup`` 读取实际加速比。
    """
    transformer = getattr(pipe, 'transformer', None)
    if transformer is None:
        raise AttributeError('pipe 上没有 transformer 属性，本补丁要求 diffusers 风格的 pipeline')

    if getattr(transformer, '_hicache_enabled', False):
        # 重复启用时先复位
        disable_hicache(pipe)

    cache = FeatureCache(interval=interval, order=order, sigma=sigma, warmup=warmup,
                         mode=mode)
    original_forward = transformer.forward
    last_output: list = []

    def hicache_forward(*args: Any, **kwargs: Any) -> Any:
        st = cache.state
        need_compute = cache.should_compute() or _tensor_of(last_output[-1] if last_output else None) is None

        if not need_compute:
            ref = _tensor_of(last_output[-1])
            pred = cache.predict(like=ref)
            out = _rebuild(last_output[-1], pred)
            last_output[:] = [out]
            cache.tick()
            if verbose:
                print('[HiCache] step=%d predicted (order=%d)' % (st.step, st.last_order_used))
            return out

        output = original_forward(*args, **kwargs)
        feat = _tensor_of(output)
        if feat is not None:
            cache.mark_computed(feat)
            last_output[:] = [output]
        cache.tick()
        if verbose:
            print('[HiCache] step=%d full compute' % st.step)
        return output

    transformer.forward = hicache_forward
    transformer._hicache = cache
    transformer._hicache_enabled = True
    transformer._hicache_original_forward = original_forward
    return cache


def disable_hicache(pipe: Any) -> None:
    """恢复原始前向。"""
    transformer = getattr(pipe, 'transformer', None)
    if transformer is None or not getattr(transformer, '_hicache_enabled', False):
        return
    transformer.forward = transformer._hicache_original_forward
    transformer._hicache_enabled = False
    for attr in ('_hicache', '_hicache_original_forward'):
        if hasattr(transformer, attr):
            delattr(transformer, attr)


def hicache_state(pipe: Any) -> Optional[HiCacheState]:
    """读取运行期统计（加速比、外推步列表等）。"""
    transformer = getattr(pipe, 'transformer', None)
    cache = getattr(transformer, '_hicache', None) if transformer is not None else None
    return cache.state if cache is not None else None