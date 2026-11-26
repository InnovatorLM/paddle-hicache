# -*- coding: utf-8 -*-
"""特征缓存队列与"完整计算 / 外推"混合调度（锚点式、局部多项式外推）。

策略（对应论文 cache-then-forecast 的工程实现）：

* 预热期（前 ``warmup`` 步）全部完整计算，建立拟合点并保证生成早期质量；
* 之后每 ``interval`` 步做一次完整计算，其余步外推替代网络前向
  （50 步 / interval=7 / warmup=2 → 完整计算 9 次，即 5.56×）；
* 只有**真实计算**的特征进入拟合点集，预测值不回填——否则窗口末的预测残差会在下一次完整计算
  处形成虚假跳变，被当作大曲率，实测会让后续外推发散到 1e8 量级；
* 拟合按真实步距做（预热期 1 步、窗口 6~7 步），基函数在 x = 目标步 − 锚点步 处取值。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import paddle

from .local_poly import extrapolate_local


@dataclass
class HiCacheState:
    """运行期状态与统计。"""

    interval: int = 7
    order: int = 2
    sigma: float = 0.5
    mode: str = 'anchored'
    warmup: int = 2

    step: int = 0
    computed: int = 0
    predicted: int = 0
    predicted_steps: list = field(default_factory=list)
    anchor_step: int = 0
    last_order_used: int = 0

    def reset(self) -> None:
        self.step = 0
        self.computed = 0
        self.predicted = 0
        self.predicted_steps = []
        self.anchor_step = 0
        self.last_order_used = 0

    @property
    def speedup(self) -> float:
        """相对"每步完整计算"的理论加速比。"""
        total = self.computed + self.predicted
        return float(total) / float(self.computed) if self.computed else 1.0


class FeatureCache:
    """记录完整计算步的特征点，给出"该步完整计算还是外推"的判定并完成外推。"""

    def __init__(self, interval: int = 7, order: int = 2, sigma: float = 0.5, warmup: int = 2,
                 mode: str = 'anchored'):
        if interval < 1:
            raise ValueError('interval 必须 >= 1')
        if order < 1:
            raise ValueError('order 必须 >= 1')
        if warmup < 1:
            raise ValueError('warmup 必须 >= 1（至少需要一步真实特征作为拟合起点）')
        self.state = HiCacheState(interval=interval, order=order, sigma=sigma, mode=mode,
                                  warmup=warmup)
        self._points: List[Tuple[int, paddle.Tensor]] = []

    # ---- 判定 ----
    def should_compute(self) -> bool:
        """当前步是否需要执行完整网络前向。"""
        st = self.state
        if st.step < st.warmup:
            return True
        if len(self._points) < 2:
            return True
        return st.step % st.interval == 0

    # ---- 更新 ----
    def mark_computed(self, feature: paddle.Tensor) -> None:
        """登记一次完整计算：写入拟合点并刷新锚点。"""
        st = self.state
        st.computed += 1
        st.anchor_step = st.step
        self._points.append((st.step, feature.detach().astype(paddle.float32)))
        if len(self._points) > st.order + 2:
            self._points = self._points[-(st.order + 2):]

    # ---- 预测 ----
    def predict(self, like: Optional[paddle.Tensor] = None) -> paddle.Tensor:
        """以最近的计算点为锚点外推到当前步。"""
        st = self.state
        if len(self._points) < 2:
            raise RuntimeError('计算点不足，无法外推；请检查 should_compute 判定逻辑')
        order_eff = min(st.order, len(self._points) - 1)
        pred = extrapolate_local(self._points, st.step, order=order_eff, sigma=st.sigma,
                                 mode=st.mode)
        st.last_order_used = order_eff
        st.predicted += 1
        st.predicted_steps.append(st.step)
        if like is not None:
            pred = pred.astype(like.dtype)
        return pred

    def tick(self) -> None:
        self.state.step += 1

    def reset(self) -> None:
        self._points = []
        self.state.reset()