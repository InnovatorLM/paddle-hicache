# -*- coding: utf-8 -*-
"""基函数形态：三种可选实现（同一套递推，仅归一化/锚定方式不同）。

记 g_n(x) = σⁿ·Hn(σx)（缩放埃尔米特多项式，递推 g_(n+1) = 2σ²x·g_n − 2nσ²·g_(n−1)）：

* ``anchored``（默认，推荐）
      B_n(x) = [g_n(x) − g_n(0)] / (2σ²)ⁿ
  性质：B_1(x) = x、B_2(x) = x²，即 **Order ≤ 2 时严格等于泰勒外推**（不引入任何偏置），
  Order ≥ 3 起才引入埃尔米特修正项（如 B_3 = x³ − 3x/(2σ²)）。σ 越小修正越强。
* ``normalized``
      B_n(x) = g_n(x) / (2σ²)ⁿ
  性质：主导项为 xⁿ，但每阶带常数偏移（B_2 = x² − 1/(2σ²)）；对应论文中"H2 负偏移可刻画
  非单调演化"的形态，代价是对平滑轨迹有系统性偏置（实测每步 ≈ −(1/2σ²)·c₂）。
* ``raw``
      B_n(x) = g_n(x)
  性质：直接替换 (−k)ⁿ 的朴素做法，全程欠外推（B_1 = 2σ²x），仅用于与其它实现逐位对比。
"""
from __future__ import annotations

from typing import List

__all__ = ['hermite_basis_values', 'MODES']

MODES = ('anchored', 'normalized', 'raw')


def _scaled_series(x: float, order: int, sigma: float) -> List[float]:
    """g_n(x) = σⁿ·Hn(σx)，n = 0..order。"""
    g = [1.0]
    if order >= 1:
        g.append(2.0 * sigma * sigma * x)
    for n in range(1, order):
        g.append(2.0 * sigma * sigma * x * g[n] - 2.0 * n * sigma * sigma * g[n - 1])
    return g[: order + 1]


def hermite_basis_values(x: float, order: int, sigma: float, mode: str = 'anchored') -> List[float]:
    """返回 B_0..B_order（B_0 恒为 1，外推时不使用）。"""
    if mode not in MODES:
        raise ValueError('mode 必须是 %s 之一，当前 %r' % (MODES, mode))
    if not 0.0 < sigma <= 1.0:
        raise ValueError('sigma 必须满足 0 < sigma <= 1')

    g = _scaled_series(x, order, sigma)
    if mode == 'raw':
        return g

    g0 = _scaled_series(0.0, order, sigma)
    out = [1.0]
    for n in range(1, order + 1):
        val = g[n]
        if mode == 'anchored':
            val -= g0[n]
        val /= (2.0 * sigma * sigma) ** n
        out.append(val)
    return out