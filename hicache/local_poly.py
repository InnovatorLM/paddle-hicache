# -*- coding: utf-8 -*-
"""局部最小二乘多项式 + 缩放埃尔米特基的外推（不等距、低阶精确）。

思路
----
锚点附近用最近若干**真实计算**点做一次 ≤Order 次多项式最小二乘拟合（允许不等距，直接用真实
step 值），得到锚点处的系数 c_j = f⁽ʲ⁾(t0)/j!；再用基函数 B_j(x) 替代泰勒项中的 xʲ：

    F̂(t0 + x) = F(t0) + Σ_{j=1..Order} c_j · B_j(x)

在 ``basis='anchored'``（默认）下 B_1=x、B_2=x²，故 Order ≤ 2 时对不超过二次的轨迹**精确**；
换用 ``normalized``/``raw`` 会引入各自的偏置或欠外推，见 :mod:`hicache.basis`。

每步额外开销为一次 (k×k) 标量最小二乘求解（k ≤ Order+1），可忽略。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import paddle

from .basis import hermite_basis_values

__all__ = ['fit_local_polynomial', 'extrapolate_local']


def fit_local_polynomial(steps: Sequence[int], features: Sequence[paddle.Tensor], order: int):
    """对 (steps, features) 做 ≤order 次多项式最小二乘拟合。

    Returns:
        (coeffs, deg)：coeffs[j] 为张量（等价于 f⁽ʲ⁾(t0)/j!），deg 为实际拟合阶数。
    """
    ts = np.asarray(steps, dtype=np.float64)
    t0 = float(ts[-1])
    deg = max(1, min(order, len(ts) - 1))
    xs = ts - t0
    A = np.vander(xs, deg + 1, increasing=True)   # 列：x^0..x^deg
    pinv = np.linalg.pinv(A)

    coeffs: List[paddle.Tensor] = []
    for j in range(deg + 1):
        acc = None
        for i, f in enumerate(features):
            w = float(pinv[j, i])
            if w == 0.0:
                continue
            term = f * w
            acc = term if acc is None else acc + term
        coeffs.append(acc)
    return coeffs, deg


def extrapolate_local(
    points: Sequence[Tuple[int, paddle.Tensor]],
    target_step: int,
    order: int,
    sigma: float = 0.5,
    mode: str = 'anchored',
) -> paddle.Tensor:
    """用局部多项式系数 + 基函数外推 target_step 处的特征。"""
    pts = sorted(points, key=lambda p: p[0])
    if len(pts) < 2:
        raise ValueError('至少需要 2 个计算点才能外推')

    steps = [p[0] for p in pts]
    feats = [p[1] for p in pts]
    coeffs, deg = fit_local_polynomial(steps, feats, order)

    x = float(target_step) - float(steps[-1])
    basis = hermite_basis_values(x, deg, sigma, mode)

    out = coeffs[0]
    for j in range(1, deg + 1):
        if basis[j] == 0.0:
            continue
        out = out + coeffs[j] * basis[j]
    return out