# -*- coding: utf-8 -*-
"""缩放埃尔米特多项式基与特征外推预测（PaddlePaddle 实现）。

对应 ICLR 2026《HiCache》：把泰勒外推中的幂函数基 ``(−k)^i`` 换成缩放埃尔米特基。

基函数定义与推导
----------------
物理学家埃尔米特多项式::

    H0 = 1,  H1 = 2y,  H(n+1)(y) = 2y·Hn(y) − 2n·H(n−1)(y)

收缩（双缩放）::

    g_n(x) = σⁿ · Hn(σx),   σ ∈ (0, 1]

``g_n`` 的主导项为 ``σⁿ·(2σx)ⁿ = (2σ²x)ⁿ``，与 ``xⁿ`` 相差 ``(2σ²)ⁿ`` 倍。为使基函数以
``xⁱ`` 为主导项（一阶项与泰勒完全一致、二阶项带来论文所述的 "H2 负偏移"），再做一次归一化::

    C_n(x) = g_n(x) / (2σ²)ⁿ = Hn(σx) / (2ⁿ σⁿ)

于是 ``C_1(x) = x``（与泰勒一阶完全相同）、``C_2(x) = x² − 1/(2σ²)``（σ=0.5 时即 ``x² − 2``，
对应论文中"负偏移可刻画非单调特征演化"的论述）。

预测公式::

    F̂(t−k) = F(t) + Σ_{i=1..Order} ΔⁱF(t)/i! · C_i(−k)

其中 ``ΔⁱF(t)`` 为向后有限差分。每步额外开销为 O(Order) 次标量运算。

``normalize=False`` 时退化为"直接替换 g_i(−k)"的朴素形式（便于与官方实现逐位比对）。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import paddle

__all__ = [
    'scaled_hermite_values',
    'hermite_coefficients',
    'backward_differences',
    'extrapolate',
    'extrapolate_from_points',
    'relative_l1',
]


def scaled_hermite_values(x: float, order: int, sigma: float) -> List[float]:
    """计算 g_n(x) = σⁿ·Hn(σx)，n = 0..order（未归一化）。

    递推形式（直接由缩放定义推出，避免每步重复乘 σ）::

        g_0 = 1,  g_1 = 2σ²x,  g_(n+1) = 2σ²x·g_n − 2nσ²·g_(n−1)
    """
    if not 0.0 < sigma <= 1.0:
        raise ValueError('sigma 必须满足 0 < sigma <= 1，当前为 %r' % sigma)
    if order < 0:
        raise ValueError('order 必须 >= 0')

    g = [1.0]
    if order >= 1:
        g.append(2.0 * sigma * sigma * x)
    for n in range(1, order):
        g.append(2.0 * sigma * sigma * x * g[n] - 2.0 * n * sigma * sigma * g[n - 1])
    return g[: order + 1]


def hermite_coefficients(k: int, order: int, sigma: float, normalize: bool = True) -> List[float]:
    """外推第 k 步时各阶差分项的系数 c_i = C_i(+k)/i!（i = 0..order，c_0 恒为 0）。

    符号约定：差分按采样方向取（``Δ¹F_t = F_t − F_{t−1}``），``k`` 为向采样前方跳过的步数，
    故基函数在 ``+k`` 处取值。论文记号中的 ``H̃i(−k)`` 对应"扩散时间递减"的反向索引，与上式等价。

    Args:
        k: 预测步长（正整数）。
        order: 外推阶数。
        sigma: 收缩因子。
        normalize: True 时对基函数除以 (2σ²)ⁱ，使一阶项与泰勒一致（推荐）。
    """
    if k <= 0:
        raise ValueError('k 必须为正整数')
    g = scaled_hermite_values(float(k), order, sigma)
    coefs = [0.0]
    fact = 1.0
    for i in range(1, order + 1):
        fact *= i
        c = g[i] / fact
        if normalize:
            c /= (2.0 * sigma * sigma) ** i
        coefs.append(c)
    return coefs


def backward_differences(features: Sequence[paddle.Tensor], order: int) -> List[paddle.Tensor]:
    """由最近的历史特征序列计算向后有限差分（1..order 阶）。

    Args:
        features: 按时间顺序排列的历史特征，最后一个元素为当前步 F_t；长度须 >= order+1。
        order: 差分阶数。

    Returns:
        list，索引 i（1..order）为第 i 阶差分；索引 0 为 None 占位。
    """
    if len(features) < order + 1:
        raise ValueError('历史特征不足：需要至少 %d 个，实际 %d 个' % (order + 1, len(features)))
    seq = list(features)[-(order + 1):]
    diffs: List[paddle.Tensor] = [None]
    level = seq
    for _ in range(order):
        level = [level[i] - level[i - 1] for i in range(1, len(level))]
        diffs.append(level[-1])
    return diffs


def extrapolate(
    features: Sequence[paddle.Tensor],
    k: int,
    order: int,
    sigma: float,
    normalize: bool = True,
) -> paddle.Tensor:
    """以缩放埃尔米特基外推未来第 k 步的特征（等步距版本）。

    Args:
        features: 历史特征序列，最后一个元素为当前步 F_t。
        k: 预测步长（>= 1）。
        order: 外推阶数。
        sigma: 收缩因子。
        normalize: 是否对基函数做 (2σ²)ⁱ 归一化（默认 True）。

    Returns:
        预测特征 F̂(t−k)，shape 与输入一致。
    """
    if len(features) < order + 1:
        raise ValueError('历史特征不足，需 >= order+1 个')

    base = features[-1]
    diffs = backward_differences(features, order)
    coefs = hermite_coefficients(k, order, sigma, normalize=normalize)

    out = base
    for i in range(1, order + 1):
        if coefs[i] == 0.0:
            continue
        out = out + diffs[i] * coefs[i]
    return out


def extrapolate_from_points(
    points: Sequence[Tuple[int, paddle.Tensor]],
    target_step: int,
    order: int,
    sigma: float,
    normalize: bool = True,
) -> paddle.Tensor:
    """按真实步距做不等距外推（用于"只在完整计算步采样"的锚点式缓存）。

    以牛顿差商估计各阶导数，避免"锚点间距 6 步却按 1 步差分放大"的尺度错配：

        f[t0]   = F(t0)
        f[t0t1] = (F1 − F0) / (t1 − t0)
        f[t0t1t2] = (f[t1t2] − f[t0t1]) / (t2 − t0)
        F(t0 + x) ≈ F0 + f[t0t1]·B1(x) + 2·f[t0t1t2]·B2(x)/2

    其中 B_i(x) 为缩放埃尔米特基（``normalize=True`` 时主导项为 xⁱ，与泰勒展开同阶）。

    Args:
        points: [(step_index, feature), ...]，按 step 递增，最后一个为锚点。
        target_step: 需要预测的时间步（> 锚点 step）。
        order: 外推阶数（不超过 len(points)-1）。
        sigma: 收缩因子。
        normalize: 是否归一化基函数。

    Returns:
        预测特征。
    """
    pts = sorted(points, key=lambda p: p[0])
    if len(pts) < 2:
        raise ValueError('至少需要 2 个计算点才能外推')
    order_eff = min(order, len(pts) - 1)
    ts = [float(p[0]) for p in pts]
    fs = [p[1] for p in pts]

    # 牛顿差商
    div = [list(fs)]
    for level in range(1, order_eff + 1):
        prev = div[level - 1]
        cur = []
        for i in range(len(prev) - 1):
            denom = ts[i + level] - ts[i]
            cur.append((prev[i + 1] - prev[i]) / denom)
        div.append(cur)

    t0 = ts[-1]
    x = float(target_step) - t0
    diffs = [div[level][-1] for level in range(order_eff + 1)]  # 0 阶为 F0，1 阶为 f[t0t1]...

    basis = scaled_hermite_values(x, order_eff, sigma)
    out = fs[-1]
    for i in range(1, order_eff + 1):
        # 差商近似 f⁽ⁱ⁾/i!，故泰勒项 f⁽ⁱ⁾xⁱ/i! = 差商·xⁱ；基函数替代 xⁱ
        coef = diffs[i] * basis[i]
        if normalize:
            coef = coef / (2.0 * sigma * sigma) ** i
        out = out + coef
    return out


def relative_l1(pred: paddle.Tensor, target: paddle.Tensor, eps: float = 1e-8) -> paddle.Tensor:
    """相对 L1 误差，用于日志与质量监控。"""
    return paddle.abs(pred - target).mean() / (paddle.abs(target).mean() + eps)
