# -*- coding: utf-8 -*-
"""NumPy 参考实现：与 :mod:`hicache.predictor` 同构，仅用于数值对齐校验（不依赖 Paddle）。"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

__all__ = [
    'scaled_hermite_values',
    'hermite_coefficients',
    'backward_differences',
    'extrapolate',
    'hermite_polynomial_exact',
]


def scaled_hermite_values(x: float, order: int, sigma: float) -> List[float]:
    """g_n(x) = σⁿ·Hn(σx)，n = 0..order。"""
    if not 0.0 < sigma <= 1.0:
        raise ValueError('sigma 必须满足 0 < sigma <= 1')
    g = [1.0]
    if order >= 1:
        g.append(2.0 * sigma * sigma * x)
    for n in range(1, order):
        g.append(2.0 * sigma * sigma * x * g[n] - 2.0 * n * sigma * sigma * g[n - 1])
    return g[: order + 1]


def hermite_coefficients(k: int, order: int, sigma: float, normalize: bool = True) -> List[float]:
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


def backward_differences(features: Sequence[np.ndarray], order: int) -> List[np.ndarray]:
    if len(features) < order + 1:
        raise ValueError('历史特征不足')
    seq = list(features)[-(order + 1):]
    diffs: List[np.ndarray] = [None]
    level = seq
    for _ in range(order):
        level = [level[i] - level[i - 1] for i in range(1, len(level))]
        diffs.append(level[-1])
    return diffs


def extrapolate(
    features: Sequence[np.ndarray],
    k: int,
    order: int,
    sigma: float,
    normalize: bool = True,
) -> np.ndarray:
    base = features[-1]
    diffs = backward_differences(features, order)
    coefs = hermite_coefficients(k, order, sigma, normalize=normalize)
    out = base.copy()
    for i in range(1, order + 1):
        if coefs[i]:
            out = out + diffs[i] * coefs[i]
    return out


def hermite_polynomial_exact(x: float, n: int) -> float:
    """物理学家埃尔米特多项式 Hn(x) 的精确值（递推，用于校验）。"""
    if n == 0:
        return 1.0
    if n == 1:
        return 2.0 * x
    h_prev, h_curr = 1.0, 2.0 * x
    for k in range(1, n):
        h_prev, h_curr = h_curr, 2.0 * x * h_curr - 2.0 * k * h_prev
    return h_curr


def normalized_basis(x: float, n: int, sigma: float) -> float:
    """C_n(x) = Hn(σx)/(2ⁿσⁿ)，主导项为 xⁿ。"""
    return hermite_polynomial_exact(sigma * x, n) / ((2.0 * sigma) ** n)
