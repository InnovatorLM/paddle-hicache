# -*- coding: utf-8 -*-
"""PyTorch 独立参考实现（用于跨框架对齐校验）。

与 Paddle 实现的**编码路径刻意不同**，避免"同一份代码抄两遍"的假对齐：

| 环节 | Paddle 实现 (hicache/) | 本文件 (torch) |
| --- | --- | --- |
| 多项式拟合 | 预计算伪逆 `np.linalg.pinv` 后线性组合 | `torch.linalg.lstsq`（QR / gelsd 路径） |
| 埃尔米特基求值 | 逐点三项递推 g(n+1)=2σ²x·g(n)−2nσ²·g(n−1) | 先求多项式升幂系数，再用 Horner 法求值 |
| 数据类型 | float32 | float32 / float64 可选 |

因此两者结果**不应逐位相同**，只应在 fp32 舍入量级（~1e-7 相对）内一致；这比对同一份公式的
逐位复现更有说服力。
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

__all__ = ['hermite_poly_coeffs', 'basis_values', 'extrapolate']


def hermite_poly_coeffs(n: int) -> List[float]:
    """物理学家埃尔米特多项式 H_n 的升幂系数（H_{n+1} = 2xH_n − 2nH_{n−1}）。"""
    if n == 0:
        return [1.0]
    if n == 1:
        return [0.0, 2.0]
    h_prev, h_cur = [1.0], [0.0, 2.0]
    for k in range(1, n):
        nxt = [0.0] * (len(h_cur) + 1)
        for i, c in enumerate(h_cur):
            nxt[i + 1] += 2.0 * c
        for i, c in enumerate(h_prev):
            nxt[i] += -2.0 * k * c
        h_prev, h_cur = h_cur, nxt
    return h_cur


def _horner(coeffs: Sequence[float], x: torch.Tensor) -> torch.Tensor:
    acc = torch.zeros_like(x)
    for c in reversed(coeffs):
        acc = acc * x + c
    return acc


def basis_values(x: torch.Tensor, order: int, sigma: float, mode: str = 'anchored') -> List[torch.Tensor]:
    """B_n(x)，与 hicache.basis 同定义但用系数+Horner 求值。"""
    out = [torch.ones_like(x)]
    for n in range(1, order + 1):
        coeffs = hermite_poly_coeffs(n)
        # H_n(σx) 的升幂系数：H_n 系数逐个乘 σ^i
        scaled = [c * (sigma ** i) for i, c in enumerate(coeffs)]
        val = _horner(scaled, x)
        if mode == 'anchored':
            val = val - _horner(scaled, torch.zeros_like(x))
        # 统一：g_n(x)=σⁿH_n(σx)；anchored/normalized 再除以 (2σ²)ⁿ，raw 不除
        if mode != 'raw':
            val = (val * (sigma ** n)) / ((2.0 * sigma * sigma) ** n)
        out.append(val)
    return out


def extrapolate(
    points: Sequence[Tuple[int, torch.Tensor]],
    target_step: int,
    order: int,
    sigma: float = 0.5,
    mode: str = 'anchored',
    dtype=torch.float32,
) -> torch.Tensor:
    pts = sorted(points, key=lambda p: p[0])
    ts = [float(p[0]) for p in pts]
    feats = [p[1].to(dtype) for p in pts]
    t0 = ts[-1]
    deg = max(1, min(order, len(ts) - 1))
    xs = torch.tensor([t - t0 for t in ts], dtype=dtype)

    A = torch.stack([xs ** j for j in range(deg + 1)], dim=1)      # [k, deg+1]
    F = torch.stack(feats, dim=0)                                   # [k, ...]
    Fmat = F.reshape(len(feats), -1)
    sol = torch.linalg.lstsq(A, Fmat).solution                      # [deg+1, N]
    coeffs = [sol[j].reshape(feats[0].shape) for j in range(deg + 1)]

    x = torch.tensor(float(target_step) - t0, dtype=dtype)
    basis = basis_values(x, deg, sigma, mode)
    out = coeffs[0]
    for j in range(1, deg + 1):
        out = out + coeffs[j] * basis[j]
    return out