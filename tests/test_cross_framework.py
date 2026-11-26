# -*- coding: utf-8 -*-
"""跨框架对齐测试：Paddle 实现 vs PyTorch 独立实现 vs 解析解。

为什么要三路比对
----------------
* 只比"Paddle 版 vs 我抄的 NumPy 版"没有意义：同一份公式、同一算子顺序、同为 fp32，
  结果逐位相同（差值恰为 0），只能证明代码抄得对，不能证明数学正确。
* 本测试因此引入两条**独立路径**：
  1) PyTorch 实现：拟合用 `torch.linalg.lstsq`、基函数用"系数 + Horner"求值，与 Paddle 版
     （伪逆 + 逐点递推）在实现上完全不同 → 二者差值应落在 fp32 舍入量级（~1e-6 相对），
     而不是 0；
  2) 解析解：对已知解析轨迹（线性 / 二次 / 指数）直接代入真值 → 误差有明确物理含义。
* 同时给出 fp32 与 fp64 的差值，说明残差来自精度而非算法分歧。

运行（需 paddle + torch）：python tests/test_cross_framework.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paddle  # noqa: E402
import torch  # noqa: E402

from hicache.local_poly import extrapolate_local  # noqa: E402
from tests import torch_reference as tref  # noqa: E402

TOL_FP32 = 2e-5
TOL_FP64 = 2e-9


def make_points(rand, steps, shape, dtype='numpy'):
    return [(s, rand(shape)) for s in steps]


def compare_once(rng, steps, order, sigma, mode='anchored'):
    shape = (2, 4, 8)
    feats = [rng.standard_normal(shape).astype(np.float32) for _ in steps]
    target = steps[-1] + 3

    # Paddle
    pts_pd = [(s, paddle.to_tensor(f)) for s, f in zip(steps, feats)]
    out_pd = extrapolate_local(pts_pd, target, order=order, sigma=sigma, mode=mode).numpy()

    # PyTorch (fp32)
    pts_t32 = [(s, torch.from_numpy(f)) for s, f in zip(steps, feats)]
    out_t32 = tref.extrapolate(pts_t32, target, order=order, sigma=sigma, mode=mode,
                              dtype=torch.float32).numpy()

    # PyTorch (fp64，近似"真值")
    pts_t64 = [(s, torch.from_numpy(f.astype(np.float64))) for s, f in zip(steps, feats)]
    out_t64 = tref.extrapolate(pts_t64, target, order=order, sigma=sigma, mode=mode,
                              dtype=torch.float64).numpy()

    scale = np.abs(out_t64).mean() + 1e-8
    d_pd_t32 = float(np.abs(out_pd - out_t32).max())
    d_pd_t64 = float(np.abs(out_pd.astype(np.float64) - out_t64).max())
    rel_pd_t32 = d_pd_t32 / scale
    return d_pd_t32, rel_pd_t32, d_pd_t64 / scale


def main() -> int:
    rng = np.random.default_rng(7)
    fails = []

    print('== 1. Paddle vs PyTorch（同公式、不同实现路径）==')
    print('%-34s %12s %12s %12s' % ('配置', 'max|Δ|', '相对差', 'vs fp64 相对差'))
    for order in (1, 2, 3):
        for sigma in (0.4, 0.5, 0.7):
            d, rel, rel64 = compare_once(rng, [0, 1, 7], order, sigma)
            flag = 'PASS' if rel < TOL_FP32 else 'FAIL'
            print('Order=%d σ=%.1f  %-16s %12.3e %12.3e %12.3e'
                  % (order, sigma, flag, d, rel, rel64))
            if rel >= TOL_FP32:
                fails.append('order=%d sigma=%.1f' % (order, sigma))

    print()
    print('== 2. 解析轨迹误差（与真值比）==')
    cases = {
        '线性 a·t+b': lambda t: 2.5 * t - 1.0,
        '二次 a·t²+b·t+c': lambda t: 0.03 * t * t - 1.5 * t + 2.0,
        '指数 exp(-0.5t/50)': lambda t: np.exp(-0.5 * t / 50.0),
    }
    steps = [0, 1, 7]
    for name, fn in cases.items():
        feats = [np.array([[fn(s)]], dtype=np.float32) for s in steps]
        target = steps[-1] + 5
        pts_pd = [(s, paddle.to_tensor(f)) for s, f in zip(steps, feats)]
        pred = float(extrapolate_local(pts_pd, target, order=2, sigma=0.5).numpy().ravel()[0])
        truth = float(fn(target))
        rel = abs(pred - truth) / (abs(truth) + 1e-9)
        print('%-22s pred=%+.6f truth=%+.6f  相对误差=%.3e' % (name, pred, truth, rel))

    print()
    print('== 3. 结论 ==')
    print('* Paddle 与 PyTorch 属不同实现路径，差值应非 0 且在 fp32 舍入量级（<2e-5 相对）；')
    print('* 线性轨迹 Order≥1 应精确（误差 0）；二次轨迹 Order=2 锚定基下应精确；')
    print('* 若某处为 0.0，说明两条路径在该配置下退化为同一算式，需换配置重测。')

    if fails:
        print('FAILURES:', fails)
        return 1
    print('跨框架对齐检查通过')
    return 0


if __name__ == '__main__':
    sys.exit(main())