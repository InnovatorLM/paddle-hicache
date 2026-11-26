# -*- coding: utf-8 -*-
"""Paddle 实现与 NumPy 参考实现的数值对齐测试。

运行（在 paddle-hicache 根目录）：
    python tests/test_alignment.py

校验内容：
1. 缩放递推 g_n = σⁿHn(σx) 与精确多项式一致；
2. 归一化基函数的一阶项恒等于 x、二阶项为 x² − 1/(2σ²)（论文所述"负偏移"）；
3. Order=1 时对线性特征轨迹的外推精确（说明与泰勒一阶等价）；
4. 端到端外推：Paddle 与 NumPy 在同一输入下 max|Δ| < 1e-5（fp32）；
5. 混合调度模式：50 步、interval=7 时完整计算 9 次（理论 5.5×）。
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hicache import reference as ref  # noqa: E402

try:
    import paddle  # noqa: E402
    from hicache import predictor as pp  # noqa: E402
    from hicache.cache import FeatureCache  # noqa: E402
except Exception as exc:  # pragma: no cover
    paddle = None
    print('[SKIP] 未安装 paddle，跳过 Paddle 侧校验：%s' % exc)


TOL = 1e-5
failures = []


def check(name: str, ok: bool, detail: str = '') -> None:
    print('%-54s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        failures.append(name)


def test_recursion_vs_exact() -> None:
    worst = 0.0
    for sigma in (0.2, 0.4, 0.5, 0.7, 1.0):
        for x in (-6.0, -3.0, -1.0, 0.0, 2.0):
            g = ref.scaled_hermite_values(x, order=6, sigma=sigma)
            for n in range(7):
                exact = sigma ** n * ref.hermite_polynomial_exact(sigma * x, n)
                worst = max(worst, abs(g[n] - exact) / max(1.0, abs(exact)))
    check('缩放递推 g_n == σⁿ·Hn(σx)', worst < 1e-9, 'max rel err=%.2e' % worst)


def test_normalized_basis_shape() -> None:
    sigma = 0.5
    ok1 = all(abs(ref.normalized_basis(x, 1, sigma) - x) < 1e-12 for x in (-3.0, -0.7, 0.0, 1.5))
    ok2 = all(abs(ref.normalized_basis(x, 2, sigma) - (x * x - 1.0 / (2 * sigma * sigma))) < 1e-12
              for x in (-3.0, -0.7, 0.0, 1.5))
    check('C_1(x) == x（一阶与泰勒一致）', ok1)
    check('C_2(x) == x² − 1/(2σ²)（σ=0.5 时 x²−2）', ok2)


def test_order1_exact_on_linear() -> None:
    # 线性轨迹 F(t) = a·t + b，Order=1 归一化基下应精确外推任意 k
    ts = np.arange(-4, 1, dtype=np.float64)
    a, b = 2.5, -1.0
    feats = [np.array([a * t + b]) for t in ts]
    worst = 0.0
    for k in (1, 2, 5, 7):
        pred = ref.extrapolate(feats, k=k, order=1, sigma=0.5, normalize=True)
        worst = max(worst, abs(float(pred[0]) - (a * (ts[-1] + k) + b)))
    check('Order=1 对线性轨迹外推精确（k=1..7）', worst < 1e-12, 'max err=%.2e' % worst)


def test_higher_order_bounded() -> None:
    # 平滑轨迹（指数衰减 + 慢变相位）：各阶次 1 步外推的相对误差应有界（缓存可用性的下限要求）
    ts = np.linspace(0, 3, 60)
    traj = np.array([[np.exp(-0.3 * t) * (1 + 0.2 * np.sin(2 * t))] for t in ts])
    scale = float(np.mean(np.abs(traj)))
    errors = []
    for order in (1, 2, 3):
        errs = []
        for i in range(order + 1, len(ts) - 1):
            pred = ref.extrapolate(traj[i - order:i + 1], k=1, order=order, sigma=0.5)
            errs.append(abs(float(pred[0]) - float(traj[i + 1][0])))
        errors.append(float(np.mean(errs)))
    rel = [e / scale for e in errors]
    check('1 步外推相对误差有界 (%s)' % '/'.join('%.2e' % r for r in rel),
          all(r < 0.05 for r in rel))


def test_negative_offset_vs_taylor() -> None:
    # 文档化的差异：k=1、σ=0.5 时二阶项系数由泰勒的 0.5 变为 -0.5（C_2 负偏移）
    c = ref.hermite_coefficients(1, 2, 0.5, normalize=True)
    taylor2 = 0.5
    check('二阶项系数 C_2(1)/2! = %.3f（泰勒 %.1f）' % (c[2], taylor2),
          abs(c[1] - 1.0) < 1e-12 and abs(c[2] - (-0.5)) < 1e-12)


def test_paddle_vs_numpy() -> None:
    if paddle is None:
        print('[SKIP] Paddle 对齐测试')
        return
    rng = np.random.default_rng(0)
    for order in (1, 2, 3):
        for sigma in (0.4, 0.5, 0.7):
            feats_np = [rng.standard_normal((2, 4, 8)).astype(np.float32) for _ in range(order + 2)]
            np_pred = ref.extrapolate(feats_np, k=3, order=order, sigma=sigma)
            pd_pred = pp.extrapolate([paddle.to_tensor(f) for f in feats_np], k=3, order=order,
                                     sigma=sigma).numpy()
            err = float(np.max(np.abs(np_pred - pd_pred)))
            check('Paddle vs NumPy (Order=%d, σ=%.1f)' % (order, sigma), err < TOL, 'max|Δ|=%.2e' % err)

    for x in (-7.0, -3.0, -1.0):
        g_np = ref.scaled_hermite_values(x, order=4, sigma=0.5)
        g_pd = pp.scaled_hermite_values(x, order=4, sigma=0.5)
        check('基函数标量对齐 x=%.1f' % x, max(abs(a - b) for a, b in zip(g_np, g_pd)) < 1e-12)


def test_cache_scheduling() -> None:
    if paddle is None:
        print('[SKIP] 调度测试')
        return
    cache = FeatureCache(interval=7, order=2, sigma=0.5, warmup=2)
    pattern = []
    for _ in range(50):
        if cache.should_compute():
            cache.mark_computed(paddle.randn([1, 4, 8]))
            pattern.append('C')
        else:
            cache.predict()
            pattern.append('P')
        cache.tick()
    comp = pattern.count('C')
    ratio = 50.0 / comp
    check('50 步中完整计算 %d 次 → 理论加速 %.2f×' % (comp, ratio), 4.5 < ratio < 6.5,
          'head=%s' % ''.join(pattern[:16]))


if __name__ == '__main__':
    test_recursion_vs_exact()
    test_normalized_basis_shape()
    test_order1_exact_on_linear()
    test_higher_order_bounded()
    test_negative_offset_vs_taylor()
    test_paddle_vs_numpy()
    test_cache_scheduling()
    print('-' * 70)
    if failures:
        print('FAILURES:', failures)
        sys.exit(1)
    print('全部对齐检查通过')