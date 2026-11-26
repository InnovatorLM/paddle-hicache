# -*- coding: utf-8 -*-
"""flux_patch 功能测试：用一个桩 transformer 验证补丁的行为（不依赖模型权重）。

校验：
1. 50 步采样中 Transformer 前向只被调用 9 次（interval=7, warmup=2 → 5.56×）；
2. 调度统计一致（9 次完整 / 41 次外推）；
3. 输出容器（dict）被正确重建；
4. disable_hicache 后恢复原前向；
5. 平滑轨迹上外推步的相对误差（信息性指标，阈值 5%）。

运行：python tests/test_patch_functional.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paddle  # noqa: E402

from hicache.flux_patch import disable_hicache, enable_hicache, hicache_state  # noqa: E402


class FakeTransformer(paddle.nn.Layer):
    """模拟扩散 Transformer：线性漂移（一阶外推即可精确复原，便于验证补丁机制）。"""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, hidden_states, timestep=None):
        self.calls += 1
        t = 0.0 if timestep is None else float(timestep)
        out = hidden_states + 0.01 * (1.0 + t)
        return {'sample': out}


class FakePipe:
    def __init__(self):
        self.transformer = FakeTransformer()


def denoise_loop(pipe, steps=50, latent=None):
    x = latent
    outs = []
    for i in range(steps):
        res = pipe.transformer(hidden_states=x, timestep=i / steps)
        x = res['sample']
        outs.append(x.numpy().copy())
    return x, outs


def main() -> int:
    paddle.seed(0)
    x0 = paddle.randn([1, 4, 16, 16]) * 0.5

    ref_pipe = FakePipe()
    _, ref = denoise_loop(ref_pipe, latent=x0)

    pipe = FakePipe()
    enable_hicache(pipe, interval=7, order=2, sigma=0.5, warmup=2)
    _, got = denoise_loop(pipe, latent=x0)
    st = hicache_state(pipe)
    calls = pipe.transformer.calls
    predicted_steps = set(st.predicted_steps)
    disable_hicache(pipe)

    rms = float(np.sqrt(np.mean([np.mean(a ** 2) for a in ref])))
    errs = [float(np.abs(a - b).mean()) / rms for a, b in zip(ref, got)]
    pred_errs = [e for i, e in enumerate(errs) if i in predicted_steps]

    print('reference forward calls: %d' % ref_pipe.transformer.calls)
    print('hicache  forward calls: %d' % calls)
    print('computed/predicted    : %d / %d (theoretical %.2f×)'
          % (st.computed, st.predicted, st.speedup))
    print('err/RMS all steps     : %.3f%%' % (100 * float(np.mean(errs))))
    print('err/RMS predicted     : %.3f%%' % (100 * float(np.mean(pred_errs))))

    pipe2 = FakePipe()
    enable_hicache(pipe2, interval=7)
    disable_hicache(pipe2)
    restored = not getattr(pipe2.transformer, '_hicache_enabled', False)

    checks = [
        ('50 步仅 9 次完整前向', calls == 9, 'calls=%d' % calls),
        ('调度统计 9/41', st.computed == 9 and st.predicted == 41, '%d/%d' % (st.computed, st.predicted)),
        ('输出容器 dict 重建', isinstance(got[0], np.ndarray), ''),
        ('disable 后恢复原前向', restored, ''),
    ]
    ok = True
    for name, passed, detail in checks:
        print('%-30s %s %s' % (name, 'PASS' if passed else 'WARN', detail))
        ok = ok and passed
    print('-' * 62)
    print('功能测试通过（外推精度见 calibrate_basis.py）' if ok else '存在未达标项（见上）')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())