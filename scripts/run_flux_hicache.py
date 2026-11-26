# -*- coding: utf-8 -*-
"""示例：FLUX.1-dev 文生图 HiCache 加速（PaddlePaddle / ppdiffusers）。

用法：
    python scripts/run_flux_hicache.py \
        --prompt "a photo of an astronaut riding a horse on the moon" \
        --steps 50 --interval 7 --order 2 --sigma 0.5 \
        --out flux_hicache.png

脚本会打印：
    * 完整计算步数 / 外推步数 / 实测墙钟加速比；
    * 提示词、超参，便于写入测试报告。
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import paddle  # noqa: E402

from hicache.flux_patch import disable_hicache, enable_hicache, hicache_state  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='FLUX.1-dev + HiCache (Paddle)')
    p.add_argument('--model', default='black-forest-labs/FLUX.1-dev')
    p.add_argument('--prompt', default='a photo of an astronaut riding a horse on the moon')
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--height', type=int, default=1024)
    p.add_argument('--width', type=int, default=1024)
    p.add_argument('--interval', type=int, default=7, help='缓存间隔 N')
    p.add_argument('--order', type=int, default=2, help='外推阶数 Order')
    p.add_argument('--sigma', type=float, default=0.5, help='收缩因子 σ')
    p.add_argument('--warmup', type=int, default=2)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='flux_hicache.png')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from ppdiffusers import FluxPipeline

    paddle.seed(args.seed)
    pipe = FluxPipeline.from_pretrained(args.model, paddle_dtype=paddle.bfloat16)

    def run(tag: str) -> float:
        paddle.seed(args.seed)
        t0 = time.perf_counter()
        image = pipe(
            args.prompt,
            num_inference_steps=args.steps,
            height=args.height,
            width=args.width,
            guidance_scale=3.5,
        ).images[0]
        dt = time.perf_counter() - t0
        image.save(args.out.replace('.png', '_%s.png' % tag))
        print('[%s] wall time = %.2fs -> %s' % (tag, dt, args.out.replace('.png', '_%s.png' % tag)))
        return dt

    # 1) 基线：原始 50 步
    t_base = run('baseline')

    # 2) 启用 HiCache
    enable_hicache(pipe, interval=args.interval, order=args.order, sigma=args.sigma, warmup=args.warmup)
    t_fast = run('hicache')
    st = hicache_state(pipe)
    disable_hicache(pipe)

    print('-' * 60)
    print('prompt    :', args.prompt)
    print('config    : steps=%d interval=%d order=%d sigma=%.2f warmup=%d'
          % (args.steps, args.interval, args.order, args.sigma, args.warmup))
    print('computed  : %d steps | predicted: %d steps' % (st.computed, st.predicted))
    print('speedup   : %.2f× (wall clock)' % (t_base / t_fast))
    print('speedup   : %.2f× (theoretical, from step counts)' % st.speedup)


if __name__ == '__main__':
    main()