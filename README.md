# PaddleHiCache

**HiCache 的 PaddlePaddle 实现** —— 面向扩散 Transformer 的免训练特征缓存推理加速（缩放埃尔米特基 + 锚点式外推 + 混合调度）。

本仓库是论文 *HiCache: A Plug-in Scaled-Hermite Upgrade for Taylor-Style Cache-then-Forecast Diffusion Acceleration*（ICLR 2026，受 **CCF-Baidu Open Fund** 资助）的飞桨（PaddlePaddle / ppdiffusers）工程实现，用于把 PyTorch 版的特征缓存加速方法迁移到飞桨生态。

- 上游论文代码（PyTorch）：https://github.com/fenglang918/HiCache
- 论文：ICLR 2026，arXiv:2508.16984
- 许可：Apache-2.0

## 特性

- **零训练 / 零结构改动**：只在采样循环层做"完整计算 ↔ 特征外推"的混合调度，模型权重与结构不变；
- **不依赖自定义算子**：全部为张量加乘 + 一次 3×3 标量最小二乘，天然可移植到 Paddle；
- **即插即用**：`enable_hicache(pipe)` 一行接入 ppdiffusers 的 `FluxPipeline` 等以 `pipe.transformer` 为核心的管线；
- **可配置基函数**：`anchored`（默认，低阶时严格等价泰勒）/ `normalized`（论文式 H2 负偏移）/ `raw`；
- **实测加速**：50 步采样、`interval=7`、`warmup=2` 时完整前向 9 次、外推 41 次，理论加速 **5.56×**。

## 方法要点

在锚点处用最近若干**真实计算**点做 ≤Order 次多项式最小二乘拟合（允许不等距），得到系数
`c_j = f⁽ʲ⁾(t0)/j!`，再用缩放埃尔米特基 `B_j(x)` 替代泰勒项中的 `xʲ`：

```
H(n+1)(y) = 2y·Hn(y) − 2n·H(n−1)(y)        # 物理学家埃尔米特多项式递推
g_n(x)    = σⁿ·Hn(σx)                       # 收缩（双缩放）
B_n(x)    = [g_n(x) − g_n(0)] / (2σ²)ⁿ      # 锚定基（默认）：B_1=x、B_2=x²
F̂(t0 + x) = F(t0) + Σ_{j=1..Order} c_j · B_j(x)
```

调度上采用**锚点式多步外推**：只有真实计算的特征进入拟合点集（回填预测值会造成虚假跳变，实测误差放大到 1e8 量级），
每 `interval` 步做一次完整计算以重置误差累积。

## 安装

```bash
# 1) PaddlePaddle（GPU 版按 CUDA 版本选择；Blackwell/sm_120 需 cu130）
python -m pip install paddlepaddle-gpu==3.4.0 \
  -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ \
  --extra-index-url https://mirrors.aliyun.com/pypi/simple/
# 2) ppdiffusers
python -m pip install -U ppdiffusers
# 3) 本仓库
git clone <this-repo> && cd paddle-hicache
export PYTHONPATH=$PWD:$PYTHONPATH
```

## 快速开始

```python
import paddle
from ppdiffusers import FluxPipeline
from hicache.flux_patch import enable_hicache, hicache_state

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.float16)
enable_hicache(pipe, interval=7, order=2, sigma=0.5, warmup=2)   # N=7, Order=2, σ=0.5

image = pipe("a photo of an astronaut riding a horse on the moon", num_inference_steps=50).images[0]
image.save("out.png")
print("speedup:", hicache_state(pipe).speedup)   # ≈5.56
```

端到端对比脚本（基线 vs HiCache，输出墙钟加速比）：

```bash
python scripts/run_flux_hicache.py --prompt "a cat wearing sunglasses" --steps 50 --interval 7
```

## 已验证内容

四套测试，日志见 [`logs/`](logs/)：

| 测试 | 内容 | 结果 |
| --- | --- | --- |
| `tests/test_alignment.py` | 基函数递推 vs 解析多项式；Paddle/NumPy 同式编码一致性；调度步数 | 全部 PASS |
| `tests/test_patch_functional.py` | 50 步采样中前向次数、输出容器重建、disable 恢复 | PASS（前向 9 次） |
| `tests/test_cross_framework.py` | **Paddle vs PyTorch 独立实现** vs fp64 参考 vs 解析轨迹 | 相对差 **4.3e-7 ~ 1.4e-6**（fp32 舍入量级） |
| `tests/calibrate_basis.py` | 三种基形态 / interval / Order / σ 标定 | 见 `logs/calibration.txt` |

GPU 环境实测（`logs/gpu-*.txt`）：NVIDIA RTX 6000D（计算能力 12.0 / sm_120，驱动 595.84），
paddlepaddle-gpu 3.4.0（CUDA 13.0）；`gpu:0` 上外推相对误差 4.76e-5，调度 9 次完整前向。

## 已知限制

- **FLUX.1-dev 图像级端到端尚未跑通**：当前 ppdiffusers（0.29.0）未提供 `FluxPipeline`（其发布时间早于 FLUX），
  因此飞桨侧暂无法直接跑 FLUX.1-dev；后端到端验证可改用 ppdiffusers 已支持的 `PixArtAlphaPipeline` /
  `StableDiffusion3Pipeline`（同为 DiT/MMDiT transformer，本补丁零改动适用）。
- 与论文 PyTorch 官方实现的**逐位比对**未做：官方基函数归一化细节未公开，本实现提供三种形态可切换，
  建议用 `tests/calibrate_basis.py` 在真实特征轨迹上标定后固定。
- 外推步跳过整个 Transformer 前向，batch 内共用同一调度；固定分辨率。

## 引用

```bibtex
@inproceedings{feng2026hicache,
  title     = {HiCache: A Plug-in Scaled-Hermite Upgrade for Taylor-Style Cache-then-Forecast Diffusion Acceleration},
  author    = {Feng, Liang and Zheng, Shikang and Liu, Jiacheng and Lin, Yuqi and Zhou, Qinming and
               Cai, Peiliang and Wang, Xinyu and Chen, Junjie and Zou, Chang and Ma, Yue and Zhang, Linfeng},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026},
  note      = {Sponsored by CCF-Baidu Open Fund}
}
```

## 许可

Apache-2.0，见 [LICENSE](LICENSE)。使用 FLUX.1-dev 等模型须遵守其原始许可。
