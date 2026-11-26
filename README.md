# HiCache · PaddlePaddle(ppdiffusers) 适配实现

基于特征缓存技术的扩散模型推理加速系统（CCF-百度松果基金 CCF-BAIDU OF202521）的 PaddlePaddle 版本实现。
对应论文：*HiCache: A Plug-in Scaled-Hermite Upgrade for Taylor-Style Cache-then-Forecast Diffusion Acceleration*（ICLR 2026）

## 一、方法要点

扩散 Transformer 相邻时间步的隐层特征高度相似，因此可以"缓存历史特征 + 外推预测"，用外推替代部分网络前向。

外推公式：在锚点 t0 处用最近若干真实计算点做 ≤Order 次多项式拟合（最小二乘，允许不等距），
得到系数 c_j = f⁽ʲ⁾(t0)/j!，再用缩放埃尔米特基 B_j(x) 替代泰勒项中的 xʲ：

```
H(n+1)(y) = 2y·Hn(y) − 2n·H(n−1)(y)           # 物理学家埃尔米特多项式递推
g_n(x)    = σⁿ·Hn(σx)                          # 收缩（双缩放）形式
B_n(x)    = [g_n(x) − g_n(0)] / (2σ²)ⁿ         # 锚定基（默认）：B_1=x、B_2=x²
F̂(t0 + x) = F(t0) + Σ_{j=1..Order} c_j · B_j(x)
```

三种基形态可切换（`mode='anchored' | 'normalized' | 'raw'`），见 `hicache/basis.py`：
`anchored` 在 Order ≤ 2 时严格等价泰勒（无偏置），Order ≥ 3 起引入埃尔米特修正；
`normalized` 即论文式"H2 负偏移"形态（每阶带常数偏移）；`raw` 为直接替换的朴素做法（欠外推）。

关键性质：**无需训练、不修改模型权重与结构**，只在采样循环层做"完整计算 / 外推"混合调度。

## 二、目录结构

```
paddle-hicache/
├── hicache/
│   ├── basis.py          # 三种基形态（anchored / normalized / raw）
│   ├── predictor.py      # 基函数递推、差商、等步距外推、相对误差
│   ├── local_poly.py     # 局部最小二乘拟合 + 基函数外推（不等距，低阶精确）
│   ├── cache.py          # 计算点队列 + 混合调度 + 运行期统计
│   ├── flux_patch.py     # ppdiffusers FluxPipeline 采样循环补丁
│   └── reference.py      # NumPy 参考实现（对齐校验用，不需要 paddle）
├── scripts/run_flux_hicache.py   # 端到端示例（基线 vs HiCache，输出墙钟加速比）
├── tests/test_alignment.py       # 数学层对齐（Paddle vs NumPy / 精确多项式）
├── tests/test_patch_functional.py# 补丁机制（前向次数、容器重建、禁用恢复）
├── tests/calibrate_basis.py      # 基形态 / interval / Order / σ 标定
├── docs/适配说明.md              # 改动清单、对齐协议、超参建议、已知限制
└── logs-*.txt                    # 在 CPU 环境实测输出的日志
```

## 三、安装与快速开始

```bash
# 1) paddlepaddle（GPU 版按 CUDA 版本选；CPU 版可直接 pip install paddlepaddle）
python -m pip install paddlepaddle          # 或 paddlepaddle-gpu==3.0.0（cu126 源）
# 2) ppdiffusers
python -m pip install -U ppdiffusers
# 3) 加入 PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

```python
import paddle
from ppdiffusers import FluxPipeline
from hicache.flux_patch import enable_hicache, hicache_state

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", paddle_dtype=paddle.bfloat16)
enable_hicache(pipe, interval=7, order=2, sigma=0.5, warmup=2)   # N=7, Order=2, σ=0.5
image = pipe("a photo of an astronaut riding a horse on the moon", num_inference_steps=50).images[0]
print(hicache_state(pipe).speedup)                               # ≈5.56
```

端到端对比脚本：

```bash
python scripts/run_flux_hicache.py --prompt "a cat wearing sunglasses" --steps 50 --interval 7
```

## 四、已验证内容（CPU 环境实测，日志见 `logs-*.txt`）

| 测试 | 内容 | 结果 |
| --- | --- | --- |
| `test_alignment.py` | 缩放递推 vs 解析多项式；Paddle/NumPy 同式编码逐位一致；调度步数 | 全部 PASS |
| `test_cross_framework.py` | Paddle vs **PyTorch 独立实现** vs fp64 参考 vs 解析轨迹 | 相对差 4.3e-7 ~ 1.4e-6（fp32 舍入量级） |
| `test_patch_functional.py` | 50 步采样中 Transformer 前向次数、容器重建、disable 恢复 | PASS，前向 **9 次**（5.56×） |
| `calibrate_basis.py` | 三种基形态、interval ∈{3,5,7,9}、Order ∈{2,3}、σ ∈{0.3..1.0} | 见下表 |

标定要点（过程 B = 缓变指数，最接近真实去噪轨迹）：

| interval | Order | 加速比 | 外推误差（B） | 外推误差（A，快衰减） |
| --- | --- | --- | --- | --- |
| 3 | 2 | 2.78× | 1.6e-5 | 1.2e-3 |
| 5 | 2 | 4.55× | 8.0e-5 | 5.5e-3 |
| 7 | 2 | 5.56× | 2.2e-4 | 1.5e-2 |
| 7 | 3 | 5.56× | 9.8e-5 | 7.7e-3 |

## 五、GPU 验证（已完成）与尚未完成

**已在真实 GPU 服务器（NVIDIA RTX 6000D，计算能力 12.0 / sm_120，驱动 595.84）验证通过**，
见 `logs-gpu-*.txt`：

| 测试 | 结果 |
| --- | --- |
| `test_alignment.py` | PASS（Paddle vs NumPy 逐位一致；调度 9 次 / 5.56×） |
| `test_patch_functional.py` | PASS（50 步→9 次前向；容器重建、disable 恢复） |
| `calibrate_basis.py` | PASS（三基形态 / interval / Order / σ 标定，与 CPU 一致） |
| `gpu_smoke.py` | PASS（gpu:0 上外推相对误差 4.76e-5，结果 Place(gpu:0)） |

> 安装要点：sm_120（Blackwell）需 paddle **cu130** 轮子（CUDA 13.0）；cu126 会报
> "no kernel image is available"。命令：
> `pip install paddlepaddle-gpu==3.4.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu130/ --extra-index-url https://mirrors.aliyun.com/pypi/simple/`

尚未完成：

- **FLUX.1-dev 图像级端到端**：该 GPU 服务器系统内存仅 15 GB，FLUX.1-dev 权重（bf16 约 24 GB）
  无法在内存中加载；`run_flux_hicache.py` 需在内存 ≥ 32 GB 的 GPU 机上执行，以产出
  "Paddle vs PyTorch 同提示集 ImageReward/PSNR/SSIM/LPIPS"对比数据。
- **与官方实现逐位比对**：论文官方代码的基函数归一化细节未公开，本实现提供三种形态并可切换，
  建议在 GPU 环境用 `calibrate_basis.py` 的思路（真实特征轨迹）选定一种后固定。

## 六、许可

仅供学术研究与百度内部评估使用；使用 FLUX.1-dev 等模型须遵守其原始许可。

仓库地址：https://github.com/InnovatorLM/paddle-hicache
