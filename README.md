# LTX-Video 2.3 本地部署 (ComfyUI + GGUF)

## 硬件要求
- GPU: RTX 3060 12GB
- RAM: 32GB

## 架构说明
使用 ComfyUI + GGUF 量化方案，解决 ltx-pipelines 在 12GB 显存下 OOM 的问题。
- 扩散模型: Q4_K_M GGUF 量化 (~17GB, 运行时按需加载)
- 文本编码器: Gemma 3 12B Q4_K_M GGUF (~7GB)
- VAE: BF16 safetensors (~1.6GB)

## 模型文件位置

```
comfyui/models/
├── diffusion_models/
│   ├── LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf   (17GB, 快速8步)
│   └── LTX-2.3-dev-Q4_K_M.gguf                  (17GB, 高质量20步)
├── text_encoders/
│   └── gemma-3-12b-it-Q4_K_M.gguf               (6.8GB)
└── vae/
    └── ltx-video-vae.safetensors                (1.6GB)
```

## 启动方法

双击 `run_comfyui.bat`，浏览器访问 http://127.0.0.1:8188

## ComfyUI 工作流搭建

### 节点连接顺序

```
UnetLoaderGGUFAdvanced ──────┐
  (选 distilled 或 dev)       │
                              ├──> ModelSamplingLTXV ──> CFGGuider ──> SamplerCustomAdvanced
CLIPLoaderGGUF ──────────────┤                                    ↑
  (gemma-3-12b-it-Q4_K_M)    │                                    │
  ├──> CLIPTextEncode (正向) ─┤                                    │
  └──> CLIPTextEncode (负向) ─┤                                    │
                              │                                    │
EmptyLTXVLatentVideo ─────────┤                                    │
  (704x1280, 121帧)           │                                    │
                              ├──> LTXVConditioning ────────────────┘
VAELoader ────────────────────┐
  (ltx-video-vae)             │
                              ↓
                    SamplerCustomAdvanced ──> VAEDecode ──> VHS_VideoCombine
                              ↑                                (保存MP4)
                    LTXVScheduler
```

### 关键参数

| 参数 | Distilled (快速) | Dev (高质量) |
|------|-----------------|-------------|
| 模型 | LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf | LTX-2.3-dev-Q4_K_M.gguf |
| 步数 | 8 | 20 |
| CFG | 1.0 | 3.0 |
| 采样器 | euler | euler |
| 调度器 | LTXVScheduler | LTXVScheduler |
| max_shift | 2.05 | 2.05 |
| base_shift | 0.95 | 0.95 |

### 分辨率与帧数
- 竖屏 720p: width=704, height=1280 (须被32整除)
- 横屏 720p: width=1280, height=704
- 5秒视频: length=121 (公式: 8n+1, 24fps)
- 3秒视频: length=73

### CLIPLoaderGGUF 设置
- type: `ltxv`

### VHS_VideoCombine 设置
- frame_rate: 24
- format: video/h264-mp4

## 已安装的自定义节点
- `ComfyUI-GGUF` - GGUF 模型加载
- `ComfyUI-VideoHelperSuite` - 视频输出

## Conda 环境
```bash
conda activate ltx-video
# Python 3.12, PyTorch 2.14.1+cu126, transformers 4.57.3
```

## 注意事项
1. ComfyUI 启动参数 `--lowvram` 会自动管理显存，将不用的模型卸载到内存
2. `--force-fp16` 强制使用 FP16 计算，适配 RTX 3060
3. CFG 值不要超过 5.0，否则会出现色彩失真和闪烁
4. 首次加载模型需要等待较长时间（GGUF 反量化）
5. 生成过程中不要关闭命令行窗口
