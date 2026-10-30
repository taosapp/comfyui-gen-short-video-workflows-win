# LTX-Video 2.3 本地部署 (ComfyUI + GGUF)

## 硬件要求
- GPU: RTX 3060 12GB
- RAM: 32GB

## 架构说明
使用 ComfyUI + GGUF 量化方案，解决 ltx-pipelines 在 12GB 显存下 OOM 的问题。
- 扩散模型: Q4_K_M GGUF 量化 (~17GB, 运行时按需加载)
- 文本编码器: Gemma 3 12B Q4_K_M GGUF (~7GB) + te-projection 投影层
- VAE: 视频VAE BF16 (~1.4GB) + 音频VAE (~0.3GB)

## 模型文件位置

```
comfyui/models/
├── diffusion_models/
│   ├── LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf   (17GB, 快速8步)
│   └── LTX-2.3-dev-Q4_K_M.gguf                  (17GB, 高质量20步)
├── text_encoders/
│   └── gemma-3-12b-it-Q4_K_M.gguf               (6.8GB)
├── vae/
│   └── ltx-2.3-video-vae.safetensors            (1.4GB)
├── latent_upscale_models/
│   └── ltx-2.3-spatial-upscaler-x2-1.1.safetensors  (0.9GB, 两段式精修用)
└── checkpoints/
    ├── ltx-2.3-audio-vae.safetensors            (0.3GB, 音频VAE)
    └── ltx-2.3-te-projection.safetensors        (2.2GB, Gemma投影层)
```

## 启动方法

双击 `run_comfyui.bat`，浏览器访问 http://127.0.0.1:8189

## 一键测试

```bash
python test_comfyui_api.py distilled   # 文生视频：蒸馏模型两段式(含音轨)，自动启停 ComfyUI
python test_comfyui_api.py dev         # 文生视频：dev 模型 20 步高质量
python test_i2v.py 2_1.jpg "提示词"     # 图生视频：首帧条件两段式(含音轨)，图片需在 comfyui/input 下
```

测试脚本会自动启动/关闭 ComfyUI（端口 8189），并提交 LTX-2.3 音视频联合工作流
（LTXAVTextEncoderLoaderGGUF 加载 Gemma+投影，EmptyLTXVLatentVideo+LTXVEmptyLatentAudio，
蒸馏模型默认两段式：352x640 8步 → 空间上采样x2 → 704x1280 精修3步）。

## 关键参数

| 参数 | Distilled (快速) | Dev (高质量) |
|------|-----------------|-------------|
| 模型 | LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf | LTX-2.3-dev-Q4_K_M.gguf |
| 步数 | 8+3 (两段式) | 20 |
| CFG | 1.0 | 1.0~3.0 |
| 采样器 | euler | euler |
| 调度 | ManualSigmas 官方蒸馏调度 | LTXVScheduler |

## 分辨率与帧数
- 竖屏 720p: 704x1280 (须被32整除)
- 5秒视频: length=121 (公式: 8n+1), 帧率 25fps
- LTX-2.3 为音视频联合模型 (LTXAV)，输出自带音轨

## 已安装的自定义节点
- `ComfyUI-GGUF` - GGUF 模型加载
- `ComfyUI-VideoHelperSuite` - 视频输出
- `ltx2_gguf_loader.py` - LTXAVTextEncoderLoaderGGUF（Gemma GGUF + 投影层双文件加载）

## Conda 环境
```bash
conda activate ltx-video
# Python 3.12, PyTorch cu126, transformers 4.57.3
```

## 注意事项
1. ComfyUI 启动参数 `--lowvram` 会自动管理显存，将不用的模型卸载到内存
2. `--force-fp16` 强制使用 FP16 计算，适配 RTX 3060
3. CFG 值不要超过 5.0，否则会出现色彩失真和闪烁
4. 首次加载模型需要等待较长时间（GGUF 反量化，17GB 模型可能需数分钟）
5. 生成过程中不要关闭命令行窗口
