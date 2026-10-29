"""
LTX-Video 2.3 FP8 Dev 高质量生成脚本
适用于 RTX 3060 12GB，单阶段全分辨率生成 + CFG引导

注意: Dev模型在12GB显存下可能需要降低分辨率或帧数。
如果OOM，建议先用 --height 960 --width 512 测试。

用法:
    python generate_fp8_dev.py --prompt "你的提示词"
    python generate_fp8_dev.py --prompt "提示词" --height 1280 --width 704 --num-frames 121
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import torch

from ltx_core.components.guiders import MultiModalGuiderParams
from ltx_pipelines.ti2vid_one_stage import TI2VidOneStagePipeline
from ltx_pipelines.utils.constants import LTX_2_3_PARAMS, DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.media_io import encode_video

logging.getLogger().setLevel(logging.INFO)

MODELS_DIR = Path(__file__).parent / "models"
OUTPUT_DIR = Path(__file__).parent / "output"

DEFAULT_DEV_FP8 = MODELS_DIR / "ltx-2.3-fp8" / "ltx-2.3-22b-dev-fp8.safetensors"
DEFAULT_GEMMA_ROOT = MODELS_DIR / "gemma-3-12b-it"


def parse_args():
    parser = argparse.ArgumentParser(description="LTX-Video 2.3 FP8 Dev 视频生成")
    parser.add_argument("--prompt", type=str, required=True, help="文本提示词")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="负面提示词")
    parser.add_argument("--output-path", type=str, default=None, help="输出视频路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子 (默认: 42)")
    parser.add_argument("--height", type=int, default=1280, help="视频高度 (默认: 1280, 竖屏)")
    parser.add_argument("--width", type=int, default=704, help="视频宽度 (默认: 704, 需被32整除)")
    parser.add_argument("--num-frames", type=int, default=121, help="帧数, 公式: 8K+1 (默认: 121 = 5秒@24fps)")
    parser.add_argument("--frame-rate", type=float, default=24.0, help="帧率 (默认: 24)")
    parser.add_argument("--num-inference-steps", type=int, default=30, help="推理步数 (默认: 30)")
    parser.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_DEV_FP8), help="Dev FP8 模型路径")
    parser.add_argument("--gemma-root", type=str, default=str(DEFAULT_GEMMA_ROOT), help="Gemma 文本编码器路径")
    parser.add_argument("--video-cfg-scale", type=float, default=3.0, help="视频CFG引导强度 (默认: 3.0)")
    parser.add_argument("--audio-cfg-scale", type=float, default=7.0, help="音频CFG引导强度 (默认: 7.0)")
    parser.add_argument("--enhance-prompt", action="store_true", help="启用提示词增强")
    parser.add_argument("--quantization", type=str, default=None, choices=["fp8-cast", "fp8-scaled-mm"], help="量化方式")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        args.output_path = str(OUTPUT_DIR / f"dev_{timestamp}.mp4")

    for path_name, path_val in [
        ("checkpoint_path", args.checkpoint_path),
        ("gemma_root", args.gemma_root),
    ]:
        if not Path(path_val).exists():
            print(f"[ERROR] {path_name} 不存在: {path_val}")
            sys.exit(1)

    quantization = None
    if args.quantization == "fp8-cast":
        from ltx_core.quantization import QuantizationPolicy
        quantization = QuantizationPolicy.fp8_cast()
    elif args.quantization == "fp8-scaled-mm":
        from ltx_core.quantization import QuantizationPolicy
        quantization = QuantizationPolicy.fp8_scaled_mm(None)

    print(f"=" * 60)
    print(f"LTX-Video 2.3 FP8 Dev 生成")
    print(f"=" * 60)
    print(f"提示词: {args.prompt}")
    print(f"分辨率: {args.width}x{args.height}")
    print(f"帧数: {args.num_frames} ({args.num_frames / args.frame_rate:.1f}秒 @ {args.frame_rate}fps)")
    print(f"推理步数: {args.num_inference_steps}")
    print(f"视频CFG: {args.video_cfg_scale} | 音频CFG: {args.audio_cfg_scale}")
    print(f"种子: {args.seed}")
    print(f"输出: {args.output_path}")
    print(f"=" * 60)

    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU VRAM: {vram_gb:.1f} GB")

    t0 = time.time()

    print("\n[1/3] 加载模型...")
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=(),
        quantization=quantization,
    )

    t1 = time.time()
    print(f"  模型加载完成 ({t1 - t0:.1f}秒)")

    print("\n[2/3] 生成视频中...")
    video_guider_params = MultiModalGuiderParams(
        cfg_scale=args.video_cfg_scale,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    )
    audio_guider_params = MultiModalGuiderParams(
        cfg_scale=args.audio_cfg_scale,
        stg_scale=1.0,
        rescale_scale=0.7,
        modality_scale=3.0,
        skip_step=0,
        stg_blocks=[28],
    )

    video, audio = pipeline(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        num_inference_steps=args.num_inference_steps,
        video_guider_params=video_guider_params,
        audio_guider_params=audio_guider_params,
        images=[],
        enhance_prompt=args.enhance_prompt,
    )

    t2 = time.time()
    print(f"  生成完成 ({t2 - t1:.1f}秒)")

    print("\n[3/3] 保存视频...")
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    encode_video(
        video=video,
        fps=int(args.frame_rate),
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=1,
    )

    t3 = time.time()
    print(f"  保存完成 ({t3 - t2:.1f}秒)")
    print(f"\n总耗时: {t3 - t0:.1f}秒")
    print(f"输出文件: {args.output_path}")


if __name__ == "__main__":
    main()
