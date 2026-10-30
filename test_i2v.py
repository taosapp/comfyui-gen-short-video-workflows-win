"""
LTX-Video 2.3 图生视频 (I2V) 测试脚本 — 基于官方两段式蒸馏配方 + LTXAV 音视频联合工作流

流程:
  1. LoadImage 读取 input 目录中的图片
  2. LTXVImgToVideo 在半分辨率 (W/2, H/2) 编码首帧图像条件 (strength=1.0, noise_mask 锁定首帧)
  3. 第一段 8 步粗采 (官方蒸馏 ManualSigmas, CFG=1.0)
  4. LTXVSeparateAVLatent 分离音视频 -> LTXVLatentUpsampler 空间上采样 x2
     (注意: 上采样节点会丢弃 noise_mask)
  5. LTXVImgToVideoInplace 在全分辨率把原图重新注入首帧并重建 noise_mask
  6. 第二段 3 步精修 (官方 REFINE_SIGMAS)
  7. 分离音视频, 分别解码, 合成 MP4 (H.264 + AAC)

服务管理逻辑 (启动/提交/轮询/关闭) 直接复用 test_comfyui_api.py。
用法: python test_i2v.py [图片文件名(需在 comfyui/input 下)] [提示词]
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # PYTHONSAFEPATH 下脚本目录不在 sys.path

from test_comfyui_api import (
    start_comfyui, queue_prompt, poll_progress,
    COMFYUI_URL, COMFYUI_DIR,
    DISTILLED_SIGMAS, REFINE_SIGMAS,
)

DEFAULT_IMAGE = "2_1.jpg"
# 运动主体前置（模型对靠前 token 服从度更高），数量/景深/方向写实
DEFAULT_PROMPT = ("固定镜头。橙色桂花花瓣从画面左上方向右下方缓缓飘落："
                  "前景有两三片虚焦的大花瓣飘过，远处空中飘着七八片较小的清晰花瓣；"
                  "前景桂花树叶在微风中明显摇曳、摆动幅度大；"
                  "背景轻微虚化，阳光明媚，电影感，高质量")

NEGATIVE = "ugly, blurry, low quality, distorted, deformed, watermark, text, bad anatomy"

NOISE_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 20261030

# 两段式: 半分辨率粗采 -> 上采样 x2 -> 全分辨率精修
HALF_W, HALF_H = 352, 640       # 704x1280 的一半 (需被 32 整除)
FULL_W, FULL_H = 704, 1280
LENGTH = 121                     # 8n+1, 25fps -> ~4.8 秒
FRAME_RATE = 25.0

UPSCALE_MODEL = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def build_i2v_workflow(prompt, image_name):
    wf = {
        # ---- 模型加载 ----
        "1": {"class_type": "UnetLoaderGGUFAdvanced", "inputs": {
            "unet_name": "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf",
            "dequant_dtype": "default", "patch_dtype": "default", "patch_on_device": False}},
        "2": {"class_type": "LTXAVTextEncoderLoaderGGUF", "inputs": {
            "text_encoder": "gemma-3-12b-it-Q4_K_M.gguf",
            "projection_ckpt": "ltx-2.3-te-projection.safetensors", "device": "default"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ltx-2.3-video-vae.safetensors"}},
        "4": {"class_type": "LTXVAudioVAELoader", "inputs": {"ckpt_name": "ltx-2.3-audio-vae.safetensors"}},

        # ---- 文本条件 ----
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}},

        # ---- 首帧图像条件 ----
        "7": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "8": {"class_type": "LTXVImgToVideo", "inputs": {
            "positive": ["5", 0], "negative": ["6", 0], "vae": ["3", 0], "image": ["7", 0],
            "width": HALF_W, "height": HALF_H, "length": LENGTH, "batch_size": 1, "strength": 1.0}},

        # ---- 音频空 latent + 合并 AV ----
        "9": {"class_type": "LTXVEmptyLatentAudio", "inputs": {
            "frames_number": LENGTH, "frame_rate": FRAME_RATE, "batch_size": 1, "audio_vae": ["4", 0]}},
        "10": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["8", 2], "audio_latent": ["9", 0]}},

        # ---- 帧率条件（运动强度/时序一致性依赖此项，不可省）----
        "28": {"class_type": "LTXVConditioning", "inputs": {
            "positive": ["5", 0], "negative": ["6", 0], "frame_rate": FRAME_RATE}},

        # ---- 采样公共件 ----
        "11": {"class_type": "LatentUpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}},
        "12": {"class_type": "CFGGuider", "inputs": {
            "cfg": 1.0, "model": ["1", 0], "positive": ["28", 0], "negative": ["28", 1]}},
        "13": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "14": {"class_type": "RandomNoise", "inputs": {"noise_seed": NOISE_SEED}},
        "15": {"class_type": "ManualSigmas", "inputs": {"sigmas": DISTILLED_SIGMAS}},

        # ---- 第一段: 半分辨率粗采 8 步 ----
        "16": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["14", 0], "guider": ["12", 0], "sampler": ["13", 0],
            "sigmas": ["15", 0], "latent_image": ["10", 0]}},

        # ---- 分离 -> 上采样 x2 -> 重新注入图像条件 ----
        "17": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["16", 0]}},
        "18": {"class_type": "LTXVLatentUpsampler", "inputs": {
            "samples": ["17", 0], "upscale_model": ["11", 0], "vae": ["3", 0]}},
        "19": {"class_type": "LTXVImgToVideoInplace", "inputs": {
            "vae": ["3", 0], "image": ["7", 0], "latent": ["18", 0], "strength": 1.0, "bypass": False}},
        "20": {"class_type": "LTXVConcatAVLatent", "inputs": {
            "video_latent": ["19", 0], "audio_latent": ["17", 1]}},

        # ---- 第二段: 全分辨率精修 3 步 ----
        "21": {"class_type": "ManualSigmas", "inputs": {"sigmas": REFINE_SIGMAS}},
        "22": {"class_type": "RandomNoise", "inputs": {"noise_seed": NOISE_SEED}},
        "23": {"class_type": "SamplerCustomAdvanced", "inputs": {
            "noise": ["22", 0], "guider": ["12", 0], "sampler": ["13", 0],
            "sigmas": ["21", 0], "latent_image": ["20", 0]}},
        "24": {"class_type": "LTXVSeparateAVLatent", "inputs": {"av_latent": ["23", 0]}},
    }

    # ---- 解码输出 (与 t2v 相同) ----
    wf.update({
        "25": {"class_type": "VAEDecodeTiled", "inputs": {
            "samples": ["24", 0], "vae": ["3", 0],
            "tile_size": 768, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8}},
        "26": {"class_type": "LTXVAudioVAEDecode", "inputs": {"samples": ["24", 1], "audio_vae": ["4", 0]}},
        "27": {"class_type": "VHS_VideoCombine", "inputs": {
            "frame_rate": int(FRAME_RATE), "loop_count": 0,
            "filename_prefix": "LTX-Video-I2V",
            "format": "video/h264-mp4", "pix_fmt": "yuv420p", "pingpong": False,
            "save_output": True, "images": ["25", 0], "audio": ["26", 0]}},
    })
    return wf


def main():
    image_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_IMAGE
    prompt = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROMPT

    print("=" * 50)
    print("  LTX-Video 2.3 图生视频测试 (I2V, 两段式)")
    print("=" * 50)
    print(f"  图片: {COMFYUI_DIR}\\input\\{image_name}")
    print(f"  提示词: {prompt}")
    print(f"  分辨率: {FULL_W}x{FULL_H} (竖屏), 帧数: {LENGTH} @25fps (~4.8秒, 含音轨)")
    print(f"  调度: 两段式 {HALF_W}x{HALF_H} 8步 -> 上采样x2 -> {FULL_W}x{FULL_H} 精修3步")
    print()

    t0 = time.time()

    proc = start_comfyui()
    if not proc:
        print("ComfyUI 启动失败!")
        sys.exit(1)

    try:
        print("\n[2/4] 提交 I2V 工作流...")
        workflow = build_i2v_workflow(prompt, image_name)
        prompt_id = queue_prompt(workflow)
        print(f"  任务ID: {prompt_id}")

        outputs = poll_progress(prompt_id)

        t1 = time.time()
        print(f"\n[4/4] 完成! 耗时: {t1-t0:.1f}秒")

        if outputs:
            for node_id, output in outputs.items():
                if "gifs" in output:
                    for gif in output["gifs"]:
                        print(f"  输出: {gif.get('filename', 'unknown')}")
                        print(f"  路径: {COMFYUI_DIR}/output/{gif.get('filename', '')}")
        else:
            print("  未获取到输出，请检查 ComfyUI 控制台日志")

    finally:
        print("\n关闭 ComfyUI...", flush=True)
        import subprocess
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("已关闭", flush=True)


if __name__ == "__main__":
    main()
