"""
LTX-Video 2.3 ComfyUI API 生成脚本（按官方 LTX-2.3 工作流结构重写 + 两段式采样）

修复/升级记录:
1. VHS_VideoCombine 在当前 VideoHelperSuite 版本要求必填 pingpong 输入（缺省提交会被 400 拒绝）。
2. 旧工作流用 CLIPLoaderGGUF 单文件加载 Gemma，ComfyUI 会构建"裸 Gemma"文本编码器
   （输出 4D 全层堆叠，缺少 text_embedding_projection 投影），导致采样前在
   LTXAV.preprocess_text_embeds 处 4D/3D 崩溃。现改用 LTXAVTextEncoderLoaderGGUF
   （custom_nodes/ltx2_gguf_loader.py）按官方双文件配方加载:
   gemma GGUF + ltx-2.3-te-projection.safetensors（从 fp8 checkpoint 抽取）。
3. LTX-2.3 是音视频联合模型 (LTXAV)：latent 必须包含视频+音频两路
   （EmptyLTXVLatentVideo + LTXVEmptyLatentAudio -> LTXVConcatAVLatent），
   采样后 LTXVSeparateAVLatent 分离，分别解码。
4. 蒸馏模型使用官方模板的 8 步 ManualSigmas 调度（CFG=1.0）。
5. 两段式采样（搬自官方 video_ltx2_3_t2v 模板）：
   第一段在半分辨率 (W/2, H/2) 采 8 步 -> LTXVLatentUpsampler 空间上采样 x2
   -> 第二段用官方精修调度 (0.85, 0.7250, 0.4219, 0.0) 采 3 步。
   音频 latent 由第一段生成并在第二段原样携带（上采样只作用于视频流）。
   需要 models/latent_upscale_models/ltx-2.3-spatial-upscaler-x2-1.1.safetensors。
"""
import json
import time
import urllib.request
import urllib.error
import sys
import os
import subprocess

COMFYUI_DIR = r"D:\ai\gen-short-video\comfyui"
# 用 8189 避开 ComfyUI Desktop（它占用 8188 且未装 GGUF 自定义节点）
COMFYUI_URL = "http://127.0.0.1:8189"
CONDA_BAT = r"C:\Users\PPXW\miniconda3\condabin\activate.bat"

# 提示词：明确运动方向，避免模型随机选方向
PROMPT = ("A giant panda running towards the camera on the green grass, "
          "sunny day, clear blue sky, cinematic, natural lighting, high quality, 4K")
NOISE_SEED = 7  # 原 42 生成的是侧向/倒行画面，换 seed 重新抽

# 蒸馏模型是否启用官方两段式采样（半分辨率粗采 + 上采样精修）
TWO_STAGE = True

# 官方 LTX-2.3 蒸馏模型 8 步调度（来自 comfyui-workflow-templates video_ltx2_3_t2v）
DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
# 官方第二段精修调度（3 步，作用于上采样后的 latent）
REFINE_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
UPSCALE_MODEL = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"


def start_comfyui():
    print("[1/4] 启动 ComfyUI...", flush=True)
    env = os.environ.copy()
    server_log = os.path.join(COMFYUI_DIR, "comfyui_server.log")
    # 子进程输出必须落盘：PIPE 无人读取时一旦填满缓冲区会卡死 ComfyUI 启动（本项目的原始坑）
    log_fh = open(server_log, "w", encoding="utf-8", errors="replace")
    cmd = (f'call "{CONDA_BAT}" ltx-video && cd /d "{COMFYUI_DIR}" && '
           f'python -u main.py --port 8189 --lowvram --force-fp16 --disable-auto-launch')
    proc = subprocess.Popen(
        cmd, shell=True, cwd=COMFYUI_DIR,
        stdout=log_fh, stderr=subprocess.STDOUT,
        env=env
    )
    try:
        for i in range(150):
            time.sleep(2)
            try:
                req = urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
                data = json.loads(req.read())
                print(f"  ComfyUI 已启动 (GPU: {data.get('system_stats', {}).get('devices', [{}])[0].get('name', 'unknown')})", flush=True)
                return proc
            except Exception:
                if i % 5 == 4:
                    print(f"  等待中... ({(i+1)*2}秒)", flush=True)
        print("  启动超时! 最近服务端日志:", flush=True)
        if os.path.exists(server_log):
            with open(server_log, encoding="utf-8", errors="replace") as f:
                for line in f.readlines()[-25:]:
                    print("   |", line.rstrip(), flush=True)
        return None
    finally:
        log_fh.close()


def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    try:
        resp = urllib.request.urlopen(req)
        result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"  提交被拒绝 (HTTP {e.code}):")
        try:
            err = json.loads(body)
            print(f"  错误: {err.get('error', {}).get('details', body)}")
            for nid, ne in (err.get('node_errors') or {}).items():
                for er in ne.get('errors', []):
                    print(f"  节点 {nid} ({ne.get('class_type')}): {er.get('type')} - {er.get('details')}")
        except Exception:
            print(body)
        raise SystemExit(1)
    return result.get("prompt_id")


def poll_progress(prompt_id):
    print("[3/4] 生成中...", flush=True)
    while True:
        try:
            req = urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5)
            history = json.loads(req.read())
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                status = history[prompt_id].get("status", {})
                if status.get("status_str") == "error":
                    print("\n  生成出错! 最近消息:")
                    for m in status.get("messages", [])[-6:]:
                        print("   ", json.dumps(m, ensure_ascii=False)[:400])
                    return None
                if status.get("completed", False) or outputs:
                    return outputs
        except Exception:
            pass
        time.sleep(3)
        print("  ...", end="", flush=True)


def _common_loaders(prompt, diffusion_gguf):
    """两段式/单段式共用的模型加载与文本编码节点。"""
    return {
        "1": {
            "class_type": "UnetLoaderGGUFAdvanced",
            "inputs": {
                "unet_name": diffusion_gguf,
                "dequant_dtype": "default",
                "patch_dtype": "default",
                "patch_on_device": False
            }
        },
        "2": {
            "class_type": "LTXAVTextEncoderLoaderGGUF",
            "inputs": {
                "text_encoder": "gemma-3-12b-it-Q4_K_M.gguf",
                "projection_ckpt": "ltx-2.3-te-projection.safetensors",
                "device": "default"
            }
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "ltx-2.3-video-vae.safetensors"}
        },
        "4": {
            "class_type": "LTXVAudioVAELoader",
            "inputs": {"ckpt_name": "ltx-2.3-audio-vae.safetensors"}
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]}
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {
                "text": "ugly, blurry, low quality, distorted, deformed, watermark, text, bad anatomy",
                "clip": ["2", 0]
            }
        },
    }


def _output_nodes(video_src, audio_src, frame_rate, prefix):
    """解码 + 输出节点（两段式/单段式共用）。video_src/audio_src: [node, slot]"""
    return {
        "24": {
            "class_type": "VAEDecodeTiled",
            "inputs": {
                "samples": video_src,
                "vae": ["3", 0],
                "tile_size": 768,
                "overlap": 64,
                "temporal_size": 64,
                "temporal_overlap": 8
            }
        },
        "25": {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"samples": audio_src, "audio_vae": ["4", 0]}
        },
        "26": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": int(frame_rate),
                "loop_count": 0,
                "filename_prefix": prefix,
                "format": "video/h264-mp4",
                "pix_fmt": "yuv420p",
                "pingpong": False,
                "save_output": True,
                "images": ["24", 0],
                "audio": ["25", 0]
            }
        },
    }


def _build_two_stage(prompt, diffusion_gguf):
    """官方两段式：半分辨率 8 步粗采 -> 空间上采样 x2 -> 全分辨率 3 步精修。"""
    width, height = 352, 640      # 全分辨率 704x1280 的一半（需被 32 整除）
    length = 121
    frame_rate = 25.0
    wf = _common_loaders(prompt, diffusion_gguf)
    wf.update({
        "7": {
            "class_type": "LatentUpscaleModelLoader",
            "inputs": {"model_name": UPSCALE_MODEL}
        },
        # ---- 第一段：半分辨率粗采 ----
        "8": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": length, "batch_size": 1}
        },
        "9": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": length,
                "frame_rate": frame_rate,
                "batch_size": 1,
                "audio_vae": ["4", 0]
            }
        },
        "10": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["8", 0], "audio_latent": ["9", 0]}
        },
        "11": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": frame_rate}
        },
        "12": {
            "class_type": "CFGGuider",
            "inputs": {"cfg": 1.0, "model": ["1", 0], "positive": ["11", 0], "negative": ["11", 1]}
        },
        "13": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"}
        },
        "14": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": NOISE_SEED}
        },
        "15": {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": DISTILLED_SIGMAS}
        },
        "16": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["14", 0], "guider": ["12", 0], "sampler": ["13", 0],
                "sigmas": ["15", 0], "latent_image": ["10", 0]
            }
        },
        # ---- 上采样 + 第二段精修 ----
        "17": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av_latent": ["16", 0]}
        },
        "18": {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {
                "samples": ["17", 0],
                "upscale_model": ["7", 0],
                "vae": ["3", 0]
            }
        },
        "19": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["18", 0], "audio_latent": ["17", 1]}
        },
        "20": {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": REFINE_SIGMAS}
        },
        "21": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": NOISE_SEED}
        },
        "22": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["21", 0], "guider": ["12", 0], "sampler": ["13", 0],
                "sigmas": ["20", 0], "latent_image": ["19", 0]
            }
        },
        "23": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av_latent": ["22", 0]}
        },
    })
    wf.update(_output_nodes(["23", 0], ["23", 1], frame_rate, "LTX-Video-2S"))
    return wf


def _build_single_stage(prompt, diffusion_gguf, is_distilled):
    """原单段式（dev 模型或显式关闭两段式时使用）。"""
    width, height = 704, 1280
    length = 121
    frame_rate = 25.0
    wf = _common_loaders(prompt, diffusion_gguf)
    wf.update({
        "7": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {"width": width, "height": height, "length": length, "batch_size": 1}
        },
        "8": {
            "class_type": "LTXVEmptyLatentAudio",
            "inputs": {
                "frames_number": length,
                "frame_rate": frame_rate,
                "batch_size": 1,
                "audio_vae": ["4", 0]
            }
        },
        "9": {
            "class_type": "LTXVConcatAVLatent",
            "inputs": {"video_latent": ["7", 0], "audio_latent": ["8", 0]}
        },
        "10": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["5", 0], "negative": ["6", 0], "frame_rate": frame_rate}
        },
        "11": {
            "class_type": "CFGGuider",
            "inputs": {"cfg": 1.0, "model": ["1", 0], "positive": ["10", 0], "negative": ["10", 1]}
        },
        "12": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"}
        },
        "13": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": NOISE_SEED}
        },
        "14": {
            "class_type": "ManualSigmas" if is_distilled else "LTXVScheduler",
            "inputs": (
                {"sigmas": DISTILLED_SIGMAS} if is_distilled else
                {"steps": 20, "max_shift": 2.05, "base_shift": 0.95,
                 "stretch": True, "terminal": 0.1, "latent": ["7", 0]}
            )
        },
        "15": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["13", 0], "guider": ["11", 0], "sampler": ["12", 0],
                "sigmas": ["14", 0], "latent_image": ["9", 0]
            }
        },
        "16": {
            "class_type": "LTXVSeparateAVLatent",
            "inputs": {"av_latent": ["15", 0]}
        },
    })
    wf.update(_output_nodes(["16", 0], ["16", 1], frame_rate, "LTX-Video"))
    return wf


def build_workflow(prompt, model_type="distilled"):
    """构建 LTX-2.3 音视频联合工作流。蒸馏模型默认官方两段式采样。"""
    is_distilled = model_type == "distilled"
    diffusion_gguf = (
        "LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf" if is_distilled
        else "LTX-2.3-dev-Q4_K_M.gguf"
    )
    if is_distilled and TWO_STAGE:
        return _build_two_stage(prompt, diffusion_gguf)
    return _build_single_stage(prompt, diffusion_gguf, is_distilled)


def main():
    prompt = PROMPT

    model_type = sys.argv[1] if len(sys.argv) > 1 else "distilled"
    two_stage = TWO_STAGE and model_type == "distilled"

    print("=" * 50)
    print(f"  LTX-Video 2.3 视频生成测试 ({model_type})")
    print("=" * 50)
    print(f"  提示词: {prompt}")
    print(f"  分辨率: 704x1280 (竖屏 720p)")
    print(f"  帧数: 121 @25fps (~4.8秒, 含音轨)")
    if two_stage:
        print(f"  调度: 两段式 352x640 8步 -> 上采样x2 -> 704x1280 精修3步")
    else:
        print(f"  调度: {'官方蒸馏 8 步 ManualSigmas' if model_type == 'distilled' else 'LTXVScheduler 20 步'}")
    print()

    t0 = time.time()

    proc = start_comfyui()
    if not proc:
        print("ComfyUI 启动失败!")
        sys.exit(1)

    try:
        print(f"\n[2/4] 提交工作流 (模型: {model_type}, 两段式: {two_stage})...")
        workflow = build_workflow(prompt, model_type)
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
                if "images" in output:
                    for img in output["images"]:
                        print(f"  图片: {img.get('filename', 'unknown')}")
        else:
            print("  未获取到输出，请检查 ComfyUI 控制台日志")

    finally:
        print("\n关闭 ComfyUI...", flush=True)
        # shell=True 产生 cmd 包装进程，terminate 杀不死 python 子进程，需整树终止
        subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                       capture_output=True)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("已关闭", flush=True)


if __name__ == "__main__":
    main()
