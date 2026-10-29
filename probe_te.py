# -*- coding: utf-8 -*-
"""探针：复刻 CLIPLoaderGGUF(type=ltxv) 的加载与编码路径，打印各阶段张量形状。"""
import os
import sys
import importlib.util

COMFY = r"D:\ai\gen-short-video\comfyui"
GGUF_NODE = os.path.join(COMFY, "custom_nodes", "ComfyUI-GGUF")
TE_PATH = os.path.join(COMFY, "models", "text_encoders", "gemma-3-12b-it-Q4_K_M.gguf")

sys.path.insert(0, COMFY)
os.chdir(COMFY)

import types
pkg = types.ModuleType("gguf_pkg")
pkg.__path__ = [GGUF_NODE]
sys.modules["gguf_pkg"] = pkg
gguf_ops = importlib.import_module("gguf_pkg.ops")
gguf_loader = importlib.import_module("gguf_pkg.loader")

print("[1] 读取 GGUF ...", flush=True)
sd = gguf_loader.gguf_clip_loader(TE_PATH)
keys = list(sd.keys())
print("    key 数:", len(keys), " 示例:", keys[:3], flush=True)
print("    含 text_embedding_projection.*:", any(k.startswith("text_embedding_projection") for k in keys), flush=True)
print("    含 model.diffusion_model.*:", any(k.startswith("model.diffusion_model") for k in keys), flush=True)

import comfy.sd as csd
import comfy.text_encoders.lt as lt_mod
import torch

print("[2] 构建 CLIP (CLIPType.LTXV + GGMLOps) ...", flush=True)
clip = csd.load_text_encoder_state_dicts(
    clip_type=csd.CLIPType.LTXV,
    state_dicts=[sd],
    model_options={"custom_operations": gguf_ops.GGMLOps, "initial_device": "cpu"},
    embedding_directory=[os.path.join(COMFY, "embeddings")],
)

te = clip.cond_stage_model
print("    TE wrapper:", type(te).__name__, flush=True)
inner = getattr(te, "diffusion_model", te)
print("    TE inner:", type(inner).__name__, flush=True)
print("    text_projection_type:", getattr(inner, "text_projection_type", "<n/a>"), flush=True)
print("    compat_mode:", getattr(inner, "compat_mode", "<n/a>"), flush=True)

# 猴子补丁：直接调 TE 的 encode_token_weights 看原始输出
tokens = clip.tokenize("A giant panda running on the green grass, sunny day.")
print("[3] 直接调 TE.encode_token_weights ...", flush=True)
with torch.no_grad():
    out, pooled, extra = inner.encode_token_weights(tokens)
print("    out.shape =", tuple(out.shape), " ndim =", out.ndim, flush=True)
print("    pooled:", None if pooled is None else tuple(pooled.shape), flush=True)
print("    extra keys:", list(extra.keys()), flush=True)

print("[4] 走完整 CLIP.encode_from_tokens ...", flush=True)
with torch.no_grad():
    cond, pooled2 = clip.encode_from_tokens(tokens, return_pooled=True)
print("    cond.shape =", tuple(cond.shape), " ndim =", cond.ndim, flush=True)
