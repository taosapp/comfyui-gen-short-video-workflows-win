# -*- coding: utf-8 -*-
"""从 ltx-2.3-22b-distilled-fp8.safetensors 中抽取小体积组件文件。

1. text_embedding_projection.*  -> models/checkpoints/ltx-2.3-te-projection.safetensors (保留前缀)
2. audio_vae.* + vocoder.*      -> models/checkpoints/ltx-2.3-audio-vae.safetensors   (保留前缀)
3. vae.*                        -> models/vae/ltx-2.3-video-vae.safetensors          (去掉 vae. 前缀)
"""
import json
import struct
import os
from safetensors import safe_open
from safetensors.torch import save_file

SRC = r"D:\ai\gen-short-video\models\ltx-2.3-fp8\ltx-2.3-22b-distilled-fp8.safetensors"
OUT_PROJ = r"D:\ai\gen-short-video\comfyui\models\checkpoints\ltx-2.3-te-projection.safetensors"
OUT_AUDIO = r"D:\ai\gen-short-video\comfyui\models\checkpoints\ltx-2.3-audio-vae.safetensors"
OUT_VAE = r"D:\ai\gen-short-video\comfyui\models\vae\ltx-2.3-video-vae.safetensors"

with open(SRC, "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))

orig_metadata = hdr.get("__metadata__", {})
print("原文件 metadata keys:", list(orig_metadata.keys()))

keys = [k for k in hdr if k != "__metadata__"]


def group_bytes(keys, hdr):
    sizes = {"F64": 8, "I64": 8, "F32": 4, "I32": 4, "BF16": 2, "F16": 2, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}
    total = 0
    for k in keys:
        info = hdr[k]
        numel = 1
        for d in info["shape"]:
            numel *= d
        total += numel * sizes.get(info["dtype"], 2)
    return total


proj_keys = [k for k in keys if k.startswith("text_embedding_projection.")]
audio_keys = [k for k in keys if k.startswith("audio_vae.") or k.startswith("vocoder.")]
vae_keys = [k for k in keys if k.startswith("vae.")]

print(f"projection: {len(proj_keys)} tensors, {group_bytes(proj_keys, hdr)/1e9:.2f} GB")
print(f"audio vae : {len(audio_keys)} tensors, {group_bytes(audio_keys, hdr)/1e9:.2f} GB")
print(f"video vae : {len(vae_keys)} tensors, {group_bytes(vae_keys, hdr)/1e9:.2f} GB")

f = safe_open(SRC, framework="pt", device="cpu")


def extract(keys, out, strip_prefix=None, metadata=None, overwrite=False):
    if os.path.exists(out) and not overwrite:
        print(f"已存在，跳过: {out}")
        return
    sd = {}
    for k in keys:
        t = f.get_tensor(k)
        new_k = k[len(strip_prefix):] if strip_prefix and k.startswith(strip_prefix) else k
        sd[new_k] = t.contiguous()
    md = {"format": "pt"}
    if metadata:
        md.update(metadata)
    save_file(sd, out, metadata=md)
    print(f"写出: {out} ({len(sd)} tensors, {os.path.getsize(out)/1e9:.2f} GB)")


extract(proj_keys, OUT_PROJ)
extract(audio_keys, OUT_AUDIO, metadata=orig_metadata, overwrite=True)
extract(vae_keys, OUT_VAE, strip_prefix="vae.", metadata=orig_metadata, overwrite=True)
print("完成")
