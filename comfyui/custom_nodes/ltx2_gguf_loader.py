# -*- coding: utf-8 -*-
"""
LTX-2.x 文本编码器加载器 (GGUF 版)

官方 LTXAVTextEncoderLoader 只支持 safetensors 文本编码器（comfy.utils.load_torch_file
不认识 .gguf），而 CLIPLoaderGGUF 单文件加载 Gemma-3-12B 时会命中 sd.py 中
te_model == GEMMA_3_12B 的分支，构建出"裸 Gemma"（layer="all" 输出 4D，
缺少 LTXAVTEModel 的 text_embedding_projection 投影），最终在
LTXAV.preprocess_text_embeds 的 embeddings_connector 里因 4D/3D 维度不匹配崩溃。

本节点按官方双文件配方加载：
  state_dicts = [gemma gguf (ComfyUI-GGUF 懒加载), text_embedding_projection safetensors]
  clip_type   = CLIPType.LTXV
=> len(clip_data)==2 分支 -> ltxav_te(**llama_detect, **sd_detect)
   (sd_detect 从第二个文件检出 dual_linear 投影) -> LTXAVTEModel 正常构建。
"""
import importlib
import os
import sys
import types

import torch

import folder_paths
import comfy.sd
import comfy.utils
import comfy.model_management

# 把 ComfyUI-GGUF 目录挂成一个可导入的包（目录名含连字符，不能直接 import）
_PKG = "gguf_pkg_shim"
if _PKG not in sys.modules:
    pkg = types.ModuleType(_PKG)
    pkg.__path__ = [os.path.join(os.path.dirname(os.path.abspath(__file__)), "ComfyUI-GGUF")]
    sys.modules[_PKG] = pkg
_gguf_ops = importlib.import_module(_PKG + ".ops")
_gguf_loader = importlib.import_module(_PKG + ".loader")
_gguf_nodes = importlib.import_module(_PKG + ".nodes")


class LTXAVTextEncoderLoaderGGUF:
    @classmethod
    def get_filename_list(s):
        # 与 CLIPLoaderGGUF 相同：合并核心 text_encoders/clip 与 ComfyUI-GGUF 注册的 clip_gguf 列表
        files = list(folder_paths.get_filename_list("text_encoders"))
        files += list(folder_paths.get_filename_list("clip"))
        files += list(folder_paths.get_filename_list("clip_gguf"))
        return sorted(set(files))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text_encoder": (s.get_filename_list(),),
                "projection_ckpt": (folder_paths.get_filename_list("checkpoints"),),
                "device": (["default", "cpu"], {"advanced": True}),
            }
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders"
    TITLE = "Load LTXV Audio Text Encoder (GGUF)"

    def load_clip(self, text_encoder, projection_ckpt, device="default"):
        clip_path1 = (folder_paths.get_full_path("text_encoders", text_encoder)
                      or folder_paths.get_full_path("clip_gguf", text_encoder)
                      or folder_paths.get_full_path("clip", text_encoder))
        if clip_path1 is None:
            raise ValueError(f"text encoder not found: {text_encoder}")
        clip_path2 = folder_paths.get_full_path_or_raise("checkpoints", projection_ckpt)

        sd1 = _gguf_loader.gguf_clip_loader(clip_path1)  # gemma 权重（GGUF 量化，懒加载）
        sd2 = comfy.utils.load_torch_file(clip_path2)   # text_embedding_projection.*（约2.3GB）

        model_options = {
            "custom_operations": _gguf_ops.GGMLOps,
            "initial_device": comfy.model_management.text_encoder_offload_device(),
        }
        if device == "cpu":
            model_options["load_device"] = model_options["offload_device"] = torch.device("cpu")

        clip = comfy.sd.load_text_encoder_state_dicts(
            clip_type=comfy.sd.CLIPType.LTXV,
            state_dicts=[sd1, sd2],
            model_options=model_options,
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
        )
        clip.patcher = _gguf_nodes.GGUFModelPatcher.clone(clip.patcher)
        return (clip,)


NODE_CLASS_MAPPINGS = {
    "LTXAVTextEncoderLoaderGGUF": LTXAVTextEncoderLoaderGGUF,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAVTextEncoderLoaderGGUF": "Load LTXV Audio Text Encoder (GGUF)",
}
