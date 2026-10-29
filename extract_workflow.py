# -*- coding: utf-8 -*-
"""从 ComfyUI 输出 PNG 中提取嵌入的工作流 JSON（UI 格式 + API 格式）。"""
import json
from PIL import Image

SRC = r"D:\ai\gen-short-video\comfyui\output\LTX-Video_00002.png"
OUT_UI = r"D:\ai\gen-short-video\workflow_ltx23_t2v_ui.json"
OUT_API = r"D:\ai\gen-short-video\workflow_ltx23_t2v_api.json"

img = Image.open(SRC)
meta = getattr(img, "text", None) or img.info
print("PNG 元数据 keys:", list(meta.keys()))

wf_ui = meta.get("workflow")
wf_api = meta.get("prompt")

if wf_ui:
    data = json.loads(wf_ui)
    with open(OUT_UI, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"UI 工作流  -> {OUT_UI} ({len(data.get('nodes', []))} 节点)")
else:
    print("PNG 中没有 workflow (UI格式) 元数据")

if wf_api:
    data = json.loads(wf_api)
    with open(OUT_API, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"API 工作流 -> {OUT_API} ({len(data)} 节点)")
else:
    print("PNG 中没有 prompt (API格式) 元数据")
