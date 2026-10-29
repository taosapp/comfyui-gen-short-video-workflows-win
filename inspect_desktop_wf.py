# -*- coding: utf-8 -*-
"""列出 Desktop 保存的工作流中所有引用模型的节点及其当前值。"""
import json

P = r"C:\Users\PPXW\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\video_ltx2_3_i2v-test3060.json"
wf = json.load(open(P, encoding="utf-8"))

MODEL_NODES = {
    "CheckpointLoaderSimple": "ckpt_name",
    "LTXAVTextEncoderLoader": "text_encoder",
    "LTXVAudioVAELoader": "ckpt_name",
    "LoraLoaderModelOnly": "lora_name",
    "LatentUpscaleModelLoader": "model_name",
    "VAELoader": "vae_name",
    "UnetLoaderGGUF": "unet_name",
    "UnetLoaderGGUFAdvanced": "unet_name",
    "CLIPLoaderGGUF": "clip_name",
    "CLIPLoader": "clip_name",
    "LTXAVTextEncoderLoaderGGUF": "text_encoder",
}


def walk(nodes, links, prefix=""):
    lm = {}
    for l in links:
        if isinstance(l, dict):
            lm[l.get("id")] = (l.get("origin_id"), l.get("target_slot"), l.get("type"))
        else:
            lm[l[0]] = (l[1], l[4], l[5])
    for n in nodes:
        nt = n.get("type", "")
        if nt in ("Reroute", "Note", "MarkdownNote"):
            continue
        wv = n.get("widgets_values")
        wvs = json.dumps(wv, ensure_ascii=False)[:200] if wv else ""
        ins = []
        for i in (n.get("inputs") or []):
            ln = i.get("link")
            if isinstance(ln, dict):
                ln = ln.get("id")
            if ln is not None and ln in lm:
                src, slot, typ = lm[ln]
                ins.append(f"{i.get('name')}<-#{src}({typ})")
        tag = " <== 模型节点" if nt in MODEL_NODES else ""
        print(f"{prefix}[{n['id']}] {nt} | in: {' '.join(ins)} | wv: {wvs}{tag}")


subs = wf.get("definitions", {}).get("subgraphs", [])
print(f"顶层节点: {len(wf.get('nodes', []))}  子图: {len(subs)}")
for sg in subs:
    print(f"===== 子图 {sg.get('name', '')} =====")
    walk(sg.get("nodes", []), sg.get("links", []), "  ")
print("===== 顶层 =====")
walk(wf.get("nodes", []), wf.get("links", []))
