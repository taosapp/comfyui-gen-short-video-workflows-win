# -*- coding: utf-8 -*-
"""清理工作流中因节点类型替换而遗留的过时 properties（cnr_id/ver/models 数组）。"""
import json

P = r"C:\Users\PPXW\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\video_ltx2_3_i2v-test3060.json"
wf = json.load(open(P, encoding="utf-8"))
sg = wf["definitions"]["subgraphs"][0]
nodes = {n["id"]: n for n in sg["nodes"]}

# [317] 已换成自定义节点：清掉核心节点身份标记与过期下载清单
p317 = nodes[317]["properties"]
p317.pop("cnr_id", None)
p317.pop("ver", None)
p317.pop("models", None)
p317["Node name for S&R"] = "LTXAVTextEncoderLoaderGGUF"

# 其余节点：模型选择已改，清掉过期的 models 下载清单（文件全在本地，不需要下载提示）
for nid in (279, 285, 316, 324):
    nodes[nid]["properties"].pop("models", None)

with open(P, "w", encoding="utf-8") as f:
    json.dump(wf, f, ensure_ascii=False, indent=1)

print("properties 清理完成")
for nid in (279, 285, 316, 317, 324, 311):
    n = nodes[nid]
    print(f"[{nid}] {n['type']} props: {json.dumps(n.get('properties'), ensure_ascii=False)[:150]}")
