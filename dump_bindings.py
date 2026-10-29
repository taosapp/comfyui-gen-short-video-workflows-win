# -*- coding: utf-8 -*-
"""完整导出 Desktop 工作流的子图定义与绑定信息。"""
import json

P = r"C:\Users\PPXW\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\video_ltx2_3_i2v-test3060.json"
wf = json.load(open(P, encoding="utf-8"))

print("=== 顶层节点 widgets_values（完整） ===")
for n in wf["nodes"]:
    if n["id"] == 320:
        print(json.dumps(n.get("widgets_values"), ensure_ascii=False, indent=1))
        print("--- 顶层节点320 inputs:", json.dumps(n.get("inputs"), ensure_ascii=False))
        print("--- 顶层节点320 outputs:", json.dumps(n.get("outputs"), ensure_ascii=False))

sg = wf["definitions"]["subgraphs"][0]
print("\n=== 子图 inputs 定义（暴露的槽位） ===")
print(json.dumps(sg.get("inputs", []), ensure_ascii=False, indent=1)[:4000])
print("\n=== 子图 outputs 定义 ===")
print(json.dumps(sg.get("outputs", []), ensure_ascii=False)[:1500])

print("\n=== 子图 links 原始（含 reroute/293） ===")
for l in sg.get("links", []):
    print(json.dumps(l, ensure_ascii=False))

print("\n=== 节点293（疑似reroute） ===")
for n in sg["nodes"]:
    if n["id"] in (293, 285, 317, 316, 279, 324, 311):
        print(json.dumps({"id": n["id"], "type": n["type"], "pos": n.get("pos"),
                          "inputs": n.get("inputs"), "outputs": n.get("outputs"),
                          "wv": n.get("widgets_values")}, ensure_ascii=False)[:1200])
        print()
