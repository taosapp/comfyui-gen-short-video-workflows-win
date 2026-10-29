# -*- coding: utf-8 -*-
"""扫描工作流中所有残留的 properties.models 数组与暴露槽位。"""
import json

P = r"C:\Users\PPXW\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\video_ltx2_3_i2v-test3060.json"
wf = json.load(open(P, encoding="utf-8"))
print("顶层 models 字段:", "models" in wf)
sg = wf["definitions"]["subgraphs"][0]


def scan(nodes, where):
    for n in nodes:
        models = (n.get("properties") or {}).get("models")
        if models:
            print(f'{where} [{n["id"]}] {n["type"]}:')
            for m in models:
                print(f'    {m.get("name")} -> {m.get("directory")}')


scan(sg["nodes"], "子图")
scan(wf["nodes"], "顶层")
print("--- 子图暴露槽位（顺序即链接引用的 slot 序号） ---")
for i, s in enumerate(sg["inputs"]):
    print(i, s["name"], "linkIds:", s.get("linkIds"))
