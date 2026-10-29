# -*- coding: utf-8 -*-
"""规范化 Desktop settings.json 的 modelsDirs 路径（单反斜杠）。"""
import json

p = r"C:\Users\PPXW\AppData\Roaming\Comfy Desktop\settings.json"
cfg = json.load(open(p, encoding="utf-8"))
cfg["modelsDirs"] = [
    r"D:\ai\gen-short-video\models",
    r"C:\Users\PPXW\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models",
]
json.dump(cfg, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

check = json.load(open(p, encoding="utf-8"))["modelsDirs"]
print("最终 modelsDirs:")
for d in check:
    print("  ", d, "| 路径有效:", __import__("os").path.isdir(d))
