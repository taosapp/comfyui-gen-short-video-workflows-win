# -*- coding: utf-8 -*-
"""探针：提交两段式工作流，打印 400 完整响应体。"""
import json
import time
import urllib.request
import urllib.error

from test_comfyui_api import build_workflow, COMFYUI_URL

# 等服务就绪
for _ in range(60):
    try:
        urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
        break
    except Exception:
        time.sleep(2)

wf = build_workflow("A giant panda running towards the camera, sunny day", "distilled")
data = json.dumps({"prompt": wf}).encode("utf-8")
req = urllib.request.Request(f"{COMFYUI_URL}/prompt", data=data,
                             headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req) as r:
        print("OK:", r.read().decode()[:300])
except urllib.error.HTTPError as e:
    print(f"HTTP {e.code} 完整响应体:")
    print(e.read().decode("utf-8", "replace"))
