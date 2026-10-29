# -*- coding: utf-8 -*-
"""
诊断脚本：提交与 test_comfyui_api.py 完全相同的工作流，
但完整打印 ComfyUI 的执行状态与节点错误信息（原脚本吞掉了这些内容）。
"""
import json
import time
import urllib.request
import urllib.error

from test_comfyui_api import build_workflow

COMFYUI_URL = "http://127.0.0.1:8188"


def wait_server(timeout=300):
    print("[*] 等待 ComfyUI 就绪 ...", flush=True)
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            urllib.request.urlopen(f"{COMFYUI_URL}/system_stats", timeout=3)
            print(f"[*] ComfyUI 已就绪 (等待 {time.time()-t0:.0f}s)", flush=True)
            return True
        except Exception:
            time.sleep(3)
    print("[!] 等待超时", flush=True)
    return False


def queue_prompt(workflow):
    data = json.dumps({"prompt": workflow}).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt", data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        print(f"[!] HTTP {e.code} 提交被拒绝, 响应体:", flush=True)
        try:
            print(json.dumps(json.loads(body), ensure_ascii=False, indent=2), flush=True)
        except Exception:
            print(body, flush=True)
        return None


def dump_messages(messages, prefix=""):
    for m in messages or []:
        try:
            print(f"{prefix}message: {json.dumps(m, ensure_ascii=False)[:3000]}", flush=True)
        except Exception:
            print(f"{prefix}message(raw): {m}", flush=True)


def main():
    if not wait_server():
        return 2

    workflow = build_workflow(
        "A giant panda running on the green grass, sunny day, "
        "clear blue sky, cinematic, natural lighting, high quality, 4K",
        "distilled")

    print("[*] 提交工作流 ...", flush=True)
    result = queue_prompt(workflow)
    if result is None:
        return 1
    prompt_id = result.get("prompt_id")
    print(f"[*] prompt_id = {prompt_id}", flush=True)
    if result.get("node_errors"):
        print("[!] 提交即报 node_errors:", flush=True)
        print(json.dumps(result["node_errors"], ensure_ascii=False, indent=2), flush=True)

    t0 = time.time()
    last_len = 0
    while True:
        try:
            with urllib.request.urlopen(f"{COMFYUI_URL}/history/{prompt_id}", timeout=5) as resp:
                history = json.loads(resp.read())
        except Exception as e:
            print(f"  (poll error: {e})", flush=True)
            time.sleep(3)
            continue

        entry = history.get(prompt_id)
        if entry is None:
            # 还在队列里，看队列状态
            try:
                with urllib.request.urlopen(f"{COMFYUI_URL}/queue", timeout=5) as resp:
                    q = json.loads(resp.read())
                running = [i[1] for i in q.get("queue_running", [])]
                pending = [i[1] for i in q.get("queue_pending", [])]
                print(f"  [{time.time()-t0:5.0f}s] running={prompt_id in running} "
                      f"pending={prompt_id in pending}", flush=True)
            except Exception:
                pass
            time.sleep(5)
            continue

        status = entry.get("status", {})
        status_str = status.get("status_str")
        completed = status.get("completed")
        msgs = status.get("messages", [])
        if len(msgs) != last_len:
            dump_messages(msgs[last_len:], prefix="  ")
            last_len = len(msgs)

        if status_str == "error" or (completed and status_str != "success"):
            print(f"\n[!] 执行失败 status_str={status_str}", flush=True)
            dump_messages(msgs)
            return 1

        if completed and status_str == "success":
            print(f"\n[OK] 执行成功, 耗时 {time.time()-t0:.0f}s", flush=True)
            print(json.dumps(entry.get("outputs", {}), ensure_ascii=False, indent=2)[:3000],
                  flush=True)
            return 0

        time.sleep(5)


if __name__ == "__main__":
    import sys
    sys.exit(main())
