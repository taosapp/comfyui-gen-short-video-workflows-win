# -*- coding: utf-8 -*-
"""修复剩余两个假缺模型警告：
1. 删除空的暴露槽位 lora_name（384 lora -> Unknown 的来源），并修正槽位序号偏移
2. 给 [317] 补 properties.models，把 gemma 正确归属到 text_encoders
"""
import json

P = r"C:\Users\PPXW\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\video_ltx2_3_i2v-test3060.json"
BAK = P + ".bak2"
shutil_reason = ""
import shutil
shutil.copy2(P, BAK)
print(f"已备份 -> {BAK}")

wf = json.load(open(P, encoding="utf-8"))
sg = wf["definitions"]["subgraphs"][0]

# ---- 1. 删除空暴露槽位 lora_name（index 9）----
slots = sg["inputs"]
idx = next(i for i, s in enumerate(slots) if s["name"] == "lora_name")
assert slots[idx].get("linkIds") == [], "槽位非空，先检查"
removed = slots.pop(idx)
print(f"已删除暴露槽位 #{idx}: {removed['name']} ({removed['id']})")

# 其后槽位序号各前移一位：更新引用 -10 的链接 origin_slot
for l in sg["links"]:
    if l["origin_id"] == -10 and l["origin_slot"] >= idx:
        old = l["origin_slot"]
        l["origin_slot"] = old - 1
        print(f"  链接 {l['id']}: origin_slot {old} -> {l['origin_slot']} (target {l['target_id']})")

# ---- 2. 顶层节点：删除对应控件输入 + widgets_values 条目 ----
top = next(n for n in wf["nodes"] if n["id"] == 320)
before = len(top["inputs"])
top["inputs"] = [i for i in top["inputs"]
                 if not (i.get("name") == "lora_name" and i.get("type") == "COMBO"
                         and i.get("label") == "distilled_lora")]
assert len(top["inputs"]) == before - 1, "顶层 lora_name 输入未找到"
del top["widgets_values"][8]  # 原 index 8 = 384 lora
print(f"顶层控件输入现为 {len(top['inputs'])} 项, widgets_values {len(top['widgets_values'])} 项")

# ---- 3. [317] models 归属提示 ----
nodes = {n["id"]: n for n in sg["nodes"]}
nodes[317]["properties"]["models"] = [{
    "name": "gemma-3-12b-it-Q4_K_M.gguf",
    "url": "https://huggingface.co/unsloth/gemma-3-12b-it-GGUF/resolve/main/gemma-3-12b-it-Q4_K_M.gguf",
    "directory": "text_encoders",
}]

# ---- 校验 ----
slot_count = len(sg["inputs"])
for l in sg["links"]:
    if l["origin_id"] == -10:
        assert 0 <= l["origin_slot"] < slot_count, f"链接 {l['id']} origin_slot 越界: {l['origin_slot']}"
link_ids = {l["id"] for l in sg["links"]}
for n in sg["nodes"]:
    for i in n.get("inputs", []):
        lid = i.get("link")
        if isinstance(lid, dict):
            lid = lid.get("id")
        if lid is not None:
            assert lid in link_ids, f"节点 {n['id']} 引用了不存在的 link {lid}"

with open(P, "w", encoding="utf-8") as f:
    json.dump(wf, f, ensure_ascii=False, indent=1)

print("修改完成并通过校验")
print("最终暴露槽位:", [s["name"] for s in sg["inputs"]])
print("顶层 widgets_values:", json.dumps(top["widgets_values"], ensure_ascii=False)[:320])
