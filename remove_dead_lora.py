# -*- coding: utf-8 -*-
"""删除已旁路的 [285] LoraLoaderModelOnly，把 [324] 的 model 输入改接 checkpoint。"""
import json

P = r"C:\Users\PPXW\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\video_ltx2_3_i2v-test3060.json"
wf = json.load(open(P, encoding="utf-8"))
sg = wf["definitions"]["subgraphs"][0]

# 1. 重接 link 641: [285].MODEL -> [316].MODEL
fixed = False
for l in sg["links"]:
    if l["id"] == 641:
        assert l["origin_id"] == 285 and l["origin_slot"] == 0, l
        l["origin_id"] = 316
        l["origin_slot"] = 0
        fixed = True
assert fixed, "link 641 未找到"

# 2. 删除节点 285
before = len(sg["nodes"])
sg["nodes"] = [n for n in sg["nodes"] if n["id"] != 285]
assert len(sg["nodes"]) == before - 1

# 3. [316] 的 MODEL 输出 links 缓存同步
for n in sg["nodes"]:
    if n["id"] == 316:
        for o in n.get("outputs", []):
            if o.get("name") == "MODEL":
                o["links"] = [478, 541, 641]

# 4. 校验: 不再有对 285 的引用；所有 input link 存在
link_ids = {l["id"] for l in sg["links"]}
node_ids = {n["id"] for n in sg["nodes"]}
for l in sg["links"]:
    assert l["origin_id"] in node_ids or l["origin_id"] in (-10,), f"link {l['id']} origin {l['origin_id']} 悬空"
for n in sg["nodes"]:
    for i in n.get("inputs", []):
        lid = i.get("link")
        if isinstance(lid, dict):
            lid = lid.get("id")
        if lid is not None:
            assert lid in link_ids, f"节点 {n['id']} 引用不存在的 link {lid}"

with open(P, "w", encoding="utf-8") as f:
    json.dump(wf, f, ensure_ascii=False, indent=1)

print("节点 285 已删除，link 641 已改接 [316] MODEL")
print("剩余节点数:", len(sg["nodes"]))
