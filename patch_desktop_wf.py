# -*- coding: utf-8 -*-
"""
把 Desktop 保存的 video_ltx2_3_i2v-test3060 工作流改为指向本项目已有模型。

改动清单:
  [316] CheckpointLoaderSimple  : ckpt -> ltx-2.3-22b-distilled-fp8 (本地有, 含VAE/投影)
  [279] LTXVAudioVAELoader      : 解除共享ckpt绑定 -> 本地小文件 ltx-2.3-audio-vae.safetensors
  [317] LTXAVTextEncoderLoader  : 换成 LTXAVTextEncoderLoaderGGUF
                                  (gemma-3-12b-it-Q4_K_M.gguf + ltx-2.3-te-projection.safetensors)
  [285] LoraLoaderModelOnly     : 旁路(两个CFGGuider直接用checkpoint的MODEL), 节点孤立保底换成本地lora
  [324] LoraLoader              : 指向本地已有 lora(该分支被开关禁用, 仅为过校验)
  [311] LatentUpscaleModelLoader: 文件名不变(本地有, 通过 extra_model_paths.yaml 暴露)
"""
import json
import shutil

P = r"C:\Users\PPXW\AppData\Local\Comfy-Desktop\ComfyUI-Installs\ComfyUI\ComfyUI\user\default\workflows\video_ltx2_3_i2v-test3060.json"
BAK = P + ".bak"

shutil.copy2(P, BAK)
print(f"已备份原文件 -> {BAK}")

wf = json.load(open(P, encoding="utf-8"))
sg = wf["definitions"]["subgraphs"][0]
nodes = {n["id"]: n for n in sg["nodes"]}
links = sg["links"]

LOCAL_LORA = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"


def unbind(node, link_id):
    """从节点上解除一个 widget 绑定输入（变回本地控件）"""
    before = len(node["inputs"])
    node["inputs"] = [i for i in node["inputs"] if i.get("link") != link_id]
    assert len(node["inputs"]) == before - 1, f"link {link_id} 未在节点 {node['id']} 上找到"


def drop_link(link_id):
    before = len(links)
    links[:] = [l for l in links if l["id"] != link_id]
    assert len(links) == before - 1, f"link {link_id} 不存在"


def slot_def(name):
    for s in sg["inputs"]:
        if s["name"] == name:
            return s
    raise KeyError(name)


# ---- [316] checkpoint -> distilled fp8 ----
nodes[316]["widgets_values"][0] = "ltx-2.3-22b-distilled-fp8.safetensors"

# ---- [279] 音频VAE: 解除共享 ckpt 绑定(604), 用小文件 ----
unbind(nodes[279], 604)
drop_link(604)
slot_def("ckpt_name")["linkIds"].remove(604)
nodes[279]["widgets_values"][0] = "ltx-2.3-audio-vae.safetensors"

# ---- [317] 文本编码器: 换 GGUF 加载节点, 解除 ckpt 绑定(605) ----
n317 = nodes[317]
unbind(n317, 605)
drop_link(605)
slot_def("ckpt_name")["linkIds"].remove(605)
n317["type"] = "LTXAVTextEncoderLoaderGGUF"
if "title" in n317:
    n317["title"] = "Load LTXV Audio Text Encoder (GGUF)"
n317["widgets_values"] = [
    "gemma-3-12b-it-Q4_K_M.gguf",
    "ltx-2.3-te-projection.safetensors",
    "default",
]
assert [i["name"] for i in n317["inputs"]] == ["text_encoder", "device"], n317["inputs"]

# ---- [285] 蒸馏lora旁路: 两个CFGGuider直接接checkpoint MODEL ----
for lid in (478, 541):
    for l in links:
        if l["id"] == lid:
            assert l["origin_id"] == 285
            l["origin_id"] = 316
            l["origin_slot"] = 0
unbind(nodes[285], 520)
drop_link(520)
unbind(nodes[285], 602)
drop_link(602)
slot_def("lora_name")["linkIds"].remove(602)
nodes[285]["widgets_values"][0] = LOCAL_LORA
# 输出/输入的 links 缓存同步
for o in nodes[285].get("outputs", []):
    if o.get("name") == "MODEL":
        o["links"] = [641]
for o in nodes[316].get("outputs", []):
    if o.get("name") == "MODEL":
        o["links"] = [478, 541]

# ---- [324] / 顶层暴露控件: 指向本地已有 lora ----
nodes[324]["widgets_values"][0] = LOCAL_LORA

top = next(n for n in wf["nodes"] if n["id"] == 320)
wv = top["widgets_values"]
assert len(wv) == 12, wv
wv[7] = "ltx-2.3-22b-distilled-fp8.safetensors"            # ckpt_name
wv[8] = LOCAL_LORA                                          # distilled_lora(已旁路,仅显示)
wv[9] = "gemma-3-12b-it-Q4_K_M.gguf"                        # text_encoder
wv[10] = "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"      # 不变
wv[11] = LOCAL_LORA                                         # lora_name_1

# ---- 校验: 所有 input 引用的 link 都存在, link 的目标节点都存在 ----
link_ids = {l["id"] for l in links}
node_ids = {n["id"] for n in sg["nodes"]}
for n in sg["nodes"]:
    for i in n.get("inputs", []):
        lid = i.get("link")
        if isinstance(lid, dict):
            lid = lid.get("id")
        if lid is not None:
            assert lid in link_ids, f"节点{n['id']}引用了不存在的link {lid}"
for l in links:
    assert l["origin_id"] == -10 or l["origin_id"] in node_ids, l
    assert l["target_id"] in node_ids or l["target_id"] in (-10, -20), l

with open(P, "w", encoding="utf-8") as f:
    json.dump(wf, f, ensure_ascii=False, indent=1)

print("修改完成并通过链接完整性校验")
print("\n=== 改后模型节点 ===")
for n in sg["nodes"]:
    if n["type"] in ("CheckpointLoaderSimple", "LTXVAudioVAELoader",
                     "LoraLoaderModelOnly", "LatentUpscaleModelLoader",
                     "LTXAVTextEncoderLoader", "LTXAVTextEncoderLoaderGGUF", "LoraLoader"):
        print(f"[{n['id']}] {n['type']}: {json.dumps(n.get('widgets_values'), ensure_ascii=False)[:160]}")
print("\n顶层参数:", json.dumps(wv, ensure_ascii=False)[:400])
