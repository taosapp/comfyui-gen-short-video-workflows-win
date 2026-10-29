# -*- coding: utf-8 -*-
"""打印官方 LTX-2.3 t2v 模板的节点图摘要。"""
import json

p = r'C:\Users\PPXW\miniconda3\envs\ltx-video\Lib\site-packages\comfyui_workflow_templates_json\templates\video_ltx2_3_t2v.json'
wf = json.load(open(p, encoding='utf-8'))
nodes = wf.get('nodes', [])
links = wf.get('links', [])
print('节点数:', len(nodes), ' 连接数:', len(links))

for n in sorted(nodes, key=lambda x: x['id']):
    nt = n.get('type')
    if nt in ('Reroute', 'Note', 'MarkdownNote'):
        continue
    wv = n.get('widgets_values')
    try:
        wvs = json.dumps(wv, ensure_ascii=False)[:260]
    except Exception:
        wvs = str(wv)[:260]
    print(f"[{n['id']}] {nt}  {wvs}")

print('\n---- 主要数据流(按输出类型) ----')
link_map = {}
for l in links:
    # [id, origin_id, origin_slot, target_id, target_slot, type]
    link_map[l[0]] = (l[1], l[4], l[5])

def input_desc(n):
    out = []
    vals = n.get('inputs') or []
    for i in vals:
        ln = i.get('link')
        if ln is not None and ln in link_map:
            src, slot, typ = link_map[ln]
            out.append(f"{i.get('name')}<-#{src}({typ})")
    return ' '.join(out)

for n in sorted(nodes, key=lambda x: x['id']):
    nt = n.get('type')
    if nt in ('Reroute', 'Note', 'MarkdownNote'):
        continue
    d = input_desc(n)
    if d:
        print(f"[{n['id']}] {nt}: {d}")
