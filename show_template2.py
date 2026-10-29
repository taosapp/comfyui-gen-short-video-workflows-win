# -*- coding: utf-8 -*-
"""展开官方 LTX-2.3 t2v 模板的子图节点。"""
import json

p = r'C:\Users\PPXW\miniconda3\envs\ltx-video\Lib\site-packages\comfyui_workflow_templates_json\templates\video_ltx2_3_t2v.json'
wf = json.load(open(p, encoding='utf-8'))

def dump_graph(nodes, links, prefix=''):
    lm = {}
    for l in links:
        if isinstance(l, dict):
            lm[l.get('id')] = (l.get('origin_id'), l.get('target_slot'), l.get('type'))
        else:
            lm[l[0]] = (l[1], l[4], l[5])
    for n in sorted(nodes, key=lambda x: x['id']):
        nt = n.get('type')
        if nt in ('Reroute', 'Note', 'MarkdownNote'):
            continue
        wv = n.get('widgets_values')
        try:
            wvs = json.dumps(wv, ensure_ascii=False)[:300]
        except Exception:
            wvs = str(wv)[:300]
        ins = []
        for i in (n.get('inputs') or []):
            ln = i.get('link')
            if isinstance(ln, dict):
                ln = ln.get('id')
            if ln is not None and ln in lm:
                src, slot, typ = lm[ln]
                ins.append(f"{i.get('name')}<-#{src}({typ})")
        print(f"{prefix}[{n['id']}] {nt}")
        if ins:
            print(f"{prefix}    in: {' '.join(ins)}")
        if wvs and wvs != 'null':
            print(f"{prefix}    wv: {wvs}")

defs = wf.get('definitions', {})
subs = defs.get('subgraphs', [])
print('顶层节点数:', len(wf.get('nodes', [])), ' 子图数:', len(subs))
for sg in subs:
    print('=' * 70)
    print('SUBGRAPH', sg.get('id'), sg.get('name', ''))
    dump_graph(sg.get('nodes', []), sg.get('links', []), prefix='  ')

print('=' * 70)
print('TOP LEVEL')
dump_graph(wf.get('nodes', []), wf.get('links', []))
