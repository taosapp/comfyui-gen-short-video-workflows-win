# -*- coding: utf-8 -*-
"""检查 GGUF 头部元数据与张量名（不加载权重）。"""
import sys
import gguf

for path in sys.argv[1:]:
    print("=" * 60)
    print(path)
    reader = gguf.GGUFReader(path)
    tensors = [t.name for t in reader.tensors]
    for k, f in reader.fields.items():
        if k in ("general.architecture", "general.name", "general.file_type"):
            try:
                v = [f.parts[idx].data for idx in f.data]
                v = b"".join(v)
                if isinstance(v, bytes):
                    v = v.decode(errors="replace")
                print(f"  {k} = {v}")
            except Exception as e:
                print(f"  {k} = <{e}>")
    proj = [n for n in tensors if "text_embedding_projection" in n]
    conn = [n for n in tensors if "embeddings_connector" in n]
    print(f"  张量总数: {len(tensors)}")
    print(f"  含 text_embedding_projection.*: {len(proj)}  {proj[:4]}")
    print(f"  含 embeddings_connector.*: {len(conn)}  {conn[:4]}")
    print(f"  前几个张量: {tensors[:6]}")
