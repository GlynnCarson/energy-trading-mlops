"""Export every picture in the docx together with the caption that follows it."""
import re
import sys
from pathlib import Path
from docx import Document
from docx.oxml.ns import qn

doc = Document(sys.argv[1])
out = Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
paras = doc.paragraphs
for i, p in enumerate(paras):
    blips = p._p.findall(".//" + qn("a:blip"))
    if not blips:
        continue
    cap = next((q.text for q in paras[i + 1:i + 3] if q.text.strip()), "")
    rid = blips[0].get(qn("r:embed"))
    part = doc.part.related_parts[rid]
    tag = re.sub(r"[^A-Za-z0-9]+", "_", cap[:40]).strip("_") or f"para{i}"
    ext = part.partname.split(".")[-1]
    (out / f"{i:03d}_{tag}.{ext}").write_bytes(part.blob)
    print(f"{i:03d}  {cap[:80]}")
