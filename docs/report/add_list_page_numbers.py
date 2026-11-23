"""
One-shot: give every entry in the List of Figures and List of Tables a live page number.

Each figure/table caption in the body gets a hidden bookmark (_LoF_Fig_4_5, _LoT_Tab_5_10, ...), and each list
entry ends with a right-aligned dot-leader tab and a PAGEREF field pointing at that bookmark, so Word computes the
page number. Refresh after any edit by updating fields (Word: Ctrl+A, F9; COM: doc.Fields.Update()), the same
way as the Table of Contents.

    python docs/report/add_list_page_numbers.py      then update fields in Word (or via COM)

Refuses to run if the bookmarks already exist.
Backup: docs/report/backup/energy_trading_mlops_proposal_v17_before_list_page_numbers.docx
"""
import copy
import re
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

F = Path(__file__).resolve().parents[2] / "energy_trading_mlops_proposal_v17.docx"
doc = Document(F)
ps = doc.paragraphs
LABEL = re.compile(r"^(Figure|Table) ([0-9]+\.[0-9]+|A[0-9]+):")

if any(b.get(qn("w:name"), "").startswith(("_LoF_", "_LoT_")) for b in doc.element.body.iter(qn("w:bookmarkStart"))):
    sys.exit("List bookmarks already present - this update has been applied.")

ch1 = next(i for i, p in enumerate(ps) if p.style.name == "Heading 1" and p.text.startswith("Chapter One"))
entries = {LABEL.match(p.text).group(0): p for p in ps[:ch1] if LABEL.match(p.text)}
captions = {}
for p in ps[ch1:]:
    m = LABEL.match(p.text)
    if m:
        assert m.group(0) not in captions, f"duplicate caption {m.group(0)}"
        captions[m.group(0)] = p
assert entries.keys() == captions.keys(), (set(entries) ^ set(captions))

# tab stop copied from the one entry that already had a page number
template_tabs = next(p._p.pPr.find(qn("w:tabs")) for p in entries.values()
                     if p._p.pPr is not None and p._p.pPr.find(qn("w:tabs")) is not None)


def run(*children):
    r = OxmlElement("w:r")
    for c in children:
        r.append(c)
    return r


def el(tag, text=None, **attrs):
    e = OxmlElement(tag)
    for k, v in attrs.items():
        e.set(qn(k), v)
    if text is not None:
        e.text = text
        if text != text.strip():
            e.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    return e


next_id = max(int(b.get(qn("w:id"))) for b in doc.element.body.iter(qn("w:bookmarkStart"))) + 1
for label, cap in captions.items():
    kind, num = LABEL.match(label).groups()
    name = ("_LoF_Fig_" if kind == "Figure" else "_LoT_Tab_") + num.replace(".", "_")
    # bookmark around the caption paragraph's content
    start = el("w:bookmarkStart", **{"w:id": str(next_id), "w:name": name})
    end = el("w:bookmarkEnd", **{"w:id": str(next_id)})
    pPr = cap._p.pPr
    (pPr.addnext(start) if pPr is not None else cap._p.insert(0, start))
    cap._p.append(end)
    next_id += 1

    ent = entries[label]
    # drop any typed page number (a run holding a tab) and the text after it
    for r in list(ent._p.findall(qn("w:r"))):
        if r.find(qn("w:tab")) is not None:
            ent._p.remove(r)
    epPr = ent._p.get_or_add_pPr()
    if epPr.find(qn("w:tabs")) is None:
        epPr.insert(0, copy.deepcopy(template_tabs))
    ent._p.append(run(el("w:tab")))
    ent._p.append(run(el("w:fldChar", **{"w:fldCharType": "begin"})))
    ent._p.append(run(el("w:instrText", f" PAGEREF {name} \\h ")))
    ent._p.append(run(el("w:fldChar", **{"w:fldCharType": "separate"})))
    ent._p.append(run(el("w:t", "0")))
    ent._p.append(run(el("w:fldChar", **{"w:fldCharType": "end"})))

doc.save(F)
print(f"added page-number fields to {len(entries)} entries")
