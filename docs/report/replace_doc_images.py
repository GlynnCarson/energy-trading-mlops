"""
Re-embed the current figure PNGs into the report docx.

    python docs/report/replace_doc_images.py energy_trading_mlops_proposal_v17.docx docs/report/figures

Word keeps its own copy of every picture, so regenerating a PNG does not change the document. This script swaps
the embedded image for each captioned figure below, keeps its displayed width (raising it to the standard 6.3 in
if it was narrower than 5.5 in) and sets the height from the new image's aspect ratio, so nothing is stretched.
"""
import io
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from PIL import Image

EMU_PER_IN = 914400
MAP = {"Figure 2.1:": "fig2_1_conceptual.png", "Figure 3.1:": "fig3_1_agile.png",
       "Figure 4.1:": "fig4_1_usecase.png", "Figure 4.2:": "fig4_2_class.png", "Figure 4.3:": "fig4_3_erd.png",
       "Figure 4.4:": "fig4_4_db_schema.png", "Figure 4.5:": "fig4_5_architecture.png",
       "Figure 4.6:": "fig4_6_ui_wireframe.png", "Figure 5.1:": "fig5_1_price_series.png",
       "Figure 5.2:": "backtest_eia_report.png", "Figure 5.3:": "fig5_2_training_curves.png",
       "Figure 5.4:": "fig5_3_test_equity.png", "Figure 5.5:": "fig5_5_paper_trading.png"}

docx_path, fig_dir = sys.argv[1], Path(sys.argv[2])
doc = Document(docx_path)
paras = doc.paragraphs
done = []
for i, p in enumerate(paras):
    blips = p._p.findall(".//" + qn("a:blip"))
    if not blips:
        continue
    cap = next((q.text for q in paras[i + 1:i + 3] if q.text.strip()), "")
    key = next((k for k in MAP if cap.startswith(k)), None)
    if not key:
        continue
    new = (fig_dir / MAP[key]).read_bytes()
    part = doc.part.related_parts[blips[0].get(qn("r:embed"))]
    part._blob = new
    w, h = Image.open(io.BytesIO(new)).size
    for tag in (".//" + qn("wp:extent"), ".//" + qn("a:ext")):
        for ext in p._p.findall(tag):
            cx = int(ext.get("cx"))
            if cx < 5.5 * EMU_PER_IN:
                cx = int(6.3 * EMU_PER_IN)
                ext.set("cx", str(cx))
            ext.set("cy", str(int(cx * h / w)))
    done.append(key)
doc.save(docx_path)
missing = [k for k in MAP if k not in done]
print("replaced:", ", ".join(k.rstrip(":") for k in done))
if missing:
    sys.exit(f"missing: {missing}")
