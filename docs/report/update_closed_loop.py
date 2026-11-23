"""
One-shot update of the report for the closed loop built on 23 November 2026: quality gate, MLflow Model
Registry (champion alias), paper trading (simulated), monitoring and champion/challenger retraining, run in the
Docker Compose DAG. Also: Terraform validation passed in GitHub Actions (TC-16), DEF-07 fixed, DEF-08 mitigated,
new defects DEF-14 and DEF-15, 102 automated tests.

Adds Section 5.4.8, Section 5.6.5, Table 5.10 (gate results), Figure 5.5 (paper trading), Table 5.11 (paper
trading and health), FR-15 to FR-18 and TC-18 to TC-22; renumbers the objectives table from Table 5.10 to
Table 5.12 (caption, list entry and bookmark). Evidence: reports/live/ (gate_initial.json, model_registry.json,
monitor_status.json, paper_trading.csv, evidence/2026-11-23/).

    python docs/report/update_closed_loop.py
    python docs/report/replace_doc_images.py energy_trading_mlops_proposal_v17.docx docs/report/figures
    then update fields and the TOC in Word (COM: Fields.Update(), TablesOfContents(1).Update())

Refuses to run if TC-18 already exists.
Backup: docs/report/backup/energy_trading_mlops_proposal_v17_before_closed_loop.docx
"""
import copy
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.table import Table, _Row
from docx.text.paragraph import Paragraph

HERE = Path(__file__).resolve().parent
F = HERE.parents[1] / "energy_trading_mlops_proposal_v17.docx"
BACKUP = HERE / "backup" / "energy_trading_mlops_proposal_v17_before_closed_loop.docx"
doc = Document(F)
P = doc.paragraphs


# ── helpers ─────────────────────────────────────────────────────────────────
def para(prefix):
    hits = [p for p in P if p.text.startswith(prefix)]
    assert len(hits) == 1, (prefix, len(hits))
    return hits[0]


def replace(p, old, new):
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new, 1)
            return
    assert old in p.text, (old[:60], p.text[:80])
    fmts = {r._r.rPr.xml if r._r.rPr is not None else "" for r in p.runs}
    assert len(fmts) == 1, "runs differ in formatting"
    p.runs[0].text = p.text.replace(old, new, 1)
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)


def set_text(p, text):
    replace(p, p.text, text)


def set_cell(cell, text):
    for extra in cell.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    runs = cell.paragraphs[0].runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def table_with(first):
    hits = [t for t in doc.tables if any(r.cells[0].text.strip() == first for r in t.rows)]
    assert len(hits) == 1, (first, len(hits))
    return hits[0]


def row(t, first):
    hits = [r for r in t.rows if r.cells[0].text.strip() == first]
    assert len(hits) == 1, (first, len(hits))
    return hits[0]


def strip_bookmarks(el):
    for tag in ("w:bookmarkStart", "w:bookmarkEnd"):
        for b in list(el.iter(qn(tag))):
            b.getparent().remove(b)
    return el


NEXT_BM = [max(int(b.get(qn("w:id"))) for b in doc.element.body.iter(qn("w:bookmarkStart"))) + 1]


def add_bookmark(p_el, name):
    bid = str(NEXT_BM[0])
    NEXT_BM[0] += 1
    start, end = OxmlElement("w:bookmarkStart"), OxmlElement("w:bookmarkEnd")
    start.set(qn("w:id"), bid)
    start.set(qn("w:name"), name)
    end.set(qn("w:id"), bid)
    pPr = p_el.find(qn("w:pPr"))
    (pPr.addnext(start) if pPr is not None else p_el.insert(0, start))
    p_el.append(end)


def clone_para(template, text):
    el = strip_bookmarks(copy.deepcopy(template._p))
    p = Paragraph(el, template._parent)
    if text is not None:
        set_text(p, text)
    return p


def caption(template, text, bookmark):
    p = clone_para(template, text)
    add_bookmark(p._p, bookmark)
    return p


def list_entry(template, text, bookmark):
    """Copy of a List of Figures / Tables entry: text before the tab, PAGEREF to `bookmark`."""
    el = copy.deepcopy(template._p)
    runs = el.findall(qn("w:r"))
    tab_i = next(i for i, r in enumerate(runs) if r.find(qn("w:tab")) is not None)
    text_runs = [r for r in runs[:tab_i] if r.find(qn("w:t")) is not None]
    text_runs[0].find(qn("w:t")).text = text
    for r in text_runs[1:]:
        el.remove(r)
    for it in el.iter(qn("w:instrText")):
        it.text = f" PAGEREF {bookmark} \\h "
    return el


def rename_bookmark(old, new):
    hits = [b for b in doc.element.body.iter(qn("w:bookmarkStart")) if b.get(qn("w:name")) == old]
    assert len(hits) == 1, old
    hits[0].set(qn("w:name"), new)
    for it in doc.element.body.iter(qn("w:instrText")):
        if f"PAGEREF {old} " in it.text:
            it.text = it.text.replace(old, new)


def fill_table(tbl_el, rows):
    """rows[0] = header. Grows or shrinks the copied table to len(rows) rows (copies of its last row)."""
    t = Table(tbl_el, doc)
    trs = tbl_el.findall(qn("w:tr"))
    while len(trs) < len(rows):
        trs[-1].addnext(copy.deepcopy(trs[-1]))
        trs = tbl_el.findall(qn("w:tr"))
    for tr in trs[len(rows):]:
        tbl_el.remove(tr)
    for r, values in zip(Table(tbl_el, doc).rows, rows):
        assert len(r.cells) == len(values), (len(r.cells), values)
        for c, v in zip(r.cells, values):
            set_cell(c, v)
    return t


def insert_before(anchor_el, *els):
    for e in els:
        anchor_el.addprevious(e._p if isinstance(e, Paragraph) else e)


def picture_para(template, image_path):
    """Copy of a picture paragraph with its own image part (the copy must not share the original's image)."""
    el = strip_bookmarks(copy.deepcopy(template._p))
    rid, _ = doc.part.get_or_add_image(str(image_path))
    for blip in el.iter(qn("a:blip")):
        blip.set(qn("r:embed"), rid)
    ids = [int(d.get("id")) for d in doc.element.body.iter(qn("wp:docPr"))]
    for d in el.iter(qn("wp:docPr")):
        d.set("id", str(max(ids) + 1))
        d.set("name", "Figure 5.5")
    return el


if any(r.cells[0].text.strip() == "TC-18" for r in table_with("TC-14").rows):
    sys.exit("TC-18 already present - this update has been applied.")
shutil.copy2(F, BACKUP)

# templates
T_BODY = para("Before the fix, PPO-300k appeared to perform very well")
T_H3 = para("5.4.7\tDeviations from the Proposed Design")
T_TCAP = para("Table 5.9: Results of the five fixed-reward PPO-300k runs")
T_FCAP = para("Figure 5.4: Cumulative return over the full test period")
fig54_i = next(i for i, p in enumerate(P) if p.text.startswith("Figure 5.4: Cumulative return over the full"))
T_PIC = P[fig54_i - 1]
assert T_PIC._p.findall(".//" + qn("a:blip")), "picture paragraph not found above Figure 5.4"
T_SPACER = P[fig54_i - 2]
assert T_SPACER.text == "" and not T_SPACER._p.findall(".//" + qn("a:blip"))


def spacer():
    return Paragraph(strip_bookmarks(copy.deepcopy(T_SPACER._p)), T_SPACER._parent)


# ── renumber the objectives table 5.10 -> 5.12 first (the new Table 5.10 reuses the name _LoT_Tab_5_10) ──
replace(para("Table 5.10 evaluates each specific objective"), "Table 5.10", "Table 5.12")
replace(para("Table 5.10: Evaluation of the specific objectives"), "Table 5.10", "Table 5.12")
rename_bookmark("_LoT_Tab_5_10", "_LoT_Tab_5_12")
replace(para("Table 5.10: Evaluation of the Specific Objectives"), "Table 5.10", "Table 5.12")


# ── Abstract ────────────────────────────────────────────────────────────────
replace(para("The pipeline ingests daily Henry Hub natural gas"),
        "The Apache Airflow pipeline ran end to end in a local Docker Compose stack; Terraform and GitHub Actions "
        "exist as code only.",
        "A quality gate on validation data decides which models enter the MLflow Model Registry, and the approved "
        "model is paper traded (simulated) on newly published prices, monitored and, if its performance degrades, "
        "retrained. The Apache Airflow pipeline, including this loop, ran end to end in a local Docker Compose "
        "stack; the AWS infrastructure was validated but never deployed.")
replace(para("Testing revealed that the original reward contained only transaction costs"),
        "The results hold for one market and one simulated test period.",
        "In 96 trading days of paper trading on prices published after its data ended, the approved agent returned "
        "+2.17 % against -0.83 % for Buy & Hold, too short a period to confirm the advantage. The results hold for "
        "one market and simulated trading only.")

# ── Chapter 4: requirements and diagrams ────────────────────────────────────
fr = table_with("FR-14")
anchor = row(fr, "FR-14")._tr
for vals in [
    ("FR-15", "The system shall apply a quality gate to a candidate model using the validation split only: the "
              "candidate passes if its return is higher than Buy & Hold's over the same days, it beats a configured "
              "share (80 %) of seeded Random Agents and its maximum drawdown stays above a configured floor "
              "(-40 %); the decision and its reasons shall be logged to the candidate's MLflow run.",
     "High", "Objective iv", "Unit test"),
    ("FR-16", "The system shall register models that pass the gate in the MLflow Model Registry as "
              "energy_trading_ppo, mark the approved model with the alias champion and keep a history of every "
              "gate decision and promotion.", "High", "Objective iv", "Unit test; inspection"),
    ("FR-17", "The system shall paper trade the champion (simulated, with no real money) on newly published prices "
              "using only data up to each day, compare it with Buy & Hold and assess its health against "
              "configured thresholds.", "High", "Objective iv", "Unit test; execution"),
    ("FR-18", "When the champion breaches a health threshold, the system shall retrain challengers with fixed seeds "
              "and replace the champion only with a challenger that passes the gate and beats the champion on the "
              "same validation window.", "Medium", "Objective iv", "Unit test; execution")]:
    tr = copy.deepcopy(anchor)
    anchor.addnext(tr)
    anchor = tr
    for c, v in zip(_Row(tr, fr).cells, vals):
        set_cell(c, v)

rtm = table_with("iv. MLflow and orchestration")
r = row(rtm, "iv. MLflow and orchestration")
set_cell(r.cells[1], "FR-10, FR-12, FR-13, FR-15 to FR-18, NFR-01, NFR-05")
set_cell(r.cells[2], "Figures 4.1, 4.3, 4.4, 4.5")
set_cell(r.cells[3], "train_drl.py, dag.py, drl_tasks.py, quality_gate.py, registry.py, paper_trade.py, monitor.py, "
                     "retrain.py")
set_cell(r.cells[4], "TC-09, TC-13, TC-14, TC-17 to TC-22")

replace(para("The use case diagram in Figure 4.1 identifies the actors"),
        "The diagram addresses requirements FR-01 to FR-14.", "The diagram addresses requirements FR-01 to FR-18.")
p = para("The Airflow Scheduler is associated with the four automated use cases")
replace(p, "The Airflow Scheduler is associated with the four automated use cases (ingest, transform, train and "
           "backtest), which shows that the routine pipeline needs no human trigger.",
        "The Airflow Scheduler is associated with the six automated use cases (ingest, transform, train, "
        "backtest, gate and paper trade), which shows that the routine pipeline, including the closed loop that "
        "gates, promotes and monitors models, needs no human trigger. The ML Engineer is associated with the gate "
        "because its thresholds and the champion rule are set in drl.yaml.")
replace(p, "The Energy Analyst interacts only with results,",
        "The Energy Analyst interacts only with results, including the paper-trading results of the champion,")

replace(para("The class diagram in Figure 4.2 shows the main classes and modules"),
        "The diagram addresses FR-01 to FR-12.",
        "The diagram addresses FR-01 to FR-12. The closed-loop modules added in November 2026 (quality_gate, "
        "registry, paper_trade, monitor and retrain, called from the DAG through closed_loop_tasks) appear in the "
        "DAG's task list and are described in Section 5.4.8; they are not drawn as separate classes so that the "
        "diagram stays legible at page width.")

replace(para("Figure 4.3 models the data entities kept in the S3 data lake"),
        "Figure 4.3 models the data entities kept in the S3 data lake and the MLflow tracking store, with their keys "
        "and the cardinality of each relationship. It addresses DR-02, DR-05 and DR-07.",
        "Figure 4.3 models the data entities kept in the S3 data lake, the MLflow tracking store and the MLflow "
        "Model Registry, with their keys and the cardinality of each relationship. It addresses DR-02, DR-05, "
        "DR-07 and FR-16.")
p = para("Every dataset is keyed by a UTC business-day timestamp")
set_text(p, p.text + " In the registry, one registered model, energy_trading_ppo, has many versions, each created "
                     "from the model files of one run, and an alias such as champion points to exactly one version, "
                     "which is how the pipeline finds the approved model without a file path.")

replace(para("Figure 4.4 shows the physical schema of the two stores"),
        "the tables of the MLflow tracking database and the external tables of the AWS Glue Data Catalog. It "
        "addresses FR-10, DR-05 and DR-07.",
        "the tables of the MLflow tracking and model registry database and the external tables of the AWS Glue "
        "Data Catalog. It addresses FR-10, FR-16, DR-05 and DR-07.")
p = para("The MLflow tables use composite primary keys")
set_text(p, p.text + " The registry tables follow the same pattern: a model version is keyed by the model name and "
                     "version number and refers to the run it came from, and an alias row maps a name such as "
                     "champion to one version. The closed loop used the file store in ./mlruns, which holds the "
                     "same registry entities as files.")

set_text(para("The shaded boxes trace the path that produced the reported results"),
         "The shaded boxes trace the path that produced the reported results: EIA data was fetched and transformed, "
         "the PPO agent was trained in the trading environment with MLflow tracking, and the trained agent was "
         "evaluated in the backtest report and the replay application, all through the local run scripts with data "
         "passed in memory. The model lifecycle row was added in November 2026 and was also used for reported "
         "results: the quality gate chose and registered the champion, which was then paper traded and monitored "
         "locally (Section 5.6.5); retraining ran only as a smoke test in the Docker Compose stack. The white boxes "
         "ran only in the local Docker Compose stack or in GitHub Actions: the Airflow DAG ingested and transformed "
         "the EIA data into S3 zones emulated by LocalStack, trained and backtested an agent with an MLflow server "
         "and sent its notification to a local Mailpit inbox (TC-14), and from November 2026 it also ran the "
         "closed loop (TC-22); GitHub Actions runs on every push, where the Terraform validation passes (TC-16) but "
         "the formatting check fails. The dashed components (the Glue crawlers and the AWS services) are defined in "
         "code but were not provisioned, so no part of the system is publicly deployed. Because the pipeline logic "
         "lives in modules that both the DAG and the local scripts call, the same code runs under Airflow and in "
         "the local scripts.")

# ── Chapter 5: introduction and environment ─────────────────────────────────
p = para("This chapter explains how the design in Chapter 4 was implemented")
replace(p, "Section 5.4 describes the implemented modules and records deviations from the proposed design.",
        "Section 5.4 describes the implemented modules, including the closed loop added in November 2026, and "
        "records deviations from the proposed design.")
replace(p, "Section 5.6 presents the test strategy, test cases, defects and the evaluation of the trained agents.",
        "Section 5.6 presents the test strategy, test cases, defects and the evaluation of the trained agents and "
        "of the closed loop.")
p = para("All results in this chapter come from recorded evidence")
set_text(p, p.text + " The quality gate, the model registry and the paper trading added in November 2026 are "
                     "evidenced by the MLflow Model Registry and the files in reports/live/.")
replace(para("The solution was developed, trained and tested on a single developer workstation."),
        "The Docker Compose stack was used only to run the Airflow DAG end to end,",
        "The Docker Compose stack was used only to run the Airflow DAG end to end, including the closed loop,")

t51 = table_with("Storage used")
r = row(t51, "Storage used")
set_cell(r.cells[1], "69 MB MLflow store, 158 MB checkpoints, 2.8 MB reports")
set_cell(r.cells[2], "Measured on disk on 23 November 2026, after the closed-loop runs.")

replace(para("Table 5.2 lists the software versions that were actually used."),
        "Several differ from the versions pinned in requirements.txt; this is recorded as defect DEF-07 in "
        "Section 5.6.3.",
        "requirements.txt originally pinned older versions (DEF-07, Section 5.6.3); in November 2026 it and the "
        "Airflow image were updated to these versions, so the DAG now runs with the same libraries as the "
        "workstation.")
t52 = table_with("pandas / NumPy / pyarrow")
set_cell(row(t52, "MLflow").cells[2], "Experiment tracking and model registry (local file store; server in Docker "
                                      "Compose for the DAG)")
set_cell(row(t52, "Apache Airflow").cells[2], "Orchestration; runs in the Docker Compose stack (Dockerfile.airflow, "
                                              "with the library versions above), not installed on the workstation")

replace(para("ii.  Run pytest tests/ to confirm the environment"), "(67 tests are expected to pass)",
        "(102 tests are expected to pass)")
vi = para("vi.  Run mlflow ui --backend-store-uri ./mlruns to inspect the runs.")
vi._p.addnext(clone_para(vi, "vii.  Run python apply_gate_to_seeds.py gate and then python apply_gate_to_seeds.py "
                             "register to apply the quality gate to the five fixed-reward runs and register the "
                             "champion, then python -m src.ml.drl.paper_trade and python -m src.ml.drl.monitor to "
                             "update the paper-trading ledger and the health check in reports/live/.")._p)

# ── 5.4: orchestration text, deviations, new Section 5.4.8 ───────────────────
p = para("The Airflow DAG (src/pipeline/dag.py")
replace(p, "For offline inspection, trading_replay_app.py provides",
        "In November 2026 the DAG was extended with the closed loop described in Section 5.4.8. For offline "
        "inspection, trading_replay_app.py provides")
t54 = table_with("Quality gate and registry")
r = row(t54, "Quality gate and registry")
set_cell(r.cells[2], "Validation-only gate (beats Buy & Hold and 80 % of 30 seeded Random Agents, drawdown above "
                     "-40 %); MLflow Model Registry with the alias champion; added in November 2026")
set_cell(r.cells[3], "A mean reward above zero could not be passed under the cost-only reward and ignores the "
                     "baselines (Section 5.4.8)")
r = row(t54, "Orchestration")
set_cell(r.cells[2], "Data-to-backtest tasks plus a closed loop that branches on the champion's health and on the "
                     "retrain outcome; EmailOperator to a local mail catcher; run in Docker Compose")
set_cell(r.cells[3], "SNS not provisioned; gate added after the evaluation")

h55 = para("5.5\tModel Training and Selection")
insert_before(h55._p,
    clone_para(T_H3, "5.4.8\tClosed Loop: Quality Gate, Model Registry, Paper Trading and Retraining"),
    clone_para(T_BODY,
        "The quality gate and model registry proposed in Chapter 3 were implemented in November 2026, after the "
        "evaluation in Section 5.6.4, together with paper trading, monitoring and retraining, which close the loop "
        "from a deployed model back to training (FR-15 to FR-18). quality_gate.py judges a candidate on the "
        "validation split only. It passes if its validation return is higher than Buy & Hold's over the same "
        "days, so that it adds something to simply holding gas; if it beats at least 80 % of 30 seeded Random "
        "Agents, so that its result is unlikely to be luck; and if its maximum drawdown stays above -40 %, ten "
        "points before the 50 % stop-loss, so that it is not close to ruin. The thresholds are set in "
        "config/drl.yaml and were fixed before the gate was run on any model. The decision and the reason for each "
        "check are logged as tags and metrics on the candidate's MLflow run. registry.py registers passing models "
        "as versions of energy_trading_ppo in the MLflow Model Registry and gives the approved model the alias "
        "champion, using aliases rather than the deprecated registry stages. Among the candidates that pass, the "
        "champion is the one with the highest validation return; test results are never used."),
    clone_para(T_BODY,
        "paper_trade.py runs the champion, unchanged, day by day on EIA prices published after its data ended and "
        "compares it with Buy & Hold on a simulated $100,000 account. No real money or orders are involved. It "
        "drives the trading environment's own code, so the costs, the position limit and the stop-loss are those "
        "of the backtest. For each day only the rows dated up to that day are used, and the outlier and "
        "normalisation statistics are frozen from the model's training split and stored with the model in MLflow, "
        "which removes on this path the look-ahead recorded as DEF-08. The daily ledger is append-only. "
        "monitor.py then compares the champion with Buy & Hold since it became active and over the last 60 trading "
        "days, and measures how far its account is below its peak. No alarm is raised before 20 trading days, "
        "because a few days of gas prices are mostly noise; a gap below -10 percentage points in either "
        "comparison, or a drawdown from peak below -30 %, marks the champion as needing retraining, and a retrain "
        "that fails is not repeated for 20 trading days."),
    clone_para(T_BODY,
        "retrain.py trains challengers with the PPO-300k settings and the fixed seeds 42, 1 and 2 on a walk-forward "
        "split in which the most recent 252 trading days are the validation window, so that new data is used for "
        "training and for judging. Each challenger's final model, rather than the checkpoint selected on that "
        "window, must pass the gate and must beat the champion on the same window before it replaces it, and every "
        "attempt is recorded in the registry history. In the DAG, paper_trade and monitor are followed by a branch "
        "that either skips retraining or trains one challenger per seed, applies the gate, promotes and sends one "
        "of four emails: retrain required, champion replaced, retrain failed the gate or champion kept. A join "
        "task lets a skipped branch through but not a failed one, so the end task keeps the ALL_SUCCESS rule "
        "introduced for DEF-12, and all training tasks share a one-slot pool (DEF-15)."))

p = para("Eleven training runs were logged to MLflow (Table 5.5).")
set_text(p, p.text + " The retraining runs of the closed loop (Section 5.6.5) are logged in a separate experiment, "
                     "energy-drl-eia-gas-retrain, and are not part of this table.")

# ── 5.6: tests, defects ─────────────────────────────────────────────────────
p = para("Testing combined four levels.")
replace(p, "Every test was executed once on the workstation in Table 5.1, except where stated.",
        "The closed loop added in November 2026 was tested in the same way (TC-18 to TC-22). Every test was "
        "executed once on the workstation in Table 5.1, except where stated.")
set_text(para("The automated suite contains 67 unit tests"),
         "The automated suite contains 102 unit tests: 8 for ingestion, 26 for transformation, 31 for the "
         "environment, training and backtest code, 4 for the pipeline configuration and 33 for the closed loop "
         "(15 for the quality gate and promotion rules, 11 for the monitor, 4 for paper trading and 3 for "
         "promotion against a temporary registry), including the regression tests added with the DEF-01, DEF-09, "
         "DEF-12 to DEF-15 fixes. All 102 passed in 134 seconds, with 66 % statement coverage of src/. Coverage was "
         "highest for the registry (99 %), the quality gate (97 %), the environment (93 %) and retraining (90 %), "
         "lower for paper trading (53 %) and the monitor (62 %), whose live-data entry points were exercised by the "
         "DAG runs instead, and zero for the DAG and the Airflow task wrappers, which need Airflow and were "
         "exercised by the DAG runs (TC-14, TC-22). Table 5.6 groups the unit tests into test cases and adds the "
         "integration, system and configuration tests.")

tests = table_with("TC-14")
set_cell(row(tests, "TC-16").cells[3],
         "Not completed on the workstation: the AWS provider plugin failed to start. Retest in GitHub Actions on "
         "23 November 2026: terraform init and validate succeeded ('Success! The configuration is valid'). Pass")
anchor = row(tests, "TC-17")._tr
for vals in [
    ("TC-18", "FR-15",
     "8 tests: the gate passes a candidate that meets all three checks and fails one that misses any of them, at "
     "the exact boundaries (a return equal to Buy & Hold, 80 % of Random Agents, a drawdown of -40 %); an "
     "always-long agent matches Buy & Hold; the frozen normaliser ignores the evaluation window.",
     "8/8 passed. Applied to the five fixed-reward runs: all passed (Table 5.10). Pass"),
    ("TC-19", "FR-17",
     "11 tests: monitor status for healthy, watch, each breached threshold, the 20-day warm-up and the 60-day "
     "window; the retrain trigger, including the cool-down after a failed retrain and a forced drill.",
     "11/11 passed. Pass"),
    ("TC-20", "FR-17, DR-04",
     "4 tests: features for day t are unchanged when later rows are altered; decisions up to day t are unchanged; "
     "the ledger continues identically from its saved state; paper-trading values equal the backtest's.",
     "4/4 passed; values equal to the cent. Pass"),
    ("TC-21", "FR-16, FR-18",
     "10 tests: champion rule (highest validation return among passing runs, ties to the smaller drawdown); a "
     "challenger replaces the champion only if it passed the gate and is strictly better; two retraining rounds "
     "against a temporary registry move the alias consistently and keep the history; a smoke test never "
     "registers or promotes.",
     "10/10 passed. Pass"),
    ("TC-22", "FR-12, FR-18",
     "Run the closed loop in the Docker Compose DAG: a normal run takes the skip branch; a forced retrain trains "
     "the challengers, applies the gate and sends the matching email; 2 configuration tests check the DAG "
     "structure and the training pool.",
     "Full-size drill stopped after 45 min: four trainings in parallel ran about 20 times slower (DEF-15) and one "
     "task failed once on an experiment-creation race (DEF-14). Fail. Retest after both fixes: the scheduled run "
     "of 22 November succeeded on the skip branch; a smoke test (4,096 steps, promotion disabled) ran all 23 "
     "tasks, the three challengers failed the gate, the champion was kept and the email arrived; 2/2 "
     "configuration tests passed. Pass")]:
    tr = copy.deepcopy(anchor)
    anchor.addnext(tr)
    anchor = tr
    for c, v in zip(_Row(tr, tests).cells, vals):
        set_cell(c, v)

defects = table_with("DEF-12")
set_cell(row(defects, "DEF-07").cells[3], "Fixed: requirements.txt and the Airflow image pin the versions in "
                                          "Table 5.2; the closed loop ran with them in Docker Compose.")
set_cell(row(defects, "DEF-08").cells[3], "Mitigated: the quality gate, paper trading and retraining fit the "
                                          "normaliser and outlier statistics on the training split only (TC-20); "
                                          "the Chapter 5 evaluation scripts keep the original behaviour so that "
                                          "their results stay reproducible.")
anchor = row(defects, "DEF-13")._tr
for vals in [
    ("DEF-14", "Medium",
     "The three challenger tasks of a retrain started at the same time and each created the MLflow experiment, "
     "giving two experiments with the same name; one task's run fell back to the Default experiment, whose "
     "artefact path was not writable in the container, and failed.",
     "Fixed: prepare_retrain creates the experiment once before the parallel tasks; regression test added; the "
     "duplicate experiment was merged."),
    ("DEF-15", "Medium",
     "Four PPO training tasks ran in parallel in the Docker Compose stack and oversubscribed the six-core CPU; "
     "each ran about 20 times slower and would have exceeded the two-hour timeout, so the full-size retraining "
     "drill was stopped.",
     "Fixed: all training tasks use a one-slot Airflow pool; regression test added; smoke test passed (TC-22).")]:
    tr = copy.deepcopy(anchor)
    anchor.addnext(tr)
    anchor = tr
    for c, v in zip(_Row(tr, defects).cells, vals):
        set_cell(c, v)

# ── new Section 5.6.5 with Table 5.10, Figure 5.5 and Table 5.11 ─────────────
gate_tbl = copy.deepcopy(table_with("Mean ± SD")._tbl)
fill_table(gate_tbl, [
    ["Seed", "Validation return", "Random Agents beaten", "Max drawdown (validation)", "Gate",
     "Test, full period (not used)"],
    ["42", "+56.10 %", "30 of 30", "-28.03 %", "Pass", "+124.36 %"],
    ["1", "+82.91 %", "30 of 30", "-24.75 %", "Pass", "+170.79 %"],
    ["2", "+44.90 %", "28 of 30", "-32.50 %", "Pass", "+121.93 %"],
    ["3", "+57.13 %", "30 of 30", "-29.11 %", "Pass", "+190.91 %"],
    ["4", "+85.10 %", "30 of 30", "-17.89 %", "Pass", "+67.02 %"],
    ["Buy & Hold", "+27.20 %", "-", "-39.61 %", "-", "+8.68 %"]])
paper_tbl = copy.deepcopy(table_with("iv. MLflow tracking, registry, quality gate and daily Airflow DAG")._tbl)
fill_table(paper_tbl, [
    ["Measure", "Paper trading, 7 July to 17 November 2026", "Monitoring threshold"],
    ["Trading days", "96 (the champion held a position on every day)", "No alarm before 20 days"],
    ["Champion return", "+2.17 % ($102,174)", "-"],
    ["Buy & Hold return", "-0.83 % ($99,172)", "-"],
    ["Gap since deployment", "+3.00 percentage points", "Retrain below -10 points"],
    ["Gap over the last 60 trading days", "+1.89 percentage points", "Retrain below -10 points"],
    ["Drawdown from peak (on 17 November)", "-3.36 %", "Retrain below -30 %"],
    ["Maximum drawdown", "-15.15 % (Buy & Hold -21.12 %)", "-"],
    ["Health", "Healthy; no retrain triggered", "-"]])

h57 = para("5.7\tVersion Control and Implementation Evidence")
insert_before(h57._p,
    clone_para(T_H3, "5.6.5\tQuality Gate, Paper Trading and Monitoring Results"),
    clone_para(T_BODY,
        "The quality gate was applied retroactively to the five fixed-reward PPO-300k runs, using each run's best "
        "checkpoint, the model evaluated in Section 5.6.4, on the whole validation period of 413 days (4 May 2023 "
        "to 2 December 2024), over which Buy & Hold returned +27.20 %. Table 5.10 gives the result. The "
        "validation returns differ from those in Table 5.9 because the gate uses the whole validation period "
        "rather than its first 252 days, and the normaliser fitted on the training split rather than on the "
        "validation data. The test returns are shown for comparison only; they played no part in the decision."),
    caption(T_TCAP, "Table 5.10: Quality-gate results of the five fixed-reward runs on the validation period",
            "_LoT_Tab_5_10"),
    gate_tbl,
    spacer(),
    clone_para(T_BODY,
        "All five runs passed, so the gate excluded no model. It shows that each run beat Buy & Hold and almost "
        "every random agent on data it was not trained on, but it did not separate good runs from weaker ones. "
        "All five were registered as versions 1 to 5 of energy_trading_ppo, and seed 4 (version 5) became the "
        "champion with the highest validation return, +85.10 % against +82.91 % for seed 1. This figure is a "
        "selection result and is optimistic by construction, because the checkpoint and the champion were both "
        "chosen on this period; it is not an estimate of future performance. Validation ranking also did not "
        "predict test ranking: seed 4 had the lowest full-test return of the five, and the rank correlation "
        "between validation and test return was about -0.1."),
    Paragraph(picture_para(T_PIC, HERE / "figures" / "fig5_5_paper_trading.png"), T_PIC._parent),
    caption(T_FCAP, "Figure 5.5: Paper trading (simulated) of the champion against Buy & Hold from 7 July to 17 "
                    "November 2026, with the Henry Hub price and the champion's position", "_LoF_Fig_5_5"),
    clone_para(T_BODY,
        "The champion was then paper traded from 7 July 2026, the first day after its data ended, to 17 November "
        "2026, the last price published when the loop was run: 96 trading days that no model had seen (Figure "
        "5.5 and Table 5.11). It returned +2.17 % against -0.83 % for Buy & Hold. It held a position on every day "
        "and a full one on 28 days, so its account followed the price closely; most of its lead was gained in the "
        "price fall of late October, when it held a smaller position on several days (12,000 MMBtu on 22 October, "
        "for example, against 25,000 for Buy & Hold), and its maximum drawdown was -15.15 % against -21.12 %. The "
        "ledger was produced on 23 November 2026 by replaying the days in order. Each decision used only data up "
        "to its day (TC-20), and the champion was chosen with data that ended in December 2024, so no information "
        "from the paper-trading period influenced the decisions, but they were not made in real time."),
    caption(T_TCAP, "Table 5.11: Paper-trading (simulated) results and health of the champion", "_LoT_Tab_5_11"),
    paper_tbl,
    spacer(),
    clone_para(T_BODY,
        "The monitor rated the champion healthy: every measure was well inside its threshold. Ninety-six days and "
        "a lead of three percentage points in a flat market are too little to show that the champion is better "
        "than Buy & Hold. What the paper trading shows is that the loop works on genuinely new data without "
        "look-ahead and that the champion has not degraded enough to need retraining. In the Docker Compose stack, "
        "the scheduled run of 22 November took the skip branch and succeeded. A full-size retraining drill was "
        "stopped after 45 minutes, because four PPO trainings ran in parallel on the six-core workstation and each "
        "ran about 20 times slower than alone (DEF-15); one challenger task had also failed once because two tasks "
        "created the MLflow experiment at the same time (DEF-14). After both were fixed, a smoke test with a "
        "4,096-step budget and promotion disabled ran the retrain branch end to end: the three undertrained "
        "challengers failed the gate on the window from 1 December 2025 to 17 November 2026, over which Buy & "
        "Hold lost 50.05 % and the champion returned +52.65 %, the champion was kept, and the 'retrain failed the "
        "quality gate' email arrived (TC-22). No full-size retrain has yet replaced the champion."))

# ── 5.7 and 5.8 ─────────────────────────────────────────────────────────────
replace(para("The repository contains the source code, tests, configuration files"),
        "and the MLflow tracking store (mlruns/) from which the Chapter 5 results were reproduced.",
        ", the MLflow tracking store and model registry (mlruns/) from which the Chapter 5 results were reproduced, "
        "and the closed-loop records in reports/live/ (registry history, paper-trading ledger, monitor status and "
        "the evidence of the Docker runs), from which a project dashboard is built and published with GitHub "
        "Pages.")

obj = table_with("iv. MLflow tracking, registry, quality gate and daily Airflow DAG")
r = row(obj, "iv. MLflow tracking, registry, quality gate and daily Airflow DAG")
set_cell(r.cells[1], "All runs tracked and reproducible (TC-09); quality gate and MLflow Model Registry with a "
                     "champion alias (TC-18, TC-21); champion paper traded and monitored (TC-19, TC-20); DAG ran "
                     "end to end in Docker Compose, including the closed loop, and alerts on failure (TC-14, TC-17, "
                     "TC-22).")
set_cell(r.cells[2], "Achieved (local deployment)")
r = row(obj, "v. Test and validate on held-out data, integration, cloud and CI/CD")
set_cell(r.cells[1], "102 unit tests passed; held-out backtests reproduced; integration and system tests run, "
                     "including the DAG (TC-14, TC-22); Terraform validated in CI (TC-16); cloud deployment and the "
                     "deployment workflow not run.")

p = para("Research question iii can now be answered for this market and period.")
p._p.addnext(clone_para(T_BODY,
    "Research question iv can also be answered more fully. With the quality gate, the registry and the monitor, "
    "MLflow no longer only records runs: it holds the evidence behind each promotion decision, and the approved "
    "model is identified by an alias rather than by a file path. The first results also show the gate's limits. "
    "All five runs passed, so it acted as a minimum standard rather than as a way to choose between good models, "
    "and the run it ranked first had the weakest test result. The 96 days of paper trading are the first "
    "evidence on data that no model or decision had seen, and so far they show a small lead over Buy & Hold "
    "(+2.17 % against -0.83 %) rather than the large margins of the test period.")._p)

set_text(para("v.  The Airflow DAG was run end to end only twice"),
         "v.  The Airflow DAG, including the closed loop, ran end to end only in a local Docker Compose stack. The "
         "retraining branch ran there only as a smoke test with undertrained challengers, so no full-size retrain "
         "has been through the gate, and the AWS infrastructure was validated (TC-16) but never deployed, so the "
         "cloud requirements are supported by configuration evidence only.")
vi = para("vi.  The idle and drawdown penalties proposed in Chapter 3 were not implemented")
vi._p.addnext(clone_para(vi,
    "vii.  The paper-trading period covers only 96 trading days in a flat market, and it was produced by replaying "
    "the days in order on 23 November 2026, after the champion had been chosen. Each decision used only data up to "
    "its day and the choice used no data after December 2024, but the decisions were not made in real time.")._p)

# ── Chapter 6 ───────────────────────────────────────────────────────────────
replace(para("The project set out to address three problems identified in Chapter 1"),
        "The Airflow DAG has been run end to end in a local Docker Compose stack, while the cloud and CI/CD "
        "components exist as code only.",
        "Since November 2026 it also gates, registers, paper trades and monitors its models and can retrain them, "
        "and the Airflow DAG, including this closed loop, has been run end to end in a local Docker Compose stack, "
        "while the cloud deployment exists as code only.")

p = para("Objective iv was partially achieved.")
replace(p, "Objective iv was partially achieved.", "Objective iv was achieved in a local deployment.")
replace(p, "The model registry and the quality gate were not implemented. The Airflow DAG ran end to end in a local "
           "Docker Compose stack (TC-14): it ingested and transformed the live EIA data, trained an agent with an "
           "MLflow server, backtested it and emailed the MLflow run ID and the backtest summary. Research question "
           "iv is therefore answered in part: MLflow tracking has been shown to work inside a scheduled Airflow "
           "pipeline that trains the evaluated PPO-300k configuration and sends an alert when a task fails (TC-17), "
           "but without the registry and the quality gate the pipeline does not yet manage the model lifecycle.",
        "The quality gate and the model registry were implemented in November 2026: candidates are judged on the "
        "validation period against Buy & Hold, 30 seeded Random Agents and a drawdown floor, passing models are "
        "registered as versions of energy_trading_ppo, and the approved model carries the alias champion (TC-18, "
        "TC-21). All five fixed-reward runs passed, and seed 4 became the champion. The Airflow DAG ran end to end "
        "in a local Docker Compose stack (TC-14): it ingested and transformed the live EIA data, trained an agent "
        "with an MLflow server, backtested it, emailed the results and alerts when a task fails (TC-17), and with "
        "the closed loop it paper trades the champion, checks its health and branches to retraining when a "
        "threshold is breached (TC-22). Research question iv is therefore answered for a local deployment: MLflow "
        "tracking, the registry and the gate manage the model lifecycle inside a scheduled pipeline. Two limits "
        "remain: the retraining branch has run only as a smoke test, so no full-size challenger has yet been "
        "through the gate, and the gate passed every run, so it has acted as a minimum standard rather than as a "
        "way to choose between good models.")

p = para("Objective v was partially achieved.")
replace(p, "All 67 unit tests pass with 62 % statement coverage", "All 102 unit tests pass with 66 % statement "
                                                              "coverage")
replace(p, "Thirteen defects were recorded in total, of which eight have been fixed and one has been worked around.",
        "Fifteen defects were recorded in total, of which eleven have been fixed and two (DEF-02 and DEF-08) have "
        "been mitigated; running the closed loop in Docker found two of them (DEF-14 and DEF-15), which the unit "
        "tests could not.")
replace(p, "The Terraform configuration and the CI/CD workflows were not run successfully, so the cloud and "
           "deployment requirements rest on configuration evidence only.",
        "The Terraform configuration now validates in GitHub Actions (TC-16), but it has never been applied and the "
        "deployment workflow has not run, so the cloud requirements still rest on configuration evidence only.")

replace(para("Taken together, the aim was partially met."),
        "and have run together under Airflow in a local container stack;",
        "and have run together under Airflow in a local container stack; the models are gated, registered, paper "
        "traded and monitored in a closed loop;")

set_text(para("iv.  Pin the software versions that were actually used"),
         "iv.  Keep the pinned software versions in step with the environment (system maintainers). requirements.txt "
         "and the Airflow image now pin the versions in Table 5.2 (DEF-07 fixed), and the closed loop ran with them "
         "in Docker Compose. Any upgrade should be made in both places and followed by the full test suite and a "
         "re-evaluation, because reproducibility (NFR-01) depends on the software environment as well as on the "
         "MLflow records.")
set_text(para("v.  Execute the deployment code before relying on it"),
         "v.  Execute the deployment code before relying on it (system maintainers). The DAG, including the closed "
         "loop, now runs end to end in the local Docker Compose stack (TC-14, TC-22), marks a run failed when any "
         "task fails (DEF-12, TC-17) and trains the evaluated configuration by default (DEF-13). Terraform "
         "validation passes in GitHub Actions (TC-16), but the CI workflow still stops at the code-formatting check "
         "and the deployment workflow has no AWS credentials; fixing the formatting and running the deployment "
         "against a test account would supply the remaining evidence.")
set_text(para("vi.  Base the planned quality gate on validation evidence and baselines"),
         "vi.  Strengthen the quality gate (system developers). The gate recommended here has been implemented "
         "(Section 5.4.8): a candidate must beat Buy & Hold and 80 % of 30 seeded Random Agents on the validation "
         "period and keep its drawdown above -40 %, with the test period kept for reporting only. All five "
         "fixed-reward runs passed, and the run it ranked first had the lowest test return of the five, so the gate "
         "works as a minimum standard but does not yet choose well between good models. It should be extended to "
         "judge candidates on a window that was not also used to select their checkpoint, to require a margin over "
         "Buy & Hold rather than any gain, and to require consistent results across several seeds of the same "
         "configuration before a model is promoted.")
replace(para("vii.  Revise data preparation for spike-driven commodities"),
        "should be fitted on the training split only (DEF-08).",
        "should be fitted on the training split only, as the closed loop already does (DEF-08, mitigated).")
p = para("x.  No hardware change is recommended")
set_text(p, p.text + " Training runs should, however, run one at a time on this machine: four in parallel ran "
                     "about 20 times slower (DEF-15).")
set_text(para("v.  A sustained deployment study."),
         "v.  A sustained deployment study. Paper trading (simulated) began on 7 July 2026, and the closed loop now "
         "runs in the Docker Compose stack (Section 5.6.5), but 96 trading days are too few to judge the champion, "
         "and no breach has yet triggered a full-size retrain. The DAG should run daily for several months, on a "
         "machine that stays on or a funded cloud account. This would measure what the present study could not: "
         "how often scheduled runs succeed, how often the monitor triggers retraining and challengers pass the "
         "gate, whether a retrained champion does better than the one it replaced, and whether the input data "
         "drift away from the training distribution. It would be judged by the run success rate, the gate and "
         "promotion decisions over time, the paper-trading gap to Buy & Hold and a drift measure on the main "
         "features.")

# ── List of Figures / List of Tables ────────────────────────────────────────
lof_54 = para("Figure 5.4: Cumulative Return over the Full Test Period")
lof_54._p.addnext(list_entry(lof_54, "Figure 5.5: Paper Trading (Simulated) of the Champion against Buy & Hold, "
                                     "7 July to 17 November 2026", "_LoF_Fig_5_5"))
lot_59 = para("Table 5.9: Results of the Five Fixed-Reward PPO-300k Runs")
e511 = list_entry(lot_59, "Table 5.11: Paper-Trading (Simulated) Results and Health of the Champion", "_LoT_Tab_5_11")
e510 = list_entry(lot_59, "Table 5.10: Quality-Gate Results of the Five Fixed-Reward Runs on the Validation Period",
                  "_LoT_Tab_5_10")
lot_59._p.addnext(e511)
lot_59._p.addnext(e510)

doc.save(F)
print("updated", F.name, "- backup:", BACKUP.name)
