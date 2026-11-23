"""
One-shot update of the report after fixing DEF-12 (end task hid failed runs) and DEF-13 (DAG trained with the
hourly drl.yaml defaults), both verified in the Docker Compose stack on 20 November 2026:
  * manual__fault_injection_def12 (invalid total_timesteps): train_drl failed 3 times, run marked failed,
    one alert email in Mailpit  -> new TC-17
  * scheduled__2026-11-19T00:00:00+00:00: 8/8 tasks succeeded with the daily PPO-300k defaults,
    MLflow run 404abd89eb804a159b405989b4b60386, training 22 min 8 s, backtest (first 252 days of its own test
    split, 2025-04-03 .. 2026-03-23) +48.22 % vs Buy & Hold -30.24 %  -> TC-14 retest
Also: 67 unit tests (2 new regression tests in tests/test_pipeline_config.py), 91 s, coverage 62 %, env 94 %.

    python docs/report/update_def12_def13.py
Refuses to run if TC-17 already exists. Backup: docs/report/backup/energy_trading_mlops_proposal_v17_before_blank_page_fix.docx
"""
import copy
import sys
from pathlib import Path

from docx import Document
from docx.table import _Row

F = Path(__file__).resolve().parents[2] / "energy_trading_mlops_proposal_v17.docx"
doc = Document(F)
P = doc.paragraphs


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


tests = table_with("TC-14")
if any(r.cells[0].text.strip() == "TC-17" for r in tests.rows):
    sys.exit("TC-17 already present - this update has been applied.")

# ── Chapter 5 text ──────────────────────────────────────────────────────────
replace(para("ii.  Run pytest tests/ to confirm the environment"), "(65 tests are expected to pass)",
        "(67 tests are expected to pass)")

p = para("The Airflow DAG (src/pipeline/dag.py")
replace(p, "because the end task runs whatever the state of the tasks before it (DEF-12).",
        "because the end task ran whatever the state of the tasks before it (DEF-12).")
replace(p, "It used the default drl.yaml settings, which were written for hourly synthetic data, so its backtest "
           "results are not comparable with the evaluation in Section 5.6 (DEF-13).",
        "That run used the drl.yaml defaults, which were still the settings for hourly synthetic data (DEF-13). "
        "After the end task was changed to fail the run when any task fails and the defaults were set to the "
        "daily settings of the PPO-300k agents, a run with a deliberately invalid training budget was marked "
        "failed and sent an alert email (TC-17), and the next scheduled run trained the PPO-300k configuration "
        "in 22 minutes. Its backtest over the first 252 days of its own test period (3 April 2025 to 23 March "
        "2026, which overlaps the Chapter 5 test period) returned +48.22 % against -30.24 % for Buy & Hold; with "
        "one seed, one Random Agent and a later data cut, this checks the pipeline and is not part of the "
        "evaluation in Section 5.6.")

p = para("The automated suite contains 65 unit tests")
replace(p, "65 unit tests: 8 for ingestion, 26 for transformation and 31 for the environment, training and backtest "
           "code, including the four regression tests added with the DEF-01 and DEF-09 fixes. All 65 passed in "
           "58 seconds,",
        "67 unit tests: 8 for ingestion, 26 for transformation, 31 for the environment, training and backtest "
        "code and 2 for the pipeline configuration, including the six regression tests added with the DEF-01, "
        "DEF-09, DEF-12 and DEF-13 fixes. All 67 passed in 91 seconds,")
replace(p, "environment (93 %)", "environment (94 %)")

replace(para("v.  The Airflow DAG was run end to end only once"),
        "v.  The Airflow DAG was run end to end only once, in a local Docker Compose stack and with training "
        "settings that differ from those of the evaluated agents (DEF-13), and",
        "v.  The Airflow DAG was run end to end only twice, in a local Docker Compose stack, and")

# ── Chapter 6 ───────────────────────────────────────────────────────────────
p = para("Objective iv was partially achieved.")
replace(p, "The Airflow DAG ran end to end once in a local Docker Compose stack (TC-14):",
        "The Airflow DAG ran end to end in a local Docker Compose stack (TC-14):")
replace(p, "MLflow tracking has been shown to work inside a scheduled Airflow pipeline, but without the registry "
           "and the quality gate the pipeline does not yet manage the model lifecycle, and the DAG's training "
           "settings still differ from those of the evaluated agents (DEF-13).",
        "MLflow tracking has been shown to work inside a scheduled Airflow pipeline that trains the evaluated "
        "PPO-300k configuration and sends an alert when a task fails (TC-17), but without the registry and the "
        "quality gate the pipeline does not yet manage the model lifecycle.")

p = para("Objective v was partially achieved.")
replace(p, "All 65 unit tests pass", "All 67 unit tests pass")
replace(p, "of which six have been fixed and one has been worked around.",
        "of which eight have been fixed and one has been worked around.")
replace(p, "although training had failed in all of them (DEF-12).",
        "although training had failed in all of them (DEF-12); after the fix, a run with an injected training "
        "failure was marked failed (TC-17).")

p = para("v.  Fix the remaining orchestration defects")
replace(p, p.text,
        "v.  Execute the deployment code before relying on it (system maintainers). The DAG now runs end to end "
        "in the local Docker Compose stack (TC-14), marks a run failed when any task fails (DEF-12, TC-17) and "
        "trains the evaluated configuration by default (DEF-13). The GitHub Actions workflows should still be run "
        "on the repository and the Terraform validation completed on a machine where the AWS provider starts "
        "(TC-16), which would supply the remaining evidence without the cost of a paid AWS deployment.")

replace(para("x.  No hardware change is recommended"), "and a 500,000-step run inside the DAG took 36 minutes,",
        "and a 300,000-step run inside the DAG took 22 minutes,")

replace(para("v.  A sustained deployment study."),
        "Once the DAG's defects have been fixed and the quality gate implemented (Section 6.3.2),",
        "Once the quality gate has been implemented (Section 6.3.2),")

# ── Tables ──────────────────────────────────────────────────────────────────
c = row(tests, "TC-14").cells[3]
assert c.text.rstrip().endswith("email received. Pass")
set_cell(c, c.text.rstrip()[:-len("Pass")] + "Retest after the DEF-12 and DEF-13 fixes: all eight tasks "
         "succeeded with the daily PPO-300k settings; 300,000-step training in 22 min 8 s. Pass")
c = row(tests, "TC-10").cells[3]
set_cell(c, c.text.replace("; 500,000 steps inside the DAG in 35 min 45 s. Pass",
                           "; inside the DAG, 500,000 steps in 35 min 45 s and 300,000 steps in 22 min 8 s. Pass"))
anchor = row(tests, "TC-16")._tr
tr = copy.deepcopy(anchor)
anchor.addnext(tr)
for cell, v in zip(_Row(tr, tests).cells, [
        "TC-17", "FR-13, NFR-05",
        "Trigger a DAG run with an invalid training budget: train_drl is retried twice, the run is marked failed "
        "and an alert email is sent.",
        "train_drl failed three times, five minutes apart; the downstream tasks were marked upstream_failed; the "
        "run was marked failed; one alert email arrived in Mailpit. Pass"]):
    set_cell(cell, v)

defects = table_with("DEF-12")
r = row(defects, "DEF-12")
set_cell(r.cells[2], "The DAG's end task used the ALL_DONE trigger rule, so a run was marked successful even when "
                     "train_drl failed, as happened in the five July 2026 runs.")
set_cell(r.cells[3], "Fixed: the end task uses the default ALL_SUCCESS rule, so a failed task fails the run "
                     "(TC-17); regression test added.")
r = row(defects, "DEF-13")
set_cell(r.cells[2], "The DAG trained with the drl.yaml defaults, written for hourly synthetic data (720-step "
                     "episodes, a 100-unit position limit, a 30 % stop-loss and hourly annualisation), not the "
                     "settings of the evaluated agents.")
set_cell(r.cells[3], "Fixed: the defaults are the daily PPO-300k settings; regression test added; DAG rerun "
                     "(TC-14).")

t = table_with("v. Test and validate on held-out data, integration, cloud and CI/CD")
c = row(t, "v. Test and validate on held-out data, integration, cloud and CI/CD").cells[1]
set_cell(c, c.text.replace("65 unit tests passed", "67 unit tests passed"))
c = row(t, "iv. MLflow tracking, registry, quality gate and daily Airflow DAG").cells[1]
set_cell(c, c.text.replace("DAG ran end to end in Docker Compose (TC-14).",
                           "DAG ran end to end in Docker Compose and alerts on failure (TC-14, TC-17)."))

t = table_with("iv. MLflow and orchestration")
c = row(t, "iv. MLflow and orchestration").cells[-1]
set_cell(c, c.text.replace("TC-09, TC-13, TC-14", "TC-09, TC-13, TC-14, TC-17"))

doc.save(F)
print("updated", F.name)
