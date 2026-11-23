"""
One-shot update of the report after the first successful end-to-end Airflow DAG run (Docker Compose stack,
20 November 2026, run scheduled__2026-11-18T00:00:00+00:00). Figure 4.5 is redrawn by make_diagrams.py and
re-embedded separately.

    python docs/report/update_dag_run.py

Changes the passages that said the DAG / Docker stack was never executed, records the July 2026 DAG failures and
the two orchestration defects found (DEF-11 fixed, DEF-12 and DEF-13 open), and updates Tables 5.2, 5.4, 5.6,
5.7 and 5.10. Refuses to run if DEF-13 is already in the defect log.
Backup: docs/report/backup/energy_trading_mlops_proposal_v17_before_dag_run.docx
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
    """Replace text inside a paragraph without touching formatting; merges runs only if they share formatting."""
    for r in p.runs:
        if old in r.text:
            r.text = r.text.replace(old, new, 1)
            return
    assert old in p.text, (old[:60], p.text[:80])
    fmts = {r._r.rPr.xml if r._r.rPr is not None else "" for r in p.runs}
    assert len(fmts) == 1, "runs differ in formatting"
    full = p.text.replace(old, new, 1)
    p.runs[0].text = full
    for r in p.runs[1:]:
        r._r.getparent().remove(r._r)


def set_cell(cell, text):
    ps = cell.paragraphs
    for extra in ps[1:]:
        extra._p.getparent().remove(extra._p)
    runs = ps[0].runs
    runs[0].text = text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


def row(table, first):
    hits = [r for r in table.rows if r.cells[0].text.strip() == first]
    assert len(hits) == 1, (first, len(hits))
    return hits[0]


def table_containing(first_cell):
    hits = [t for t in doc.tables if any(r.cells[0].text.strip() == first_cell for r in t.rows)]
    assert len(hits) == 1, (first_cell, len(hits))
    return hits[0]


defects = table_containing("DEF-10")
if any(r.cells[0].text.strip() == "DEF-13" for r in defects.rows):
    sys.exit("DEF-13 already present - this update has been applied.")

# ── Abstract ────────────────────────────────────────────────────────────────
replace(para("The pipeline ingests daily Henry Hub"),
        "Apache Airflow orchestration, Docker containers, Terraform infrastructure and GitHub Actions workflows "
        "were implemented as code but were not used for the reported results.",
        "The Apache Airflow pipeline was run end to end in a local Docker Compose stack, while the Terraform "
        "infrastructure and GitHub Actions workflows exist as code only; the reported results come from local "
        "run scripts.")

# ── Chapter 4: architecture text ────────────────────────────────────────────
replace(para("The orchestration layer contains the Airflow DAG"),
        "The diagram distinguishes the components used for the reported results from those that are defined but "
        "were not executed.",
        "The diagram distinguishes the components used for the reported results, those that ran only in the "
        "local Docker Compose stack, and those that are defined in code but were not run successfully.")

p = para("The solid boxes trace the path that produced the reported results")
replace(p, "The solid boxes trace", "The shaded boxes trace")
replace(p, "The dashed components (the S3 zones, the Glue crawlers, the Airflow DAG, the email notification and the "
           "AWS services) are implemented in code or configuration but were not executed in this study, so no part "
           "of the system is publicly deployed.",
        "The white boxes ran end to end only in the local Docker Compose stack: on 20 November 2026 the Airflow "
        "DAG ingested and transformed the EIA data into S3 zones emulated by LocalStack, trained and backtested "
        "an agent with an MLflow server, and sent its notification to a local Mailpit inbox (TC-14). The dashed "
        "components (the Glue crawlers and the AWS services) are defined in code or configuration but were not "
        "provisioned, so no part of the system is publicly deployed.")
replace(p, "the same code would run unchanged under Airflow.",
        "the same code runs under Airflow and in the local scripts.")

# ── Chapter 5 ───────────────────────────────────────────────────────────────
replace(para("A rerun of step iii will not reproduce"),
        "The Docker Compose stack can be started with docker compose up, and its configuration was validated "
        "with docker compose config.",
        "The Docker Compose stack is started with docker compose up; the DAG is then triggered from the Airflow "
        "web interface (localhost:8080), the MLflow server is at localhost:5000, and the notification emails "
        "appear in the Mailpit inbox (localhost:8026).")

replace(para("The Airflow DAG (src/pipeline/dag.py"),
        "The DAG was not executed in this study: Airflow was not installed on the workstation, and the DAG "
        "imported DummyOperator, which is deprecated in Airflow 2.8 (DEF-04, since fixed).",
        "Airflow was not installed on the workstation, so the DAG runs in the Docker Compose stack, which also "
        "contains the MLflow server, LocalStack in place of S3 and Mailpit, a local mail catcher that receives "
        "the notification emails. Five runs in July 2026 failed at train_drl, first because the MLflow server, "
        "which installed MLflow when it started, was not yet reachable, and then because the server gave the "
        "Airflow container a local artefact path that it could not write to (DEF-11). Airflow still marked these "
        "runs successful, because the end task runs whatever the state of the tasks before it (DEF-12). After "
        "the server was switched to the official MLflow image with artefacts uploaded through the server, and "
        "the deprecated DummyOperator had been replaced (DEF-04), the DAG ran end to end on 20 November 2026 "
        "(TC-14). It used the default drl.yaml settings, which were written for hourly synthetic data, so its "
        "backtest results are not comparable with the evaluation in Section 5.6 (DEF-13).")

replace(para("Testing combined four levels."),
        "System tests ran the full data-to-backtest path through train_on_eia.py and evaluate_eia_runs.py.",
        "System tests ran the full data-to-backtest path through train_on_eia.py and evaluate_eia_runs.py, and "
        "the whole DAG in the Docker Compose stack.")

replace(para("The automated suite contains 65 unit tests"),
        "and zero for the DAG and the Airflow task wrappers, which need Airflow to run.",
        "and zero for the DAG and the Airflow task wrappers, which need Airflow to run and were instead "
        "exercised by the DAG run (TC-14).")

replace(para("v.  The Airflow DAG, the Docker Compose stack, the AWS infrastructure"),
        "v.  The Airflow DAG, the Docker Compose stack, the AWS infrastructure and the CI/CD workflows were not "
        "executed, so the orchestration and deployment requirements are supported by configuration evidence only.",
        "v.  The Airflow DAG was run end to end only once, in a local Docker Compose stack and with training "
        "settings that differ from those of the evaluated agents (DEF-13), and the AWS infrastructure and the "
        "CI/CD workflows were not run successfully, so the deployment requirements are supported by configuration "
        "evidence only.")

# ── Chapter 6 ───────────────────────────────────────────────────────────────
replace(para("The project set out to address three problems"),
        "The orchestration, container, cloud and CI/CD components were written as code but were not executed.",
        "The Airflow DAG has been run end to end in a local Docker Compose stack, while the cloud and CI/CD "
        "components exist as code only.")

replace(para("Objective iv was partially achieved."),
        "The model registry and the quality gate were not implemented, and the Airflow DAG was defined but never "
        "run (TC-14). Research question iv is therefore answered only at the design level: the DAG passes the "
        "MLflow run ID and the backtest summary between tasks, but this integration has not been shown to work "
        "under Airflow.",
        "The model registry and the quality gate were not implemented. The Airflow DAG ran end to end once in a "
        "local Docker Compose stack (TC-14): it ingested and transformed the live EIA data, trained an agent "
        "with an MLflow server, backtested it and emailed the MLflow run ID and the backtest summary. Research "
        "question iv is therefore answered in part: MLflow tracking has been shown to work inside a scheduled "
        "Airflow pipeline, but without the registry and the quality gate the pipeline does not yet manage the "
        "model lifecycle, and the DAG's training settings still differ from those of the evaluated agents "
        "(DEF-13).")

p = para("Objective v was partially achieved.")
replace(p, "the Docker Compose file was validated (TC-15)",
        "the Docker Compose stack was validated (TC-15) and ran the whole DAG (TC-14)")
replace(p, "Ten defects were recorded in total, of which four have been fixed.",
        "Thirteen defects were recorded in total, of which six have been fixed and one has been worked around. "
        "The DAG runs also showed that a run marked successful by Airflow is not evidence on its own: the five "
        "July 2026 runs were marked successful although training had failed in all of them (DEF-12).")
replace(p, "The DAG, the Terraform configuration and the CI/CD workflows were not executed for the reported "
           "results, so the orchestration, cloud and deployment requirements rest on configuration evidence only.",
        "The Terraform configuration and the CI/CD workflows were not run successfully, so the cloud and "
        "deployment requirements rest on configuration evidence only.")

replace(para("Taken together, the aim was partially met."),
        "Ingestion, transformation, training, tracking and backtesting work and are reproducible;",
        "Ingestion, transformation, training, tracking and backtesting work, are reproducible and have run "
        "together under Airflow in a local container stack;")

p = para("v.  Execute the orchestration and deployment code before relying on it")
replace(p, p.text,
        "v.  Fix the remaining orchestration defects and execute the deployment code before relying on it "
        "(system maintainers). The DAG has run end to end in the local Docker Compose stack (TC-14), but its end "
        "task hides failed runs (DEF-12) and it trains with the hourly defaults in drl.yaml rather than the "
        "settings of the evaluated agents (DEF-13). The end task should succeed only when every task before it "
        "has succeeded, and the daily settings should become the defaults. The GitHub Actions workflows should "
        "then be run on the repository and the Terraform validation completed on a machine where the AWS "
        "provider starts (TC-16), which would supply the remaining evidence without the cost of a paid AWS "
        "deployment.")

replace(para("x.  No hardware change is recommended"),
        "or about 30 minutes with four runs in parallel,",
        "or about 30 minutes with four runs in parallel, and a 500,000-step run inside the DAG took 36 minutes,")

p = para("v.  A sustained deployment study.")
replace(p, "Once the DAG has been executed and the quality gate implemented (Section 6.3.2),",
        "Once the DAG's defects have been fixed and the quality gate implemented (Section 6.3.2),")
replace(p, "It would need a long-running Airflow installation or a funded cloud account,",
        "It would need a long-running Airflow installation, such as the Docker Compose stack on a machine that "
        "stays on, or a funded cloud account,")

# ── Tables ──────────────────────────────────────────────────────────────────
t = table_containing("Docker / Terraform")                   # Table 5.2 software versions
r = row(t, "Apache Airflow")
set_cell(r.cells[2], "Orchestration; runs in the Docker Compose stack (Dockerfile.airflow), not installed on the "
                     "workstation")
r = row(t, "MLflow")
set_cell(r.cells[1], "3.12.0 (server image 3.14.0)")
set_cell(r.cells[2], "Experiment tracking (local file store; server in Docker Compose for the DAG)")

t = table_containing("Orchestration")                         # Table 5.4 deviations
r = row(t, "Orchestration")
set_cell(r.cells[2], "Six tasks without branching; EmailOperator to a local mail catcher; run in Docker Compose")

t = table_containing("TC-14")                                 # Table 5.6 test cases
set_cell(row(t, "TC-14").cells[3],
         "Five July 2026 runs failed at train_drl (DEF-11). Retest on 20 November 2026 in Docker Compose: all "
         "eight tasks succeeded; the first ingest attempt failed at the EIA API and the automatic retry passed; "
         "4,262 rows ingested, 500,000-step training in 35 min 45 s, run and models stored in MLflow, backtest "
         "report written to S3 (LocalStack), email received. Pass")
r = row(t, "TC-10")
set_cell(r.cells[3], r.cells[3].text.rstrip().removesuffix("Pass").rstrip().rstrip(".")
         + "; 500,000 steps inside the DAG in 35 min 45 s. Pass")

set_cell(row(defects, "DEF-04").cells[3],
         "Fixed: the DAG imports EmptyOperator and ran under Airflow 2.8.1 (TC-14).")
new_rows = [
    ("DEF-11", "High", "In the Docker Compose stack the MLflow server installed MLflow when it started and returned "
     "a local artefact path that the Airflow container could not write to, so train_drl failed in all five DAG "
     "runs of July 2026.", "Fixed: official MLflow image with artefacts uploaded through the server; the scheduler "
     "waits until MLflow is healthy."),
    ("DEF-12", "Medium", "The DAG's end task uses the ALL_DONE trigger rule, so a run is marked successful even "
     "when train_drl fails, as happened in the five July 2026 runs.", "Open"),
    ("DEF-13", "Medium", "The DAG trains with the drl.yaml defaults, written for hourly synthetic data (720-step "
     "episodes, a 100-unit position limit, a 30 % stop-loss and hourly annualisation), not the settings of the "
     "evaluated agents, so its backtest results are not comparable with Section 5.6.", "Open"),
]
anchor = row(defects, "DEF-10")._tr           # DEF-10 is the last row; new rows copy its formatting
for values in new_rows:
    tr = copy.deepcopy(anchor)
    anchor.addnext(tr)
    anchor = tr
    for c, v in zip(_Row(tr, defects).cells, values):
        set_cell(c, v)

t = table_containing("iv. MLflow tracking, registry, quality gate and daily Airflow DAG")   # Table 5.10
set_cell(row(t, "iv. MLflow tracking, registry, quality gate and daily Airflow DAG").cells[1],
         "All runs tracked and reproducible (TC-09); registry and quality gate not implemented; DAG ran end to end "
         "in Docker Compose (TC-14).")
set_cell(row(t, "v. Test and validate on held-out data, integration, cloud and CI/CD").cells[1],
         "65 unit tests passed; held-out backtests reproduced; integration and system tests run, including the "
         "DAG (TC-14); cloud and CI/CD defined but not run successfully (TC-16).")

doc.save(F)
print("updated", F.name)
