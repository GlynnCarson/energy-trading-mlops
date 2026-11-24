"""One-shot (24 Nov 2026): make every description of the technology stack match the implemented system.

Chapters 4 to 6 already matched; this updates Chapters 1 to 3 (scope, limitations, justification,
literature-review comparisons, conceptual framework, tools, Table 3.1, deliverables) and two
details in Chapters 4 and 5. Proposed designs that changed stay mentioned in Chapter 3 with a
pointer to Table 5.4, so the deviation record in Section 5.4.7 still has its reference.
Objectives, research questions and the aim are left as proposed (Chapter 6 evaluates them).
Backup: docs/report/backup/energy_trading_mlops_proposal_v17_before_stack_update.docx
Run from the repo root:  python docs/report/update_stack_text.py
"""
import copy
import sys
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn

DOC = Path(__file__).resolve().parents[2] / "energy_trading_mlops_proposal_v17.docx"
d = Document(str(DOC))
P = d.paragraphs

if any(p.style.name.startswith("Heading") and p.text.startswith("3.6.7") for p in P):
    sys.exit("Already applied (Section 3.6.7 exists)")


def find(start):
    hits = [p for p in P if start in p.text and not p.style.name.startswith(("TOC", "Heading"))]
    assert len(hits) == 1, f"{len(hits)} paragraphs start with {start!r}"
    return hits[0]


def set_text(p, text):
    runs = p.runs
    assert runs, p.text[:60]
    for r in runs[1:]:
        assert bool(r.bold) == bool(runs[0].bold) and bool(r.italic) == bool(runs[0].italic), p.text[:60]
        r._r.getparent().remove(r._r)
    runs[0].text = text


def sub(start, old, new):
    p = find(start)
    assert old in p.text, f"not found in {start!r}: {old!r}"
    set_text(p, p.text.replace(old, new, 1))


def replace(start, text):
    set_text(find(start), text)


def find_heading(start):
    hits = [p for p in P if p.style.name.startswith("Heading") and p.text.startswith(start)]
    assert len(hits) == 1, f"{len(hits)} headings start with {start!r}"
    return hits[0]


def heading(start, new):
    """Rename a numbered heading, keeping its number, tab and run formatting."""
    p = find_heading(start)
    num = p.text.split("\t")[0]
    set_text(p, f"{num}\t{new}")


def insert_after(p, style_src, text):
    """New paragraph after p with the formatting of style_src."""
    el = copy.deepcopy(style_src._p)
    for r in el.findall(qn("w:r")):
        el.remove(r)
    p._p.addnext(el)
    from docx.text.paragraph import Paragraph
    new = Paragraph(el, p._parent)
    run_src = style_src.runs[0]._r
    new_r = copy.deepcopy(run_src)
    for t in new_r.findall(qn("w:t")):
        new_r.remove(t)
    new._p.append(new_r)
    new.runs[0].text = text
    return new


# ── Chapter 1 ────────────────────────────────────────────────────────────────
sub("From an academic standpoint", "from data collection to cloud deployment",
    "from data collection to deployment and monitoring")
sub("From a practical standpoint",
    "The use of infrastructure-as-code and CI/CD automation means the system can be reliably deployed and "
    "updated by any team familiar with standard software engineering practices.",
    "Because the pipeline is packaged in containers, the infrastructure is defined as code and CI/CD workflows "
    "are provided, the system can be recreated and updated by any team familiar with standard software "
    "engineering practices.")
sub("This project covers an end-to-end pipeline",
    "Orchestration covers a daily Airflow DAG with email notification, packaged to run locally in Docker Compose, "
    "with LocalStack emulating Amazon Simple Storage Service (S3). Infrastructure covers Terraform definitions of "
    "S3, Glue, CloudWatch, Amazon Simple Notification Service (SNS) and Elastic Container Registry (ECR) on AWS.",
    "The model lifecycle covers a quality gate on the validation split, the MLflow Model Registry with a champion "
    "alias, paper trading (simulated) of the champion on newly published prices, health monitoring and the "
    "retraining of challengers. Orchestration covers a daily Airflow DAG that runs the pipeline and this loop and "
    "sends email notifications, packaged to run locally in a Docker Compose stack with a PostgreSQL metadata "
    "database, an MLflow tracking server, LocalStack emulating Amazon Simple Storage Service (S3) and a Mailpit "
    "inbox. Infrastructure covers Terraform definitions of S3, Glue, IAM, CloudWatch, Amazon Simple Notification "
    "Service (SNS) and Elastic Container Registry (ECR) on AWS, which were validated but not provisioned.")
sub("The agent has been tested only on daily Henry Hub",
    "The proposed 24-hour price forecast was not built, because the agent outputs trading positions rather than "
    "forecasts, and the MLflow model registry and the automated quality gate were not implemented (Table 5.4).",
    "The proposed 24-hour price forecast was not built, because the agent outputs trading positions rather than "
    "forecasts (Table 5.4). The quality gate, the model registry, paper trading and retraining were added late in "
    "the project, so the champion has been paper traded for only 96 trading days and retraining has run only as a "
    "smoke test.")
sub("The agent has been tested only on daily Henry Hub",
    "so the reported results were produced on a single workstation by local run scripts rather than by the "
    "scheduled Airflow DAG.",
    "so the system runs only on a single workstation: the reported results were produced by local run scripts, "
    "and the Airflow DAG ran end to end only in a local Docker Compose stack.")

# ── Chapter 2 ────────────────────────────────────────────────────────────────
sub("Machine Learning Operations (MLOps) is a relatively recent",
    "with daily automated retraining and quality gating.",
    "with an automated quality gate and retraining triggered by performance monitoring.")
sub("The application of MLOps specifically to energy systems",
    "the automated daily retraining pipeline proposed in this project.",
    "the monitoring and automated retraining built into this project.")
sub("FinRL is an open-source deep reinforcement learning library",
    "Terraform-provisioned AWS infrastructure.", "AWS infrastructure defined in Terraform.")
sub("The Electricity Market Forecasting Platform is",
    "with a fully automated daily retraining pipeline deployed on AWS.",
    "with a scheduled pipeline, a model registry and monitoring-triggered retraining, run in containers and "
    "defined as code for AWS.")
sub("MLflow Projects is an open-source component",
    "Terraform infrastructure provisioning, resulting", "Terraform infrastructure definitions, resulting")
sub("The review above reveals several consistent gaps",
    "This project addresses all four gaps.",
    "This project addresses all four gaps, although its cloud infrastructure was defined and validated but not "
    "deployed (Section 5.4.6).")
sub("The conceptual framework for this project is built around",
    "a feedback loop from the output layer back to the data collection layer to trigger model retraining.",
    "a feedback loop from the output layer back to the data collection layer that retrains the model when "
    "monitoring shows that its performance has degraded.")
replace("Layer 1, Actors and Users",
        "Layer 1, Actors and Users, represents the human roles that interact with the system. The Data Engineer is "
        "responsible for monitoring the ingestion and transformation stages and resolving data quality issues when "
        "they arise. The ML Engineer reviews training experiment results in MLflow, sets the quality-gate "
        "thresholds and the rule for choosing the champion, and reviews promotion decisions. The Energy Analyst "
        "uses the results: the backtest reports and the paper-trading record of the approved model. The Airflow "
        "Scheduler is a system actor that triggers the full pipeline automatically every day without human "
        "intervention.")
replace("Layer 2, Data Collection",
        "Layer 2, Data Collection, covers all activities related to acquiring raw energy data. The primary data "
        "source is the EIA API v2, which provides daily Henry Hub natural gas and West Texas Intermediate crude oil "
        "spot prices. A seeded synthetic data generator produces repeatable data for development and testing when "
        "API access is unavailable. Raw data is serialised to Parquet and stored in the raw zone of the S3 data "
        "lake under the raw/energy/ prefix; in the local Docker Compose stack the bucket is emulated by LocalStack.")
replace("Layer 3, Processing and AI",
        "Layer 3, Processing and AI, is the analytical core of the system. The transformation module cleans the "
        "raw data, masks outliers, regularises the series to business days and engineers 38 leakage-free calendar, "
        "lag, rolling-statistic and technical-indicator features. The engineered dataset is passed to the "
        "EnergyTradingEnv Gymnasium environment, where the PPO agent is trained, and every run's parameters, "
        "metrics and model artefacts are logged to MLflow. A quality gate then judges each trained agent on the "
        "validation split against Buy & Hold and seeded Random Agents, and only an agent that passes can be "
        "registered.")
replace("Layer 4, Storage and Infrastructure",
        "Layer 4, Storage and Infrastructure, covers the persistent data and compute resources that support the "
        "pipeline. The S3 data lake stores raw and processed data in separate zones. The MLflow tracking store "
        "keeps every run, and the MLflow Model Registry keeps a versioned history of the models that passed the "
        "gate, with the alias champion marking the approved one. Locally, these services run as containers in a "
        "Docker Compose stack. Terraform defines the equivalent AWS resources, including a versioned and encrypted "
        "S3 bucket, a Glue Data Catalog over the data lake and an ECR repository for the pipeline image; they were "
        "validated but not provisioned.")
replace("Layer 5, Outputs and Feedback",
        "Layer 5, Outputs and Feedback, covers the products that the system delivers and the mechanisms by which "
        "results flow back into the pipeline. Each trained agent produces a backtest report against Buy & Hold and "
        "Random baselines. The approved model is paper traded (simulated, with no real money) on each newly "
        "published price, and a monitor compares it with Buy & Hold. The pipeline sends email notifications over "
        "SMTP when a run succeeds or fails and when the champion is retrained, replaced or kept; locally they are "
        "delivered to a Mailpit inbox. The feedback loop shown at the top of Figure 2.1 represents the retraining "
        "cycle: when the monitor reports that the champion's performance has degraded, challengers are trained on "
        "the newest data, and one replaces the champion only if it passes the gate and beats it.")

# ── Chapter 3 ────────────────────────────────────────────────────────────────
sub("Data is acquired from two sources.",
    "All raw data is saved as Parquet files under the raw/energy/ prefix in S3.",
    "All raw data is saved as Parquet files under the raw/energy/ prefix in S3, which LocalStack emulates in the "
    "local Docker Compose stack.")
sub("The project uses an Agile iterative approach",
    "The GitHub Actions CI pipeline runs all tests and style checks on every commit.",
    "The GitHub Actions CI workflow is triggered on every push and runs the style checks before the tests.")
sub("Every training run logs its parameters",
    "The model is registered under the name energy_trading_ppo in the MLflow Model Registry. The Airflow quality "
    "gate reads the pass/fail run tag from MLflow to determine branching.",
    "A model that passes the quality gate is registered as a version of energy_trading_ppo in the MLflow Model "
    "Registry, and the approved version carries the alias champion, so the pipeline loads the approved model by "
    "name rather than by file path. The gate decision and its reasons are logged as tags and metrics on the "
    "candidate's run.")
sub("The Airflow DAG separates pipeline logic",
    "The BranchPythonOperator reads the MLflow quality gate tag and routes execution to either the promote_model "
    "path or the model_rejected terminal task.",
    "Two BranchPythonOperator tasks control the closed loop: branch_on_health skips retraining while the champion "
    "is healthy, and branch_on_outcome selects the email sent after a retrain (champion replaced, retrain failed "
    "the gate or champion kept). The proposal routed a single quality-gate branch to a promote_model or "
    "model_rejected task; Section 5.4.8 explains the change.")
replace("The test suite uses pytest with moto",
        "The test suite uses pytest with moto for AWS S3 mocking, and the CI workflow is configured to upload "
        "coverage to Codecov. Locally, Docker Compose runs the whole pipeline in containers (Section 3.6.7). The "
        "deploy.yml GitHub Actions workflow is defined to build the Docker image, push it to ECR, apply the "
        "Terraform plan and copy the DAG to the MWAA DAG bucket on every push to the main branch; it has not run "
        "successfully because no AWS account was provisioned (Section 5.4.6).")
sub("The second store is the AWS Glue Data Catalog",
    "so that queries through Amazon Athena scan only the partitions they need.",
    "so that queries through Amazon Athena scan only the partitions they need. The crawlers and the catalog are "
    "defined in Terraform but were not provisioned, so this store exists as a design; locally, the Parquet files "
    "are read directly from the LocalStack bucket.")

heading("3.6.2", "Apache Airflow 2.8.1")
replace("Apache Airflow provides DAG-based pipeline orchestration",
        "Apache Airflow provides DAG-based pipeline orchestration with configurable task retry logic and a "
        "monitoring web interface. It runs in the local Docker Compose stack from an image built on "
        "apache/airflow:2.8.1, with a webserver, a scheduler and a PostgreSQL metadata database. Amazon MWAA, the "
        "managed Airflow service on AWS, was proposed as the production host but was not used because of its cost "
        "(Table 5.4).")
heading("3.6.3", "MLflow 3.12.0")
replace("MLflow provides four core functions",
        "MLflow provides four core functions: logging of training parameters, metrics and artefacts; storage of "
        "trained model files linked to their parent runs; model registration and versioning in the Model "
        "Registry; and aliases, such as champion, that point to the approved model version. Aliases replace the "
        "registry stages (None, Staging and Production) planned in the proposal, which MLflow has deprecated. The "
        "reported runs use a local file store (./mlruns); in Docker Compose the pipeline logs to an MLflow server "
        "(image 3.14.0).")
heading("3.6.4", "stable-baselines3 2.8.0 and gymnasium 1.2.3")
replace("stable-baselines3 provides the PPO implementation",
        "stable-baselines3 provides the PPO implementation used to train the DRL trading agent, built on PyTorch "
        "(2.12.0, CPU build) with a consistent interface for training, evaluation and serialisation, and its "
        "DummyVecEnv wrapper runs four copies of the environment for vectorised training. gymnasium provides the "
        "Env base class from which EnergyTradingEnv inherits.")
replace("Terraform provisions all AWS infrastructure as code.",
        "Terraform defines the AWS infrastructure as code. The configuration in terraform/main.tf covers S3, IAM "
        "roles, the Glue Data Catalog, CloudWatch, SNS and ECR in us-east-1. Using Terraform means infrastructure "
        "changes are reviewed through a plan before being applied, and the entire environment can be recreated "
        "identically in any AWS account. The configuration is validated in CI but has not been applied, so no AWS "
        "resources exist (Section 5.4.6).")
p_gha = find("GitHub Actions automates the CI/CD workflow.")
set_text(p_gha,
         "GitHub Actions automates the CI/CD workflow. The CI workflow runs on every push and defines lint checks, "
         "the full test suite, a smoke test, a Docker image build and Terraform validation. Terraform validation "
         "passes, but the lint job currently stops at the formatting check, so the jobs that depend on it do not "
         "run. The CD workflow is defined to run on every push to the main branch, push the Docker image to ECR, "
         "apply the Terraform plan and deploy the DAG to MWAA, but it has not run successfully because no AWS "
         "credentials are configured.")
h366 = find_heading("3.6.6")
h367 = insert_after(p_gha, h366, "3.6.7\tDocker Compose, LocalStack and Mailpit")
insert_after(h367, p_gha,
             "Docker Compose runs the complete pipeline locally as containers, so that the DAG can be executed end "
             "to end without a cloud account. The stack contains the Airflow webserver and scheduler, a PostgreSQL "
             "15 database for Airflow's metadata, an MLflow 3.14.0 tracking server with a SQLite backend, "
             "LocalStack 3.2, which emulates the S3 API so that the same boto3 code writes to a local bucket, and "
             "Mailpit 1.21, a local mail catcher that receives the pipeline's emails over SMTP in place of a real "
             "mailbox. An initialisation service prepares the Airflow database and a one-slot pool so that only "
             "one training task runs at a time.")

heading("3.7.3", "DRL Training and Model Lifecycle Modules")
replace("A working EnergyTradingEnv gymnasium environment",
        "A working EnergyTradingEnv gymnasium environment and PPO training pipeline integrated with MLflow, and a "
        "backtest module that compares the agent with Buy & Hold and Random baselines. The closed-loop modules "
        "(quality_gate, registry, paper_trade, monitor and retrain) gate and register trained models, and paper "
        "trade, monitor and retrain the approved one. The proposal also listed a run_prediction entry point and a "
        "recursive 24-hour forecast generator writing Parquet forecasts to S3; these were not built because the "
        "agent outputs trading positions rather than forecasts (Table 5.4).")
replace("A production-ready Airflow DAG with seven tasks",
        "An Airflow DAG with eight pipeline tasks (start, ingest, transform, data_quality_report, train_drl, "
        "backtest_drl, notify_success and end) and a closed loop that paper trades and monitors the champion and, "
        "when needed, retrains, gates and promotes challengers. Tasks pass data through XCom, the DAG branches on "
        "the champion's health and on the retrain outcome, and notifications are sent by the Airflow "
        "EmailOperator. The DAG runs in the local Docker Compose stack; deploy.yml packages it for MWAA, but that "
        "workflow has not been run (Section 5.4.6). The proposal planned seven tasks with SNS notification "
        "(Table 5.4).")
replace("A complete Terraform configuration provisioning all AWS resources",
        "A Terraform configuration defining the AWS resources in us-east-1 (S3, Glue, IAM, CloudWatch, SNS and "
        "ECR), validated in CI but not applied; the proposal planned eu-west-2 with MWAA (Table 5.4). A multi-stage "
        "Dockerfile for the pipeline image, an Airflow image (Dockerfile.airflow) and a Docker Compose stack with "
        "Airflow, PostgreSQL, an MLflow server, LocalStack and Mailpit. GitHub Actions CI and CD workflows covering "
        "linting, testing, Docker build, Terraform validation and deployment.")

# ── Table 3.1 ────────────────────────────────────────────────────────────────
ROWS = [
    ("Python", "3.11", "Primary programming language"),
    ("stable-baselines3", "2.8.0", "PPO DRL agent training"),
    ("gymnasium", "1.2.3", "Custom trading environment"),
    ("PyTorch", "2.12.0 (CPU)", "Neural network back end"),
    ("MLflow", "3.12.0 (server 3.14.0)", "Experiment tracking and model registry"),
    ("Apache Airflow", "2.8.1", "Pipeline orchestration in Docker Compose"),
    ("PostgreSQL", "15", "Airflow metadata database"),
    ("Docker / Docker Compose", "29.6.1", "Containers and the local stack"),
    ("LocalStack", "3.2", "Local emulation of S3"),
    ("Mailpit", "1.21", "Local inbox for pipeline emails"),
    ("Terraform", "1.15.2 (CI 1.6.6)", "Infrastructure as code; validated, not applied"),
    ("AWS S3, Glue, IAM", "N/A", "Target data lake, catalogue and roles; defined, not provisioned"),
    ("AWS ECR", "N/A", "Target image registry; defined, not provisioned"),
    ("AWS CloudWatch, SNS", "N/A", "Target logging and alerts; defined, not provisioned"),
    ("GitHub Actions", "N/A", "CI/CD workflows"),
    ("pytest + moto", "9.0.3 / 5.2.2", "Testing and AWS S3 mocking"),
    ("pyarrow / Parquet", "23.0.1", "Serialisation and storage format"),
]
t = next(tb for tb in d.tables if tb.rows[0].cells[0].text.strip() == "Tool / Technology")
assert t.rows[2].cells[0].text.strip() == "stable-baselines3"
template = copy.deepcopy(t.rows[1]._tr)
for r in list(t.rows)[1:]:
    t._tbl.remove(r._tr)
for vals in ROWS:
    tr = copy.deepcopy(template)
    t._tbl.append(tr)
    row = t.rows[-1]
    for c, v in zip(row.cells, vals):
        set_text(c.paragraphs[0], v)

# ── Chapters 4 and 5 ─────────────────────────────────────────────────────────
ir = next(tb for tb in d.tables if tb.rows[0].cells[0].text.strip() == "ID"
          and tb.rows[1].cells[0].text.strip() == "IR-01")
cell = next(r.cells[2] for r in ir.rows if r.cells[0].text.strip() == "IR-06")
old = "MLflow UI (port 5000), Airflow UI (port 8080), PNG backtest dashboard"
assert old in cell.text
set_text(cell.paragraphs[0], cell.text.replace(old, "MLflow UI (port 5000), Airflow UI (port 8080), Mailpit inbox "
                                                    "(port 8026), PNG backtest dashboard"))
sub("The Airflow DAG (src/pipeline/dag.py, FR-12 and FR-13)",
    "which also contains the MLflow server, LocalStack in place of S3 and Mailpit",
    "which also contains a PostgreSQL database for Airflow's metadata, the MLflow server, LocalStack in place of "
    "S3 and Mailpit")

d.save(str(DOC))
print("Stack text updated")
