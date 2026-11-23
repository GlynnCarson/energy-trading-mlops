"""
HISTORICAL - do not rerun. This script performed the first insertion of Chapters 4 and 5. The document has
since been changed (section 5.8 removed, section 5.7 updated, diagrams restructured by restructure_ch3_ch4.py
after supervisor feedback), so its Chapter 4 content is out of date. Edit the docx directly instead.

Inserts Chapter 4 and Chapter 5 into energy_trading_mlops_proposal_v17.docx,
between Chapter 3 and References, using the document's own styles (proposal styles).

    python docs/report/make_diagrams.py
    python docs/report/make_charts.py
    python docs/report/build_chapters.py

It also adds the new figures, tables, abbreviations and the ISO/IEC 25010
reference to the front-matter lists and reference list, and applies the
formatting fixes listed in the project notes (Chapter Three and 2.5.x headings, Fig 3.1).
A copy of the document as it was before the first insertion is kept in
docs/report/backup/. The script refuses to run if Chapter Four already exists.
All result numbers are read from reports/evaluation/evaluation_summary.json.
"""

import copy
import json
import re
import shutil
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Pt, Inches

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIG = HERE / "figures"
TARGET = ROOT / "energy_trading_mlops_proposal_v17.docx"
BACKUP = HERE / "backup" / "energy_trading_mlops_proposal_v17_before_ch4_ch5.docx"
EVAL = json.load(open(ROOT / "reports" / "evaluation" / "evaluation_summary.json"))
SUFFIX = " – Energy Trading MLOps Platform"

# ─────────────────────────────────────────────────────────────────────────────
# Document helpers (formatting values from the proposal styles)
# ─────────────────────────────────────────────────────────────────────────────

doc = Document(str(TARGET))
body = doc.element.body


def _text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


if any(_text(el).strip().startswith("Chapter Four") for el in body):
    sys.exit("Chapter Four already exists in the document - nothing changed. Restore the backup to rebuild.")
if not BACKUP.exists():
    BACKUP.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TARGET, BACKUP)

# Everything added below is appended at the end of the body, then moved to just
# before the page-break paragraph that precedes "References".
_ref_idx = [i for i, el in enumerate(body) if _text(el).strip() == "References"][0]
INSERT_BEFORE = body[_ref_idx - 1]
_appended_from = len([el for el in body if el.tag != qn("w:sectPr")])
_first_h1 = [False]


def _runs(par, text, size=None, italic_all=False, bold_all=False):
    """Add text with **bold** and *italic* markup."""
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not tok:
            continue
        if tok.startswith("**"):
            r = par.add_run(tok[2:-2]); r.bold = True
        elif tok.startswith("*"):
            r = par.add_run(tok[1:-1]); r.italic = True
        else:
            r = par.add_run(tok)
        if size:
            r.font.size = Pt(size)
        if italic_all:
            r.italic = True
        if bold_all:
            r.bold = True
    return par


def _fmt(par, align=WD_ALIGN_PARAGRAPH.JUSTIFY, after=None, keep_next=False):
    pf = par.paragraph_format
    pf.line_spacing = 1.5
    if align is not None:
        pf.alignment = align
    if after is not None:
        pf.space_after = Pt(after)
    if keep_next:
        pf.keep_with_next = True
    return par


def H1(text):
    # Same pattern as the proposal: an empty paragraph holding a page break, then the heading.
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    p = doc.add_paragraph(style="Heading 1")
    p.add_run(text)
    _fmt(p, WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)


def H2(num, text):
    p = doc.add_paragraph(style="Heading 2")
    p.add_run(f"{num}\t{text}")
    _fmt(p, None, keep_next=True)


def H3(num, text):
    p = doc.add_paragraph(style="Heading 3")
    p.add_run(f"{num}\t{text}")
    _fmt(p, None, keep_next=True)


def P(text):
    _fmt(_runs(doc.add_paragraph(style="Normal"), text))


ROMAN = ["i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii"]


def LIST(items):
    for i, t in enumerate(items):
        _fmt(_runs(doc.add_paragraph(style="Normal"), f"{ROMAN[i]}.  {t}"))


def CAPTION(text, keep_next=False):
    p = doc.add_paragraph(style="Normal")
    _runs(p, text, size=11, italic_all=True)
    _fmt(p, WD_ALIGN_PARAGRAPH.CENTER, keep_next=keep_next)


def FIGURE(name, caption, width=6.3):
    p = doc.add_paragraph(style="Normal")
    p.add_run().add_picture(str(FIG / name), width=Inches(width))
    _fmt(p, WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
    CAPTION(caption)


def _cell_borders_and_margins(cell, fill=None):
    tcPr = cell._tc.get_or_add_tcPr()
    borders = OxmlElement("w:tcBorders")
    for edge in ("top", "left", "bottom", "right"):
        e = OxmlElement(f"w:{edge}")
        e.set(qn("w:val"), "single"); e.set(qn("w:sz"), "4"); e.set(qn("w:space"), "0"); e.set(qn("w:color"), "000000")
        borders.append(e)
    tcPr.append(borders)
    if fill:
        shd = OxmlElement("w:shd")
        shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), fill)
        tcPr.append(shd)
    mar = OxmlElement("w:tcMar")
    for edge, w in (("top", 80), ("left", 120), ("bottom", 80), ("right", 120)):
        e = OxmlElement(f"w:{edge}"); e.set(qn("w:w"), str(w)); e.set(qn("w:type"), "dxa"); mar.append(e)
    tcPr.append(mar)


def TABLE(caption, headers, rows, widths):
    CAPTION(caption, keep_next=True)
    t = doc.add_table(rows=1 + len(rows), cols=len(headers))
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.autofit = False
    tblPr = t._tbl.tblPr
    look = OxmlElement("w:tblLook"); look.set(qn("w:val"), "04A0"); tblPr.append(look)
    for ci, h in enumerate(headers):
        c = t.rows[0].cells[ci]
        c.text = ""
        _runs(c.paragraphs[0], h, bold_all=True)
        _fmt(c.paragraphs[0], WD_ALIGN_PARAGRAPH.LEFT)
        _cell_borders_and_margins(c, "D9D9D9")
    trPr = t.rows[0]._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader"); trPr.append(hdr)
    for ri, row in enumerate(rows, start=1):
        for ci, val in enumerate(row):
            c = t.rows[ri].cells[ci]
            c.text = ""
            _runs(c.paragraphs[0], str(val))
            _fmt(c.paragraphs[0], WD_ALIGN_PARAGRAPH.LEFT)
            _cell_borders_and_margins(c)
    for row in t.rows:
        for ci, w in enumerate(widths):
            row.cells[ci].width = Inches(w)
    doc.add_paragraph(style="Normal")  # spacing after table, as in the proposal


# ─────────────────────────────────────────────────────────────────────────────
# Evidence values
# ─────────────────────────────────────────────────────────────────────────────
S = EVAL["splits"]
R10, R300 = EVAL["runs"]["PPO-10k"], EVAL["runs"]["PPO-300k"]
F3, F1 = R300["full_test"], R10["full_test"]
W3, W1 = R300["first_252"], R10["first_252"]


def pct(x, sign=True):
    return f"{x:+.2f} %" if sign else f"{x:.2f} %"


# =============================================================================
# CHAPTER FOUR
# =============================================================================
H1("Chapter Four: System Analysis and Design")

H2("4.1", "Introduction")
P("This chapter presents the analysis and design of the Energy Trading MLOps Platform. It states what the system "
  "must do, the quality conditions it must satisfy and the data and interfaces it depends on, and it presents the "
  "diagrams used to communicate the structure and behaviour of the solution. Section 4.2 defines the functional, "
  "non-functional, data and interface requirements and traces them to the objectives in Section 1.3.1. Section 4.3 "
  "presents the analysis diagrams: a context diagram, a Level 0 data flow diagram, a use case diagram and an activity "
  "diagram of the daily pipeline run. Section 4.4 presents the design diagrams: the reinforcement learning design, "
  "the class diagram, the logical data design and the deployment diagram.")
P("Chapter 3 justified an Agile, design science approach in which the pipeline itself is the research artefact. The "
  "analysis therefore combines two complementary views. Structured analysis (context diagram and data flow diagram) "
  "is used because the platform is fundamentally a data pipeline in which data is transformed in stages. "
  "Object-oriented models (use case and class diagrams) are used because the implementation is organised into "
  "Python classes and modules with clear responsibilities. A data-centric design view replaces a conventional "
  "relational schema, because the platform stores Parquet datasets and experiment records rather than "
  "transactional data. The design in this chapter reflects the system as refined during development; components "
  "proposed in Chapter 3 that were not implemented are recorded, with reasons, in Section 5.4.7.")

H2("4.2", "System Requirements")
P("The system requirements identified for this project are presented in the following subsections. Each "
  "requirement has a unique identifier, a priority, a source or rationale, and a verification method, so that it "
  "can be traced forward to the implementation and test cases in Chapter 5. Thresholds are stated only where they "
  "can be justified from the project configuration or the intended operating environment.")

H3("4.2.1", "Functional Requirements")
P("Functional requirements describe the services the platform must provide. They are grouped by pipeline stage "
  "and are summarised in Table 4.1.")
TABLE("Table 4.1: Functional requirements",
      ["ID", "Requirement", "Priority", "Source", "Verification"],
      [["FR-01", "The system shall retrieve daily Henry Hub natural gas (series RNGWHHD) and WTI crude oil (series "
                 "RWTC) spot prices from the EIA API v2 from a configurable start date, and shall use the selected "
                 "commodity as the trading target.", "High", "Objective ii", "Demonstration"],
       ["FR-02", "The system shall retry a failed API request up to the configured number of attempts (three) with "
                 "exponential back-off, and shall raise an error when all attempts fail.", "Medium",
        "Objective ii", "Inspection"],
       ["FR-03", "The system shall provide a seeded synthetic data generator so that the pipeline can run without "
                 "API access.", "Medium", "Section 1.5.2", "Unit test"],
       ["FR-04", "The system shall clean raw data by converting timestamps to UTC, sorting them, removing duplicate "
                 "timestamps, masking values beyond the configured standard-deviation threshold and regularising "
                 "the series to its detected frequency.", "High", "Objective ii", "Unit test"],
       ["FR-05", "The system shall reject a dataset that fails a critical validation rule: missing, null or "
                 "non-numeric target; constant price; fewer than 100 rows; or a non-datetime or non-monotonic "
                 "index.", "High", "Objective ii", "Unit test"],
       ["FR-06", "The system shall engineer calendar, lag, rolling-statistic and technical-indicator features using "
                 "only information available before each time step.", "High", "Research question ii",
        "Unit test"],
       ["FR-07", "The system shall split the engineered dataset chronologically into training, validation and test "
                 "sets in the configured 80/10/10 ratio.", "High", "Objective ii", "Unit test"],
       ["FR-08", "The system shall provide a Gymnasium trading environment that accepts a position action, "
                 "simulates trades with transaction costs and a stop-loss, and returns an observation, a reward "
                 "and portfolio information.", "High", "Objective iii", "Unit test"],
       ["FR-09", "The system shall train a PPO agent (with SAC, TD3 and A2C selectable through configuration), "
                 "evaluate it periodically on the validation split and retain the best checkpoint.", "High",
        "Objective iii", "Demonstration"],
       ["FR-10", "The system shall record the parameters, metrics, configuration file and model artefacts of every "
                 "training run in MLflow.", "High", "Objective iv", "Inspection"],
       ["FR-11", "The system shall backtest a trained agent on the held-out test split and compare it with Buy & "
                 "Hold and Random baselines using return, Sharpe ratio, maximum drawdown, Calmar ratio, win rate, "
                 "value at risk and profit factor.", "High", "Objective v", "Demonstration"],
       ["FR-12", "The system shall orchestrate ingestion, transformation, data-quality reporting, training, "
                 "backtesting and notification as a daily Apache Airflow DAG.", "High", "Objective iv",
        "Execution"],
       ["FR-13", "The system shall send an email containing the MLflow run ID and backtest summary after a "
                 "successful run, and an alert when a task fails.", "Medium", "Objective iv", "Execution"],
       ["FR-14", "The system shall allow a user to replay a backtest trajectory offline in a desktop application.",
        "Low", "Energy Analyst", "Demonstration"]],
      [0.75, 2.6, 0.8, 1.05, 1.3])

H3("4.2.2", "Non-Functional Requirements")
P("The non-functional requirements in Table 4.2 were selected from the ISO/IEC 25010 quality characteristics that "
  "are relevant to an automated ML pipeline: reproducibility, performance efficiency, maintainability and "
  "testability, security, reliability, portability and human oversight. The performance threshold is taken from the "
  "two-hour execution_timeout configured for every task in the Airflow DAG, because a training run that exceeds it "
  "would be terminated by the scheduler.")
TABLE("Table 4.2: Non-functional requirements",
      ["ID", "Requirement", "Rationale", "Verification"],
      [["NFR-01", "Given the same data snapshot, configuration and seed, re-evaluating a logged model shall "
                  "reproduce its MLflow-logged validation return to within 0.01 percentage points.",
        "Auditability claim in Section 3.3.4", "Re-evaluation test"],
       ["NFR-02", "A 300,000-timestep PPO training run on the daily dataset shall complete within the 2-hour task "
                  "timeout on a CPU-only development machine.", "DAG execution_timeout", "Measurement"],
       ["NFR-03", "The automated test suite shall pass in full before results are reported, and the CI workflow "
                  "shall run linting, tests, a smoke test, a Docker build and Terraform validation on every push.",
        "Section 3.3.7", "Test run; inspection"],
       ["NFR-04", "The EIA API key and AWS credentials shall be supplied through environment variables and shall "
                  "not appear in source code, configuration files, logs or reports.", "Credential protection",
        "Inspection"],
       ["NFR-05", "Every pipeline task shall be retried up to two times with a five-minute delay, and only one DAG "
                  "run shall be active at a time.", "Transient API or storage failures", "Inspection"],
       ["NFR-06", "The pipeline shall run locally without AWS (local MLflow file store) and as containers through "
                  "Docker Compose.", "Development without cloud cost", "Demonstration"],
       ["NFR-07", "Every backtest shall produce a visual report of prices, trades, positions, drawdown and action "
                  "distribution, and the agent shall only act in simulation.", "Human oversight of agent "
                                                                                 "behaviour", "Inspection"]],
      [0.8, 2.9, 1.5, 1.3])

H3("4.2.3", "Data Requirements")
P("The platform consumes public market data and produces derived datasets, models and reports. Table 4.3 sets out "
  "the data requirements. Because the project is an ML system, particular attention is given to chronological "
  "partitioning, because a random split of a price series would allow information from the future to leak into "
  "training.")
TABLE("Table 4.3: Data requirements",
      ["ID", "Requirement"],
      [["DR-01", "Source and ownership: daily spot prices published by the U.S. Energy Information Administration "
                 "through the EIA Open Data API v2, accessed with a free personal API key."],
       ["DR-02", "Format and units: one record per business day; target price_usd in $/MMBtu (Henry Hub); exogenous "
                 "oil_price_usd in $/bbl (WTI); UTC timestamps; storage as Parquet."],
       ["DR-03", "Quality: no null target values and no duplicate timestamps; values beyond the configured "
                 "threshold (6 standard deviations) are masked and forward-filled for at most five business days; "
                 "at least 100 rows are required."],
       ["DR-04", "Partitioning and leakage: chronological 80/10/10 split; every engineered feature uses lagged "
                 "prices only; the test split is never used for training or checkpoint selection."],
       ["DR-05", "Storage and retention: raw and processed datasets under separate S3 prefixes in a versioned, "
                 "encrypted bucket; experiment records and models in the MLflow store and local checkpoints."],
       ["DR-06", "Privacy: no personal data is processed; the API key is the only confidential item."],
       ["DR-07", "Reproducibility: each MLflow run stores the full configuration file, the random seed and the "
                 "split sizes, so that the dataset used can be reconstructed and checked."]],
      [0.8, 5.7])

H3("4.2.4", "Interface and Integration Requirements")
P("The platform integrates several external services and libraries. Table 4.4 records each interface, the protocol "
  "or contract it uses, and how failures are handled.")
TABLE("Table 4.4: Interface and integration requirements",
      ["ID", "Interface", "Protocol and behaviour"],
      [["IR-01", "EIA Open Data API v2", "HTTPS GET returning JSON; parameters frequency=daily, series facet, start "
                                          "date, ascending sort and length=5000; 30-second timeout; three retries "
                                          "with exponential back-off."],
       ["IR-02", "Object storage (AWS S3 or LocalStack)", "boto3 S3 client; Parquet serialisation through pyarrow; "
                                                          "endpoint and credentials from environment variables."],
       ["IR-03", "MLflow tracking", "Configurable tracking URI: local file store (./mlruns) or HTTP server on port "
                                    "5000; parameters, step metrics, tags and artefacts."],
       ["IR-04", "Apache Airflow", "PythonOperator callables; XCom for S3 paths, run IDs and summaries; "
                                   "EmailOperator over SMTP."],
       ["IR-05", "Gymnasium API", "reset() and step() return values as specified by Gymnasium; verified with "
                                  "check_env before training."],
       ["IR-06", "User interfaces", "MLflow UI (port 5000), Airflow UI (port 8080), PNG backtest dashboard and the "
                                    "Tkinter trade-replay application."]],
      [0.8, 1.9, 3.8])

H3("4.2.5", "Requirements Traceability")
P("Table 4.5 links each specific objective to the requirements that realise it, the design artefacts in this "
  "chapter, the implementing modules and the test cases reported in Chapter 5. Objective i was addressed by the "
  "literature review in Chapter 2 and therefore has no software requirement.")
TABLE("Table 4.5: Requirements traceability matrix",
      ["Objective", "Requirements", "Design", "Implementation", "Tests"],
      [["ii. Ingestion and transformation", "FR-01 to FR-07, DR-01 to DR-05", "Figures 4.2, 4.7",
        "ingest.py, transform.py", "TC-01 to TC-04, TC-08"],
       ["iii. PPO agent and environment", "FR-08, FR-09", "Figures 4.5, 4.6", "env.py, train_drl.py",
        "TC-05, TC-06, TC-11"],
       ["iv. MLflow and orchestration", "FR-10, FR-12, FR-13, NFR-01, NFR-05", "Figures 4.4, 4.7, 4.8",
        "train_drl.py, dag.py, drl_tasks.py", "TC-09, TC-13, TC-14"],
       ["v. Testing and validation", "FR-11, NFR-02, NFR-03, NFR-06", "Figures 4.4, 4.8",
        "backtest.py, evaluate_eia_runs.py, ci.yml", "TC-07, TC-10, TC-12, TC-15, TC-16"]],
      [1.4, 1.5, 1.0, 1.5, 1.1])

H2("4.3", "System Analysis Diagrams")
P("This section presents the diagrams used to analyse the system boundary, the processes and data stores, the "
  "actors and their goals, and the control flow of a pipeline run. Names are kept consistent with the code so that "
  "every element can be traced to the implementation.")

H3("4.3.1", "Context Diagram")
P("The context diagram in Figure 4.1 represents the whole platform as a single process and shows its exchanges with "
  "external entities. It establishes the system boundary: the EIA API, object storage, the email service and the "
  "Airflow scheduler lie outside the platform, while ingestion, transformation, training and evaluation lie inside "
  "it. The diagram addresses requirements FR-01, FR-10, FR-12 and FR-13.")
FIGURE("fig4_1_context.png", "Figure 4.1: Context Diagram" + SUFFIX)
P("The diagram shows that the platform has a single external data provider, which makes the EIA API the main point "
  "of failure and justifies the retry behaviour in FR-02. The human actors only supply configuration and consume "
  "results; none of them intervenes during a run, which reflects the goal of an automated pipeline. Object storage "
  "both receives and returns data because each stage reads the output of the previous one from storage rather "
  "than from memory, which allows a stage to be re-run independently.")

H3("4.3.2", "Data Flow Diagram – Level 0")
P("Figure 4.2 decomposes the single process of the context diagram into six major processes, five data stores and "
  "their data flows, using Gane-Sarson notation. Each process corresponds to a module or Airflow task. The diagram "
  "addresses requirements FR-01 to FR-13 and DR-05.")
FIGURE("fig4_2_dfd0.png", "Figure 4.2: Data Flow Diagram (Level 0)" + SUFFIX)
P("Data moves through the system in one direction: raw prices (D1) become processed splits (D2), which feed "
  "training (4.0) and backtesting (5.0). D2 is drawn twice, marked with the duplicate-store line, to avoid "
  "crossing flows. The training and validation splits flow only to process 4.0 and the test split only to "
  "process 5.0, which shows the separation required by DR-04. Process 4.0 writes to two stores: experiment "
  "metadata to the MLflow store (D3) and model files to the checkpoint store (D4). Process 5.0 reads the best "
  "checkpoint from D4, which makes the checkpoint store the hand-over point between training and evaluation.")

H3("4.3.3", "Use Case Diagram")
P("The use case diagram in Figure 4.3 identifies the actors that interact with the platform and the services the "
  "platform provides to them. Three human actors and three external systems were identified from the requirements. "
  "The Data Engineer maintains the data pipeline, the ML Engineer trains and evaluates agents, and the Energy "
  "Analyst consumes results. The Airflow Scheduler, the EIA API and the Email Service are external systems.")
FIGURE("fig4_3_usecase.png", "Figure 4.3: Use Case Diagram" + SUFFIX)
P("The Airflow Scheduler is associated with the four automated use cases (ingest, transform, train and backtest), "
  "which shows that the routine pipeline needs no human trigger. The «include» relationship from *Train DRL "
  "trading agent* to *Track experiment in MLflow* is used because every training run always logs to MLflow; "
  "tracking is not optional behaviour. Configuration is shared by the Data Engineer and the ML Engineer, since "
  "pipeline.yaml and drl.yaml control the data and model stages respectively. The Energy Analyst interacts only "
  "with results, which reflects the delimitation in Section 1.5.3 that the system does not execute live trades.")

H3("4.3.4", "Activity Diagram of the Daily Pipeline Run")
P("Figure 4.4 models the control flow of one scheduled run as it is defined in the Airflow DAG. It is included "
  "because the order of tasks, the parallel branch and the failure behaviour are not visible in the data flow "
  "diagram. It addresses FR-05, FR-12, FR-13 and NFR-05.")
FIGURE("fig4_4_activity.png", "Figure 4.4: Activity Diagram of the Daily Pipeline Run" + SUFFIX)
P("After transformation the flow forks: the data-quality report runs in parallel with training, so the report is "
  "produced even if training fails. The join uses Airflow's all_done trigger rule so the run always reaches the end "
  "node. The decision node after cleaning represents validate_dataframe: if a critical rule fails, the task raises "
  "an error, Airflow retries it twice at five-minute intervals and then sends a failure email. Only one run can be "
  "active at a time, which prevents two training runs from overwriting the same checkpoint directory.")

H2("4.4", "System Design Diagrams")
P("The design diagrams describe how the solution is organised to satisfy the analysed requirements. They follow "
  "the same narrative, diagram and interpretation sequence as Section 4.3.")

H3("4.4.1", "Reinforcement Learning Design")
P("Figure 4.5 shows the design of the learning problem, which is the core of objective iii. It defines what the "
  "agent observes, what it can do and how it is rewarded. The values shown are those used for the daily Henry Hub "
  "runs reported in Chapter 5.")
FIGURE("fig4_5_rl_design.png", "Figure 4.5: Reinforcement Learning Design" + SUFFIX)
P("The action is a single continuous value in [-1, 1] that sets a *target* position rather than an order size: -1 "
  "means no position, +1 means the maximum position, and the environment buys or sells the difference. This "
  "formulation keeps the action space bounded and lets the agent hold a position simply by repeating the same "
  "action. The design is long-only; short selling is out of scope. The observation combines 38 Z-score-normalised "
  "market features with three portfolio variables, so the agent knows both the market state and its own "
  "exposure. The designed reward is the scaled change in portfolio value from one step to the next, minus "
  "transaction costs, so profitable positions are rewarded and over-trading is penalised. A stop-loss terminates "
  "the episode if the portfolio falls 50 % below its initial value. Section 5.6.3 reports that the implemented "
  "reward did not match this design.")

H3("4.4.2", "Class Diagram")
P("The class diagram in Figure 4.6 shows the main classes and modules, their key attributes and operations, and "
  "their relationships. Python modules that expose functions rather than classes are shown with the «module» "
  "stereotype so that the whole implementation structure can be seen in one view.")
FIGURE("fig4_6_class.png", "Figure 4.6: Class Diagram" + SUFFIX, width=6.0)
P("EnergyTradingEnv inherits from the abstract gymnasium.Env, which is what allows any Stable-Baselines3 algorithm "
  "to train on it without modification. The environment owns a RunningNorm instance (composition), so "
  "normalisation statistics live and die with the environment. Both data sources produce a DataFrame for the "
  "transform module and are otherwise independent, which allows the synthetic generator to replace the API during "
  "testing. The train_drl and backtest modules depend on the environment but not on each other; they communicate "
  "only through saved checkpoints. The DAG calls the task wrappers in drl_tasks, which keeps orchestration code "
  "separate from pipeline logic, as proposed in Section 3.3.5.")

H3("4.4.3", "Logical Data Design")
P("The platform does not use a transactional database, so a relational schema is not appropriate. Figure 4.7 "
  "instead presents a logical data design of the two stores the platform relies on: the Parquet datasets in the "
  "S3 data lake and the entities of the MLflow tracking store. It addresses DR-02, DR-05 and DR-07.")
FIGURE("fig4_7_data_design.png", "Figure 4.7: Logical Data Design" + SUFFIX)
P("Every dataset is keyed by a UTC business-day timestamp, and the split label is carried by the file name rather "
  "than by a column, so each split can be loaded without reading the others. In the tracking store, an experiment "
  "groups many runs, and each run owns its parameters, metrics and artefacts. The dashed lineage link is "
  "deliberately weak: a run records the training and validation row counts but not a hash or path of the exact "
  "processed files. Chapter 5 shows that these row counts were nonetheless sufficient to reconstruct the dataset "
  "exactly; storing a dataset hash is recommended to make this link explicit.")

H3("4.4.4", "Deployment Diagram")
P("Figure 4.8 shows the three environments in which the platform is defined, and distinguishes the environment "
  "that was actually used for the reported results from those that were only configured. This addresses NFR-06 "
  "and the requirement in the course guidance that the deployment status of a system be stated accurately.")
FIGURE("fig4_8_deployment.png", "Figure 4.8: Deployment Diagram" + SUFFIX)
P("All reported training and evaluation runs were executed on the developer workstation using a local MLflow file "
  "store. The Docker Compose stack packages the same code with Airflow, an MLflow server backed by PostgreSQL and "
  "LocalStack for S3 emulation; its configuration was validated but the stack was not used for the reported runs. "
  "The AWS environment is fully described in Terraform but was not provisioned, so no part of the system is "
  "publicly deployed. GitHub Actions workflows define the path from a push to the main branch to an AWS deployment.")

# =============================================================================
# CHAPTER FIVE
# =============================================================================
H1("Chapter Five: System Implementation, Testing and Evaluation")

H2("5.1", "Introduction")
P("This chapter explains how the design in Chapter 4 was implemented, tested and evaluated. Because the project is "
  "an ML system, the chapter follows the data-to-model route recommended in the course guidance. Section 5.2 "
  "describes the implementation environment. Section 5.3 describes the dataset and its preparation. Section 5.4 "
  "describes the implemented modules and records deviations from the proposed design. Section 5.5 covers model "
  "training. Section 5.6 presents the test strategy, test cases, defects and the evaluation of the trained agents. "
  "Section 5.7 covers version control, and Section 5.8 discusses the results against the objectives and states "
  "the limitations.")
P("All results in this chapter come from recorded evidence: the MLflow tracking store (./mlruns), the pytest output, "
  "and a re-evaluation script (evaluate_eia_runs.py) that rebuilt the dataset and re-ran the saved models. The "
  "re-evaluation reproduced the logged MLflow metrics exactly (Section 5.6.4), which is the basis for trusting the "
  "numbers reported here.")

H2("5.2", "Implementation Environment")
P("The solution was developed, trained and tested on a single developer workstation. Container and cloud "
  "environments were defined but not used for the reported runs (Figure 4.8). The system is therefore a local "
  "prototype and is not publicly available.")

H3("5.2.1", "Hardware and Deployment Environment")
TABLE("Table 5.1: Implementation environment",
      ["Item", "Tested specification", "Purpose and evidence"],
      [["Development and training machine", "AMD Ryzen 5 5625U (6 cores, 12 threads), 15.4 GB usable RAM, "
                                             "Windows 11 Home (build 26200)", "All training, testing and "
                                                                              "evaluation. Values read from the "
                                                                              "operating system and from the "
                                                                              "system_info.txt saved inside each "
                                                                              "model file."],
       ["GPU", "AMD Radeon integrated graphics; not used", "PyTorch was the CPU build (GPU Enabled: False in the "
                                                          "saved models). Training on CPU met NFR-02."],
       ["Storage used", "29 MB MLflow store, 35 MB checkpoints, 1.8 MB reports", "Measured on disk after all runs."],
       ["Network", "Internet access to api.eia.gov", "Required only for ingestion (IR-01)."],
       ["Cloud (target)", "AWS us-east-1: S3, Glue, IAM, CloudWatch, SNS, ECR", "Defined in terraform/main.tf; not "
                                                                               "provisioned, so no cost was "
                                                                               "incurred."]],
      [1.4, 2.4, 2.7])

H3("5.2.2", "Software, Services and Configuration")
P("Table 5.2 lists the software versions that were actually used. These were read from the Python environment and "
  "confirmed by the system information stored inside each saved model. Several differ from the versions pinned in "
  "requirements.txt; this is recorded as defect DEF-07 in Section 5.6.3.")
TABLE("Table 5.2: Software versions used",
      ["Component", "Version used", "Role"],
      [["Python", "3.11.5", "Implementation language"],
       ["pandas / NumPy / pyarrow", "2.1.4 / 1.26.4 / 23.0.1", "Data handling and Parquet I/O"],
       ["Gymnasium", "1.2.3", "Trading environment interface"],
       ["Stable-Baselines3", "2.8.0", "PPO implementation"],
       ["PyTorch", "2.12.0 (CPU)", "Neural network back end"],
       ["MLflow", "3.12.0", "Experiment tracking (local file store)"],
       ["boto3 / moto", "1.34.103 / 5.2.2", "S3 access and S3 mocking in tests"],
       ["requests", "2.32.3", "EIA API client"],
       ["matplotlib", "3.8.2", "Backtest reports and figures"],
       ["pytest", "9.0.3", "Automated testing"],
       ["Apache Airflow", "2.8.1 (container image)", "Orchestration; defined in Dockerfile.airflow, not installed "
                                                     "on the workstation"],
       ["Docker / Terraform", "29.6.1 / 1.15.2", "Containers and infrastructure as code"]],
      [1.9, 1.8, 2.8])
P("Configuration is held in two YAML files. config/pipeline.yaml selects the data source (eia_api), the commodity "
  "(gas), the history start date (2010-01-01), the feature windows, the outlier threshold (6.0 standard deviations) "
  "and the split ratios. config/drl.yaml holds the environment, algorithm and MLflow settings. The EIA API key is "
  "read from the EIA_API_KEY environment variable, which is loaded from a local .env file; a search of the source "
  "code, configuration files and reports found no copy of the key outside .env (NFR-04).")

H3("5.2.3", "Deployment and Reproducibility")
P("The reported results can be reproduced on a machine with Python 3.11 and the versions in Table 5.2 by running "
  "the following steps from the project root with PYTHONPATH set to the project root:")
LIST(["Copy .env.example to .env and set EIA_API_KEY to a valid EIA key.",
      "Run pytest tests/ to confirm the environment (61 tests are expected to pass).",
      "Run python train_on_eia.py 300000 to fetch the data, train the agent, log the run to ./mlruns and write the "
      "backtest report to reports/backtest_eia/.",
      "Run python evaluate_eia_runs.py to rebuild the dataset as it was on the training date, re-evaluate the "
      "logged models and write reports/evaluation/evaluation_summary.json.",
      "Run mlflow ui --backend-store-uri ./mlruns to inspect the runs."])
P("A rerun of step iii will not reproduce the reported numbers exactly unless the data is cut at the same date, "
  "because the EIA series grows every day; evaluate_eia_runs.py handles this by trimming the data until the split "
  "sizes match those logged in MLflow. The Docker Compose stack can be started with docker compose up, and its "
  "configuration was validated with docker compose config.")

H2("5.3", "Dataset Description and Preparation")

H3("5.3.1", "Dataset Description")
P("The dataset consists of two daily price series published by the U.S. Energy Information Administration through "
  "the EIA Open Data API v2. The trading target is the Henry Hub natural gas spot price (series RNGWHHD, in $/MMBtu), "
  "which is the main benchmark price for U.S. natural gas. The second series is the WTI crude oil spot price "
  "(series RWTC, in $/bbl), which was included as an exogenous feature because oil and gas prices are related "
  "through fuel substitution and production. Both series are publicly available and contain no personal data.")
P(f"Each raw record contains a date, the gas price, the oil price and a source label. The data used for the "
  f"reported runs started on 4 January 2010 and ended on {EVAL['last_raw_date'][8:10].lstrip('0')} July 2026, "
  f"the last value published on the training date of 14 July 2026, giving {EVAL['raw_rows']:,} raw business-day "
  f"records with no missing values. Figure 5.1 shows the two series and the chronological split.")
FIGURE("fig5_1_price_series.png", "Figure 5.1: Henry Hub and WTI daily spot prices with the training, validation "
                                  "and test periods")
P("The figure shows why this series is a demanding test for a trading agent. Henry Hub prices were low and stable "
  "for long periods but are interrupted by sharp, short-lived spikes, many of them in winter. The validation period "
  "(May 2023 to December 2024) was one of the lowest-priced periods in the sample, with prices between $1.21 and "
  "$3.39, while the test period (December 2024 to July 2026) contains two large spikes. The three periods therefore "
  "have different price behaviour, which matters when the validation and test results are compared in Section 5.6.4.")

H3("5.3.2", "Data Preparation and Partitioning")
P("The raw data passed through the cleaning, validation and feature-engineering steps designed in Section 3.2.2, "
  "with the deviations listed in Section 5.4.7. Table 5.3 records the effect of each step on the reported dataset.")
TABLE("Table 5.3: Effect of each preparation step",
      ["Step", "Result on the reported dataset"],
      [["Fetch and join", f"{EVAL['raw_rows']:,} business days (4 January 2010 to 6 July 2026); WTI joined by date and "
                          "forward-filled; no missing values."],
       ["Duplicate removal", "No duplicate timestamps found."],
       ["Outlier masking (> 6 SD)", "5 Henry Hub values masked and then forward-filled: 17 February 2021 ($23.86), "
                                    "12 January 2024 ($13.20) and 23, 26 and 27 January 2026 ($30.72, $25.01, "
                                    "$17.19)."],
       ["Frequency regularisation", "Business-day frequency detected; 140 calendar gaps (mainly public holidays) "
                                    "added and forward-filled, giving 4,306 rows."],
       ["Validation", "All eight rules passed (target present, numeric, no nulls, non-negative, non-constant, at "
                      "least 100 rows, datetime index, monotonic index)."],
       ["Feature engineering", "38 numeric features; the first 168 rows dropped because the longest lag and rolling "
                               "window needed 168 previous values; 4,138 rows remain."],
       ["Chronological split", f"Training {S['train']['rows']:,} rows ({S['train']['start']} to {S['train']['end']}); "
                               f"validation {S['val']['rows']:,} ({S['val']['start']} to {S['val']['end']}); test "
                               f"{S['test']['rows']:,} ({S['test']['start']} to {S['test']['end']})."]],
      [1.8, 4.7])
P("The 38 features comprise 12 calendar features, six price lags (1, 6, 12, 24, 48 and 168 steps), 12 rolling "
  "statistics (mean, standard deviation, minimum and maximum over 6, 24 and 168 steps), seven technical indicators "
  "(RSI-14, MACD, three Bollinger Band values and two rates of change) and the WTI price. Every lag, rolling "
  "statistic and indicator is computed on the price shifted by one step, so no feature uses the price of the day "
  "being traded. This was verified by the unit test test_lag_values_shifted_correctly and by the split test "
  "test_no_data_leakage.")
P("Two side-effects of reusing a pipeline designed for hourly data were found. First, the feature names still "
  "carry an hourly suffix (for example price_usd_lag_168h), although on daily data a lag of 168 is 168 trading "
  "days. Second, four calendar features (hour_of_day, hour_sin, hour_cos and is_weekend) are constant on "
  "business-day data and carry no information. The outlier rule also had an important effect: the five masked "
  "values were genuine price spikes, so the cleaned series understates the January 2026 spike, whose true peak "
  "($30.72) was replaced by the previous value ($8.42). Both issues are recorded in Section 5.6.3.")

H2("5.4", "System Implementation")
P("The implementation follows the module structure in Figure 4.6 and the data flow in Figure 4.2. This section "
  "describes each component, links it to its requirements and identifies the evidence that it works.")

H3("5.4.1", "Data Ingestion Module")
P("The ingestion module (src/ingestion/ingest.py, FR-01 to FR-03) implements the EIAIngester class. During "
  "development, the route proposed in Chapter 3 was found to return only monthly data and no price target, so the "
  "client was rewritten to use two daily routes that were verified against the live API: natural-gas/pri/fut for "
  "series RNGWHHD and petroleum/pri/spt for series RWTC. The build_dataset method fetches the target commodity and "
  "the other commodity, joins them by date and forward-fills the exogenous series. Each request uses a 30-second "
  "timeout and up to three attempts with exponential back-off. The SyntheticEnergyGenerator class produces a seeded, "
  "hourly synthetic series with five variables, which is used by the unit tests and by the synthetic training runs. "
  "The run_ingestion entry point used by Airflow writes the result to S3 as Parquet.")

H3("5.4.2", "Transformation and Validation Module")
P("The transformation module (src/transformation/transform.py, FR-04 to FR-07) implements clean_raw, "
  "validate_dataframe, engineer_features and temporal_split. Because the project moved from hourly to daily data, "
  "clean_raw was extended to detect the data frequency from the median spacing of the timestamps and to regularise "
  "the series to hourly or business-day frequency, with a forward-fill limit of three hours or five business days. "
  "validate_dataframe applies eight rules; a failed critical rule raises an error that stops the Airflow task "
  "(Figure 4.4). The module was written as stateless functions that take and return a DataFrame, as proposed in "
  "Section 3.3.2, which made each step easy to unit-test.")

H3("5.4.3", "Trading Environment")
P("EnergyTradingEnv (src/ml/drl/env.py, FR-08) implements the Gymnasium interface. On each step it converts the "
  "action into a target position, clamps the trade to the available cash or the position held, ignores trades "
  "smaller than the minimum trade size, charges a transaction cost of 0.1 % of the trade value and updates cash, "
  "position and average entry price. The observation is built from the feature vector, normalised with a "
  "running mean and variance (RunningNorm) and clipped to five standard deviations, followed by the cash ratio, "
  "position ratio and normalised unrealised profit. In training mode each episode starts at a random point; in "
  "test mode it starts at the beginning of the data. The environment passes the Gymnasium check_env validation "
  "(test test_gymnasium_check_env).")

H3("5.4.4", "Training and Experiment Tracking")
P("The training module (src/ml/drl/train_drl.py, FR-09 and FR-10) builds the agent from configuration, so that PPO, "
  "SAC, TD3 or A2C can be selected without code changes. run_drl_training validates the environment, creates four "
  "parallel training environments, and attaches four callbacks: EvalCallback evaluates the agent on the "
  "validation split every 10,000 steps and saves the best checkpoint; CheckpointCallback saves a checkpoint every "
  "50,000 steps; MLflowCallback logs the PPO training statistics; and PortfolioMetricsCallback logs the portfolio "
  "return on the validation split. At the end of training the final model, the best model and the complete "
  "configuration are logged to MLflow as artefacts, and a final evaluation on the validation split is logged with "
  "the prefix final_. The helper script train_on_eia.py runs this module on the EIA data with daily settings: an "
  "episode length of 252 trading days, a maximum position of 25,000 units (so that $100,000 can be fully invested "
  "at about $3 to $4 per MMBtu) and a stop-loss of 50 %.")

H3("5.4.5", "Backtesting and Reporting")
P("The backtest module (src/ml/drl/backtest.py, FR-11 and NFR-07) walks the trained agent through the test split "
  "with deterministic actions and runs the two baselines in the same environment: Buy & Hold, which takes the "
  "maximum position on the first day and keeps it, and a Random Agent, which samples actions uniformly. It computes "
  "total and annualised return, Sharpe ratio, maximum drawdown, Calmar ratio, win rate, 95 % value at risk and "
  "profit factor. The annualisation was corrected during development to use 252 periods per year for daily data "
  "rather than the hourly default. The module produces the six-panel dashboard shown in Figure 5.2, which is the "
  "main way a human inspects what the agent did.")
FIGURE("backtest_eia_report.png", "Figure 5.2: Backtest dashboard produced by backtest.py for the PPO-300k agent "
                                  "(first 252 test days)")
P("The dashboard shows the cumulative return of the agent and the baselines, the price with buy and sell markers, "
  "the position held, the drawdown and the distribution of actions. Its axis labels still refer to hours and "
  "USD/MWh from the original hourly design, although the data is daily and in $/MMBtu (DEF-05). The figure is "
  "interpreted in Section 5.6.4.")

H3("5.4.6", "Orchestration, Infrastructure and Delivery")
P("The Airflow DAG (src/pipeline/dag.py, FR-12 and FR-13) defines the tasks start, ingest, transform, "
  "data_quality_report, train_drl, backtest_drl, notify_success and end, scheduled daily with two retries, a "
  "five-minute retry delay, a two-hour execution timeout and one active run at a time. Tasks pass S3 paths, the "
  "MLflow run ID and the backtest summary to each other through XCom. The DAG was not executed in this study: "
  "Airflow was not installed on the workstation, and the DAG imports DummyOperator, which is deprecated in Airflow "
  "2.8 (DEF-04). Infrastructure is defined in terraform/main.tf, which declares an S3 data lake with versioning, "
  "encryption, lifecycle rules and a public-access block, Glue crawlers and database, IAM roles, a CloudWatch log "
  "group with 30-day retention, an SNS topic with an email subscription and an ECR repository. The GitHub Actions "
  "workflow ci.yml defines lint, test, smoke-test, Docker build and Terraform validation jobs, and deploy.yml "
  "defines the image push, Terraform apply and DAG deployment. For offline inspection, trading_replay_app.py "
  "provides a Tkinter desktop application that replays a saved trajectory with play, pause, speed and scrub "
  "controls (FR-14).")

H3("5.4.7", "Deviations from the Proposed Design")
P("Several parts of the design proposed in Chapter 3 changed during implementation. Table 5.4 records each "
  "material deviation and its reason.")
TABLE("Table 5.4: Deviations from the Chapter 3 design",
      ["Area", "Proposed (Chapter 3)", "Implemented", "Reason"],
      [["Data", "Hourly electricity demand and daily gas price", "Daily Henry Hub gas price (target) and WTI oil "
                                                                  "price", "Proposed route returned monthly data "
                                                                           "with no price target"],
       ["Outlier rule", "4 standard deviations", "6 standard deviations", "Keep more genuine price spikes"],
       ["Reward", "10 % fee, idle penalty, drawdown penalty", "0.1 % fee and 50 % stop-loss; no idle or drawdown "
                                                             "penalty", "Simplified during development"],
       ["Network and budget", "[256, 256]; 200,000 steps", "[256, 256, 128]; 300,000 steps (daily) and 500,000 "
                                                           "(synthetic); 4 environments", "Configured in drl.yaml"],
       ["Quality gate and registry", "Mean reward > 0 gate; MLflow Model Registry", "Not implemented; models "
                                                                                   "stored as run artefacts",
        "Time constraints"],
       ["Forecasting", "Recursive 24-hour forecast to S3", "Not implemented; the agent outputs positions, not "
                                                           "forecasts", "Scope focused on the trading agent"],
       ["Orchestration", "Seven tasks with gate branching; SNS", "Six tasks without branching; EmailOperator; "
                                                                "not executed", "Gate not implemented"],
       ["Cloud", "AWS eu-west-2 with MWAA", "Terraform for us-east-1 without MWAA; not provisioned",
        "Cost of a student project"],
       ["User interface", "Web forecast dashboard (Figure 3.7)", "PNG backtest dashboard and Tkinter replay app",
        "No forecasting service to display"],
       ["Tracking store", "SQLite or PostgreSQL", "Local file store (./mlruns); PostgreSQL configured in Docker "
                                                  "Compose", "Local development"]],
      [1.1, 1.75, 2.05, 1.6])

H2("5.5", "Model Training and Selection")
P("Five training runs were logged to MLflow. Table 5.5 summarises them; the two smallest runs were smoke tests used "
  "to check the pipeline end to end. PPO was the only algorithm trained, using the hyperparameters in Figure 4.5, "
  "seed 42 and the default Stable-Baselines3 PPO settings for anything not listed. No hyperparameter search was "
  "performed. The checkpoint used for testing was the best checkpoint chosen by EvalCallback, which selects the "
  "checkpoint with the highest mean *reward* on the validation split; the test split played no part in selection.")
TABLE("Table 5.5: Training runs logged in MLflow",
      ["Run", "Data", "Steps (envs)", "Duration", "Validation return"],
      [["PPO smoke test", "Synthetic hourly", "500 (1)", "18 s", "+0.33 %"],
       ["PPO", "Synthetic hourly", "20,000 (1)", "2 min 4 s", "-6.97 %"],
       ["PPO", "Synthetic hourly", "500,000 (4)", "35 min 53 s", "-0.55 %"],
       ["PPO-10k", "EIA daily gas", "10,000 (4)", "1 min 0 s", pct(R10['val_logged_return_pct'])],
       ["PPO-300k", "EIA daily gas", "300,000 (4)", "14 min 57 s", pct(R300['val_logged_return_pct'])]],
      [1.3, 1.4, 1.3, 1.2, 1.3])
P("Figure 5.3 shows how the PPO-300k run evolved. The return on the validation episode rose to +16.7 % at 60,000 "
  "steps and then fell, ending at -6.9 % at 300,000 steps. At the same time the entropy loss rose steadily towards "
  "zero, which means the policy became more and more deterministic.")
FIGURE("fig5_2_training_curves.png", "Figure 5.3: Validation return and entropy loss during the PPO-300k training run")
P("The two curves together show that longer training did not produce a better trader on the validation data; the "
  "policy instead converged to a fixed behaviour. Section 5.6.3 explains why: the implemented reward did not "
  "contain the profit or loss from price movements, so the training signal could not favour profitable positions.")

H2("5.6", "Testing and Evaluation")

H3("5.6.1", "Test Strategy and Acceptance Criteria")
P("Testing combined four levels. Unit tests checked individual functions and classes with pytest, using the "
  "synthetic generator for repeatable data and moto to mock S3. Integration tests checked the live EIA client and "
  "the interaction between the environment, the trained models and MLflow. System tests ran the full data-to-"
  "backtest path through train_on_eia.py and evaluate_eia_runs.py. Configuration checks validated the Docker Compose "
  "and Terraform files. Most tests were black-box tests against expected outputs; the reward test (TC-11) was a "
  "white-box test designed after inspecting the step function. A test passed only if the observed result matched a "
  "result defined before the test was run. Every test was executed once on the workstation in Table 5.1, except "
  "where stated.")

H3("5.6.2", "Test Cases and Results")
P("The automated suite contained 61 unit tests: 8 for ingestion, 26 for transformation and 27 for the "
  "environment, training and backtest code. All 61 passed in 85 seconds, with 62 % statement coverage of src/. "
  "Coverage was highest for the environment (93 %), training (86 %) and transformation (83 %) modules, and zero for "
  "the DAG and the Airflow task wrappers, which need Airflow to run. Table 5.6 groups the unit tests into test cases "
  "and adds the integration, system and configuration tests.")
TABLE("Table 5.6: Test cases and results",
      ["ID", "Requirement", "Scenario and expected result", "Actual result and verdict"],
      [["TC-01", "FR-04, FR-05", "9 tests: cleaning removes injected duplicates, gives a sorted UTC index and no null "
                                 "target; invalid data is rejected.", "9/9 passed. Pass"],
       ["TC-02", "FR-06", "14 tests: calendar, lag, rolling and indicator features are created, bounded and "
                          "shifted correctly; no nulls remain.", "14/14 passed. Pass"],
       ["TC-03", "FR-07, DR-04", "3 tests: split sizes, chronological order and no overlap between splits.",
        "3/3 passed. Pass"],
       ["TC-04", "FR-03", "8 tests: synthetic data has the expected rows, columns, bounds and is identical for the "
                          "same seed.", "8/8 passed. Pass"],
       ["TC-05", "FR-08", "18 tests: spaces, step outputs, buy and sell behaviour, no negative cash or position, "
                          "termination, check_env.", "18/18 passed. Pass"],
       ["TC-06", "FR-09", "5 tests: PPO and A2C agents build, an invalid algorithm is rejected, the agent learns and "
                          "is evaluated.", "5/5 passed. Pass"],
       ["TC-07", "FR-11", "4 tests: metrics keys, positive return sign, Buy & Hold and Random baselines run.",
        "4/4 passed. Pass"],
       ["TC-08", "FR-01, IR-01", "Live call to build_dataset('2010-01-01', 'gas'): both series returned and joined "
                                 "with no nulls.", f"{EVAL['raw_rows']:,} rows to the cut-off date; no nulls. Pass"],
       ["TC-09", "NFR-01", "Rebuild the July dataset and re-evaluate both EIA models; logged validation return "
                           "reproduced within 0.01 points.", f"{pct(R10['val_reproduced_return_pct'])} and "
                                                             f"{pct(R300['val_reproduced_return_pct'])}, identical "
                                                             f"to the logged values. Pass"],
       ["TC-10", "NFR-02", "300,000-step PPO run completes within 2 hours on CPU.", "14 min 57 s. Pass"],
       ["TC-11", "FR-08", "Hold a full position while the price rises from 20 to 160 with zero fees; the reward "
                          "should be positive.", "Portfolio rose from 100,000 to 107,000 but every reward was 0. "
                                                 "Fail (DEF-01)"],
       ["TC-12", "FR-11", "Backtest walks the whole test split (414 steps).", "Stopped after 252 steps. Fail "
                                                                              "(DEF-02); retest with corrected "
                                                                              "episode length: 414 steps. Pass"],
       ["TC-13", "FR-10", "Logged portfolio_max_drawdown_pct lies between -100 and 0.", "-64,450,496.7 logged for "
                                                                                       "PPO-300k. Fail (DEF-03)"],
       ["TC-14", "FR-12", "DAG imports and runs under Airflow 2.8.1.", "Not executed; Airflow not installed "
                                                                      "locally (DEF-04)"],
       ["TC-15", "NFR-06", "docker compose config validates docker-compose.yml.", "Valid; warning that the version "
                                                                                  "attribute is obsolete. Pass"],
       ["TC-16", "NFR-03", "terraform init and validate succeed.", "Not completed: the AWS provider plugin failed to "
                                                                   "start on the workstation"]],
      [0.75, 1.15, 2.4, 2.2])

H3("5.6.3", "Defects and Retesting")
P("Table 5.7 records the defects found during testing and evaluation. None of the failed results was removed from "
  "this report.")
TABLE("Table 5.7: Defect log",
      ["ID", "Severity", "Defect and effect", "Status"],
      [["DEF-01", "Critical", "In step(), the previous and new portfolio values are both calculated at the same "
                              "day's price, and the price change happens only after the step ends. The reward "
                              "therefore contains only transaction costs and the stop-loss penalty, never the "
                              "profit or loss from price movements, so the agent was trained to avoid trading "
                              "costs rather than to trade profitably.", "Open. Fix: compute the reward after "
                                                                        "advancing to the next price, then "
                                                                        "retrain."],
       ["DEF-02", "High", "run_backtest stops at episode_length (252), so the original backtest covered only 252 of "
                          "the 414 test days.", "Mitigated in evaluate_eia_runs.py; source fix pending."],
       ["DEF-03", "Medium", "PortfolioMetricsCallback computes drawdown on a percentage-return series whose peak is "
                            "close to zero, producing meaningless values.", "Open"],
       ["DEF-04", "Medium", "The DAG imports the deprecated DummyOperator instead of EmptyOperator.", "Open"],
       ["DEF-05", "Low", "Backtest chart labels refer to hours and USD/MWh for daily $/MMBtu data.", "Open"],
       ["DEF-06", "Low", "Four calendar features are constant on daily data; feature names carry an hourly suffix; "
                         "the outlier rule masks genuine spikes.", "Open"],
       ["DEF-07", "Medium", "requirements.txt pins Stable-Baselines3 2.2.1, Gymnasium 0.29.1, PyTorch 2.2.0 and "
                            "MLflow 2.10.2, but the results were produced with the versions in Table 5.2.", "Open"],
       ["DEF-08", "Low", "Each evaluation environment fits its normaliser on its own split, and the outlier "
                         "statistics are computed on the whole series, so some statistics of later data reach "
                         "earlier stages.", "Open"],
       ["DEF-09", "Low", "The Random Agent seeds the environment but not the action sampler, so its results differ "
                         "between executions.", "Open; results reported as distributions."]],
      [0.8, 0.95, 3.4, 1.35])
P("DEF-01 is the most important finding of the project. It was discovered while trying to explain why the PPO-300k "
  "agent stopped trading. Test TC-11 confirmed it directly: with zero fees, a full position held while the price "
  "rose eightfold produced a reward of exactly zero on every step. Because the rewards in every reported run "
  "followed the same code, none of the trained agents received a signal that linked positions to price "
  "movements.")

H3("5.6.4", "Model Evaluation")
P("The two agents trained on real data were evaluated on the held-out test period with their best checkpoints. "
  "Because of DEF-02 the original backtest covered only the first 252 test days, so the evaluation was repeated on "
  f"the full test period of {F3['agent']['n_steps']} trading steps (4 December 2024 to 6 July 2026). Table 5.8 "
  "reports both windows, and Figure 5.4 shows the full-period results.")
rows = []
for label, w in [("First 252 days", "first_252"), ("Full test period", "full_test")]:
    for name, R in [("PPO-300k", R300), ("PPO-10k", R10)]:
        a = R[w]["agent"]
        rows.append([label, name, pct(a["total_return_pct"]), f"{a['sharpe']:.2f}", pct(a["max_drawdown_pct"]),
                     f"{a['total_trades']}"])
    b = R300[w]["buy_and_hold"]
    rows.append([label, "Buy & Hold", pct(b["total_return_pct"]), f"{b['sharpe']:.2f}", pct(b["max_drawdown_pct"]),
                 "1"])
TABLE("Table 5.8: Test-period results of the EIA-trained agents and Buy & Hold",
      ["Window", "Strategy", "Total return", "Sharpe", "Max drawdown", "Trades"], rows,
      [1.25, 1.1, 1.15, 0.8, 1.25, 0.75])
FIGURE("fig5_3_test_equity.png", "Figure 5.4: Cumulative return over the full test period (top) and PPO-300k "
                                 "position with Henry Hub price (bottom)")
P(f"On paper, PPO-300k performed very well: a {pct(F3['agent']['total_return_pct'])} return over the full test "
  f"period against {pct(F3['buy_and_hold']['total_return_pct'])} for Buy & Hold, with a maximum drawdown of "
  f"{pct(F3['agent']['max_drawdown_pct'])} instead of {pct(F3['buy_and_hold']['max_drawdown_pct'])}. Figure 5.4 "
  "shows how this result arose. The agent held a large position during the first weeks of the test period, was "
  "still holding it when the price jumped from $4.30 per MMBtu on 16 January 2025 to $9.86 on 17 January (the value "
  "was carried forward over the 20 January public holiday before falling to $4.40 on 21 January), sold at the "
  f"high price and then stayed out of the market. It held no position on {F3['agent']['days_flat']} of the "
  f"{F3['agent']['n_steps']} test days, and almost all of its return came from that single price movement. The "
  "action histogram in Figure 5.2 confirms this: most actions were exactly -1, the action for holding nothing.")
P(f"Three further pieces of evidence show that this result does not demonstrate a learned trading skill. First, "
  f"the same agent lost {abs(R300['val_logged_return_pct']):.2f} % on the validation period. Second, the "
  f"distribution of 30 Random Agents on the same test period was very wide (mean "
  f"{pct(F3['random_return_mean'])}, standard deviation {F3['random_return_std']:.1f} points, best "
  f"{pct(F3['random_return_max'])}), and {F3['random_beats_agent_pct']:.0f} % of the random agents beat PPO-300k. "
  f"Because of DEF-09, a second execution gave a different random distribution (mean "
  f"{pct(F1['random_return_mean'])}, standard deviation {F1['random_return_std']:.1f} points); the conclusion that "
  "random positions can produce returns of this size on this test period holds in both. Third, because of DEF-01 the "
  "agent was never rewarded for profit, so the profitable exit cannot be attributed to the policy having learned "
  "from prices. The most likely explanation is that the policy converged towards holding nothing, which is the "
  "cheapest behaviour under a cost-only reward, and that the early holding period happened to coincide with the "
  "spike.")
P(f"PPO-10k, trained for only 10,000 steps, kept an almost constant position of 10,000 to 15,000 units "
  f"(about half the maximum) and traded every day. It returned {pct(F1['agent']['total_return_pct'])} over the "
  f"full test period against {pct(F1['buy_and_hold']['total_return_pct'])} for Buy & Hold, with a similar "
  "drawdown. This is the behaviour expected of an almost untrained policy whose actions stay near zero, which "
  "maps to a half-sized position. On the synthetic hourly data, the 500,000-step agent returned +0.11 % on its "
  "720-hour test window against +0.03 % for Buy & Hold, with a smaller drawdown (-1.67 % against -3.21 %); the "
  "Random Agent curve in the same saved report finished at about +2 %, read from the chart because its exact "
  "value was not logged.")

H2("5.7", "Version Control and Implementation Evidence")
P("Project repository: https://github.com/GlynnCarson/energy-trading-mlops (private; the supervisor is added as a "
  "collaborator to give access).")
P("The repository contains the source code, tests, configuration files, a README, the requirements file, an "
  "environment template (.env.example), Dockerfiles, a Docker Compose file, the Terraform configuration, the two "
  "GitHub Actions workflows, the report-generation scripts in docs/report/ and the MLflow tracking store (mlruns/) "
  "from which the Chapter 5 results were reproduced. A .gitignore file excludes the .env file that holds the API key, "
  "Python build artefacts, intermediate training checkpoints, Terraform state and local tool settings. The repository "
  "was created in November 2026, after the implementation was complete, so its commit history records the finished "
  "system rather than the development process; the MLflow run records, with their start times and stored "
  "configurations, are the main evidence of how the models were developed. The README still describes the original "
  "hourly synthetic setup and should be updated to the daily EIA configuration.")
P("Beyond source control, the MLflow store provides the implementation evidence for the model work. Each run "
  "records its parameters, step-by-step metrics, the full configuration file and the trained models, and the saved "
  "models record the library versions used. This evidence made it possible to reconstruct the dataset, reproduce "
  "the logged results exactly and locate defects DEF-01 to DEF-03 months after the runs were made.")

H2("5.8", "Evaluation Results, Discussion and Limitations")
P("Table 5.9 evaluates each specific objective against its evidence.")
TABLE("Table 5.9: Evaluation of the specific objectives",
      ["Objective", "Evidence", "Outcome"],
      [["i. Investigate forecasting and trading techniques", "Literature review in Chapter 2", "Achieved"],
       ["ii. Ingestion and transformation pipeline", "Live EIA ingestion (TC-08); 4,138-row leakage-free feature set; "
                                                     "34 data tests passed (TC-01 to TC-04). Daily data used instead "
                                                     "of hourly.", "Achieved, with a change of data frequency"],
       ["iii. PPO agent in a custom environment with costs, idle and drawdown penalties",
        "Environment and PPO training implemented and tested (TC-05, TC-06); idle and drawdown penalties not "
        "implemented; reward excluded profit and loss (DEF-01).", "Partially achieved"],
       ["iv. MLflow tracking, registry, quality gate and daily Airflow DAG",
        "All runs tracked and reproducible (TC-09); registry and quality gate not implemented; DAG defined but not "
        "executed (TC-14).", "Partially achieved"],
       ["v. Test and validate on held-out data, integration, cloud and CI/CD",
        "61 unit tests passed; held-out backtests reproduced; integration and system tests run; cloud and CI/CD "
        "defined but not executed (TC-16).", "Partially achieved"]],
      [2.0, 3.3, 1.2])
P("The data and MLOps parts of the project produced the strongest evidence. The pipeline retrieved and prepared "
  "16 years of real market data without leakage, and the experiment tracking was good enough to reconstruct the "
  "exact dataset of a run made four months earlier and reproduce its logged results to four decimal places. This "
  "supports the argument in Chapter 2, drawn from Kreuzberger et al. (2023) and Sculley et al. (2015), that "
  "tracking and reproducibility are what separate a maintainable ML system from a one-off script. In this project "
  "it was the tracking evidence that exposed the reward defect: without the logged configuration and models, the "
  "+124 % backtest result could easily have been reported as a success.")
P("The trading results do not answer research question iii as it was posed. With the implemented reward, the "
  "cumulative reward cannot be positive, because every reward is either zero or a cost. Measured by portfolio "
  "return instead, the evidence is mixed and does not show that the agent learned to trade: the best test result "
  "came from a single spike, the same agent lost money on the validation period, and random positions produced "
  "returns of similar size. Cao, Chen and Fu (2020) found that reward shaping with transaction costs and risk "
  "penalties produced realistic trading behaviour; the present results neither support nor contradict that "
  "finding, because the profit term of the reward was missing.")
P("The findings are also limited by the following factors:")
LIST(["Only one market (Henry Hub natural gas) and one period were tested, and the 414-day test period was "
      "dominated by two weather-season price spikes, so results are highly sensitive to a few days.",
      "Each configuration was trained once with a single seed and without a hyperparameter search, so the "
      "variability of the training process is unknown.",
      "Trades were executed at the same day's published spot price with a fixed 0.1 % cost, without bid-ask "
      "spreads, market impact or limits on position size relative to market volume. Rows added for public "
      "holidays carry forward-filled prices, and the simulator allows trading on them.",
      "The outlier rule replaced five genuine price spikes, including the true January 2026 peak, so the test data "
      "understates the most extreme event in the test period.",
      "The Airflow DAG, the Docker Compose stack, the AWS infrastructure and the CI/CD workflows were not executed, "
      "so the orchestration and deployment requirements are supported by configuration evidence only.",
      "Because of DEF-01, all trading results describe agents trained without a profit signal; they should be "
      "regarded as a test of the pipeline rather than of PPO as a trading method."])
P("The conclusions, recommendations and future work that follow from these results are presented in Chapter 6.")

# ─────────────────────────────────────────────────────────────────────────────
# Move the new chapters into place (after Chapter 3, before References)
# ─────────────────────────────────────────────────────────────────────────────
_new = [el for el in body if el.tag != qn("w:sectPr")][_appended_from:]
for el in _new:
    INSERT_BEFORE.addprevious(el)


def _find(pred):
    return [el for el in body if el.tag == qn("w:p") and pred(_text(el).strip())]


def _clone_with_text(src, parts):
    """Copy paragraph `src` and replace the text of its w:t elements in order with `parts`."""
    new = copy.deepcopy(src)
    ts = list(new.iter(qn("w:t")))
    for t, s in zip(ts, parts):
        t.text = s
        t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in ts[len(parts):]:
        t.text = ""
    return new


def _set_style(p, style_id):
    pPr = p.find(qn("w:pPr"))
    if pPr is None:
        pPr = OxmlElement("w:pPr"); p.insert(0, pPr)
    st = pPr.find(qn("w:pStyle"))
    if st is None:
        st = OxmlElement("w:pStyle"); pPr.insert(0, st)
    st.set(qn("w:val"), style_id)


# ── List of Figures ──────────────────────────────────────────────────────────
fig37 = _find(lambda t: t.startswith("Figure 3.7:") and len(t) < 60)[0]
fig32 = _find(lambda t: t.startswith("Figure 3.2:") and len(t) < 60)[0]
fig32.addprevious(_clone_with_text(fig37, ["Figure 3.1: Agile Development Lifecycle"]))
new_figs = ["Figure 4.1: Context Diagram", "Figure 4.2: Data Flow Diagram (Level 0)", "Figure 4.3: Use Case Diagram",
            "Figure 4.4: Activity Diagram of the Daily Pipeline Run", "Figure 4.5: Reinforcement Learning Design",
            "Figure 4.6: Class Diagram", "Figure 4.7: Logical Data Design", "Figure 4.8: Deployment Diagram",
            "Figure 5.1: Henry Hub and WTI Daily Spot Prices with the Training, Validation and Test Periods",
            "Figure 5.2: Backtest Dashboard for the PPO-300k Agent (First 252 Test Days)",
            "Figure 5.3: Validation Return and Entropy Loss during PPO-300k Training",
            "Figure 5.4: Cumulative Return over the Full Test Period and PPO-300k Position"]
anchor = fig37
for f in new_figs:
    n = _clone_with_text(fig37, [f]); anchor.addnext(n); anchor = n

# ── List of Tables ───────────────────────────────────────────────────────────
t31 = _find(lambda t: t.startswith("Table 3.1:") and len(t) < 80)[0]
new_tabs = ["Table 4.1: Functional Requirements", "Table 4.2: Non-Functional Requirements",
            "Table 4.3: Data Requirements", "Table 4.4: Interface and Integration Requirements",
            "Table 4.5: Requirements Traceability Matrix", "Table 5.1: Implementation Environment",
            "Table 5.2: Software Versions Used", "Table 5.3: Effect of Each Preparation Step",
            "Table 5.4: Deviations from the Chapter 3 Design", "Table 5.5: Training Runs Logged in MLflow",
            "Table 5.6: Test Cases and Results", "Table 5.7: Defect Log",
            "Table 5.8: Test-Period Results of the EIA-Trained Agents and Buy & Hold",
            "Table 5.9: Evaluation of the Specific Objectives"]
anchor = t31
for t in new_tabs:
    n = _clone_with_text(t31, [t, ""])  # same tab-leader layout as Table 3.1, page number left blank
    for r in list(n.iter(qn("w:r")))[1:]:
        n.remove(r)
    anchor.addnext(n); anchor = n

# ── List of Abbreviations (alphabetical insertion) ───────────────────────────
abbrev_new = {"CPU": "Central Processing Unit", "FR": "Functional Requirement", "GPU": "Graphics Processing Unit",
              "HTTPS": "Hypertext Transfer Protocol Secure", "IAM": "Identity and Access Management",
              "JSON": "JavaScript Object Notation", "MMBtu": "Million British Thermal Units",
              "NFR": "Non-Functional Requirement", "SD": "Standard Deviation",
              "SMTP": "Simple Mail Transfer Protocol", "UI": "User Interface", "WTI": "West Texas Intermediate"}
def _abbr_key(p):
    return (list(p.iter(qn("w:t")))[0].text or "").strip().upper()


start = _find(lambda t: t == "List of Abbreviations")[0]
abbr_paras, el = [], start.getnext()
while el is not None and not _text(el).strip().startswith("Chapter One"):
    if len(list(el.iter(qn("w:t")))) >= 2 and len(list(el.iter(qn("w:tab")))) >= 1:
        abbr_paras.append(el)
    el = el.getnext()
template = abbr_paras[0]
for ab, full in abbrev_new.items():
    n = _clone_with_text(template, [ab, " " + full])
    before = [p for p in abbr_paras if _abbr_key(p) < ab.upper()]
    if before:
        before[-1].addnext(n)
    else:
        abbr_paras[0].addprevious(n)
    abbr_paras.append(n)
    abbr_paras.sort(key=lambda p: list(body).index(p))

# ── References ───────────────────────────────────────────────────────────────
hossain = _find(lambda t: t.startswith("Hossain, E."))[0]
hossain.addnext(copy.deepcopy(hossain.getnext()))  # empty spacer paragraph, as between the other references
hossain.getnext().addnext(_clone_with_text(hossain, [
    "International Organization for Standardization (2023) ISO/IEC 25010:2023 Systems and software engineering "
    "– Systems and software Quality Requirements and Evaluation (SQuaRE) – Product quality model. "
    "Geneva: ISO."]))

# ── Formatting fixes listed in the project notes ─────────────────────────────────────
# Chapter Three: make it a real Heading 1 preceded by a page-break paragraph, like the other chapters.
ch3 = _find(lambda t: t == "Chapter Three: Development Methodology")[0]
for r in list(ch3.iter(qn("w:r"))):
    if r.find(qn("w:br")) is not None and not _text(r):
        ch3.remove(r)
for r in ch3.iter(qn("w:r")):
    rPr = r.find(qn("w:rPr"))
    if rPr is not None:
        for tag in ("w:b", "w:bCs", "w:sz", "w:szCs", "w:color"):
            e = rPr.find(qn(tag))
            if e is not None:
                rPr.remove(e)
_set_style(ch3, "Heading1")
ch3_par = doc.paragraphs[[p._p for p in doc.paragraphs].index(ch3)]
_fmt(ch3_par, WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
pb = OxmlElement("w:p"); r = OxmlElement("w:r"); br = OxmlElement("w:br"); br.set(qn("w:type"), "page")
r.append(br); pb.append(r); ch3.addprevious(pb)

# Section 2.5 and its subsections: real Heading 2 / Heading 3 so they appear in the Table of Contents.
for p in _find(lambda t: re.match(r"^2\.5(\.\d)?[A-Z\s]", t) is not None and len(t) < 70):
    num = re.match(r"^(2\.5(?:\.\d)?)", _text(p).strip()).group(1)
    style = "Heading3" if num.count(".") == 2 else "Heading2"
    _set_style(p, style)
    for r in p.iter(qn("w:r")):
        rPr = r.find(qn("w:rPr"))
        if rPr is not None:
            for tag in ("w:b", "w:bCs", "w:color"):
                e = rPr.find(qn(tag))
                if e is not None:
                    rPr.remove(e)
    pPr = p.find(qn("w:pPr"))
    jc = pPr.find(qn("w:jc")) if pPr is not None else None
    if jc is not None:
        pPr.remove(jc)

# Fig 3.1 caption: proposal caption format, Times New Roman 11 pt italic, centred.
f31 = _find(lambda t: t.startswith("Fig 3.1"))[0]
for child in list(f31):
    if child.tag == qn("w:r"):
        f31.remove(child)
pPr = f31.find(qn("w:pPr"))
if pPr is not None and pPr.find(qn("w:rPr")) is not None:
    pPr.remove(pPr.find(qn("w:rPr")))
f31_par = doc.paragraphs[[p._p for p in doc.paragraphs].index(f31)]
_runs(f31_par, "Figure 3.1: Agile Development Lifecycle" + SUFFIX, size=11, italic_all=True)
_fmt(f31_par, WD_ALIGN_PARAGRAPH.CENTER)

doc.save(str(TARGET))
print("Updated", TARGET)
print("Backup of the original:", BACKUP)
