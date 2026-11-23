"""
Applies the supervisor's feedback (17 Nov 2026) to energy_trading_mlops_proposal_v17.docx, in place:

  * Chapter 3 keeps only the discussion of the analysis and design diagrams; Figures 3.2-3.7 are removed
    and the text of Sections 3.4 and 3.5 is rewritten to describe the diagrams drawn in Chapter 4.
  * Chapter 4 contains exactly the diagram types discussed in Chapter 3 - use case, class, entity
    relationship, database schema, system architecture and user interface (Figures 4.1-4.6) - all drawn
    from the implemented system. The context, data-flow, activity, RL-design and deployment diagrams are removed.
  * Cross-references in Chapters 4 and 5, the traceability table and the List of Figures are updated.

    python docs/report/make_diagrams.py
    python docs/report/restructure_ch3_ch4.py

One-shot: refuses to run if Figure 3.2 is no longer in Chapter 3.
"""

import copy
import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Pt, Inches

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DOCX = ROOT / "energy_trading_mlops_proposal_v17.docx"
FIG = HERE / "figures"
SUFFIX = " – Energy Trading MLOps Platform"

doc = Document(str(DOCX))
body = doc.element.body
W_P = qn("w:p")


def text(el):
    """Paragraph text including tab characters (heading numbers are separated from titles by a w:tab)."""
    out = []
    for r in el.iter(qn("w:r")):
        for child in r:
            if child.tag == qn("w:t"):
                out.append(child.text or "")
            elif child.tag == qn("w:tab"):
                out.append("\t")
    return "".join(out)


def style_of(el):
    s = el.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return s.get(qn("w:val")) if s is not None else None


def find_one(pred, start=None):
    hits = [el for el in body if el.tag == W_P and pred(text(el).strip(), el)]
    if start is not None:
        idx = list(body).index(start)
        hits = [h for h in hits if list(body).index(h) > idx]
    if not hits:
        raise SystemExit("not found")
    return hits[0]


def heading(prefix, level):
    return find_one(lambda t, el: style_of(el) == f"Heading{level}" and t.startswith(prefix))


ch3 = find_one(lambda t, el: t == "Chapter Three: Development Methodology" and style_of(el) == "Heading1")
_ch3_idx = list(body).index(ch3)
captions_ch3 = [el for i, el in enumerate(body) if i > _ch3_idx and el.tag == W_P
                and text(el).startswith("Figure 3.2: Use Case Diagram")]
if not captions_ch3:
    sys.exit("Figure 3.2 is no longer in Chapter 3 - this restructuring has already been applied.")

# ── paragraph builders (formatting as in the proposal styles) ──────────────────────────


def _fmt(p, align, keep_next=False):
    pf = p.paragraph_format
    pf.line_spacing = 1.5
    if align is not None:
        pf.alignment = align
    if keep_next:
        pf.keep_with_next = True


def make_para(txt, style="Normal", align=WD_ALIGN_PARAGRAPH.JUSTIFY, size=None, italic=False, keep_next=False):
    p = doc.add_paragraph(style=style)
    r = p.add_run(txt)
    if size:
        r.font.size = Pt(size)
    if italic:
        r.italic = True
    _fmt(p, align, keep_next)
    el = p._p
    el.getparent().remove(el)
    return el


def make_figure(fname, caption, width=6.3):
    p = doc.add_paragraph(style="Normal")
    p.add_run().add_picture(str(FIG / fname), width=Inches(width))
    _fmt(p, WD_ALIGN_PARAGRAPH.CENTER, keep_next=True)
    pic = p._p
    pic.getparent().remove(pic)
    return [pic, make_para(caption, align=WD_ALIGN_PARAGRAPH.CENTER, size=11, italic=True)]


def H2(num, t):
    return make_para(f"{num}\t{t}", style="Heading 2", align=None, keep_next=True)


def H3(num, t):
    return make_para(f"{num}\t{t}", style="Heading 3", align=None, keep_next=True)


def insert_before(anchor, elements):
    for el in elements:
        anchor.addprevious(el)


def remove_between(start_el, stop_el, include_start=False):
    """Remove the siblings strictly between start_el and stop_el (optionally start_el itself)."""
    el = start_el if include_start else start_el.getnext()
    while el is not None and el is not stop_el:
        nxt = el.getnext()
        body.remove(el)
        el = nxt


# =============================================================================
# 1. Chapter 3: remove Figures 3.2-3.7 and rewrite Sections 3.4 and 3.5
# =============================================================================
CH3_TEXT = {
    "3.4\tSystem Analysis": [
        "This section discusses the analysis models for the system. Three diagrams are used: a use case diagram "
        "showing the actors and the functions the system provides to them, a class diagram showing the principal "
        "classes and modules and how they relate, and an entity relationship diagram showing the data entities and "
        "how they are linked. The diagrams themselves are presented in Section 4.3 of Chapter 4, where they are "
        "drawn from the implemented system."],
    "3.4.1\tUse Case Diagram": [
        "The use case diagram (Figure 4.1) identifies three human actors and three external systems. The Data "
        "Engineer configures the pipeline, ingests energy price data, cleans, validates and engineers the features, "
        "and reviews the data-quality report. The ML Engineer configures and trains the DRL trading agent, tracks "
        "each experiment in MLflow and backtests the agent against baseline strategies. The Energy Analyst views "
        "the backtest report and the trade replay and receives pipeline notifications. The Airflow Scheduler "
        "triggers the automated use cases every day, the EIA API v2 supplies the price data, and an email service "
        "delivers the notifications. Training always records the experiment in MLflow, which is modelled as an "
        "include relationship."],
    "3.4.2\tClass Diagram": [
        "The class diagram (Figure 4.2) shows the main classes and modules of the system and how they relate. "
        "EIAIngester and SyntheticEnergyGenerator are the two data sources; both produce a DataFrame for the "
        "transform module, which cleans, validates, engineers features and splits the data chronologically. "
        "EnergyTradingEnv inherits from the abstract gymnasium.Env class and owns a RunningNorm instance that "
        "normalises its observations. The train_drl module creates the trading environments, builds the PPO agent "
        "and registers two callbacks, MLflowCallback and PortfolioMetricsCallback, which log training and "
        "portfolio metrics to MLflow. The backtest module walks a trained agent through the test split and compares "
        "it with Buy & Hold and Random baselines. The Airflow DAG calls the task wrappers in drl_tasks, and the "
        "helpers module provides configuration loading and S3 input and output to the other components."],
    "3.4.3\tEntity Relationship Diagram": [
        "The entity relationship diagram (Figure 4.3) models the data entities kept in the S3 data lake and the "
        "MLflow tracking store. RAW_ENERGY holds the ingested daily observations keyed by timestamp, and each "
        "ingestion produces three PROCESSED_SPLIT datasets (training, validation and test) that carry the "
        "engineered features and the target price. Each processed dataset is summarised by a DQ_REPORT, and the "
        "test split is evaluated in a BACKTEST_REPORT. On the tracking side, an EXPERIMENT contains many RUN "
        "records, and each run logs many PARAM, METRIC and ARTIFACT entries; the artefacts include the trained "
        "models and the full configuration file. A run is linked to the dataset it used through the training and "
        "validation row counts it logs."],
    "3.5.1\tDatabase Schema": [
        "The physical database schema is shown in Figure 4.4 in Chapter 4. The system uses two data stores. The "
        "first is the MLflow tracking database. When MLflow runs as a server in the Docker Compose stack it stores "
        "its records in SQLite; when it runs locally, the same entities are kept as files in a local file store. "
        "The schema used by the platform comprises five tables: experiments, runs, params, metrics and tags. The "
        "experiments table has a one-to-many relationship with runs, which in turn has one-to-many relationships "
        "with params, metrics and tags through the run_uuid foreign key. Model files and configuration files are "
        "stored as artefacts under each run's artifact location rather than in the database.",
        "The second store is the AWS Glue Data Catalog, which provides an external schema over the Parquet files "
        "in the S3 data lake. Two crawlers, each scheduled hourly, maintain one table over the raw zone "
        "(raw/energy/) and one over the processed zone (processed/energy/). Each table is partitioned by its "
        "folder name, which records the ingestion date range for raw data and the processing time for processed "
        "data, so that queries through Amazon Athena scan only the partitions they need.",
        "Lineage between the two stores is recorded through the training and validation row counts that every "
        "MLflow run logs as parameters, which identify the processed dataset on which a model was trained."],
    "3.5.2\tSystem Architecture": [
        "The system architecture is presented in Figure 4.5 in Chapter 4. It follows a layered design: five "
        "horizontal layers describe the flow of data through the platform, and a vertical platform layer provides "
        "the supporting infrastructure and DevOps services.",
        "The data sources layer contains the EIA API v2, which supplies daily Henry Hub natural gas and WTI crude "
        "oil prices, and a seeded synthetic data generator used for testing. The ingestion and storage layer writes "
        "raw and processed Parquet files to separate zones of an S3 data lake, catalogued by AWS Glue crawlers. The "
        "processing and machine learning layer cleans and validates the data, engineers 38 features, wraps the "
        "result in the Gymnasium trading environment, trains the PPO agent with Stable-Baselines3 and logs every run "
        "to MLflow.",
        "The orchestration layer contains the Airflow DAG, which runs the ingest, transform, train, backtest and "
        "notify tasks daily, and the local run scripts that call the same modules directly. The evaluation and "
        "delivery layer produces the backtest report against Buy & Hold and Random baselines, the offline "
        "trade-replay application and the email notification. Alongside these, the platform layer provides "
        "Terraform for AWS provisioning in us-east-1, Docker Compose for the local container stack, GitHub Actions "
        "for CI/CD, CloudWatch and SNS for logging and alerts, and ECR and IAM for container images and access "
        "control. The diagram distinguishes the components used for the reported results from those that are "
        "defined but were not executed."],
    "3.5.3\tUser Interface Design": [
        "Although the platform is primarily an automated backend pipeline, a desktop trade-replay application is "
        "provided so that the ML Engineer and the Energy Analyst can inspect how a trained agent behaved without "
        "reading raw logs. Its wireframe is shown in Figure 4.6 in Chapter 4.",
        "The window has a title bar with a clock, a row of six indicator cards (time step, spot price, position, "
        "agent return, lead over Buy & Hold and number of trades) and three stacked charts. The first chart shows "
        "the spot price with buy and sell markers, the second the position held by the agent, and the third the "
        "cumulative return of the agent against the Buy & Hold and Random baselines. A control bar provides play "
        "and pause, reset, four playback speeds and a timeline slider, so that the user can replay the test period "
        "or move to any step. The application runs offline and reads the trajectory saved by the backtest."],
}
order = list(CH3_TEXT)
stop_ch3 = heading("3.6\t", 2)
for i, key in enumerate(order):
    level = 2 if key.count(".") == 1 else 3
    h = heading(key, level)
    nxt_key = order[i + 1] if i + 1 < len(order) else None
    if key == "3.4.3\tEntity Relationship Diagram":
        stop = heading("3.5\tSystem Design", 2)
    elif nxt_key:
        stop = heading(nxt_key, 2 if nxt_key.count(".") == 1 else 3)
    else:
        stop = stop_ch3
    if key == "3.4\tSystem Analysis":
        stop = heading("3.4.1\t", 3)
    template = h.getnext()                       # an existing Chapter 3 body paragraph
    remove_between(h, stop)
    insert_before(stop, [make_para(t) for t in CH3_TEXT[key]])

# =============================================================================
# 2. Chapter 4: rewrite the introduction, replace Sections 4.3 and 4.4
# =============================================================================
intro = heading("4.1\tIntroduction", 2)
remove_between(intro, heading("4.2\tSystem Requirements", 2))
insert_before(heading("4.2\tSystem Requirements", 2), [make_para(t) for t in [
    "This chapter presents the analysis and design of the Energy Trading MLOps Platform. Section 4.2 defines the "
    "functional, non-functional, data and interface requirements and traces them to the objectives in Section "
    "1.3.1. Section 4.3 presents the analysis diagrams discussed in Section 3.4 (the use case, class and entity "
    "relationship diagrams), and Section 4.4 presents the design diagrams discussed in Section 3.5 (the database "
    "schema, the system architecture and the user interface design).",
    "Chapter 3 justified an Agile, design science approach in which the pipeline itself is the research artefact. "
    "The diagrams combine an object-oriented view, because the implementation is organised into Python classes "
    "and modules with clear responsibilities, with a data-centric view of the datasets and experiment records the "
    "platform stores. All diagrams are drawn from the implemented system; components proposed earlier that were "
    "not implemented are recorded, with reasons, in Section 5.4.7."]])

h43 = heading("4.3\tSystem Analysis Diagrams", 2)
ch5_break = find_one(lambda t, el: t.startswith("Chapter Five") and style_of(el) == "Heading1").getprevious()
remove_between(h43, ch5_break, include_start=True)

new = []
new += [H2("4.3", "System Analysis Diagrams"), make_para(
    "This section presents the analysis diagrams discussed in Section 3.4: the use case diagram, the class diagram "
    "and the entity relationship diagram. Each is drawn from the implemented system, and names are kept consistent "
    "with the code so that every element can be traced to the implementation.")]
new += [H3("4.3.1", "Use Case Diagram"), make_para(
    "The use case diagram in Figure 4.1 identifies the actors that interact with the platform and the services the "
    "platform provides to them. Three human actors and three external systems were identified from the "
    "requirements. The Data Engineer maintains the data pipeline, the ML Engineer trains and evaluates agents, and "
    "the Energy Analyst consumes results. The Airflow Scheduler, the EIA API and the Email Service are external "
    "systems. The diagram addresses requirements FR-01 to FR-14.")]
new += make_figure("fig4_1_usecase.png", "Figure 4.1: Use Case Diagram" + SUFFIX)
new += [make_para(
    "The Airflow Scheduler is associated with the four automated use cases (ingest, transform, train and "
    "backtest), which shows that the routine pipeline needs no human trigger. The include relationship from Train "
    "DRL trading agent to Track experiment in MLflow is used because every training run always logs to MLflow; "
    "tracking is not optional behaviour. Configuration is shared by the Data Engineer and the ML Engineer, since "
    "pipeline.yaml and drl.yaml control the data and model stages respectively. The Energy Analyst interacts only "
    "with results, which reflects the delimitation in Section 1.5.3 that the system does not execute live trades.")]
new += [H3("4.3.2", "Class Diagram"), make_para(
    "The class diagram in Figure 4.2 shows the main classes and modules, their key attributes and operations, and "
    "their relationships. Python modules that expose functions rather than classes are shown with the module "
    "stereotype so that the whole implementation structure can be seen in one view. The diagram addresses FR-01 "
    "to FR-12.")]
new += make_figure("fig4_2_class.png", "Figure 4.2: Class Diagram" + SUFFIX, width=6.0)
new += [make_para(
    "EnergyTradingEnv inherits from the abstract gymnasium.Env, which is what allows any Stable-Baselines3 "
    "algorithm to train on it without modification. The environment owns a RunningNorm instance (composition), so "
    "normalisation statistics live and die with the environment. Both data sources produce a DataFrame for the "
    "transform module and are otherwise independent, which allows the synthetic generator to replace the API "
    "during testing. The train_drl and backtest modules depend on the environment but not on each other; they "
    "communicate only through saved checkpoints. The DAG calls the task wrappers in drl_tasks, which keeps "
    "orchestration code separate from pipeline logic, as proposed in Section 3.3.5.")]
new += [H3("4.3.3", "Entity Relationship Diagram"), make_para(
    "Figure 4.3 models the data entities kept in the S3 data lake and the MLflow tracking store, with their keys "
    "and the cardinality of each relationship. It addresses DR-02, DR-05 and DR-07.")]
new += make_figure("fig4_3_erd.png", "Figure 4.3: Entity Relationship Diagram" + SUFFIX)
new += [make_para(
    "Every dataset is keyed by a UTC business-day timestamp, and the split label is carried by the file name rather "
    "than by a column, so each split can be loaded without reading the others. One ingestion produces three "
    "processed splits, each summarised by one data-quality report. In the tracking store, an experiment groups many "
    "runs, and each run owns its parameters, metrics and artefacts. The dashed lineage link is deliberately weak: a "
    "run records the training and validation row counts but not a hash or path of the exact processed files. "
    "Chapter 5 shows that these row counts were nonetheless sufficient to reconstruct the dataset exactly; storing a "
    "dataset hash is recommended to make this link explicit.")]
new += [H2("4.4", "System Design Diagrams"), make_para(
    "This section presents the design diagrams discussed in Section 3.5: the database schema, the system "
    "architecture and the user interface design. They follow the same narrative, diagram and interpretation "
    "sequence as Section 4.3.")]
new += [H3("4.4.1", "Database Schema"), make_para(
    "Figure 4.4 shows the physical schema of the two stores described in Section 3.5.1: the tables of the MLflow "
    "tracking database and the external tables of the AWS Glue Data Catalog. It addresses FR-10, DR-05 and DR-07.")]
new += make_figure("fig4_4_db_schema.png", "Figure 4.4: Database Schema" + SUFFIX)
new += [make_para(
    "The MLflow tables use composite primary keys: a parameter or tag is identified by its run and key, and a "
    "metric value by its run, key, timestamp and step, which is what allows a metric such as the validation return "
    "to be logged many times during one training run. Only metadata is stored in the database; model files and the "
    "configuration file are artefacts on disk, referenced by artifact_uri. The two Glue tables have no keys "
    "because they are external tables over Parquet files, and their partition column comes from the folder name. "
    "Because the crawlers do not set a table prefix, Glue infers the table names from the S3 paths; setting a "
    "prefix is recommended so that the raw and processed tables are named explicitly. The reported runs used the "
    "local MLflow file store, which holds the same entities as files; the SQLite database is used when the MLflow "
    "server runs in Docker Compose.")]
new += [H3("4.4.2", "System Architecture"), make_para(
    "Figure 4.5 presents the layered architecture described in Section 3.5.2 and marks which components were used "
    "for the results reported in Chapter 5. It addresses FR-12, NFR-06 and the requirement in the course guidance "
    "that the deployment status of a system be stated accurately.")]
new += make_figure("fig4_5_architecture.png", "Figure 4.5: System Architecture" + SUFFIX)
new += [make_para(
    "The solid boxes trace the path that produced the reported results: EIA data was fetched and transformed, the "
    "PPO agent was trained in the trading environment with MLflow tracking, and the trained agent was evaluated "
    "in the backtest report and the replay application, all through the local run scripts with data passed in "
    "memory. The dashed components (the S3 zones, the Glue crawlers, the Airflow DAG, the email notification and "
    "the AWS services) are implemented in code or configuration but were not executed in this study, so no part "
    "of the system is publicly deployed. The GitHub Actions workflows ran when the repository was first pushed but "
    "failed at the formatting check and at the AWS deployment step, which has no credentials. Because the pipeline "
    "logic lives in modules that both the DAG and the local scripts call, the same code would run unchanged under "
    "Airflow.")]
new += [H3("4.4.3", "User Interface Design"), make_para(
    "Figure 4.6 is the wireframe of the trade-replay application described in Section 3.5.3. It addresses FR-14 "
    "and NFR-07.")]
new += make_figure("fig4_6_ui_wireframe.png", "Figure 4.6: User Interface Design" + SUFFIX)
new += [make_para(
    "The layout puts the information a reviewer needs first: the indicator cards give the current state at a "
    "glance, and the three charts share a time axis with a cursor at the current step, so that a trade, the "
    "resulting position and its effect on return can be read along one vertical line. Showing the agent against "
    "both baselines in the bottom chart keeps every result in context. Some labels (Hour, USD/MWh and MWh) were "
    "inherited from the original hourly electricity design and do not match the daily gas data; this is recorded "
    "as DEF-05 in Chapter 5. The static backtest dashboard produced by the backtest module is shown with the "
    "results in Figure 5.2.")]
insert_before(ch5_break, new)

# =============================================================================
# 3. Cross-references in Chapters 4 and 5, tables, List of Figures
# =============================================================================
REPLACE = {
    "(Figure 4.8)": "(Figure 4.5)",
    "The implementation follows the module structure in Figure 4.6 and the data flow in Figure 4.2.":
        "The implementation follows the class structure in Figure 4.2 and the architecture in Figure 4.5.",
    "stops the Airflow task (Figure 4.4)": "stops the Airflow task",
    "using the hyperparameters in Figure 4.5, seed 42":
        "using the hyperparameters in config/drl.yaml (learning rate 3e-4, 2,048 steps per rollout, batch size 64, "
        "10 epochs, discount factor 0.99, clip range 0.2, entropy coefficient 0.01 and a [256, 256, 128] network), "
        "seed 42",
    "Web forecast dashboard (Figure 3.7)": "Web forecast dashboard",
    "Local file store (./mlruns); PostgreSQL configured in Docker Compose":
        "Local file store (./mlruns); SQLite for the MLflow server in Docker Compose",
}
TABLE_DESIGN = {"Figures 4.2, 4.7": "Figures 4.1, 4.3, 4.4", "Figures 4.5, 4.6": "Figures 4.2, 4.5",
                "Figures 4.4, 4.7, 4.8": "Figures 4.3, 4.4, 4.5", "Figures 4.4, 4.8": "Figures 4.5, 4.6"}


def all_paragraphs():
    for p in doc.paragraphs:
        yield p
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                yield from c.paragraphs


def set_text(p, new_text):
    runs = p.runs
    runs[0].text = new_text
    for r in runs[1:]:
        r._r.getparent().remove(r._r)


used = set()
for p in all_paragraphs():
    t = p.text
    for old, new_ in REPLACE.items():
        if old in t:
            t = t.replace(old, new_); used.add(old)
    if t.strip() in TABLE_DESIGN:
        t = TABLE_DESIGN[t.strip()]; used.add(t)
    if t != p.text:
        set_text(p, t)
missing = [k for k in REPLACE if k not in used]
assert not missing, missing

# List of Figures: drop 3.2-3.7 and the old 4.x entries, add the new 4.1-4.6 after Figure 3.1
lof = [el for el in body if el.tag == W_P and len(text(el)) < 120]
f31 = [el for el in lof if text(el).strip() == "Figure 3.1: Agile Development Lifecycle"][0]
for el in [el for el in lof if text(el).startswith(("Figure 3.2:", "Figure 3.3:", "Figure 3.4:", "Figure 3.5:",
                                                   "Figure 3.6:", "Figure 3.7:", "Figure 4."))
           and list(body).index(el) < list(body).index(ch3)]:
    body.remove(el)
anchor = f31
for t in ["Figure 4.1: Use Case Diagram", "Figure 4.2: Class Diagram", "Figure 4.3: Entity Relationship Diagram",
          "Figure 4.4: Database Schema", "Figure 4.5: System Architecture", "Figure 4.6: User Interface Design"]:
    n = copy.deepcopy(f31)
    ts = list(n.iter(qn("w:t")))
    ts[0].text = t
    for extra in ts[1:]:
        extra.text = ""
    anchor.addnext(n)
    anchor = n

doc.save(str(DOCX))
print("Restructured", DOCX)
