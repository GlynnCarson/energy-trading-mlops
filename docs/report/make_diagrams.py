"""
Generates the report's diagrams: Figure 2.1 (conceptual framework), Figure 3.1 (Agile lifecycle) and
the Chapter 4 analysis and design diagrams (Figures 4.1 - 4.6): use case,
class, entity relationship, database schema, system architecture and user interface.
These are the diagram types discussed in Chapter 3 (Sections 3.4 and 3.5).

    python docs/report/make_diagrams.py

Output: docs/report/figures/fig4_*.png
Every element is traced to the implemented code in src/ and config/.
"""

from pathlib import Path
from diagram_kit import (canvas, save, box, header_box, arrow, poly_arrow, actor, system_actor, usecase,
                         datastore, process, legend, BLACK, DARK, MID, LIGHT, PALE, WHITE)
from matplotlib.patches import FancyBboxPatch, Circle, Polygon, Rectangle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.1  Use case diagram
# ─────────────────────────────────────────────────────────────────────────────
def use_case():
    fig, ax, H = canvas("Use Case Diagram", w=14, h=12.9)
    # boundary
    bx, by, bw, bh = 31, 6, 38, 72.2
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=1.2", fc="white",
                                ec=BLACK, lw=2.0, zorder=1))
    ax.add_patch(FancyBboxPatch((bx, by + bh - 4.2), bw, 4.2, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=BLACK, ec=BLACK, lw=2.0, zorder=2))
    ax.add_patch(Rectangle((bx, by + bh - 4.2), bw, 1.5, fc=BLACK, ec="none", zorder=2))
    ax.text(50, by + bh - 2.1, "Energy Trading MLOps Platform", ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold", zorder=3)

    ys = [70.5 - 6.1 * i for i in range(11)]
    uc = {
        "cfg": (50, ys[0], "Configure pipeline\n(YAML)", WHITE),
        "ing": (50, ys[1], "Ingest energy price data", WHITE),
        "tr": (50, ys[2], "Clean, validate and\nengineer features", WHITE),
        "dq": (50, ys[3], "Review data-quality\nreport", WHITE),
        "train": (50, ys[4], "Train DRL trading agent", PALE),
        "track": (50, ys[5], "Track experiment\nin MLflow", PALE),
        "bt": (50, ys[6], "Backtest agent against\nbaselines", PALE),
        "gate": (50, ys[7], "Gate, register and\npromote model", PALE),
        "paper": (50, ys[8], "Paper trade and monitor\nthe champion", PALE),
        "view": (50, ys[9], "View backtest report\nand trade replay", LIGHT),
        "notify": (50, ys[10], "Receive pipeline\nnotification", LIGHT),
    }
    for k, (cx, cy, t, f) in uc.items():
        usecase(ax, cx, cy, t, w=26, h=4.9, fill=f)

    actor(ax, 13, 57, "Data Engineer", "pipeline and data quality")
    actor(ax, 13, 33, "ML Engineer", "training and evaluation")
    actor(ax, 13, 10, "Energy Analyst", "consumes results")
    system_actor(ax, 88, 60, "Airflow Scheduler", "daily trigger")
    system_actor(ax, 88, 37, "EIA API v2", "price data source")
    system_actor(ax, 88, 11, "Email Service", "SMTP")

    def link(a, b):
        # zorder 1.5: above the boundary's white fill (1) but below the ellipses (3), so every line visibly
        # reaches the edge of its use case
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=MID, lw=1.1, zorder=1.5))

    de, ml, an = (16, 63), (16, 39), (16, 16)
    for k in ["cfg", "ing", "tr", "dq"]:
        link(de, (37, uc[k][1]))
    for k in ["cfg", "train", "track", "bt", "gate", "view"]:
        link(ml, (37, uc[k][1]))
    for k in ["paper", "view", "notify"]:
        link(an, (37, uc[k][1]))
    sch, eia, mail = (83.5, 63.5), (83.5, 40.5), (83.5, 14.5)
    for k in ["ing", "tr", "train", "bt", "gate", "paper"]:
        link(sch, (63, uc[k][1]))
    link(eia, (63, uc["ing"][1]))
    link(mail, (63, uc["notify"][1]))

    # «include» Train -> Track
    arrow(ax, (50, ys[4] - 2.45), (50, ys[5] + 2.45), dashed=True, open_head=True)
    ax.text(57.5, (ys[4] + ys[5]) / 2, "«include»", fontsize=8.5, style="italic", color=DARK, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=8)
    legend(ax, [("box_white", "Data pipeline"), ("box_pale", "Model lifecycle"),
                ("box_mid", "Results and delivery"), ("solid", "association"), ("dash", "«include»")], y=1.3)
    save(fig, OUT / "fig4_1_usecase.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.2  Class diagram
# ─────────────────────────────────────────────────────────────────────────────
def class_diagram():
    fig, ax, H = canvas("Class Diagram", w=14, h=16.5)
    top = H - 9
    a = header_box(ax, 1.5, top, 29, "EIAIngester", sections=[
        ["- api_key : str", "- session : requests.Session", "+ SERIES : dict{gas, oil}"],
        ["- _get(route, params) : dict", "- _fetch_series(...) : DataFrame", "+ build_dataset(start, commodity)"]])
    b = header_box(ax, 35.5, top, 29, "SyntheticEnergyGenerator", sections=[
        ["- seed : int = 42"], ["+ generate(days) : DataFrame"]])
    c = header_box(ax, 69.5, top, 29, "transform", stereo="module", head_fill=BLACK, sections=[
        ["TARGET = 'price_usd'"],
        ["+ clean_raw(df)", "+ validate_dataframe(df) : dict", "+ engineer_features(df)",
         "+ temporal_split(df) : (train, val, test)", "+ run_transformation(raw_path)"]])
    row2 = top - 31
    d = header_box(ax, 2, row2, 22, "gymnasium.Env", stereo="abstract", head_fill=MID, body_fill=PALE, sections=[
        ["# observation_space", "# action_space"], ["+ reset()", "+ step(action)"]])
    e = header_box(ax, 28, row2, 34, "EnergyTradingEnv", head_fill=DARK, sections=[
        ["- df : DataFrame", "- cash, position_mwh : float", "- norm : RunningNorm", "- trade_log : list"],
        ["+ reset(seed) : (obs, info)", "+ step(action) : (obs, r, term, trunc, info)",
         "- _execute_trade(delta) : float", "- _compute_reward(tx_cost) : float", "+ get_episode_summary() : dict"]])
    f = header_box(ax, 66, row2, 20, "RunningNorm", head_fill=DARK, sections=[
        ["- mean, var : ndarray", "- clip : float = 5"], ["+ update(x)", "+ normalise(x)"]])
    row3 = row2 - 32
    g = header_box(ax, 1.5, row3, 29, "train_drl", stereo="module", head_fill=DARK, sections=[
        ["SB3_ALGOS = {PPO, SAC, TD3, A2C}"],
        ["+ load_drl_config() : dict", "+ build_agent(algo, env, cfg)", "+ run_drl_training(train, val) : run_id",
         "+ evaluate_agent(agent, env, n) : dict"]])
    h = header_box(ax, 35.5, row3, 29, "MLflowCallback /\nPortfolioMetricsCallback", head_fill=DARK, head_h=5.0, sections=[
        ["- log_freq / eval_freq : int"], ["+ _on_step() : bool", "  -> mlflow.log_metrics()"]])
    i = header_box(ax, 69.5, row3, 29, "backtest", stereo="module", head_fill=DARK, sections=[
        ["+ run_backtest(agent, test_df, cfg)", "+ run_buy_and_hold(test_df, cfg)", "+ run_random_agent(test_df, cfg)",
         "- _compute_metrics(values, ...) : dict", "+ run_full_backtest(...) : dict"]])
    row4 = row3 - 22
    j = header_box(ax, 1.5, row4, 29, "energy_trading_mlops_pipeline", stereo="Airflow DAG", head_fill=MID,
                   body_fill=PALE, sections=[["schedule : @daily, retries : 2"],
                                             ["ingest, transform, data_quality_report,", "train_drl, backtest_drl, notify_success,",
                                              "paper_trade, monitor, retrain branch"]])
    k = header_box(ax, 35.5, row4, 29, "drl_tasks", stereo="module", head_fill=MID, body_fill=PALE, sections=[
        ["+ task_train_drl(**context)", "+ task_backtest_drl(**context)"]])
    l = header_box(ax, 69.5, row4, 29, "helpers", stereo="utility", head_fill=MID, body_fill=PALE, sections=[
        ["+ load_pipeline_config() : dict", "+ upload_df_to_s3(df, bucket, key)", "+ download_df_from_s3(bucket, key)",
         "+ get_env(key) : str"]])

    def top_mid(r): return (r[0] + r[2] / 2, r[1] + r[3])
    def bot_mid(r): return (r[0] + r[2] / 2, r[1])

    # dependencies (dashed, open arrow)
    poly_arrow(ax, [(a[0] + a[2] / 2, a[1]), (a[0] + a[2] / 2, a[1] - 2.0), (c[0] + 6, a[1] - 2.0), (c[0] + 6, c[1])],
               dashed=True, open_head=True)
    yb = b[1] + b[3] / 2
    arrow(ax, (b[0] + b[2], yb), (c[0], yb), dashed=True, open_head=True)
    ax.text((b[0] + b[2] + c[0]) / 2, yb + 1.3, "DataFrame", ha="center", fontsize=8.0, color=DARK)
    poly_arrow(ax, [(c[0] + c[2] / 2, c[1]), (c[0] + c[2] / 2, e[1] + e[3] + 3), (e[0] + e[2] / 2 + 6, e[1] + e[3] + 3),
                    (e[0] + e[2] / 2 + 6, e[1] + e[3])], dashed=True, open_head=True, label="engineered frame", lidx=1,
               loff=(0, 1.3))
    # inheritance Env <- EnergyTradingEnv (hollow triangle)
    ey = e[1] + e[3] - 9
    ax.add_line(Line2D([e[0], d[0] + d[2] + 1.6], [ey, ey], color=BLACK, lw=1.4, zorder=2))
    ax.add_patch(Polygon([[d[0] + d[2], ey], [d[0] + d[2] + 1.8, ey + 1.0], [d[0] + d[2] + 1.8, ey - 1.0]],
                         fc="white", ec=BLACK, lw=1.4, zorder=3))
    ax.text((e[0] + d[0] + d[2]) / 2 + 0.8, ey + 2.2, "extends", fontsize=8.5, color=DARK, ha="center")
    # composition Env <>- RunningNorm (filled diamond at Env side)
    fy = f[1] + f[3] - 6
    ax.add_line(Line2D([e[0] + e[2] + 2.4, f[0]], [fy, fy], color=BLACK, lw=1.4, zorder=2))
    ax.add_patch(Polygon([[e[0] + e[2], fy], [e[0] + e[2] + 1.2, fy + 0.9], [e[0] + e[2] + 2.4, fy],
                          [e[0] + e[2] + 1.2, fy - 0.9]], fc=BLACK, ec=BLACK, zorder=3))
    ax.text(e[0] + e[2] + 2.0, fy + 1.6, "1", fontsize=8.5, ha="center")
    # train_drl uses env
    poly_arrow(ax, [(g[0] + g[2] / 2, g[1] + g[3]), (g[0] + g[2] / 2, g[1] + g[3] + 3.0), (e[0] + 8, g[1] + g[3] + 3.0),
                    (e[0] + 8, e[1])], dashed=True, open_head=True, label="creates n_envs", lidx=1, loff=(0, 1.2))
    poly_arrow(ax, [(i[0] + i[2] / 2, i[1] + i[3]), (i[0] + i[2] / 2, i[1] + i[3] + 3.0), (e[0] + e[2] - 6, i[1] + i[3] + 3.0),
                    (e[0] + e[2] - 6, e[1])], dashed=True, open_head=True, label="walks test split", lidx=1, loff=(0, 1.2))
    arrow(ax, (g[0] + g[2], g[1] + g[3] - 6), (h[0], g[1] + g[3] - 6), "registers", dashed=True, open_head=True,
          loff=(0, 1.3), fs=8.0)
    # bottom row
    arrow(ax, (j[0] + j[2], j[1] + j[3] - 7), (k[0], j[1] + j[3] - 7), "calls", dashed=True, open_head=True, loff=(0, 1.2))
    poly_arrow(ax, [(k[0] + 8, k[1] + k[3]), (k[0] + 8, g[1] - 1.5), (g[0] + g[2] / 2, g[1] - 1.5), (g[0] + g[2] / 2, g[1])],
               dashed=True, open_head=True)
    poly_arrow(ax, [(k[0] + k[2] - 8, k[1] + k[3]), (k[0] + k[2] - 8, i[1] - 1.5), (i[0] + i[2] / 2, i[1] - 1.5),
                    (i[0] + i[2] / 2, i[1])], dashed=True, open_head=True)
    arrow(ax, (k[0] + k[2], k[1] + k[3] - 7), (l[0], k[1] + k[3] - 7), "S3 I/O,\nconfig", dashed=True, open_head=True,
          loff=(0, 2.1), fs=8.0)
    legend(ax, [("box_dark", "ML layer"), ("box_white", "Data layer"), ("box_mid", "Orchestration / utility"),
                ("dash", "dependency"), ("solid", "inheritance / composition")], y=2.2)
    save(fig, OUT / "fig4_2_class.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.3  Entity relationship diagram
# ─────────────────────────────────────────────────────────────────────────────
def erd():
    fig, ax, H = canvas("Entity Relationship Diagram",
                        subtitle="Data lake datasets, MLflow tracking entities and the MLflow Model Registry",
                        w=14, h=13.95)
    top = H - 9.5
    ax.text(25, top + 0.4, "Data lake  (bucket: energy-trading-mlops-data)", ha="center", fontsize=10.5,
            fontweight="bold", color=DARK)
    ax.text(78, top + 0.4, "MLflow tracking store", ha="center", fontsize=10.5, fontweight="bold", color=DARK)
    raw = header_box(ax, 3, top - 1.5, 44, "RAW_ENERGY", stereo="raw/energy/{start}_to_{end}/raw_energy.parquet",
                     sections=[["PK  timestamp : datetime (UTC, business day)", "     price_usd : float  (Henry Hub $/MMBtu)",
                                "     oil_price_usd : float  (WTI $/bbl)", "     data_source : str  {eia_api, synthetic}",
                                "     ingestion_timestamp : datetime"]])
    proc = header_box(ax, 3, raw[1] - 6, 44, "PROCESSED_SPLIT", stereo="processed/energy/{YYYYMMDD_HH}/{split}.parquet",
                      sections=[["PK  timestamp : datetime", "     split : {train, val, test}  (from file name)",
                                 "     price_usd : float  (target, not an input)", "     38 numeric features:",
                                 "       time (12), lags (6), rolling (12),", "       technical (7: RSI, MACD, BB x3, ROC x2),",
                                 "       oil_price_usd (exogenous)"]])
    dq = header_box(ax, 3, proc[1] - 6, 21, "DQ_REPORT", stereo="reports/dq/{ds}.json", head_fill=MID, body_fill=PALE,
                    sections=[["train_rows, n_features", "null_pct, price_stats"]])
    rep = header_box(ax, 26, proc[1] - 6, 21, "BACKTEST_REPORT", stereo="reports/drl/{ds}/", head_fill=MID,
                     body_fill=PALE, sections=[["backtest_report.png", "metrics in XCom"]])

    exp = header_box(ax, 60, top - 1.5, 36, "EXPERIMENT", sections=[["PK  experiment_id", "     name  e.g. energy-drl-eia-gas"]])
    run = header_box(ax, 60, exp[1] - 6, 36, "RUN", head_fill=DARK, sections=[
        ["PK  run_id", "FK  experiment_id", "     run_name, status, start/end time"]])
    par = header_box(ax, 60, run[1] - 6, 11, "PARAM", head_fill=DARK, sections=[["FK run_id", "key, value", "e.g. seed"]])
    met = header_box(ax, 72.5, run[1] - 6, 11, "METRIC", head_fill=DARK, sections=[["FK run_id", "key, value,", "step"]])
    tag = header_box(ax, 85, run[1] - 6, 11, "TAG", head_fill=DARK, sections=[["FK run_id", "key, value", "e.g. algorithm"]])
    art = header_box(ax, 60, par[1] - 6, 36, "ARTIFACT", head_fill=DARK, sections=[
        ["FK  run_id", "     model/ppo_final.zip", "     best_model/best_model.zip", "     config/*.yaml (full drl.yaml)"]])

    def crow(p1, p2, label, lab_off=(0, 1.2)):
        arrow(ax, p1, p2, label, head=False, lw=1.4, color=BLACK, loff=lab_off, fs=8.3)

    crow((25, raw[1]), (25, proc[1] + proc[3]), "1 : 3  (one ingestion -> train/val/test)", lab_off=(0, 0))
    crow((13.5, proc[1]), (13.5, dq[1] + dq[3]), "1 : 1", lab_off=(3, 0))
    crow((36.5, proc[1]), (36.5, rep[1] + rep[3]), "1 : 1 (test)", lab_off=(5, 0))
    crow((78, exp[1]), (78, run[1] + run[3]), "1 : N", lab_off=(2.6, 0))
    for bx_ in (par, met, tag):
        crow((bx_[0] + bx_[2] / 2, run[1]), (bx_[0] + bx_[2] / 2, bx_[1] + bx_[3]), "1 : N", lab_off=(2.6, 0))
    # RUN 1 : N ARTIFACT, drawn as plain lines (same notation as the other relationships)
    ya, yb_ = run[1] + 1.5, art[1] + art[3] - 3
    for (x1, y1), (x2, y2) in (((60, ya), (57.5, ya)), ((57.5, ya), (57.5, yb_)), ((57.5, yb_), (60, yb_))):
        ax.add_line(Line2D([x1, x2], [y1, y2], color=BLACK, lw=1.4, zorder=2))
    ax.text(55.6, (ya + yb_) / 2 - 6, "1 : N", ha="center", va="center", fontsize=8.3, color=DARK)
    # lineage link (dashed): processed split -> run, via the logged row counts
    yP = proc[1] + proc[3] - 6
    poly_arrow(ax, [(47, yP), (53.5, yP), (53.5, run[1] + 4), (60, run[1] + 4)], dashed=True, open_head=True)
    ax.text(51.6, (yP + run[1] + 4) / 2, "lineage", ha="center", va="center", fontsize=8.0, color=DARK, rotation=90)

    # MLflow Model Registry (closed loop): versions of energy_trading_ppo, each created from one run's model
    yt = min(art[1], dq[1]) - 7.5
    ax.text(25, yt + 2.4, "MLflow Model Registry  (model energy_trading_ppo)", ha="center", fontsize=10.5,
            fontweight="bold", color=DARK)
    ali = header_box(ax, 3, yt, 20, "MODEL_ALIAS", head_fill=DARK, sections=[
        ["PK FK name", "PK  alias  e.g. champion", "     version"]])
    reg = header_box(ax, 27, yt, 20, "REGISTERED_MODEL", head_fill=DARK, sections=[
        ["PK  name", "     description", "     creation_time"]])
    mv = header_box(ax, 60, yt, 36, "MODEL_VERSION", head_fill=DARK, sections=[
        ["PK FK name,  PK version", "FK  run_id", "     source  e.g. runs:/{run_id}/best_model",
         "     tags: gate result, seed, data window"]])
    yr = ali[1] + ali[3] - 6.5
    crow((23, yr), (27, yr), "N : 1", lab_off=(0, 1.2))
    crow((47, yr), (60, yr), "1 : N", lab_off=(0, 1.2))
    # RUN 1 : N MODEL_VERSION, routed along the right margin so it crosses no box
    yr1, yr2 = run[1] + run[3] / 2, mv[1] + mv[3] - 6.5
    for (x1, y1), (x2, y2) in (((96, yr1), (98.4, yr1)), ((98.4, yr1), (98.4, yr2)), ((98.4, yr2), (96, yr2))):
        ax.add_line(Line2D([x1, x2], [y1, y2], color=BLACK, lw=1.4, zorder=2))
    ax.text(99.3, (yr1 + yr2) / 2, "1 : N", ha="center", va="center", fontsize=8.3, color=DARK, rotation=90)
    save(fig, OUT / "fig4_3_erd.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.4  Database schema (physical)
# ─────────────────────────────────────────────────────────────────────────────
def db_schema():
    fig, ax, H = canvas("Database Schema", subtitle="MLflow tracking and model registry database and AWS Glue "
                                                    "Data Catalog", w=14, h=12.1)
    top = H - 9.5
    ax.text(30, top + 0.4, "MLflow tracking database  (SQLite, MLflow server in Docker Compose)", ha="center",
            fontsize=10.5, fontweight="bold", color=DARK)
    ax.text(81, top + 0.4, "AWS Glue Data Catalog  (database energy_trading_db)", ha="center", fontsize=10.5,
            fontweight="bold", color=DARK)
    exp = header_box(ax, 3, top - 1.5, 26, "experiments", sections=[[
        "PK  experiment_id : INTEGER", "     name : VARCHAR (unique)", "     artifact_location : VARCHAR",
        "     lifecycle_stage : VARCHAR", "     creation_time : BIGINT"]])
    run = header_box(ax, 33, top - 1.5, 27, "runs", head_fill=DARK, sections=[[
        "PK  run_uuid : VARCHAR(32)", "FK  experiment_id : INTEGER", "     name : VARCHAR", "     status : VARCHAR",
        "     start_time, end_time : BIGINT", "     artifact_uri : VARCHAR", "     lifecycle_stage : VARCHAR"]])
    par = header_box(ax, 3, run[1] - 5, 18, "params", head_fill=DARK, sections=[[
        "PK FK run_uuid", "PK  key : VARCHAR", "     value : VARCHAR"]])
    met = header_box(ax, 23, run[1] - 5, 19, "metrics", head_fill=DARK, sections=[[
        "PK FK run_uuid", "PK  key : VARCHAR", "PK  timestamp, step : BIGINT", "     value : FLOAT"]])
    tag = header_box(ax, 44, run[1] - 5, 16, "tags", head_fill=DARK, sections=[[
        "PK FK run_uuid", "PK  key : VARCHAR", "     value : VARCHAR"]])
    ax.text(31.5, met[1] - 2.5, "Artefacts (model .zip files, drl.yaml) are stored as files under each run's "
                                "artifact_uri, not in the database.", ha="center", fontsize=8.6, color=MID,
            style="italic")

    raw = header_box(ax, 66, top - 1.5, 31, "raw zone table", stereo="crawler: raw-crawler (hourly)",
                     head_fill=MID, body_fill=PALE, sections=[[
                         "LOCATION  s3://energy-trading-mlops-data/", "                   raw/energy/  (Parquet)",
                         "     timestamp : TIMESTAMP", "     price_usd : DOUBLE", "     oil_price_usd : DOUBLE",
                         "     data_source : STRING", "     ingestion_timestamp : TIMESTAMP",
                         "PART  partition_0 : STRING  ({start}_to_{end})"]])
    proc = header_box(ax, 66, raw[1] - 4, 31, "processed zone table", stereo="crawler: processed-crawler (hourly)",
                      head_fill=MID, body_fill=PALE, sections=[[
                          "LOCATION  s3://energy-trading-mlops-data/", "                   processed/energy/  (Parquet)",
                          "     timestamp : TIMESTAMP", "     price_usd : DOUBLE", "     38 feature columns : DOUBLE",
                          "PART  partition_0 : STRING  ({YYYYMMDD_HH})", "     split : file name train | val | test"]])

    def link(p1, p2, lab):
        arrow(ax, p1, p2, lab, head=False, lw=1.4, color=BLACK, fs=8.3, loff=(2.8, 0))

    poly_arrow(ax, [(29, exp[1] + exp[3] - 6), (33, exp[1] + exp[3] - 6)], dashed=False)
    ax.text(31, exp[1] + exp[3] - 4.6, "1 : N", ha="center", fontsize=8.3, color=DARK)
    for box_ in (par, met, tag):
        x = box_[0] + box_[2] / 2
        poly_arrow(ax, [(46.5, run[1]), (46.5, run[1] - 2.5), (x, run[1] - 2.5), (x, box_[1] + box_[3])])
    ax.text(49.5, run[1] - 1.3, "1 : N", ha="center", fontsize=8.3, color=DARK)

    # MLflow Model Registry tables (closed loop)
    yt = met[1] - 8.0
    ax.text(31.5, yt + 2.4, "MLflow Model Registry tables", ha="center", fontsize=10.5, fontweight="bold",
            color=DARK)
    rma = header_box(ax, 3, yt, 16.5, "registered_model_aliases", head_fill=DARK, sections=[[
        "PK FK name : VARCHAR", "PK  alias : VARCHAR", "     version : INTEGER"]])
    rm = header_box(ax, 23.5, yt, 15.5, "registered_models", head_fill=DARK, sections=[[
        "PK  name : VARCHAR", "     description : VARCHAR", "     creation_time : BIGINT"]])
    mv = header_box(ax, 43, yt, 17, "model_versions", head_fill=DARK, sections=[[
        "PK FK name : VARCHAR", "PK  version : INTEGER", "FK  run_id : VARCHAR(32)", "     source : VARCHAR",
        "     status : VARCHAR"]])
    yk = rm[1] + rm[3] - 6.5
    poly_arrow(ax, [(rm[0], yk), (rma[0] + rma[2], yk)])
    poly_arrow(ax, [(rm[0] + rm[2], yk), (mv[0], yk)])
    ax.text(21.5, yk + 1.3, "1 : N", ha="center", fontsize=8.3, color=DARK)
    ax.text(41.0, yk + 1.3, "1 : N", ha="center", fontsize=8.3, color=DARK)
    # runs 1 : N model_versions, routed in the gap between the MLflow and Glue tables
    yr1, yr2 = run[1] + run[3] - 3.5, mv[1] + mv[3] - 6.5
    poly_arrow(ax, [(run[0] + run[2], yr1), (62.8, yr1), (62.8, yr2), (mv[0] + mv[2], yr2)])
    ax.text(64.0, (yr1 + yr2) / 2, "1 : N", ha="center", va="center", fontsize=8.3, color=DARK, rotation=90)
    ax.text(31.5, mv[1] - 2.5, "Version tags (gate result, seed, data window) are stored in model_version_tags "
                               "(PK name, version, key; value).", ha="center", fontsize=8.6, color=MID,
            style="italic")
    legend(ax, [("box_dark", "MLflow tables"), ("box_mid", "Glue external tables (defined in Terraform)"),
                ("solid", "foreign key, arrow at the 'many' side")], y=1.6)
    save(fig, OUT / "fig4_4_db_schema.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.5  System architecture (layered)
# ─────────────────────────────────────────────────────────────────────────────
def architecture():
    fig, ax, H = canvas("System Architecture", subtitle="Energy Trading MLOps Platform - as implemented",
                        w=14, h=13.1)
    rh, rho = 8.6, 11.0  # row height; the orchestration row is taller (two task lanes)
    rows = [("DATA\nSOURCES", 74.4, rh), ("INGESTION\nAND STORAGE", 63.4, rh), ("PROCESSING\nAND ML", 52.4, rh),
            ("MODEL\nLIFECYCLE", 41.4, rh), ("ORCHESTRATION", 28.0, rho), ("EVALUATION\nAND DELIVERY", 17.0, rh)]
    for label, y, h in rows:
        box(ax, 1.5, y, 14, h, label, fill=WHITE, edge="#BBBBBB", fs=9.5, lw=1.2)

    # Three states: "used" = produced the reported results (pale, solid); "stack" = ran only in the local
    # Docker Compose stack or in GitHub Actions (white, solid); "code" = defined in code, not deployed (dashed).
    def comp(x, y, w, t, s, state="used", h=rh, fs=10, sub_fs=8.2):
        box(ax, x, y, w, h, t, sub=s, fill=PALE if state == "used" else WHITE, dashed=state == "code",
            fs=fs, sub_fs=sub_fs, lw=1.4)

    X0 = 17.5  # usable width for the layers: 17.5 .. 75.5
    comp(X0, 74.4, 28, "EIA Open Data API v2", "daily Henry Hub gas + WTI oil")
    comp(X0 + 30, 74.4, 28, "Synthetic generator", "seeded hourly data for tests")
    comp(X0, 63.4, 18, "S3 raw zone", "raw/energy/ Parquet\n(LocalStack)", state="stack")
    comp(X0 + 20, 63.4, 18, "S3 processed zone", "processed/energy/ Parquet\n(LocalStack)", state="stack")
    comp(X0 + 40, 63.4, 18, "Glue crawlers", "Data Catalog tables", state="code")
    comp(X0, 52.4, 13.5, "Transform", "clean, validate,\n38 features")
    comp(X0 + 14.6, 52.4, 13.5, "TradingEnv", "Gymnasium,\ncosts, stop-loss")
    comp(X0 + 29.2, 52.4, 13.5, "PPO trainer", "SB3, 4 envs,\nfixed seeds")
    comp(X0 + 43.8, 52.4, 14.2, "MLflow tracking", "params, metrics, models\n(./mlruns; server in\nDocker for the DAG)")
    # model lifecycle (closed loop, November 2026)
    lw_, lg = 10.8, 1.0
    for i, (t, s, st) in enumerate([("Quality gate", "validation only:\nvs B&H, Random", "used"),
                                    ("Model Registry", "energy_trading_ppo,\nalias champion", "used"),
                                    ("Paper trading", "simulated, from\n7 Jul 2026", "used"),
                                    ("Monitor", "gap vs B&H,\ndrawdown", "used"),
                                    ("Retrain", "challengers;\nsmoke test only", "stack")]):
        comp(X0 + i * (lw_ + lg), 41.4, lw_, t, s, state=st, fs=9.6, sub_fs=7.9)
    # orchestration: Airflow DAG (run in the Docker stack) + local scripts (used for the reported results)
    ax.add_patch(FancyBboxPatch((X0, 28.0), 36, rho, boxstyle="round,pad=0,rounding_size=1.0", fc=WHITE,
                                ec=BLACK, lw=1.4, zorder=3))
    ax.text(X0 + 18, 28.0 + rho - 1.6, "Airflow DAG  (@daily, retries 2)", ha="center", fontsize=10,
            fontweight="bold", zorder=4)
    lanes = [(32.4, ["ingest", "transform", "train_drl", "backtest", "notify"]),
             (28.8, ["paper_trade", "monitor", "retrain", "promote", "alert"])]
    for ly, tasks in lanes:
        for i, t in enumerate(tasks):
            bx = X0 + 1.2 + i * 6.9
            box(ax, bx, ly, 6.3, 3.0, t, fill=WHITE, fs=7.9, bold=False, lw=1.0, z=5)
            if i:
                ax.add_line(Line2D([bx - 0.6, bx], [ly + 1.5, ly + 1.5], color=DARK, lw=1.0, zorder=5))
    comp(X0 + 38, 28.0, 20, "Local run scripts", "train_on_eia.py,\nevaluate_eia_runs.py,\napply_gate_to_seeds.py",
         h=rho)
    comp(X0, 17.0, 18, "Email notification", "EmailOperator to a\nlocal Mailpit inbox", state="stack")
    comp(X0 + 20, 17.0, 18, "Trade replay app", "Tkinter, offline")
    comp(X0 + 40, 17.0, 18, "Backtest report", "vs Buy & Hold and\nRandom (PNG)")

    # vertical flow arrows between layers
    arrow(ax, (X0 + 9, 74.4), (X0 + 9, 72.0))      # EIA -> raw zone
    arrow(ax, (X0 + 9, 63.4), (X0 + 9, 61.0))      # raw zone -> transform
    arrow(ax, (X0 + 9, 28.0), (X0 + 9, 25.6))      # Airflow notify -> email
    arrow(ax, (X0 + 48, 28.0), (X0 + 48, 25.6))    # local scripts -> backtest report

    # platform column
    ax.add_patch(FancyBboxPatch((78, 17.0), 20.5, 66.0, boxstyle="round,pad=0,rounding_size=1.0", fc="white",
                                ec="#BBBBBB", lw=1.2, ls="--", zorder=1))
    ax.text(88.25, 79.8, "PLATFORM\nAND DEVOPS", ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=MID, linespacing=1.3)
    plat = [("Terraform", "AWS us-east-1; validated\nin CI, never applied", "code"),
            ("Docker Compose", "Airflow, MLflow,\nLocalStack, Mailpit", "stack"),
            ("GitHub Actions", "CI on every push; lint\nfails, deploy not run", "stack"),
            ("CloudWatch, SNS", "logs and alerts", "code"),
            ("ECR, IAM", "image registry, roles", "code")]
    for i, (t, s, st) in enumerate(plat):
        box(ax, 79.5, 66.0 - i * 11.6, 17.5, 8.4, t, sub=s, fill=WHITE, dashed=st == "code", fs=9.6, sub_fs=7.8,
            lw=1.3)
    legend(ax, [("box_pale", "used for the reported results"),
                ("box_white", "ran in Docker Compose or CI only"),
                ("dash", "defined in code; not deployed")], y=10.5)
    ax.text(50, 5.6, "The reported results, the quality gate and the paper trading came from local runs. The Airflow DAG, "
                     "including the closed loop, ran end to end in Docker Compose\n(November 2026) with LocalStack in "
                     "place of S3 and a local Mailpit inbox; retraining ran there only as a smoke test, and AWS was "
                     "not provisioned.",
            ha="center", va="center", fontsize=9, color=MID, style="italic", linespacing=1.5)
    save(fig, OUT / "fig4_5_architecture.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.6  User interface design (wireframe of the trade-replay application)
# ─────────────────────────────────────────────────────────────────────────────
def ui_wireframe():
    fig, ax, H = canvas("User Interface Design", subtitle="Wireframe of the trade-replay application "
                                                         "(trading_replay_app.py)", w=14, h=10.4)
    # window frame
    wx, wy, ww, wh = 3, 3, 72, 63.5
    ax.add_patch(FancyBboxPatch((wx, wy), ww, wh, boxstyle="round,pad=0,rounding_size=0.8", fc="white", ec=BLACK,
                                lw=1.8, zorder=1))
    ax.add_patch(Rectangle((wx, wy + wh - 3.2), ww, 3.2, fc=LIGHT, ec=BLACK, lw=1.2, zorder=2))
    ax.text(wx + 1.5, wy + wh - 1.6, "PPO Agent - Live Trading Replay", fontsize=10, fontweight="bold",
            va="center", zorder=3)
    ax.text(wx + ww - 1.5, wy + wh - 1.6, "step / date clock", fontsize=8.8, color=MID, ha="right",
            va="center", zorder=3)
    # KPI cards
    kpis = ["HOUR", "SPOT PRICE", "POSITION MWH", "AGENT RETURN", "LEAD VS B&H", "TRADES"]
    cw = (ww - 3 - 5 * 1.0) / 6
    for i, k in enumerate(kpis):
        x = wx + 1.5 + i * (cw + 1.0)
        box(ax, x, wy + wh - 10.8, cw, 6.4, k, sub="value", fill=PALE, fs=8.0, sub_fs=9.5, lw=1.0, z=3)
    # chart panels
    panels = [("Energy Spot Price - USD/MWh  (buy and sell markers)", 19.0), ("Position Held - MWh", 6.6),
              ("Cumulative Return - Agent vs Buy&Hold vs Random", 13.8)]
    y = wy + wh - 12.8
    for title, ph in panels:
        y -= ph
        ax.add_patch(Rectangle((wx + 1.5, y), ww - 3, ph, fc="white", ec=MID, lw=1.1, zorder=3))
        ax.text(wx + 2.5, y + ph - 1.3, title, fontsize=8.8, color=DARK, va="center", zorder=4)
        ax.add_line(Line2D([wx + 3, wx + ww - 4], [y + ph * 0.35, y + ph * 0.55], color="#BBBBBB", lw=1.0,
                           zorder=4))
        ax.add_line(Line2D([wx + ww * 0.55, wx + ww * 0.55], [y + 0.5, y + ph - 2.2], color=MID, lw=0.8,
                           ls=(0, (2, 2)), zorder=4))
        y -= 1.2
    # control bar
    cy = wy + 1.4
    for i, (t, w_) in enumerate([("Play / Pause", 9), ("Reset", 6), ("1x", 3), ("2x", 3), ("4x", 3), ("8x", 3)]):
        x = wx + 1.5 + sum(v for _, v in [("", 9), ("", 6), ("", 3), ("", 3), ("", 3), ("", 3)][:i]) + i * 0.8
        box(ax, x, cy, w_, 3.6, t, fill=PALE if i < 2 else WHITE, fs=8.2, bold=(i < 2), lw=1.0, z=3)
    sx = wx + 1.5 + 27 + 6 * 0.8 + 1.5
    ax.add_line(Line2D([sx, wx + ww - 2], [cy + 1.8, cy + 1.8], color=DARK, lw=2.0, zorder=3))
    ax.add_patch(Circle((sx + 12, cy + 1.8), 0.8, fc=BLACK, zorder=4))
    ax.text(sx + 18, cy + 3.6, "timeline slider (scrub)", fontsize=8.2, color=MID, ha="center", zorder=4)

    # annotations
    notes = [(wy + wh - 1.6, "1  Title and clock"),
             (wy + wh - 7.6, "2  Six live KPI cards, updated\n    every frame"),
             (wy + wh - 22.0, "3  Price chart; the dashed line\n    marks the current step"),
             (wy + wh - 36.3, "4  Position held by the agent"),
             (wy + wh - 47.7, "5  Return race against the two\n    baselines"),
             (cy + 1.8, "6  Playback controls: play,\n    reset, speed, scrub")]
    for ny, t in notes:
        ax.text(78, ny, t, fontsize=9.2, va="center", color=DARK, linespacing=1.25)
        ax.add_line(Line2D([wx + ww, 77.3], [ny, ny], color="#BBBBBB", lw=0.8, ls=(0, (2, 2)), zorder=0))
    ax.text(87, 9.0, "Reads reports/animation/\ntrajectory.json; runs fully\noffline (Tkinter + matplotlib).",
            fontsize=8.6, color=MID, ha="center", va="center", style="italic", linespacing=1.3)
    save(fig, OUT / "fig4_6_ui_wireframe.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2.1  Conceptual framework (matches the text of Section 2.7)
# ─────────────────────────────────────────────────────────────────────────────
def conceptual_framework():
    fig, ax, H = canvas("Conceptual Framework", subtitle="Five layers of the MLOps lifecycle, orchestrated daily "
                                                        "by Apache Airflow", w=14, h=7.4)
    layers = [
        ("Layer 1", "ACTORS AND USERS", ["Data Engineer", "ML Engineer", "Airflow Scheduler"]),
        ("Layer 2", "DATA COLLECTION", ["EIA API v2", "Synthetic generator", "Raw Parquet in S3", "Glue crawler (hourly)"]),
        ("Layer 3", "PROCESSING AND AI", ["Cleaning, outliers", "50+ features", "PPO in trading env",
                                          "MLflow tracking", "Quality gate"]),
        ("Layer 4", "STORAGE AND INFRA", ["S3 data lake", "MLflow Model Registry", "Glue Data Catalog",
                                          "Terraform, ECR"]),
        ("Layer 5", "OUTPUTS AND FEEDBACK", ["24-hour price forecast", "SNS email alert", "CloudWatch logs",
                                             "Daily retraining"]),
    ]
    bw, gap, x0 = 17.4, 2.4, 1.8
    top, bh = H - 13.0, 26.0
    fills = [BLACK, DARK, BLACK, DARK, BLACK]
    xs = []
    for i, (lab, title, items) in enumerate(layers):
        x = x0 + i * (bw + gap)
        xs.append(x)
        ax.add_patch(FancyBboxPatch((x, top - bh), bw, bh, boxstyle="round,pad=0,rounding_size=0.9", fc=PALE,
                                    ec=BLACK, lw=1.5, zorder=3))
        ax.add_patch(FancyBboxPatch((x, top - 7.4), bw, 7.4, boxstyle="round,pad=0,rounding_size=0.9",
                                    fc=fills[i], ec=BLACK, lw=1.5, zorder=4))
        ax.add_patch(Rectangle((x, top - 7.4), bw, 2.5, fc=fills[i], ec="none", zorder=4))
        ax.text(x + bw / 2, top - 2.2, lab, ha="center", va="center", fontsize=9, color="#CCCCCC", zorder=5)
        ax.text(x + bw / 2, top - 4.9, title, ha="center", va="center", fontsize=9.6, fontweight="bold",
                color="white", zorder=5)
        for k, item in enumerate(items):
            bullet = True
            ty = top - 10.6 - k * 3.4
            if bullet:
                ax.add_patch(Circle((x + 1.6, ty), 0.45, fc=BLACK, zorder=5))
            ax.text(x + 2.8, ty, item, ha="left", va="center", fontsize=9.2, color=BLACK, zorder=5)
        if i:
            arrow(ax, (x - gap + 0.15, top - bh / 2), (x - 0.15, top - bh / 2), lw=1.6, color=BLACK)

    # feedback loop: Layer 5 back to Layer 2 (data collection), drawn above the layers so it crosses nothing
    yl = top + 2.4
    l2c, l5c = xs[1] + bw / 2, xs[4] + bw / 2
    poly_arrow(ax, [(l5c, top), (l5c, yl), (l2c, yl), (l2c, top)], color=MID, lw=1.4)
    ax.text((l2c + l5c) / 2, yl + 1.3, "feedback loop: new data each day triggers retraining", ha="center",
            va="center", fontsize=9, color=DARK, style="italic")

    # orchestration and platform bars
    yb = top - bh - 3.0
    box(ax, x0, yb - 4.6, 5 * bw + 4 * gap, 4.6, "APACHE AIRFLOW DAG  -  runs every stage daily:  ingest -> transform -> "
        "train -> quality gate -> forecast -> notify", fill=BLACK, text_color="white", fs=9.6, lw=1.2)
    box(ax, x0, yb - 10.0, 5 * bw + 4 * gap, 4.2, "PLATFORM  -  AWS cloud  |  Terraform IaC  |  GitHub Actions "
        "CI/CD  |  Docker and ECR", fill=LIGHT, fs=9.6, lw=1.2)
    save(fig, OUT / "fig2_1_conceptual.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3.1  Agile development lifecycle (sprints match Figure A1 and the real files)
# ─────────────────────────────────────────────────────────────────────────────
def agile_lifecycle():
    fig, ax, H = canvas("Agile Development Lifecycle", subtitle="Six two- to three-week sprints, one pipeline stage "
                                                               "per sprint", w=14, h=10.4)
    # 1. the cycle
    stages = ["Plan", "Design", "Develop", "Test", "Review", "Deploy"]
    sw, sg, sx0, sy = 12.5, 3.9, 2.5, H - 16.0
    for i, s in enumerate(stages):
        x = sx0 + i * (sw + sg)
        box(ax, x, sy, sw, 4.6, s, fill=PALE if i % 2 else WHITE, fs=10.5, lw=1.4)
        if i:
            arrow(ax, (x - sg + 0.2, sy + 2.3), (x - 0.2, sy + 2.3), lw=1.4, color=BLACK)
    last_c = sx0 + 5 * (sw + sg) + sw / 2
    first_c = sx0 + sw / 2
    poly_arrow(ax, [(last_c, sy), (last_c, sy - 2.4), (first_c, sy - 2.4), (first_c, sy)], dashed=True, color=MID)
    ax.text((first_c + last_c) / 2, sy - 3.6, "feedback loop: each sprint's review informs the next sprint",
            ha="center", va="center", fontsize=9, color=DARK, style="italic")
    ax.text(2.5, sy + 6.6, "1  Sprint cycle", fontsize=10.5, fontweight="bold", color=DARK)

    # 2. sprint breakdown
    sprints = [
        ("Sprint 1  |  weeks 1-2", "Plan, Design", "Requirements and literature review; environment and EIA API setup",
         "Deliverable: project scope, config/*.yaml"),
        ("Sprint 2  |  weeks 3-4", "Develop, Test", "Data ingestion and transformation pipeline",
         "Deliverable: ingest.py, transform.py, unit tests"),
        ("Sprint 3  |  weeks 5-7", "Develop, Test, Review", "Trading environment, PPO training, MLflow, backtest",
         "Deliverable: env.py, train_drl.py, backtest.py"),
        ("Sprint 4  |  weeks 8-9", "Develop, Test", "Airflow DAG and Terraform infrastructure",
         "Deliverable: dag.py, drl_tasks.py, main.tf"),
        ("Sprint 5  |  weeks 10-11", "Develop, Deploy", "CI/CD workflows and Docker stack",
         "Deliverable: ci.yml, deploy.yml, Dockerfile"),
        ("Sprint 6  |  weeks 12-14", "Test, Review", "Integration testing, documentation and report",
         "Deliverable: test suite, README, final report"),
    ]
    ax.text(2.5, sy - 7.6, "2  Sprint breakdown (weeks as in Figure A1)", fontsize=10.5, fontweight="bold",
            color=DARK)
    cw, ch, cgx, cgy = 44.5, 9.6, 6.0, 2.6
    ytop = sy - 9.4
    centres = []
    for n, (title, acts, work, deliv) in enumerate(sprints):
        col, row = n % 2, n // 2
        x = 2.5 + col * (cw + cgx)
        y = ytop - row * (ch + cgy) - ch
        ax.add_patch(FancyBboxPatch((x, y), cw, ch, boxstyle="round,pad=0,rounding_size=0.9",
                                    fc=PALE if col else WHITE, ec=BLACK, lw=1.4, zorder=3))
        ax.text(x + 1.4, y + ch - 1.7, title, fontsize=10, fontweight="bold", va="center", zorder=4)
        ax.text(x + cw - 1.4, y + ch - 1.7, acts, fontsize=8.8, color=MID, ha="right", va="center", zorder=4,
                style="italic")
        ax.text(x + 1.4, y + ch - 4.6, work, fontsize=9.2, va="center", zorder=4)
        ax.text(x + 1.4, y + ch - 7.4, deliv, fontsize=9.2, va="center", zorder=4, family="DejaVu Sans")
        centres.append((x, y))
    # order arrows: 1->2 (right), 2->3 (down-left), 3->4, 4->5, 5->6
    for n in range(5):
        (x1, y1), (x2, y2) = centres[n], centres[n + 1]
        if y1 == y2:                      # same row, left -> right
            arrow(ax, (x1 + cw + 0.2, y1 + ch / 2), (x2 - 0.2, y2 + ch / 2), lw=1.4, color=BLACK)
        else:                             # next row: from right box down-left to the next row's left box
            poly_arrow(ax, [(x1 + cw / 2, y1), (x1 + cw / 2, y1 - cgy / 2), (x2 + cw / 2, y1 - cgy / 2),
                            (x2 + cw / 2, y2 + ch)], color=BLACK, lw=1.2)

    # 3. principles
    py = centres[-1][1] - 9.0
    ax.text(2.5, py + 5.8, "3  Agile principles applied", fontsize=10.5, fontweight="bold", color=DARK)
    princ = [("Iterative delivery", "one pipeline stage per sprint"), ("Continuous testing", "pytest on every commit"),
             ("Working software first", "pipeline runs end to end"), ("Feedback integration", "supervisor review "
                                                                                              "each sprint")]
    pw, pg = 21.5, 3.0
    for i, (t, s) in enumerate(princ):
        box(ax, 2.5 + i * (pw + pg), py - 0.6, pw, 5.6, t, sub=s, fill=WHITE, fs=10, sub_fs=8.8, lw=1.3)
    save(fig, OUT / "fig3_1_agile.png")


if __name__ == "__main__":
    conceptual_framework()
    agile_lifecycle()
    use_case()
    class_diagram()
    erd()
    db_schema()
    architecture()
    ui_wireframe()
    print("Diagrams written to", OUT)
