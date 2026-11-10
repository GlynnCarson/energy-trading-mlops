"""
Generates the Chapter 4 analysis and design diagrams (Figures 4.1 - 4.8).

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
# Figure 4.1  Context diagram
# ─────────────────────────────────────────────────────────────────────────────
def context_diagram():
    fig, ax, H = canvas("Context Diagram", w=14, h=9.6)
    # central system
    ax.add_patch(Circle((50, 31), 12.5, fc=PALE, ec=BLACK, lw=2.0, zorder=3))
    ax.text(50, 36.5, "0", ha="center", va="center", fontsize=12, fontweight="bold", zorder=4)
    ax.add_line(Line2D([39.5, 60.5], [34.2, 34.2], color=BLACK, lw=1.1, zorder=4))
    ax.text(50, 29.5, "Energy Trading\nMLOps Platform", ha="center", va="center", fontsize=12.5,
            fontweight="bold", zorder=4, linespacing=1.3)

    ent = dict(fill=WHITE, round_=False, lw=1.8, fs=10.5)
    box(ax, 2, 45, 22, 8, "EIA Open Data API v2", sub="external data provider", **ent)
    box(ax, 2, 10, 22, 8, "Airflow Scheduler", sub="time trigger (@daily)", **ent)
    box(ax, 39, 54, 22, 6.5, "Object Storage", sub="AWS S3 / LocalStack", **ent)
    box(ax, 76, 45, 22, 8, "Data Engineer", sub="maintains data pipeline", **ent)
    box(ax, 76, 27.5, 22, 8, "ML Engineer", sub="trains and evaluates agent", **ent)
    box(ax, 76, 10, 22, 8, "Energy Analyst", sub="reviews trading results", **ent)
    box(ax, 39, 3, 22, 6.5, "Email Service", sub="SMTP (Airflow EmailOperator)", **ent)

    # EIA
    arrow(ax, (40.0, 39.5), (24, 51.5), None)
    ax.text(30.5, 54.6, "price request\n(series ID, start date, API key)", ha="center", va="center", fontsize=8.5,
            color=DARK, linespacing=1.2)
    arrow(ax, (24, 46.5), (39.3, 37.4), None)
    ax.text(28.0, 38.6, "daily price records\n(JSON)", ha="center", va="center", fontsize=8.5, color=DARK,
            linespacing=1.2)
    # Scheduler
    arrow(ax, (24, 14), (40, 24), "daily run trigger", lpos=0.45, loff=(-1.5, 1.8))
    # Object storage
    arrow(ax, (47, 43.5), (47, 54), "raw / processed\nParquet, reports", lpos=0.5, loff=(-7.5, 0))
    arrow(ax, (53, 54), (53, 43.5), "stored splits", lpos=0.5, loff=(5.5, 0))
    # Data engineer
    arrow(ax, (76, 51.0), (59.5, 39.0), None)
    ax.text(70.0, 51.8, "pipeline.yaml settings", ha="center", va="center", fontsize=8.5, color=DARK)
    arrow(ax, (61.0, 35.5), (76, 46.5), None)
    ax.text(72.5, 40.0, "data-quality report", ha="center", va="center", fontsize=8.5, color=DARK)
    # ML engineer
    arrow(ax, (76, 33.5), (62.5, 32.5), "drl.yaml, training request", lpos=0.5, loff=(0, 1.8))
    arrow(ax, (62.5, 29.5), (76, 30), "MLflow runs, metrics", lpos=0.5, loff=(0, -1.8))
    # Analyst
    arrow(ax, (59.5, 23.0), (76, 15), "backtest report,\ntrade replay", lpos=0.5, loff=(1.0, 2.6))
    # Email
    arrow(ax, (50, 18.5), (50, 9.5), "success / failure\nnotification", lpos=0.5, loff=(7, 0))
    save(fig, OUT / "fig4_1_context.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.2  DFD Level 0
# ─────────────────────────────────────────────────────────────────────────────
def dfd_level0():
    fig, ax, H = canvas("Data Flow Diagram - Level 0", w=14, h=10.5)
    ent = dict(fill=WHITE, round_=False, lw=1.8, fs=10.5)
    box(ax, 2, 55, 17, 7, "EIA API v2", sub="external entity", **ent)
    box(ax, 81, 55, 17, 7, "ML Engineer", sub="external entity", **ent)
    box(ax, 81, 4, 17, 7, "Email Service", sub="external entity", **ent)

    P = dict(w=19, h=11)
    process(ax, 24, 52, num="1.0", text="Ingest market\ndata", sub="EIAIngester", **P)
    process(ax, 24, 30, num="2.0", text="Clean and\nvalidate", sub="clean_raw, validate_dataframe", **P)
    process(ax, 24, 8, num="3.0", text="Engineer features\nand split", sub="engineer_features, temporal_split", **P)
    process(ax, 56, 52, num="4.0", text="Train DRL\nagent", sub="run_drl_training (PPO)", **P)
    process(ax, 56, 30, num="5.0", text="Backtest and\nevaluate", sub="run_full_backtest", **P)
    process(ax, 56, 8, num="6.0", text="Notify\nstakeholders", sub="notify_success", **P)

    datastore(ax, 2, 37, 19, 5.5, "D1", "Raw data", "s3://.../raw/energy/")
    datastore(ax, 2, 19, 19, 5.5, "D2", "Processed splits", "train / val / test Parquet", dup=True)
    datastore(ax, 56, 44.5, 19, 5.0, "D2", "Processed splits", "duplicate of D2", dup=True)
    datastore(ax, 79, 41, 19.5, 5.5, "D3", "MLflow store", "runs, params, metrics")
    datastore(ax, 79, 31, 19.5, 5.5, "D4", "Model checkpoints", "best_model, ppo_final")
    datastore(ax, 79, 21, 19.5, 5.5, "D5", "Reports", "backtest PNG, DQ JSON")

    arrow(ax, (19, 58.5), (24, 58.5), "prices", loff=(0, 1.4))
    arrow(ax, (33.5, 52), (14, 42.5), "raw Parquet", lpos=0.55, loff=(2.5, 1.4))
    arrow(ax, (12, 37), (24, 36), "raw records", lpos=0.45, loff=(0, -1.6))
    arrow(ax, (33.5, 30), (33.5, 19), "validated series", lpos=0.5, loff=(-6.5, 0))
    arrow(ax, (24, 13), (12, 19), "split files", lpos=0.5, loff=(-1, -1.8))
    arrow(ax, (65.5, 49.5), (65.5, 52), None)
    ax.text(70.5, 50.75, "train + val", ha="center", va="center", fontsize=8.3, color=DARK)
    arrow(ax, (65.5, 44.5), (65.5, 41), None)
    ax.text(69.5, 42.75, "test split", ha="center", va="center", fontsize=8.3, color=DARK)
    arrow(ax, (81, 58.5), (75, 58.5), "drl.yaml", loff=(0, 1.4))
    arrow(ax, (75, 56.0), (82, 46.5), None)
    ax.text(83.0, 51.0, "params, metrics,\nartifacts", ha="left", va="center", fontsize=8.3, color=DARK,
            linespacing=1.2)
    arrow(ax, (75, 53.0), (79.5, 36.5), None)
    ax.text(77.6, 39.3, "checkpoints", ha="left", va="center", fontsize=8.3, color=DARK,
            bbox=dict(fc="white", ec="none", pad=0.3))
    arrow(ax, (79, 33.0), (75, 33.0), None)
    ax.text(77.0, 30.3, "best\nmodel", ha="center", va="center", fontsize=8.0, color=DARK, linespacing=1.1)
    arrow(ax, (75, 31.0), (79.5, 26.5), None)
    ax.text(79.5, 28.6, "report", ha="left", va="center", fontsize=8.3, color=DARK)
    arrow(ax, (65.5, 30), (65.5, 19), "backtest summary", lpos=0.5, loff=(-7, 0))
    arrow(ax, (75, 13.5), (81, 8.5), "email", lpos=0.5, loff=(-0.5, 1.7))
    arrow(ax, (93, 46.5), (93, 55), None)
    ax.text(95.0, 50.8, "run results", rotation=90, ha="center", va="center", fontsize=8.3, color=DARK)
    save(fig, OUT / "fig4_2_dfd0.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.3  Use case diagram
# ─────────────────────────────────────────────────────────────────────────────
def use_case():
    fig, ax, H = canvas("Use Case Diagram", w=14, h=11.2)
    # boundary
    bx, by, bw, bh = 31, 6, 38, 60
    ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle="round,pad=0,rounding_size=1.2", fc="white",
                                ec=BLACK, lw=2.0, zorder=1))
    ax.add_patch(FancyBboxPatch((bx, by + bh - 4.2), bw, 4.2, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=BLACK, ec=BLACK, lw=2.0, zorder=2))
    ax.add_patch(Rectangle((bx, by + bh - 4.2), bw, 1.5, fc=BLACK, ec="none", zorder=2))
    ax.text(50, by + bh - 2.1, "Energy Trading MLOps Platform", ha="center", va="center", color="white",
            fontsize=11.5, fontweight="bold", zorder=3)

    ys = [58.3 - 6.1 * i for i in range(9)]
    uc = {
        "cfg": (50, ys[0], "Configure pipeline\n(YAML)", WHITE),
        "ing": (50, ys[1], "Ingest energy price data", WHITE),
        "tr": (50, ys[2], "Clean, validate and\nengineer features", WHITE),
        "dq": (50, ys[3], "Review data-quality\nreport", WHITE),
        "train": (50, ys[4], "Train DRL trading agent", PALE),
        "track": (50, ys[5], "Track experiment\nin MLflow", PALE),
        "bt": (50, ys[6], "Backtest agent against\nbaselines", PALE),
        "view": (50, ys[7], "View backtest report\nand trade replay", LIGHT),
        "notify": (50, ys[8], "Receive pipeline\nnotification", LIGHT),
    }
    for k, (cx, cy, t, f) in uc.items():
        usecase(ax, cx, cy, t, w=26, h=4.9, fill=f)

    actor(ax, 13, 47, "Data Engineer", "pipeline and data quality")
    actor(ax, 13, 28, "ML Engineer", "training and evaluation")
    actor(ax, 13, 10, "Energy Analyst", "consumes results")
    system_actor(ax, 88, 50, "Airflow Scheduler", "daily trigger")
    system_actor(ax, 88, 30, "EIA API v2", "price data source")
    system_actor(ax, 88, 11, "Email Service", "SMTP")

    def link(a, b):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=MID, lw=1.1, zorder=0.5))

    de, ml, an = (16, 53), (16, 34), (16, 16)
    for k in ["cfg", "ing", "tr", "dq"]:
        link(de, (37, uc[k][1]))
    for k in ["cfg", "train", "track", "bt", "view"]:
        link(ml, (37, uc[k][1]))
    for k in ["view", "notify"]:
        link(an, (37, uc[k][1]))
    sch, eia, mail = (83.5, 53.5), (83.5, 33.5), (83.5, 14.5)
    for k in ["ing", "tr", "train", "bt"]:
        link(sch, (63, uc[k][1]))
    link(eia, (63, uc["ing"][1]))
    link(mail, (63, uc["notify"][1]))

    # «include» Train -> Track
    arrow(ax, (50, ys[4] - 2.45), (50, ys[5] + 2.45), dashed=True, open_head=True)
    ax.text(57.5, (ys[4] + ys[5]) / 2, "«include»", fontsize=8.5, style="italic", color=DARK, ha="center", va="center",
            bbox=dict(fc="white", ec="none", pad=0.4), zorder=8)
    legend(ax, [("box_white", "Data pipeline"), ("box_pale", "Model lifecycle"),
                ("box_mid", "Results and delivery"), ("solid", "association"), ("dash", "«include»")], y=1.3)
    save(fig, OUT / "fig4_3_usecase.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.4  Activity diagram of the orchestrated pipeline
# ─────────────────────────────────────────────────────────────────────────────
def activity():
    fig, ax, H = canvas("Activity Diagram - Daily Pipeline Run", w=14, h=11.0)
    cx = 38

    def act(y, text, sub=None, x=cx, w=26, h=5.2):
        box(ax, x - w / 2, y - h / 2, w, h, text, sub=sub, fill=WHITE, fs=10.2, sub_fs=8.2)

    ax.add_patch(Circle((cx, 66), 1.4, fc=BLACK, zorder=4))
    act(59.5, "ingest", "EIA API v2 -> raw Parquet (S3)")
    act(51.0, "transform: clean_raw", "UTC index, de-duplicate, outliers, asfreq")
    # decision
    ax.add_patch(Polygon([[cx, 46.3], [cx + 3.2, 43.5], [cx, 40.7], [cx - 3.2, 43.5]], fc="white", ec=BLACK,
                         lw=1.5, zorder=4))
    act(35.0, "engineer_features + temporal_split", "lags, rolling stats, RSI, MACD, BB; 80/10/10")
    # fork bar
    ax.add_patch(Rectangle((cx - 22, 29.4), 44, 0.9, fc=BLACK, zorder=4))
    act(23.5, "train_drl", "PPO, MLflow logging, checkpoints", x=cx - 11, w=22)
    act(15.5, "backtest_drl", "vs Buy & Hold and Random", x=cx - 11, w=22)
    act(23.5, "data_quality_report", "rows, nulls, price stats -> JSON", x=cx + 15, w=22)
    act(7.5, "notify_success", "email with run ID + summary", x=cx - 11, w=22)
    ax.add_patch(Rectangle((cx - 22, 2.5), 44, 0.9, fc=BLACK, zorder=4))
    ax.add_patch(Circle((cx, -0.5 + 0.0), 0.01, fc=BLACK))

    # end node next to join (right)
    ax.add_patch(Circle((cx + 30, 2.95), 1.5, fc="white", ec=BLACK, lw=1.6, zorder=4))
    ax.add_patch(Circle((cx + 30, 2.95), 0.9, fc=BLACK, zorder=5))

    arrow(ax, (cx, 64.6), (cx, 62.1))
    arrow(ax, (cx, 56.9), (cx, 53.6))
    arrow(ax, (cx, 48.4), (cx, 46.3))
    arrow(ax, (cx, 40.7), (cx, 37.6), "[critical checks passed]", loff=(9.5, 0.3))
    arrow(ax, (cx, 32.4), (cx, 30.3))
    arrow(ax, (cx - 11, 29.4), (cx - 11, 26.1))
    arrow(ax, (cx + 15, 29.4), (cx + 15, 26.1))
    arrow(ax, (cx - 11, 20.9), (cx - 11, 18.1))
    arrow(ax, (cx - 11, 12.9), (cx - 11, 10.1))
    arrow(ax, (cx - 11, 4.9), (cx - 11, 3.4))
    poly_arrow(ax, [(cx + 15, 20.9), (cx + 15, 3.4)])
    arrow(ax, (cx + 22, 2.95), (cx + 28.5, 2.95))
    ax.text(cx + 24.8, 4.6, "end", fontsize=8.5, color=DARK, ha="center")
    ax.text(cx - 23.0, 29.85, "fork", fontsize=8.5, color=DARK, ha="right", va="center")
    ax.text(cx - 23.0, 2.95, "join (trigger: all_done)", fontsize=8.5, color=DARK, ha="right", va="center")

    # failure branch
    box(ax, 72, 40.5, 25, 6, "Task fails", sub="validation error / exception", fill=PALE, fs=10, sub_fs=8.2)
    arrow(ax, (cx + 3.2, 43.5), (72, 43.5), "[critical check failed]", loff=(0, 1.5))
    box(ax, 72, 28.5, 25, 7.2, "Retry (max 2, 5 min)", sub="then email_on_failure", fill=PALE, fs=10, sub_fs=8.2)
    arrow(ax, (84.5, 40.5), (84.5, 35.7))
    ax.text(90.0, 59.5, "Applies to every task:\nretries = 2, retry_delay = 5 min,\nexecution_timeout = 2 h,\nmax_active_runs = 1",
            ha="center", va="center", fontsize=8.6, color=DARK, linespacing=1.35,
            bbox=dict(fc=PALE, ec=MID, boxstyle="round,pad=0.6", lw=1.0))
    save(fig, OUT / "fig4_4_activity.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.5  Reinforcement-learning design (agent-environment loop)
# ─────────────────────────────────────────────────────────────────────────────
def rl_design():
    fig, ax, H = canvas("Reinforcement Learning Design", subtitle="PPO agent and EnergyTradingEnv (Gymnasium)",
                        w=14, h=8.6)
    # agent
    header_box(ax, 3, 52, 32, "PPO Agent", stereo="stable-baselines3", head_fill=BLACK, sections=[
        ["policy: MlpPolicy (actor-critic)", "net_arch: [256, 256, 128]", "learning_rate: 3e-4, gamma: 0.99",
         "n_steps: 2048, batch_size: 64", "clip_range: 0.2, ent_coef: 0.01", "n_envs: 4 (DummyVecEnv), seed: 42"]])
    # env
    header_box(ax, 64, 52, 33, "EnergyTradingEnv", stereo="gymnasium.Env", head_fill=DARK, sections=[
        ["data: engineered daily price frame", "initial_cash: 100,000 USD", "max_position: 25,000 units",
         "transaction_cost: 0.1 % of trade value", "stop_loss: 50 % portfolio drop", "episode: 252 trading days"]])

    # action arrow (top)
    poly_arrow(ax, [(35, 47), (64, 47)])
    ax.text(49.5, 49.6, "Action  a_t  in  [-1, 1]", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(49.5, 44.4, "target position = (a_t + 1) / 2 x max_position\n(long-only: -1 = flat, +1 = fully long)",
            ha="center", va="center", fontsize=8.6, color=DARK, linespacing=1.3)
    # observation / reward arrows (bottom)
    poly_arrow(ax, [(64, 37), (35, 37)])
    ax.text(49.5, 39.4, "Observation  s_t+1  (41 values)", ha="center", fontsize=10.5, fontweight="bold")
    ax.text(49.5, 34.3, "38 Z-score-normalised market features\n+ cash ratio, position ratio, unrealised PnL",
            ha="center", va="center", fontsize=8.6, color=DARK, linespacing=1.3)
    poly_arrow(ax, [(80, 32.6), (80, 23.5), (19, 23.5), (19, 32.6)])
    ax.text(49.5, 25.4, "Reward  r_t+1", ha="center", fontsize=10.5, fontweight="bold",
            bbox=dict(fc="white", ec="none", pad=0.4))

    # reward design note
    box(ax, 8, 2.5, 84, 12.0, "", fill=PALE, lw=1.2)
    ax.text(50, 12.0, "Designed reward (per step)", ha="center", va="center", fontsize=10.5, fontweight="bold")
    ax.text(50, 8.4, "r_t+1 = scale x [ V(t+1) - V(t) ]  -  transaction costs  ;   V = cash + position x price ;   "
                     "scale = 1e-4",
            ha="center", va="center", fontsize=9.6, family="DejaVu Sans Mono")
    ax.text(50, 5.0, "Terminate when the portfolio falls 50 % below initial cash (penalty -1);  "
                     "truncate at the end of the 252-day window.",
            ha="center", va="center", fontsize=9.0, color=DARK)
    save(fig, OUT / "fig4_5_rl_design.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.6  Class diagram
# ─────────────────────────────────────────────────────────────────────────────
def class_diagram():
    fig, ax, H = canvas("Class Diagram", w=14, h=16.0)
    top = H - 9
    a = header_box(ax, 2, top, 30, "EIAIngester", sections=[
        ["- api_key : str", "- session : requests.Session", "+ SERIES : dict{gas, oil}"],
        ["- _get(route, params) : dict", "- _fetch_series(...) : DataFrame", "+ build_dataset(start, commodity)"]])
    b = header_box(ax, 35, top, 30, "SyntheticEnergyGenerator", sections=[
        ["- seed : int = 42"], ["+ generate(days) : DataFrame"]])
    c = header_box(ax, 68, top, 30, "transform", stereo="module", head_fill=BLACK, sections=[
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
    g = header_box(ax, 2, row3, 30, "train_drl", stereo="module", head_fill=DARK, sections=[
        ["SB3_ALGOS = {PPO, SAC, TD3, A2C}"],
        ["+ load_drl_config() : dict", "+ build_agent(algo, env, cfg)", "+ run_drl_training(train, val) : run_id",
         "+ evaluate_agent(agent, env, n) : dict"]])
    h = header_box(ax, 35, row3, 30, "MLflowCallback /\nPortfolioMetricsCallback", head_fill=DARK, head_h=5.0, sections=[
        ["- log_freq / eval_freq : int"], ["+ _on_step() : bool", "  -> mlflow.log_metrics()"]])
    i = header_box(ax, 68, row3, 30, "backtest", stereo="module", head_fill=DARK, sections=[
        ["+ run_backtest(agent, test_df, cfg)", "+ run_buy_and_hold(test_df, cfg)", "+ run_random_agent(test_df, cfg)",
         "- _compute_metrics(values, ...) : dict", "+ run_full_backtest(...) : dict"]])
    row4 = row3 - 22
    j = header_box(ax, 2, row4, 30, "energy_trading_mlops_pipeline", stereo="Airflow DAG", head_fill=MID,
                   body_fill=PALE, sections=[["schedule : @daily, retries : 2"],
                                             ["ingest, transform, data_quality_report,", "train_drl, backtest_drl, notify_success"]])
    k = header_box(ax, 35, row4, 30, "drl_tasks", stereo="module", head_fill=MID, body_fill=PALE, sections=[
        ["+ task_train_drl(**context)", "+ task_backtest_drl(**context)"]])
    l = header_box(ax, 68, row4, 30, "helpers", stereo="utility", head_fill=MID, body_fill=PALE, sections=[
        ["+ load_pipeline_config() : dict", "+ upload_df_to_s3(df, bucket, key)", "+ download_df_from_s3(bucket, key)",
         "+ get_env(key) : str"]])

    def top_mid(r): return (r[0] + r[2] / 2, r[1] + r[3])
    def bot_mid(r): return (r[0] + r[2] / 2, r[1])

    # dependencies (dashed, open arrow)
    poly_arrow(ax, [(a[0] + a[2] / 2, a[1]), (a[0] + a[2] / 2, a[1] - 2.0), (c[0] + 6, a[1] - 2.0), (c[0] + 6, c[1])],
               dashed=True, open_head=True)
    poly_arrow(ax, [(b[0] + b[2] / 2, b[1]), (b[0] + b[2] / 2, b[1] - 1.2), (c[0] + 4, b[1] - 1.2), (c[0] + 4, c[1])],
               dashed=True, open_head=True, label="DataFrame", lidx=1, loff=(0, -1.3))
    poly_arrow(ax, [(c[0] + c[2] / 2, c[1]), (c[0] + c[2] / 2, e[1] + e[3] + 3), (e[0] + e[2] / 2 + 6, e[1] + e[3] + 3),
                    (e[0] + e[2] / 2 + 6, e[1] + e[3])], dashed=True, open_head=True, label="engineered frame", lidx=1,
               loff=(0, 1.3))
    # inheritance Env <- EnergyTradingEnv (hollow triangle)
    ey = e[1] + e[3] - 9
    ax.add_line(Line2D([e[0], d[0] + d[2] + 1.6], [ey, ey], color=BLACK, lw=1.4, zorder=2))
    ax.add_patch(Polygon([[d[0] + d[2], ey], [d[0] + d[2] + 1.8, ey + 1.0], [d[0] + d[2] + 1.8, ey - 1.0]],
                         fc="white", ec=BLACK, lw=1.4, zorder=3))
    ax.text((e[0] + d[0] + d[2]) / 2 + 0.8, ey + 1.4, "extends", fontsize=8.5, color=DARK, ha="center")
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
          loff=(0, 1.2))
    # bottom row
    arrow(ax, (j[0] + j[2], j[1] + j[3] - 7), (k[0], j[1] + j[3] - 7), "calls", dashed=True, open_head=True, loff=(0, 1.2))
    poly_arrow(ax, [(k[0] + 8, k[1] + k[3]), (k[0] + 8, g[1] - 1.5), (g[0] + g[2] / 2, g[1] - 1.5), (g[0] + g[2] / 2, g[1])],
               dashed=True, open_head=True)
    poly_arrow(ax, [(k[0] + k[2] - 8, k[1] + k[3]), (k[0] + k[2] - 8, i[1] - 1.5), (i[0] + i[2] / 2, i[1] - 1.5),
                    (i[0] + i[2] / 2, i[1])], dashed=True, open_head=True)
    arrow(ax, (k[0] + k[2], k[1] + k[3] - 7), (l[0], k[1] + k[3] - 7), "S3 I/O,\nconfig", dashed=True, open_head=True,
          loff=(0, -2.6))
    legend(ax, [("box_dark", "ML layer"), ("box_white", "Data layer"), ("box_mid", "Orchestration / utility"),
                ("dash", "dependency"), ("solid", "inheritance / composition")], y=2.2)
    save(fig, OUT / "fig4_6_class.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.7  Logical data design
# ─────────────────────────────────────────────────────────────────────────────
def data_design():
    fig, ax, H = canvas("Logical Data Design", subtitle="S3 data lake datasets and MLflow tracking store",
                        w=14, h=11.0)
    top = H - 9.5
    ax.text(25, top + 0.4, "Data lake  (bucket: energy-trading-mlops-data)", ha="center", fontsize=10.5,
            fontweight="bold", color=DARK)
    ax.text(76, top + 0.4, "MLflow tracking store", ha="center", fontsize=10.5, fontweight="bold", color=DARK)
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

    exp = header_box(ax, 56, top - 1.5, 40, "EXPERIMENT", sections=[["PK  experiment_id", "     name  e.g. energy-drl-eia-gas"]])
    run = header_box(ax, 56, exp[1] - 6, 40, "RUN", head_fill=DARK, sections=[
        ["PK  run_id", "FK  experiment_id", "     run_name, status, start/end time"]])
    par = header_box(ax, 56, run[1] - 6, 19, "PARAM", head_fill=DARK, sections=[["FK run_id", "key, value", "e.g. train_rows"]])
    met = header_box(ax, 77, run[1] - 6, 19, "METRIC", head_fill=DARK, sections=[["FK run_id", "key, value, step", "e.g. final_*"]])
    art = header_box(ax, 56, par[1] - 6, 40, "ARTIFACT", head_fill=DARK, sections=[
        ["FK  run_id", "     model/ppo_final.zip", "     best_model/best_model.zip", "     config/*.yaml (full drl.yaml)"]])

    def crow(p1, p2, label, lab_off=(0, 1.2)):
        arrow(ax, p1, p2, label, head=False, lw=1.4, color=BLACK, loff=lab_off, fs=8.3)

    crow((25, raw[1]), (25, proc[1] + proc[3]), "1 : 3  (one ingestion -> train/val/test)", lab_off=(0, 0))
    crow((13.5, proc[1]), (13.5, dq[1] + dq[3]), "1 : 1", lab_off=(3, 0))
    crow((36.5, proc[1]), (36.5, rep[1] + rep[3]), "1 : 1 (test)", lab_off=(5, 0))
    crow((76, exp[1]), (76, run[1] + run[3]), "1 : N", lab_off=(3, 0))
    crow((65.5, run[1]), (65.5, par[1] + par[3]), "1 : N", lab_off=(3, 0))
    crow((86.5, run[1]), (86.5, met[1] + met[3]), "1 : N", lab_off=(3, 0))
    poly_arrow(ax, [(96, run[1] + 3), (98, run[1] + 3), (98, art[1] + art[3] - 3), (96, art[1] + art[3] - 3)],
               dashed=False, label="1 : N", lidx=1, loff=(-1.6, 0))
    # lineage link
    poly_arrow(ax, [(47, proc[1] + proc[3] - 6), (52, proc[1] + proc[3] - 6), (52, run[1] + 4), (56, run[1] + 4)],
               dashed=True, open_head=True, label="lineage: params\ntrain_rows / val_rows", lidx=1, loff=(-0.2, -3),
               fs=8)
    save(fig, OUT / "fig4_7_data_design.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4.8  Deployment diagram
# ─────────────────────────────────────────────────────────────────────────────
def deployment():
    fig, ax, H = canvas("Deployment Diagram", subtitle="Development, container and cloud environments",
                        w=14, h=10.0)

    def node(x, y, w, h, title, status, dashed=False):
        ax.add_patch(Polygon([[x, y + h], [x + 1.5, y + h + 1.5], [x + w + 1.5, y + h + 1.5], [x + w, y + h]],
                             fc=LIGHT, ec=BLACK, lw=1.3, zorder=2))
        ax.add_patch(Polygon([[x + w, y], [x + w + 1.5, y + 1.5], [x + w + 1.5, y + h + 1.5], [x + w, y + h]],
                             fc=LIGHT, ec=BLACK, lw=1.3, zorder=2))
        ax.add_patch(Rectangle((x, y), w, h, fc="white", ec=BLACK, lw=1.6, ls="--" if dashed else "-", zorder=2))
        ax.text(x + 1.2, y + h - 2.0, f"«node»  {title}", fontsize=10.5, fontweight="bold", va="center", zorder=3)
        ax.text(x + 1.2, y + h - 4.2, status, fontsize=8.6, style="italic", color=MID, va="center", zorder=3)

    def comp(x, y, w, h, t, s=None, fill=PALE):
        box(ax, x, y, w, h, t, sub=s, fill=fill, fs=9.4, sub_fs=7.8, lw=1.2)

    node(2, 23, 30, 40, "Developer workstation", "Windows 11, Python 3.11.5, CPU only - used for all reported runs")
    comp(4, 47, 26, 6, "Pipeline code (src/)", "ingest, transform, env, train, backtest")
    comp(4, 39.5, 26, 6, "Local MLflow file store", "./mlruns (experiments, runs, artifacts)")
    comp(4, 32, 26, 6, "Checkpoints and reports", "./checkpoints/drl, ./reports")
    comp(4, 24.5, 26, 6, "Tkinter replay app", "trading_replay_app.py (offline)")

    node(37, 23, 30, 40, "Docker Compose stack", "defined in docker-compose.yml - configuration validated", dashed=True)
    comp(39, 47, 12.5, 6, "Airflow", "webserver, scheduler")
    comp(53, 47, 12, 6, "MLflow server", "+ PostgreSQL")
    comp(39, 39.5, 12.5, 6, "LocalStack", "S3 emulation")
    comp(53, 39.5, 12, 6, "pipeline", "Dockerfile image")
    comp(39, 32, 26, 6, "Airflow image", "Dockerfile.airflow (2.8.1)")

    node(72, 23, 26, 40, "AWS us-east-1", "defined in Terraform - not provisioned", dashed=True)
    comp(74, 49, 22, 4.6, "S3 data lake", "versioned, encrypted")
    comp(74, 43.3, 10.5, 4.6, "Glue", "DB + crawlers")
    comp(85.5, 43.3, 10.5, 4.6, "ECR", "image repo")
    comp(74, 37.6, 10.5, 4.6, "SNS", "email alerts")
    comp(85.5, 37.6, 10.5, 4.6, "CloudWatch", "30-day logs")
    comp(74, 31.9, 22, 4.6, "IAM roles", "Glue, SageMaker")

    box(ax, 2, 6.5, 22, 8, "EIA Open Data API v2", sub="HTTPS, JSON, API key", fill=WHITE, round_=False, fs=10)
    box(ax, 39, 6.5, 26, 8, "GitHub Actions", sub="ci.yml (lint, test, smoke, build,\nTF validate) / deploy.yml", fill=WHITE,
        round_=False, fs=10)
    arrow(ax, (13, 14.5), (13, 23), "HTTPS GET", loff=(-5, 0))
    arrow(ax, (32, 43), (37, 43), None, head=True)
    ax.text(34.5, 37.2, "docker\ncompose\nup", ha="center", fontsize=8.0, color=DARK, linespacing=1.15)
    arrow(ax, (67, 43), (72, 43), None, head=True)
    ax.text(69.5, 37.2, "terraform\napply,\nECR push", ha="center", fontsize=8.0, color=DARK, linespacing=1.15)
    arrow(ax, (65, 10.5), (85, 10.5), "deploy.yml (on push to main)", loff=(0, 1.4), dashed=True, head=False)
    arrow(ax, (85, 10.5), (85, 23), dashed=True)
    legend(ax, [("solid", "implemented and executed"), ("dash", "defined, not executed in this study")], y=1.4)
    save(fig, OUT / "fig4_8_deployment.png")


if __name__ == "__main__":
    context_diagram()
    dfd_level0()
    use_case()
    activity()
    rl_design()
    class_diagram()
    data_design()
    deployment()
    print("Diagrams written to", OUT)
