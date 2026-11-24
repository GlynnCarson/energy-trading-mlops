"""One-shot (24 Nov 2026): Section 3.2 (methodology) and 3.3.3 describe the implemented method.

3.2 intro, 3.2.1 Data Acquisition, 3.2.2 Data Preprocessing, 3.2.3 Model Training, 3.2.4 Model
Validation and Testing, and 3.3.3 Reinforcement Learning Environment Design now match the code and
config (daily EIA gas + WTI, 6 SD outliers, business-day regularisation, 38 features, long-only
action, 0.1 % cost, 50 % stop-loss, PPO [256, 256, 128] for 300k steps, EvalCallback selection,
baselines, quality gate). Each changed proposal is still named with a pointer to Table 5.4, and
Table 5.4 gains an "Action" row because the long-only action was not recorded there before.
Backup: docs/report/backup/energy_trading_mlops_proposal_v17_before_methodology.docx
Run from the repo root:  python docs/report/update_methodology.py
"""
import copy
import sys
from pathlib import Path

from docx import Document

DOC = Path(__file__).resolve().parents[2] / "energy_trading_mlops_proposal_v17.docx"
d = Document(str(DOC))
P = d.paragraphs


def find(key):
    hits = [p for p in P if key in p.text and not p.style.name.startswith(("TOC", "Heading"))]
    assert len(hits) == 1, f"{len(hits)} paragraphs contain {key!r}"
    return hits[0]


def set_text(p, text):
    runs = p.runs
    for r in runs[1:]:
        assert bool(r.bold) == bool(runs[0].bold) and bool(r.italic) == bool(runs[0].italic), p.text[:60]
        r._r.getparent().remove(r._r)
    runs[0].text = text


if any("EvalCallback evaluates the agent on the validation split every 10,000 steps and keeps" in p.text
       for p in P):
    sys.exit("Already applied")

NEW = {
    "This project follows a design science research approach.":
        "This project follows a design science research approach. The primary output is a working artefact, the "
        "MLOps pipeline, rather than a theory or statistical hypothesis. The artefact is evaluated against utility "
        "criteria: does the pipeline ingest, validate and transform the data correctly and without leaking future "
        "information; does the trained DRL agent do better than the Buy & Hold and Random baselines on held-out "
        "data it has never seen; does the closed loop gate, register, paper trade and monitor models as designed; "
        "does the pipeline run end to end under Airflow; and do the automated tests and CI workflows check every "
        "code change.",

    "Quantitative evaluation metrics are also used":
        "Quantitative metrics are used throughout. The agent is judged by its total return, Sharpe ratio and "
        "maximum drawdown, always next to Buy & Hold over the same days, and by the share of seeded Random Agents "
        "it beats; data quality is measured by the validation-rule results, the number of values masked as "
        "outliers and the feature count. Every metric is logged to MLflow together with the configuration and the "
        "random seed, so each result can be reproduced. The proposal used the mean episode reward as the main "
        "metric; it was replaced because the reward alone does not show whether the agent did better than simply "
        "holding gas.",

    "Data is acquired from two sources.":
        "Data is acquired from two sources. The primary source is the EIA API v2, from which the EIAIngester class "
        "retrieves two daily spot-price series from 2010 onwards: Henry Hub natural gas (series RNGWHHD, in "
        "$/MMBtu), which is the trading target, and WTI crude oil (series RWTC, in $/bbl), which is an exogenous "
        "feature. The two series are joined by date and the oil price is forward-filled. Each request uses a "
        "30-second timeout and up to three attempts with exponential back-off. When EIA access is unavailable, a "
        "seeded synthetic generator produces repeatable hourly data with five variables (electricity price, "
        "demand, gas price, temperature and renewable share), which the unit tests use. In the Airflow DAG the raw "
        "data is saved as Parquet files under the raw/energy/ prefix in S3, which LocalStack emulates in the local "
        "Docker Compose stack; the local run scripts pass it in memory. The proposal planned hourly US electricity "
        "demand with daily gas prices, but that route returned only monthly data and no price target (Table 5.4).",

    "The transformation pipeline processes data in six ordered steps.":
        "The transformation pipeline processes data in six ordered steps. First, all timestamps are converted to "
        "Coordinated Universal Time (UTC) and sorted chronologically. Second, duplicate timestamps are removed by "
        "keeping the last observation. Third, values more than six standard deviations from the mean are masked "
        "as missing, so that the series stays continuous. Fourth, the data frequency is detected from the median "
        "spacing of the timestamps and the series is regularised to it (business days for the EIA data, hours for "
        "the synthetic data), with gaps of up to five business days or three hours forward-filled. Fifth, the "
        "dataset is checked against eight validation rules (the target is present, numeric, non-null, "
        "non-negative and not constant; there are at least 100 rows; the index is a datetime index and "
        "monotonic), and a failed critical rule stops the pipeline. Sixth, feature engineering adds 38 features: "
        "calendar features, price lags of 1, 6, 12, 24, 48 and 168 steps, rolling means, standard deviations, "
        "minima and maxima over 6, 24 and 168 steps, technical indicators including Relative Strength Index "
        "(RSI)-14, Moving Average Convergence Divergence (MACD), Bollinger Bands and rates of change, and the WTI "
        "price. Every feature is computed from the price shifted by one step, so no feature uses the price of the "
        "day being traded, and rows without enough history for the longest lag or window are dropped. The dataset "
        "is then split 80 % for training, 10 % for validation and 10 % for testing in chronological order. The "
        "proposal used a four-standard-deviation outlier rule on an hourly series; Table 5.4 records why both "
        "changed.",

    "The PPO agent is trained inside EnergyTradingEnv":
        "The PPO agent is trained inside EnergyTradingEnv, a custom Gymnasium environment. On each trading day the "
        "agent observes the 38 normalised features together with its cash ratio, position ratio and unrealised "
        "profit, and outputs a continuous action between -1 and +1 that sets its target position between zero and "
        "the maximum of 25,000 MMBtu, so it can hold gas or stay out of the market but cannot sell short. Each "
        "trade is charged a transaction cost of 0.1 % of its value, trades below the minimum size are ignored, and "
        "an episode ends early if the portfolio falls by 50 % (the stop-loss). The reward is the change in "
        "portfolio value after the price has moved to the next day, scaled by 0.0001, so it contains both the "
        "profit or loss of the position and the transaction cost; a penalty of 1 is subtracted when the stop-loss "
        "is triggered. Training episodes last 252 trading days and start at a random point in the training split. "
        "The PPO model uses a Multi-Layer Perceptron (MLP) policy with three hidden layers of 256, 256 and 128 "
        "units and the hyperparameters in config/drl.yaml (learning rate 3e-4, 2,048 steps per rollout, batch size "
        "64, ten epochs, discount factor 0.99, clip range 0.2 and entropy coefficient 0.01). It trains for 300,000 "
        "timesteps on four parallel environments, and the main configuration is trained with five fixed random "
        "seeds (42 and 1 to 4) to measure how much the result depends on the seed. All hyperparameters, metrics "
        "and models are logged to MLflow. The proposal planned a 10 % fee, idle and drawdown penalties, positions "
        "from fully short to fully long, and 200,000 timesteps on two 256-unit layers (Table 5.4).",

    "After training, the agent is evaluated using the evaluate_policy utility":
        "During training, EvalCallback evaluates the agent on the validation split every 10,000 steps and keeps "
        "the checkpoint with the highest mean validation reward, so the test split plays no part in model "
        "selection. The selected agent is then backtested on the held-out test split with deterministic actions "
        "and compared with two baselines run in the same environment: Buy & Hold, which takes the maximum "
        "position on the first day and keeps it, and 30 Random Agents with fixed seeds. The comparison uses total "
        "and annualised return, Sharpe ratio, maximum drawdown, Calmar ratio, win rate, 95 % value at risk and "
        "profit factor, annualised with 252 trading days per year. Before a model can be registered, the quality "
        "gate judges it on the validation split only: it must return more than Buy & Hold over the same days, "
        "beat at least 80 % of 30 seeded Random Agents and keep its maximum drawdown above -40 %. Software "
        "correctness is tested separately with pytest unit and integration tests, and a CI smoke test trains a "
        "very small agent on 90 days of synthetic data to check that the pipeline runs from data to evaluation. "
        "The proposal's gate, a mean reward above zero over five test episodes, was replaced because it ignored "
        "the baselines (Table 5.4).",

    "The gymnasium environment was designed to discourage degenerate agent strategies.":
        "The Gymnasium environment was designed so that the reward reflects real trading outcomes and the agent "
        "cannot take unrealistic positions. The reward is the change in portfolio value after the price has "
        "moved, so the agent is rewarded for the profit or loss of its position and pays for every trade through "
        "the transaction cost, which discourages excessive trading; the stop-loss ends an episode that loses half "
        "of the portfolio. Position size is limited by the maximum position and by the cash available. The input "
        "features are normalised with a running mean and variance and clipped to five standard deviations, which "
        "prevents scale-related training instability; for the quality gate and paper trading the normalisation "
        "statistics are frozen from the training split, so that no later data leaks into the agent's inputs. The "
        "idle and drawdown penalties planned in the proposal were not used (Table 5.4).",
}
for key, text in NEW.items():
    set_text(find(key), text)

# Table 5.4: record the long-only action (not recorded before)
t = next(tb for tb in d.tables if tb.rows[0].cells[0].text.strip() == "Area"
         and "Proposed" in tb.rows[0].cells[1].text)
reward_row = next(r for r in t.rows if r.cells[0].text.strip() == "Reward")
new_tr = copy.deepcopy(reward_row._tr)
reward_row._tr.addnext(new_tr)
row = next(r for r in t.rows if r._tr is new_tr)
for c, v in zip(row.cells, ("Action", "Position from fully short (-1) to fully long (+1)",
                            "Long only: the action sets a position from 0 to 25,000 MMBtu",
                            "Short selling was not implemented in the environment")):
    set_text(c.paragraphs[0], v)

d.save(str(DOC))
print("Methodology updated")
