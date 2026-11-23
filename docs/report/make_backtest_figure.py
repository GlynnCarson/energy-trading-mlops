"""
Regenerates Figure 5.2 (the backtest dashboard) for the fixed-reward PPO-300k agent, seed 42, on the first 252
test days, with axis labels in the data's real units (trading days, $/MMBtu). Seed 42 is the seed used for the
July runs and is the median of the five fixed-reward runs on the full test period. Uses the saved MLflow model
and the same rebuilt July-2026 dataset as evaluate_eia_runs.py / retrain_fixed_reward.py. The Random Agent is
seeded (DEF-09 fix), so the figure is reproducible.

    python docs/report/make_backtest_figure.py      (needs EIA_API_KEY in .env)

Output: docs/report/figures/backtest_eia_report.png
The original July report (cost-only reward) in reports/backtest_eia/ is kept unchanged as evidence.
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import mlflow  # noqa: E402
from stable_baselines3 import PPO  # noqa: E402
from evaluate_eia_runs import build_dataset  # noqa: E402
from retrain_fixed_reward import make_cfg, artifacts_dir  # noqa: E402
from src.ml.drl.backtest import run_full_backtest  # noqa: E402

SEED, BUDGET = 42, 300_000
RUN = json.load(open(ROOT / "reports" / "evaluation_reward_fix" / "runs" / f"PPO-300k-fixed-s{SEED}.json"))
OUT = Path(__file__).resolve().parent / "figures" / "backtest_eia_report.png"

mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
_, _, _, _, _, test = build_dataset()
cfg = make_cfg(BUDGET, SEED)
agent = PPO.load(str(artifacts_dir(RUN["run_id"]) / "best_model" / "best_model.zip"), device="cpu")
with tempfile.TemporaryDirectory() as tmp:
    summary = run_full_backtest(agent, test, cfg, output_dir=tmp)
    shutil.copy(summary["report_path"], OUT)
print("agent:", {k: summary["agent"][k] for k in ("total_return_pct", "sharpe", "max_drawdown_pct")})
print("saved", OUT)
