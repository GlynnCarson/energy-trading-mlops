"""
evaluate_training_period.py
─────────────────────────────────────────────────────────────────────────────
Replays the five fixed-reward PPO-300k agents (seeds 42, 1, 2, 3, 4) over the
period they were TRAINED on, next to Buy & Hold on the same days, for the
dashboard. No training happens here: the saved best_model of each MLflow run
is loaded and walked through the training split with the backtest code.

This is IN-SAMPLE. The agents learned from these prices (in random 252-day
episodes), so the curves show how well they fit their training data, not
whether they can trade. Evidence of skill is the validation gate and the
held-out test period, never this.

Protocol (same as the test-period evaluation in retrain_fixed_reward.py):
  * dataset: the July 2026 EIA series rebuilt by evaluate_eia_runs.build_dataset()
    (train 3,310 rows); needs the EIA API
  * settings: make_cfg() of retrain_fixed_reward.py (max position 25,000 MMBtu,
    stop-loss 50 %, 0.1 % cost), deterministic actions
  * two views:
      - full: one continuous run over the whole training split
      - yearly: a fresh $100,000 account in each calendar year, i.e. episodes
        of about 252 trading days, the length the agents were trained on

    python evaluate_training_period.py

Outputs (reports/evaluation_reward_fix/):
  training_period_trajectory.csv   date, price, Buy & Hold value, value of each seed
  training_period_yearly.csv       year, days, return of Buy & Hold and of each seed
"""

import json
from pathlib import Path

import mlflow
import pandas as pd
from stable_baselines3 import PPO

from evaluate_eia_runs import build_dataset
from retrain_fixed_reward import OUT, RESULTS, artifacts_dir, make_cfg
from src.ml.drl.backtest import run_backtest, run_buy_and_hold

SEEDS = (42, 1, 2, 3, 4)
BUDGET = 300_000


def full_cfg(seed: int, steps: int) -> dict:
    cfg = make_cfg(BUDGET, seed)
    cfg["environment"]["episode_length"] = int(steps)
    return cfg


def replay(agent, df: pd.DataFrame, cfg: dict) -> list[float]:
    # norm_stats=None fits the normaliser on df; for the full run df IS the training split,
    # so the agent sees exactly the scaling it was trained with.
    return list(run_backtest(agent, df, cfg)["portfolio_values"])


def main():
    mlflow.set_tracking_uri("./mlruns")
    _, last_raw_date, _, train, _, _ = build_dataset()
    print(f"[TRAIN-PERIOD] July dataset rebuilt (last raw date {last_raw_date}); "
          f"train {train.index[0].date()} to {train.index[-1].date()}, {len(train)} rows")
    run_ids = {json.loads(f.read_text())["seed"]: json.loads(f.read_text())["run_id"]
               for f in RESULTS.glob(f"PPO-{BUDGET // 1000}k-fixed-s*.json")}
    agents = {s: PPO.load(str(artifacts_dir(run_ids[s]) / "best_model" / "best_model.zip"), device="cpu")
              for s in SEEDS}

    # Full training split, one continuous account
    steps = len(train) - 1
    cfg = full_cfg(SEEDS[0], steps)
    bh = run_buy_and_hold(train, cfg)
    # values are recorded after each step, i.e. at the NEXT day's price (as in the test trajectories)
    out = pd.DataFrame({"date": train.index[1:].date, "price": train["price_usd"].iloc[1:].to_numpy()})
    b = list(bh["portfolio_values"])
    out["buy_hold_value"] = b + [b[-1]] * (len(out) - len(b))
    print(f"  Buy & Hold: {len(b)} days, final ${b[-1]:,.0f}")
    for s in SEEDS:
        v = replay(agents[s], train, full_cfg(s, steps))
        out[f"seed_{s}"] = v + [v[-1]] * (len(out) - len(v))   # a stopped-out account stays flat
        print(f"  seed {s:>2}: {len(v)} days, final ${v[-1]:,.0f}")
    out.round(2).to_csv(OUT / "training_period_trajectory.csv", index=False)

    # One fresh account per calendar year (about one training episode each)
    rows = []
    for year, g in train.groupby(train.index.year):
        if len(g) < 60:
            continue
        c = full_cfg(SEEDS[0], len(g) - 1)
        row = {"year": year, "days": len(g) - 1,
               "buy_hold": run_buy_and_hold(g, c)["metrics"]["total_return_pct"]}
        for s in SEEDS:
            # normaliser frozen from the full training split, as during training
            bt = run_backtest(agents[s], g, full_cfg(s, len(g) - 1),
                              norm_stats=_train_norm(train, full_cfg(s, steps)))
            row[f"seed_{s}"] = bt["metrics"]["total_return_pct"]
        rows.append(row)
        print(f"  {year}: B&H {row['buy_hold']:+.1f} %, seeds "
              + ", ".join(f"{row[f'seed_{s}']:+.1f}" for s in SEEDS))
    pd.DataFrame(rows).round(2).to_csv(OUT / "training_period_yearly.csv", index=False)
    print(f"[TRAIN-PERIOD] wrote {OUT / 'training_period_trajectory.csv'} and training_period_yearly.csv")


_NORM = {}


def _train_norm(train: pd.DataFrame, cfg: dict) -> dict:
    """Normaliser statistics fitted on the whole training split (what the agents trained with)."""
    if "stats" not in _NORM:
        from src.ml.drl.env import EnergyTradingEnv
        _NORM["stats"] = EnergyTradingEnv(train, cfg, mode="test").norm.to_dict()
    return _NORM["stats"]


if __name__ == "__main__":
    main()
