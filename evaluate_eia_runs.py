"""
evaluate_eia_runs.py
─────────────────────────────────────────────────────────────────────────────
Reproducible re-evaluation of the PPO agents trained on real EIA Henry Hub
data (MLflow experiment `energy-drl-eia-gas`), used as evidence for the
Chapter 5 results.

    python evaluate_eia_runs.py

Steps
  1. Rebuild the dataset exactly as it was at training time: fetch EIA data,
     cut it at the training date, then clean / engineer / split. The script
     checks that the split sizes match the MLflow-logged train_rows/val_rows.
  2. Re-run the deterministic validation episode with each run's final model
     and check it matches the MLflow-logged final_mean_return_pct.
  3. Backtest each run's best checkpoint on (a) the first 252 test days, the
     window the original backtest covered, and (b) the full test period.
     Benchmarks: Buy & Hold and 30 seeded Random Agents.

Outputs go to reports/evaluation/ (JSON summary + CSV trajectories).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import PPO

from src.ingestion.ingest import EIAIngester
from src.transformation.transform import clean_raw, engineer_features, temporal_split, validate_dataframe
from src.ml.drl.env import EnergyTradingEnv
from src.ml.drl.train_drl import evaluate_agent
from src.ml.drl.backtest import run_backtest, run_buy_and_hold, run_random_agent

ROOT = Path(__file__).resolve().parent
EXP_DIR = ROOT / "mlruns" / "724537757993030752"
OUT = ROOT / "reports" / "evaluation"
CUTOFF = "2026-07-14"           # date the EIA runs were trained
N_RANDOM_SEEDS = 30

RUNS = {
    "PPO-10k": {"run_id": "ef3a6400d97345ceab55ab6e0ac0b481", "logged_val_return": -2.9479},
    "PPO-300k": {"run_id": "dec62fade7684182a8bf58ab7fc13a0b", "logged_val_return": -15.0433},
}


def load_run_cfg(run_id: str) -> dict:
    cfg_file = next((EXP_DIR / run_id / "artifacts" / "config").glob("*.yaml"))
    with open(cfg_file) as f:
        return yaml.safe_load(f)


LOGGED_TRAIN_ROWS, LOGGED_VAL_ROWS = 3310, 413   # MLflow params of both EIA runs


def _prepare(raw: pd.DataFrame):
    df = clean_raw(raw.copy())
    validate_dataframe(df)
    df = engineer_features(df)
    return (df, *temporal_split(df))


def build_dataset():
    """EIA publishes with a lag, so the series available on the training day
    ended a few days before CUTOFF. Trim trailing rows until the splits match
    the sizes MLflow logged for the runs."""
    full = EIAIngester().build_dataset(start="2010-01-01", commodity="gas")
    full = full[full.index < pd.Timestamp(CUTOFF, tz="UTC")]
    for trim in range(0, 30):
        raw = full.iloc[: len(full) - trim] if trim else full
        df, train, val, test = _prepare(raw)
        if len(train) == LOGGED_TRAIN_ROWS and len(val) == LOGGED_VAL_ROWS:
            return len(raw), str(raw.index.max().date()), df, train, val, test
    raise RuntimeError("Could not reproduce the logged split sizes.")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw_rows, last_raw_date, df, train, val, test = build_dataset()

    summary = {
        "cutoff": CUTOFF,
        "last_raw_date": last_raw_date,
        "raw_rows": raw_rows,
        "featured_rows": len(df),
        "n_columns": int(df.shape[1]),
        "n_features": int(df.select_dtypes(include=np.number).shape[1] - 1),
        "splits": {
            name: {"rows": len(s), "start": str(s.index.min().date()), "end": str(s.index.max().date()),
                   "price_min": float(s["price_usd"].min()), "price_max": float(s["price_usd"].max()),
                   "price_mean": float(s["price_usd"].mean())}
            for name, s in [("train", train), ("val", val), ("test", test)]
        },
        "runs": {},
    }
    print(json.dumps(summary["splits"], indent=2))
    df[["price_usd", "oil_price_usd"]].to_csv(OUT / "eia_price_series.csv")

    for name, meta in RUNS.items():
        rid = meta["run_id"]
        cfg = load_run_cfg(rid)
        art = EXP_DIR / rid / "artifacts"
        res = {"run_id": rid}

        # (2) validation reproduction with the final model
        final_agent = PPO.load(str(art / "model" / "ppo_final.zip"), device="cpu")
        val_metrics = evaluate_agent(final_agent, EnergyTradingEnv(val, cfg, mode="test"), n_episodes=1)
        res["val_reproduced_return_pct"] = val_metrics["mean_return_pct"]
        res["val_logged_return_pct"] = meta["logged_val_return"]
        res["val_reproduced_ok"] = abs(val_metrics["mean_return_pct"] - meta["logged_val_return"]) < 0.01

        # (3) backtests with the best checkpoint
        best_agent = PPO.load(str(art / "best_model" / "best_model.zip"), device="cpu")
        for window, ep_len in [("first_252", cfg["environment"]["episode_length"]), ("full_test", len(test))]:
            c = json.loads(json.dumps(cfg))
            c["environment"]["episode_length"] = int(ep_len)
            agent_bt = run_backtest(best_agent, test, c)
            bh = run_buy_and_hold(test, c)
            rand = [run_random_agent(test, c, seed=s)["metrics"]["total_return_pct"] for s in range(N_RANDOM_SEEDS)]
            m = agent_bt["metrics"]
            m["total_trades"] = len(agent_bt["trade_log"])
            m["days_flat"] = int(np.sum(agent_bt["positions"] < 1.0))
            res[window] = {
                "agent": m,
                "buy_and_hold": bh["metrics"],
                "random_return_mean": float(np.mean(rand)),
                "random_return_std": float(np.std(rand)),
                "random_return_min": float(np.min(rand)),
                "random_return_max": float(np.max(rand)),
                "random_beats_agent_pct": float(np.mean(np.array(rand) > m["total_return_pct"]) * 100),
            }
            n = len(agent_bt["portfolio_values"])
            pd.DataFrame({
                "date": test.index[1:n + 1].date,
                "price": agent_bt["prices"],
                "agent_value": agent_bt["portfolio_values"],
                "buy_hold_value": bh["portfolio_values"][:n],
                "position": agent_bt["positions"],
            }).to_csv(OUT / f"{name}_{window}_trajectory.csv", index=False)
        summary["runs"][name] = res
        print(name, json.dumps(res, indent=2, default=str))

    with open(OUT / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved -> {OUT / 'evaluation_summary.json'}")
    ok = all(r["val_reproduced_ok"] for r in summary["runs"].values())
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
