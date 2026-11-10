"""
train_on_eia.py
─────────────────────────────────────────────────────────────────────────────
Train the DRL trading agent on REAL daily energy prices from the EIA API
(Henry Hub natural gas by default), instead of synthetic data.

    python train_on_eia.py [total_timesteps]

Pulls ~15 years of daily prices, engineers features, trains PPO, and backtests
against Buy & Hold / Random on a held-out test period. Tracks to local MLflow.
Everything runs locally — no S3/AWS needed.
"""

import sys
from datetime import datetime
from pathlib import Path

import mlflow

from src.ingestion.ingest import EIAIngester
from src.transformation.transform import clean_raw, engineer_features, temporal_split, validate_dataframe
from src.ml.drl.train_drl import load_drl_config, run_drl_training
from src.ml.drl.backtest import run_full_backtest
from src.utils.helpers import load_pipeline_config
from stable_baselines3 import PPO, SAC, TD3, A2C

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}


def main():
    total_timesteps = int(sys.argv[1]) if len(sys.argv) > 1 else 300_000
    t0 = datetime.now()
    print(f"[EIA TRAIN] Started {t0:%Y-%m-%d %H:%M:%S} | timesteps={total_timesteps:,}")

    # ── 1. Real data from EIA ────────────────────────────────────────────────
    pcfg = load_pipeline_config()["ingestion"]
    ing = EIAIngester()
    df = ing.build_dataset(start=pcfg.get("eia_start_date", "2010-01-01"),
                           commodity=pcfg.get("commodity", "gas"))

    # ── 2. Transform (clean_raw auto-detects daily frequency) ────────────────
    df = clean_raw(df)
    validate_dataframe(df)
    df = engineer_features(df)
    train, val, test = temporal_split(df)
    print(f"[EIA TRAIN] Data ready | train={len(train)} val={len(val)} test={len(test)} "
          f"| features={len(train.columns)}")

    # ── 3. DRL config, adapted for DAILY commodity trading ───────────────────
    cfg = load_drl_config()
    cfg["training"]["total_timesteps"] = total_timesteps
    cfg["environment"]["episode_length"] = 252          # ~1 trading year per episode
    cfg["environment"]["max_position_mwh"] = 25_000     # so ~$100k cash can be deployed at ~$3/MMBtu
    cfg["environment"]["stop_loss_pct"] = 0.50          # commodities are volatile; loosen the kill-switch
    cfg["mlflow"]["tracking_uri"] = "./mlruns"
    cfg["mlflow"]["experiment_name"] = "energy-drl-eia-gas"
    # daily-frequency annualisation for the backtest metrics
    cfg.setdefault("backtest", {})
    cfg["backtest"]["periods_per_year"] = 252
    cfg["backtest"]["steps_per_day"] = 1
    cfg["backtest"]["output_dir"] = "./reports/backtest_eia"

    mlflow.set_tracking_uri("./mlruns")

    # ── 4. Train ─────────────────────────────────────────────────────────────
    run_id = run_drl_training(train, val, cfg)
    print(f"[EIA TRAIN] Training complete. MLflow run: {run_id}")

    # ── 5. Backtest the best checkpoint on held-out data ─────────────────────
    algo = cfg["training"]["algorithm"]
    best = Path(cfg["training"]["best_model_path"]) / "best_model"
    final = Path(cfg["training"]["checkpoint_dir"]) / f"{algo.lower()}_final"
    ckpt = best if best.with_suffix(".zip").exists() else final
    agent = ALGO_MAP[algo].load(str(ckpt))
    print(f"[EIA TRAIN] Loaded {ckpt}.zip")

    summary = run_full_backtest(agent, test, cfg, output_dir=cfg["backtest"]["output_dir"])

    print(f"\n[EIA TRAIN] DONE in {datetime.now() - t0}. Report: {summary['report_path']}")
    print(f"[EIA TRAIN] Agent: {summary['agent']}")


if __name__ == "__main__":
    main()
