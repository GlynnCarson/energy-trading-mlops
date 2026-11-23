"""
run_real_training.py
─────────────────────────────────────────────────────────────────────────────
Full ("real") DRL training run with the original hourly settings
(500k timesteps, 4 parallel envs, PPO), on a full year of synthetic energy data.
Tracks to a local MLflow store (./mlruns) and runs a full backtest at the end.

Run:  python run_real_training.py
"""

from datetime import datetime
import mlflow

from src.ingestion.ingest import SyntheticEnergyGenerator
from src.transformation.transform import clean_raw, engineer_features, temporal_split
from src.ml.drl.train_drl import load_drl_config, run_drl_training
from src.ml.drl.backtest import run_full_backtest
from stable_baselines3 import PPO, SAC, TD3, A2C
from pathlib import Path

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}


def main():
    t0 = datetime.now()
    print(f"[REAL TRAIN] Started {t0:%Y-%m-%d %H:%M:%S}")

    # ── Data: full year of synthetic hourly data ──────────────────────────────
    gen = SyntheticEnergyGenerator(seed=42)
    df = gen.generate(days=365)
    df = clean_raw(df)
    df = engineer_features(df)
    train, val, test = temporal_split(df)
    print(f"[REAL TRAIN] Data ready | train={len(train)} val={len(val)} test={len(test)}")

    # ── Config: config/drl.yaml with the original hourly settings ─────────────
    drl_cfg = load_drl_config()
    # drl.yaml now holds the daily EIA settings used by the DAG (DEF-13 fix, Nov 2026).
    # This script reproduces the earlier synthetic run, so it pins the hourly values
    # that were the drl.yaml defaults when that run was made.
    drl_cfg["training"]["total_timesteps"] = 500_000
    drl_cfg["environment"].update({"episode_length": 720, "max_position_mwh": 100.0, "stop_loss_pct": 0.30})
    drl_cfg["backtest"].update({"periods_per_year": 8760, "steps_per_day": 24})
    drl_cfg["backtest"].pop("units", None)
    # Write to a LOCAL mlflow store (the configured http://mlflow-server:5000 only exists inside docker-compose).
    drl_cfg["mlflow"]["tracking_uri"] = "./mlruns"
    mlflow.set_tracking_uri("./mlruns")

    print(
        f"[REAL TRAIN] algorithm={drl_cfg['training']['algorithm']} "
        f"timesteps={drl_cfg['training']['total_timesteps']:,} "
        f"n_envs={drl_cfg['training']['n_envs']} "
        f"device={drl_cfg['training']['device']}"
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    run_id = run_drl_training(train, val, drl_cfg)
    print(f"[REAL TRAIN] Training complete. MLflow run: {run_id}")

    # ── Backtest the best checkpoint on the held-out test set ─────────────────
    algo = drl_cfg["training"]["algorithm"]
    best_path = Path(drl_cfg["training"]["best_model_path"]) / "best_model"
    if best_path.with_suffix(".zip").exists():
        agent = ALGO_MAP[algo].load(str(best_path))
        print(f"[REAL TRAIN] Loaded best model from {best_path}.zip")
    else:
        # Fall back to the final model if no "best" was checkpointed
        final_path = Path(drl_cfg["training"]["checkpoint_dir"]) / f"{algo.lower()}_final"
        agent = ALGO_MAP[algo].load(str(final_path))
        print(f"[REAL TRAIN] No best_model found, using final model {final_path}.zip")

    summary = run_full_backtest(agent, test, drl_cfg, output_dir="./reports/backtest")

    elapsed = datetime.now() - t0
    print(f"\n[REAL TRAIN] DONE in {elapsed}. Report: {summary['report_path']}")
    print(f"[REAL TRAIN] Agent metrics: {summary['agent']}")


if __name__ == "__main__":
    main()
