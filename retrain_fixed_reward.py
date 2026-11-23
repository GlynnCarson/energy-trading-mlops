"""
retrain_fixed_reward.py
─────────────────────────────────────────────────────────────────────────────
Retrains the EIA Henry Hub agents after the DEF-01 reward fix and evaluates
them with the same protocol as evaluate_eia_runs.py (Chapter 5), so the only
thing that differs from the July 2026 runs is the reward (and, for the extra
runs, the random seed).

    python retrain_fixed_reward.py train --budget 300000 --seed 1   # train + evaluate one run
    python retrain_fixed_reward.py eval  --budget 300000 --seed 42 --run-id <id>
                                                                    # evaluate an existing run
    python retrain_fixed_reward.py summary                          # pool all results

Held constant from the July runs (PPO-10k / PPO-300k):
  * dataset: the July 2026 EIA series, rebuilt by evaluate_eia_runs.build_dataset()
    (train 3,310 / val 413 / test 415 rows)
  * settings of train_on_eia.py: episode 252, max position 25,000, stop-loss 50 %,
    0.1 % cost, PPO hyperparameters from config/drl.yaml, 4 environments
  * checkpoint selection: EvalCallback best mean reward on the validation split
  * evaluation: validation episode, first 252 and full 414 test days, Buy & Hold,
    and 30 seeded Random Agents (action sampler seeded since the DEF-09 fix)

Changed: src/ml/drl/env.py values the portfolio at the NEXT price before
computing the reward, so rewards include profit and loss from price moves.

MLflow experiment: energy-drl-eia-gas-reward-fix (the Chapter 5 experiment
energy-drl-eia-gas is untouched). Outputs: reports/evaluation_reward_fix/.
"""

import argparse
import json
from pathlib import Path
from urllib.parse import unquote, urlparse

import mlflow
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

from evaluate_eia_runs import build_dataset, N_RANDOM_SEEDS
from src.ml.drl.backtest import run_backtest, run_buy_and_hold, run_random_agent
from src.ml.drl.env import EnergyTradingEnv
from src.ml.drl.train_drl import evaluate_agent, load_drl_config, run_drl_training

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "reports" / "evaluation_reward_fix"
RESULTS = OUT / "runs"
EXPERIMENT = "energy-drl-eia-gas-reward-fix"


def run_name(budget: int, seed: int) -> str:
    return f"PPO-{budget // 1000}k-fixed-s{seed}"


def make_cfg(budget: int, seed: int) -> dict:
    """Same overrides as train_on_eia.py, separate checkpoint folder per run."""
    name = run_name(budget, seed)
    cfg = load_drl_config()
    cfg["training"]["total_timesteps"] = budget
    cfg["training"]["seed"] = seed
    ckpt = f"./checkpoints/drl_reward_fix/{name}"
    cfg["training"]["checkpoint_dir"] = ckpt
    cfg["training"]["best_model_path"] = f"{ckpt}/best_model"
    cfg["environment"]["episode_length"] = 252
    cfg["environment"]["max_position_mwh"] = 25_000
    cfg["environment"]["stop_loss_pct"] = 0.50
    cfg["mlflow"]["tracking_uri"] = "./mlruns"
    cfg["mlflow"]["experiment_name"] = EXPERIMENT
    cfg["mlflow"].setdefault("tags", {})["reward_fix"] = "DEF-01"
    cfg.setdefault("backtest", {})
    cfg["backtest"].update({"periods_per_year": 252, "steps_per_day": 1,
                            "output_dir": f"./reports/evaluation_reward_fix/{name}",
                            "units": {"step": "trading days", "price": "$/MMBtu", "position": "MMBtu"}})
    return cfg


def artifacts_dir(run_id: str) -> Path:
    uri = mlflow.get_run(run_id).info.artifact_uri
    p = urlparse(uri)
    return Path(unquote(p.path.lstrip("/")) if p.scheme == "file" else uri)


def evaluate(budget: int, seed: int, run_id: str, val, test) -> dict:
    name, cfg = run_name(budget, seed), make_cfg(budget, seed)
    art = artifacts_dir(run_id)
    res = {"name": name, "run_id": run_id, "seed": seed, "total_timesteps": budget}

    final_agent = PPO.load(str(art / "model" / "ppo_final.zip"), device="cpu")
    best_agent = PPO.load(str(art / "best_model" / "best_model.zip"), device="cpu")
    for label, agent in [("final", final_agent), ("best", best_agent)]:
        m = evaluate_agent(agent, EnergyTradingEnv(val, cfg, mode="test"), n_episodes=1)
        res[f"val_return_pct_{label}_model"] = m["mean_return_pct"]
    res["val_buy_and_hold_return_pct"] = run_buy_and_hold(val, cfg)["metrics"]["total_return_pct"]

    for window, ep_len in [("first_252", cfg["environment"]["episode_length"]), ("full_test", len(test))]:
        c = json.loads(json.dumps(cfg))
        c["environment"]["episode_length"] = int(ep_len)
        bt = run_backtest(best_agent, test, c)
        bh = run_buy_and_hold(test, c)
        rand = [run_random_agent(test, c, seed=s)["metrics"]["total_return_pct"] for s in range(N_RANDOM_SEEDS)]
        m = bt["metrics"]
        m["total_trades"] = len(bt["trade_log"])
        m["days_flat"] = int(np.sum(bt["positions"] < 1.0))

        n = len(bt["portfolio_values"])
        gains = np.diff(np.r_[cfg["environment"]["initial_cash"], bt["portfolio_values"]])
        net = gains.sum()
        m["top5_days_share_of_gain_pct"] = float(np.sort(gains)[-5:].sum() / net * 100) if net > 0 else None

        res[window] = {
            "agent": m,
            "buy_and_hold": bh["metrics"],
            "random_return_mean": float(np.mean(rand)),
            "random_return_std": float(np.std(rand)),
            "random_return_min": float(np.min(rand)),
            "random_return_max": float(np.max(rand)),
            "random_beats_agent_pct": float(np.mean(np.array(rand) > m["total_return_pct"]) * 100),
        }
        pd.DataFrame({
            "date": test.index[1:n + 1].date,
            "price": bt["prices"],
            "agent_value": bt["portfolio_values"],
            "buy_hold_value": bh["portfolio_values"][:n],
            "position": bt["positions"],
            "action": bt["actions"][:n],
        }).to_csv(OUT / f"{name}_{window}_trajectory.csv", index=False)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(res, indent=2, default=str))
    print(name, json.dumps(res, indent=2, default=str))
    return res


def summary():
    rows = []
    for f in sorted(RESULTS.glob("*.json")):
        r = json.loads(f.read_text())
        ft, f252 = r["full_test"], r["first_252"]
        rows.append({
            "run": r["name"], "seed": r["seed"], "steps": r["total_timesteps"],
            "val_final": r["val_return_pct_final_model"], "val_best": r["val_return_pct_best_model"],
            "val_bh": r["val_buy_and_hold_return_pct"],
            "test252": f252["agent"]["total_return_pct"], "bh252": f252["buy_and_hold"]["total_return_pct"],
            "test_full": ft["agent"]["total_return_pct"], "bh_full": ft["buy_and_hold"]["total_return_pct"],
            "sharpe": ft["agent"]["sharpe"], "bh_sharpe": ft["buy_and_hold"]["sharpe"],
            "max_dd": ft["agent"]["max_drawdown_pct"], "days_flat": ft["agent"]["days_flat"],
            "top5_share": ft["agent"]["top5_days_share_of_gain_pct"],
            "rand_mean": ft["random_return_mean"], "rand_beats_pct": ft["random_beats_agent_pct"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "seed_results.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.round(2).to_string(index=False))
    for steps, g in df.groupby("steps"):
        if len(g) > 1:
            print(f"\n{steps:,} steps, {len(g)} seeds: mean ± sd")
            for col in ["val_final", "test252", "test_full", "sharpe", "max_dd", "rand_beats_pct"]:
                print(f"  {col:15s} {g[col].mean():+8.2f} ± {g[col].std(ddof=1):6.2f}"
                      f"   (min {g[col].min():+.2f}, max {g[col].max():+.2f})")
            print(f"  beats Buy & Hold on full test: {(g.test_full > g.bh_full).sum()}/{len(g)}"
                  f"; on validation: {(g.val_final > g.val_bh).sum()}/{len(g)}")
    (OUT / "evaluation_summary.json").write_text(df.to_json(orient="records", indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["train", "eval", "summary"])
    ap.add_argument("--budget", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--run-id")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    if a.mode == "summary":
        return summary()

    mlflow.set_tracking_uri("./mlruns")
    _, last_raw_date, _, train, val, test = build_dataset()
    print(f"[RETRAIN] July dataset rebuilt: last date {last_raw_date}, "
          f"train={len(train)} val={len(val)} test={len(test)}")

    if a.mode == "train":
        print(f"[RETRAIN] Training {run_name(a.budget, a.seed)}")
        run_id = run_drl_training(train, val, make_cfg(a.budget, a.seed))
    else:
        run_id = a.run_id
    evaluate(a.budget, a.seed, run_id, val, test)


if __name__ == "__main__":
    main()
