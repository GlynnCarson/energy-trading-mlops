"""
apply_gate_to_seeds.py
─────────────────────────────────────────────────────────────────────────────
Applies the quality gate retroactively to the five fixed-reward PPO-300k runs
(MLflow experiment energy-drl-eia-gas-reward-fix, seeds 42, 1, 2, 3, 4).

    python apply_gate_to_seeds.py gate        # evaluate + log gate result to each run (no registry change)
    python apply_gate_to_seeds.py register    # register the passing runs, choose and alias the champion

Data: the July 2026 dataset, rebuilt exactly as for Chapter 5 (train 3,310 /
val 413 / test 414 rows). Only the validation split is used. Candidate model:
each run's best_model checkpoint (the model Chapter 5 backtested).

The normaliser and outlier statistics are fitted on the training split / the
raw data up to the end of the training split and saved to each run as
preprocessing/preprocessing.json; paper trading reuses them unchanged.

Output: reports/live/gate_initial.json and a table on stdout.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from evaluate_eia_runs import build_dataset
from retrain_fixed_reward import make_cfg
from src.ingestion.ingest import EIAIngester
from src.ml.drl import registry
from src.ml.drl.env import EnergyTradingEnv
from src.ml.drl.quality_gate import decide, evaluate_on_window, log_gate_to_mlflow
from src.ml.drl.train_drl import load_drl_config

ROOT = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "reports" / "evaluation_reward_fix" / "runs"
OUT = ROOT / "reports" / "live" / "gate_initial.json"
SEEDS = [42, 1, 2, 3, 4]
BUDGET = 300_000


def training_preprocessing(train: pd.DataFrame, last_raw_date: str, cfg: dict) -> dict:
    """Statistics fitted on training data only (DEF-08 fix for the closed-loop path)."""
    raw = EIAIngester().build_dataset(start="2010-01-01", commodity="gas")
    raw_train = raw[raw.index <= train.index[-1]]
    num = raw_train.select_dtypes("number")
    env = EnergyTradingEnv(train, cfg, mode="test")
    return {
        "feature_cols": env.feature_cols,
        "norm_stats": env.norm.to_dict(),
        "outlier_stats": {c: {"mean": float(num[c].mean()), "std": float(num[c].std())} for c in num.columns},
        "raw_columns": list(raw.columns),
        "train_start": str(train.index[0].date()), "train_end": str(train.index[-1].date()),
        "data_last_date": last_raw_date,
        "fitted_on": "training split only",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gate", "register"])
    mode = ap.parse_args().mode
    base = load_drl_config()
    gate_cfg = base["quality_gate"]

    if mode == "gate":
        _, last_raw_date, _, train, val, test = build_dataset()
        print(f"July dataset: last date {last_raw_date}, train={len(train)} val={len(val)} test={len(test)}")
        prep = None
        results = []
        for seed in SEEDS:
            name = f"PPO-{BUDGET // 1000}k-fixed-s{seed}"
            old = json.loads((RUNS_DIR / f"{name}.json").read_text())
            run_id = old["run_id"]
            cfg = {**make_cfg(BUDGET, seed), **{k: base[k] for k in ("quality_gate", "registry")}}
            if prep is None:
                prep = training_preprocessing(train, last_raw_date, cfg)
            agent = registry.load_agent(run_id, "best_model")
            m = evaluate_on_window(agent, val, cfg, prep["norm_stats"], gate_cfg["n_random_agents"])
            res = decide(m, gate_cfg)
            log_gate_to_mlflow(run_id, res, gate_cfg, registry.tracking_uri(cfg), context="retroactive (July runs)")
            registry.save_preprocessing(cfg, run_id, {**prep, "val_start": m["window_start"],
                                                      "val_end": m["window_end"]})
            registry.record_event({"type": "gate", "context": "retroactive", "name": name, "seed": seed,
                                   "run_id": run_id, "passed": res.passed, "reasons": res.reasons,
                                   "window": f"{m['window_start']} to {m['window_end']}",
                                   "val_return_pct": m["return_pct"],
                                   "buy_and_hold_return_pct": m["buy_and_hold_return_pct"],
                                   "random_beaten_pct": res.metrics["random_beaten_pct"],
                                   "max_drawdown_pct": m["max_drawdown_pct"]})
            results.append({"name": name, "seed": seed, "run_id": run_id, **res.to_dict(),
                            "old_protocol_val_best_pct": old["val_return_pct_best_model"],
                            "test_full_return_pct_not_used": old["full_test"]["agent"]["total_return_pct"]})
            print(f"{name}: {'PASS' if res.passed else 'FAIL'} | {res.reasons}")
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps({"thresholds": gate_cfg, "results": results}, indent=2), encoding="utf-8")

        print("\nseed  val%    B&H%   rand beaten  maxDD%   (a)  (b)  (c)  gate   [old protocol val%]")
        for r in results:
            mm, ck = r["metrics"], {c["name"]: c["passed"] for c in r["checks"]}
            f = lambda x: "ok" if x else "--"
            print(f"{r['seed']:>4} {mm['return_pct']:+7.2f} {mm['buy_and_hold_return_pct']:+7.2f} "
                  f"{mm['random_beaten_pct']:>9.0f}%  {mm['max_drawdown_pct']:+7.2f}   "
                  f"{f(ck['beats_buy_and_hold']):>3}  {f(ck['beats_random_agents']):>3}  {f(ck['drawdown_floor']):>3}  "
                  f"{'PASS' if r['passed'] else 'FAIL'}   [{r['old_protocol_val_best_pct']:+.2f}]")
        return

    # mode == register: uses the saved gate results, never the test split
    data = json.loads(OUT.read_text())
    cands = [{"name": r["name"], "seed": r["seed"], "run_id": r["run_id"], "passed": r["passed"],
              "reasons": r["reasons"], "val_return_pct": r["metrics"]["return_pct"],
              "max_drawdown_pct": r["metrics"]["max_drawdown_pct"], "metrics": r["metrics"]} for r in data["results"]]
    versions = {}
    for c in cands:
        if not c["passed"]:
            continue
        p = registry.load_preprocessing(c["run_id"])
        versions[c["run_id"]] = registry.register(base, c["run_id"], "best_model", {
            "seed": c["seed"], "run_name": c["name"], "model_file": "best_model",
            "experiment": "energy-drl-eia-gas-reward-fix", "total_timesteps": BUDGET,
            "train_window": f"{p['train_start']} to {p['train_end']}",
            "val_window": f"{c['metrics']['window_start']} to {c['metrics']['window_end']}",
            "data_last_date": p["data_last_date"], "gate_passed": "true", "gate_reasons": c["reasons"],
            "val_return_pct": round(c["val_return_pct"], 4),
            "val_buy_and_hold_pct": round(c["metrics"]["buy_and_hold_return_pct"], 4),
            "val_random_beaten_pct": round(c["metrics"]["random_beaten_pct"], 2),
            "val_max_drawdown_pct": round(c["max_drawdown_pct"], 4), "origin": "retroactive gate (July runs)"})
        registry.record_event({"type": "registered", "name": c["name"], "run_id": c["run_id"],
                               "version": versions[c["run_id"]]})
        print(f"registered {c['name']} as version {versions[c['run_id']]}")
    champ = registry.choose_champion(cands)
    if champ is None:
        print("No run passed the gate: no champion.")
        registry.record_event({"type": "no_champion", "reason": "no retroactive candidate passed the gate"})
    else:
        others = sorted((c for c in cands if c["passed"] and c is not champ), key=lambda c: -c["val_return_pct"])
        reason = (f"highest validation return among the {len([c for c in cands if c['passed']])} runs that passed "
                  f"the gate ({champ['val_return_pct']:+.2f}%"
                  + (f"; next best {others[0]['name']} {others[0]['val_return_pct']:+.2f}%" if others else "")
                  + "); test results not used")
        registry.promote(base, versions[champ["run_id"]], reason)
        print(f"champion: {champ['name']} -> version {versions[champ['run_id']]}: {reason}")
    registry.export_registry(base)


if __name__ == "__main__":
    main()
