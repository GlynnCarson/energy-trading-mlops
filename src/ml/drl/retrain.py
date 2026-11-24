"""
src/ml/drl/retrain.py
─────────────────────────────────────────────────────────────────────────────
Champion/challenger retraining.

1. prepare()   Fetch EIA data up to the latest date and make a walk-forward
               split: the most recent `validation_days` (252) trading days
               are the validation window, everything before is training.
               There is no separate test split: the paper trading after a
               promotion is the out-of-sample test. Outlier and normaliser
               statistics are fitted on the training part only. The raw
               snapshot is saved so every seed trains on identical data.
2. train()     Train PPO-300k with the usual settings for one fixed seed
               (several seeds = several calls), logged to MLflow experiment
               energy-drl-eia-gas-retrain.
3. evaluate_and_promote()
               Each challenger's FINAL model (not the EvalCallback best
               checkpoint, which was picked on the validation window) goes
               through the quality gate on the validation window. The
               current champion is evaluated on the same window with its own
               frozen preprocessing. The best passing challenger replaces the
               champion only if its validation return is higher. Passing
               challengers are registered; every attempt is recorded in the
               history (reports/live/model_registry.json).

    python -m src.ml.drl.retrain [--drill] [--timesteps N] [--seeds 42 1 2]
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd
from loguru import logger

from src.ml.drl import registry
from src.ml.drl.env import EnergyTradingEnv
from src.ml.drl.quality_gate import decide, evaluate_on_window, log_gate_to_mlflow
from src.ml.drl.train_drl import load_drl_config
from src.transformation.transform import clean_raw, engineer_features

EXPERIMENT = "energy-drl-eia-gas-retrain"
WORK = registry.ROOT / "checkpoints" / "retrain"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Data
# ─────────────────────────────────────────────────────────────────────────────


def walk_forward(raw: pd.DataFrame, cfg: dict) -> dict:
    """Split and frozen preprocessing for a retrain (no statistic sees the validation window)."""
    n_val = int(cfg["retraining"]["validation_days"])
    rough = engineer_features(clean_raw(raw.copy()))  # only to locate the split date
    val_start = rough.index[-n_val]
    num = raw[raw.index < val_start].select_dtypes("number")
    outlier_stats = {c: {"mean": float(num[c].mean()), "std": float(num[c].std())} for c in num.columns}
    feats = engineer_features(clean_raw(raw.copy(), outlier_stats=outlier_stats))
    train, val = feats[feats.index < val_start], feats[feats.index >= val_start]
    env = EnergyTradingEnv(train, cfg, mode="test")
    prep = {
        "feature_cols": env.feature_cols,
        "norm_stats": env.norm.to_dict(),
        "outlier_stats": outlier_stats,
        "raw_columns": list(raw.columns),
        "train_start": str(train.index[0].date()),
        "train_end": str(train.index[-1].date()),
        "val_start": str(val.index[0].date()),
        "val_end": str(val.index[-1].date()),
        "data_last_date": str(raw.index[-1].date()),
        "fitted_on": "training split only",
        "split": f"walk-forward: last {n_val} trading days = validation",
    }
    return {"train": train, "val": val, "prep": prep}


def prepare(
    cfg: dict | None = None, raw: pd.DataFrame | None = None, drill: bool = False, smoke_test: bool = False
) -> dict:
    """smoke_test: a wiring check of the retrain branch with a tiny training budget. The gate still runs
    and is logged, but nothing is registered or promoted, so it can never change the champion."""
    cfg = cfg or load_drl_config()
    if raw is None:
        from src.ingestion.ingest import EIAIngester

        raw = EIAIngester().build_dataset(start="2010-01-01", commodity="gas")
    last = str(raw.index[-1].date())
    rid = f"retrain-{last}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    d = WORK / rid
    d.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(d / "raw.pkl")
    # Create the MLflow experiment here, once, before the per-seed training tasks start in parallel:
    # concurrent creation in the file store produced two experiments with the same name (DEF-14).
    cl = registry.client(cfg)
    if cl.get_experiment_by_name(EXPERIMENT) is None:
        cl.create_experiment(EXPERIMENT)
    split = walk_forward(raw, cfg)
    meta = {
        "retrain_id": rid,
        "drill": drill or smoke_test,
        "smoke_test": smoke_test,
        "data_last_date": last,
        **{k: split["prep"][k] for k in ("train_start", "train_end", "val_start", "val_end")},
        "train_rows": len(split["train"]),
        "val_rows": len(split["val"]),
    }
    (d / "meta.json").write_text(json.dumps(meta, indent=2))
    registry.record_event({"type": "retrain_started", **meta})
    logger.info(f"[RETRAIN] prepared {meta}")
    return meta


def _load(rid: str, cfg: dict) -> tuple[pd.DataFrame, dict, dict]:
    d = WORK / rid
    raw = pd.read_pickle(d / "raw.pkl")
    return raw, walk_forward(raw, cfg), json.loads((d / "meta.json").read_text())


# ─────────────────────────────────────────────────────────────────────────────
# 2. Training one challenger
# ─────────────────────────────────────────────────────────────────────────────


def challenger_cfg(cfg: dict, rid: str, seed: int, timesteps: int | None = None) -> dict:
    c = json.loads(json.dumps(cfg))
    c["training"]["seed"] = int(seed)
    c["training"]["total_timesteps"] = int(timesteps or cfg["retraining"]["total_timesteps"])
    ck = WORK / rid / f"seed{seed}"
    c["training"]["checkpoint_dir"] = str(ck)
    c["training"]["best_model_path"] = str(ck / "best_model")
    c["mlflow"]["tracking_uri"] = registry.tracking_uri(cfg)
    c["mlflow"]["experiment_name"] = EXPERIMENT
    return c


def train(rid: str, seed: int, cfg: dict | None = None, timesteps: int | None = None) -> str:
    import mlflow

    from src.ml.drl.train_drl import run_drl_training

    cfg = cfg or load_drl_config()
    raw, split, meta = _load(rid, cfg)
    c = challenger_cfg(cfg, rid, seed, timesteps)
    mlflow.set_tracking_uri(c["mlflow"]["tracking_uri"])
    run_id = run_drl_training(split["train"], split["val"], c)
    cl = registry.client(cfg)
    for k, v in {
        "retrain_id": rid,
        "role": "challenger",
        "drill": str(meta["drill"]).lower(),
        "smoke_test": str(meta.get("smoke_test", False)).lower(),
        "data_last_date": meta["data_last_date"],
        "split": split["prep"]["split"],
    }.items():
        cl.set_tag(run_id, k, v)
    registry.save_preprocessing(cfg, run_id, split["prep"])
    cl.log_text(run_id, raw.to_csv(), "data/raw_snapshot.csv")
    logger.success(f"[RETRAIN] seed {seed} -> run {run_id}")
    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# 3. Gate, champion/challenger comparison, promotion
# ─────────────────────────────────────────────────────────────────────────────


def select_challenger(challengers: list[dict], champion_return_pct: float | None) -> tuple[dict | None, bool, str]:
    """Best passing challenger (highest validation return) and whether it replaces the champion."""
    best = registry.choose_champion(challengers)
    if best is None:
        return None, False, f"none of the {len(challengers)} challengers passed the quality gate"
    ok, why = registry.challenger_beats_champion(best, champion_return_pct)
    return best, ok, f"best passing challenger seed {best['seed']}: {why}"


def evaluate_and_promote(rid: str, run_ids: dict, cfg: dict | None = None) -> dict:
    """run_ids: {seed: run_id}. Returns the outcome (also recorded in the history)."""
    cfg = cfg or load_drl_config()
    gate_cfg = cfg["quality_gate"]
    raw, split, meta = _load(rid, cfg)
    val = split["val"]
    uri = registry.tracking_uri(cfg)
    which = "model" if cfg["retraining"].get("use_final_model", True) else "best_model"

    challengers = []
    for seed, run_id in sorted(run_ids.items(), key=lambda kv: int(kv[0])):
        agent = registry.load_agent(run_id, which)
        m = evaluate_on_window(agent, val, cfg, split["prep"]["norm_stats"], gate_cfg["n_random_agents"])
        res = decide(m, gate_cfg)
        log_gate_to_mlflow(run_id, res, gate_cfg, uri, context=f"challenger ({rid})")
        challengers.append(
            {
                "seed": int(seed),
                "run_id": run_id,
                "passed": res.passed,
                "reasons": res.reasons,
                "val_return_pct": m["return_pct"],
                "max_drawdown_pct": m["max_drawdown_pct"],
                "buy_and_hold_return_pct": m["buy_and_hold_return_pct"],
                "random_beaten_pct": res.metrics["random_beaten_pct"],
            }
        )
        logger.info(f"[RETRAIN] seed {seed}: {'PASS' if res.passed else 'FAIL'} {res.reasons}")

    # the current champion on the SAME window, with its own frozen preprocessing
    champ = registry.current_champion(cfg)
    champ_ret = None
    if champ:
        cp = registry.load_preprocessing(champ["run_id"])
        cf = engineer_features(clean_raw(raw.copy(), outlier_stats=cp["outlier_stats"]))
        cwin = cf.loc[cf.index.intersection(val.index)]
        if len(cwin) != len(val):
            raise RuntimeError("champion features do not cover the validation window")
        cagent = registry.load_agent(champ["run_id"], champ["tags"].get("model_file", "best_model"))
        champ_ret = evaluate_on_window(cagent, cwin, cfg, cp["norm_stats"], 0)["return_pct"]

    best, promote, why = select_challenger(challengers, champ_ret)
    smoke = bool(meta.get("smoke_test", False))
    if smoke:  # wiring check: the decision is reported, but nothing is registered or promoted
        promote, why = False, f"smoke test, promotion disabled (decision would have been: {why})"
    versions = {}
    for c in challengers:
        if not c["passed"] or smoke:
            continue
        p = split["prep"]
        versions[c["seed"]] = registry.register(
            cfg,
            c["run_id"],
            which,
            {
                "seed": c["seed"],
                "run_name": f"challenger-s{c['seed']}",
                "model_file": which,
                "experiment": EXPERIMENT,
                "retrain_id": rid,
                "drill": str(meta["drill"]).lower(),
                "total_timesteps": cfg["retraining"]["total_timesteps"],
                "train_window": f"{p['train_start']} to {p['train_end']}",
                "val_window": f"{p['val_start']} to {p['val_end']}",
                "data_last_date": meta["data_last_date"],
                "gate_passed": "true",
                "gate_reasons": c["reasons"],
                "val_return_pct": round(c["val_return_pct"], 4),
                "val_buy_and_hold_pct": round(c["buy_and_hold_return_pct"], 4),
                "val_random_beaten_pct": round(c["random_beaten_pct"], 2),
                "val_max_drawdown_pct": round(c["max_drawdown_pct"], 4),
                "champion_val_return_pct_same_window": round(champ_ret, 4) if champ_ret is not None else "",
                "origin": "retrain (drill)" if meta["drill"] else "retrain (monitor trigger)",
            },
        )
        c["version"] = versions[c["seed"]]

    new_version = None
    if promote:
        new_version = versions[best["seed"]]
        registry.promote(cfg, new_version, why)
    outcome = {
        "retrain_id": rid,
        "drill": meta["drill"],
        "smoke_test": smoke,
        "data_last_date": meta["data_last_date"],
        "val_window": f"{meta['val_start']} to {meta['val_end']}",
        "model_file": which,
        "champion_version": champ["version"] if champ else None,
        "champion_val_return_pct": champ_ret,
        "challengers": challengers,
        "promoted": promote,
        "new_champion_version": new_version,
        "reason": why,
        "result": "champion_replaced" if promote else ("failed_gate" if best is None else "champion_kept"),
    }
    registry.record_event({"type": "retrain", **outcome})
    registry.export_registry(cfg)
    logger.info(f"[RETRAIN] outcome: {outcome['result']} - {why}")
    return outcome


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--drill", action="store_true")
    ap.add_argument("--timesteps", type=int)
    ap.add_argument("--seeds", type=int, nargs="*")
    a = ap.parse_args()
    cfg = load_drl_config()
    meta = prepare(cfg, drill=a.drill)
    ids = {s: train(meta["retrain_id"], s, cfg, a.timesteps) for s in (a.seeds or cfg["retraining"]["seeds"])}
    print(json.dumps(evaluate_and_promote(meta["retrain_id"], ids, cfg), indent=2, default=str))
