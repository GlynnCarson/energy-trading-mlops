"""
tests/test_promotion.py
Champion/challenger promotion against a temporary MLflow registry (the project's ./mlruns and
reports/live are never touched): alias moves, old champion is retired, history is kept, and a
full (tiny) retrain round goes through gate -> comparison -> promotion consistently.
"""

import json

import numpy as np
import pandas as pd
import pytest

from src.ml.drl import registry, retrain
from src.ml.drl.quality_gate import GateResult
from src.ml.drl.train_drl import load_drl_config


@pytest.fixture()
def sandbox(tmp_path, monkeypatch):
    mlruns = tmp_path / "mlruns"
    monkeypatch.setattr(registry, "MLRUNS", mlruns)
    monkeypatch.setattr(registry, "LIVE_DIR", tmp_path / "live")
    monkeypatch.setattr(registry, "REGISTRY_JSON", tmp_path / "live" / "model_registry.json")
    monkeypatch.setattr(retrain, "WORK", tmp_path / "work")
    cfg = load_drl_config()
    cfg["registry"]["tracking_uri"] = mlruns.as_uri()
    cfg["training"].update({"n_envs": 1, "eval_freq": 256, "save_freq": 10_000, "n_eval_episodes": 1})
    cfg["PPO"].update({"n_steps": 128, "batch_size": 32, "n_epochs": 1})
    cfg["environment"]["episode_length"] = 60
    cfg["retraining"]["validation_days"] = 60
    cfg["quality_gate"]["n_random_agents"] = 3
    return cfg


@pytest.fixture(scope="module")
def raw():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2019-01-01", periods=480, tz="UTC")
    gas = np.clip(3 + np.cumsum(rng.normal(0, 0.06, len(idx))), 0.5, None)
    oil = np.clip(70 + np.cumsum(rng.normal(0, 0.8, len(idx))), 10, None)
    return pd.DataFrame({"price_usd": gas, "oil_price_usd": oil, "data_source": "eia_api"}, index=idx)


def _run(cfg, seed):
    import mlflow
    mlflow.set_tracking_uri(cfg["registry"]["tracking_uri"])
    mlflow.set_experiment("t")
    with mlflow.start_run() as r:
        mlflow.log_text("x", "best_model/placeholder.txt")
    return r.info.run_id


def test_promotion_moves_alias_and_keeps_history(sandbox):
    cfg = sandbox
    v1 = registry.register(cfg, _run(cfg, 1), "best_model", {"seed": 1, "data_last_date": "2026-07-06"})
    v2 = registry.register(cfg, _run(cfg, 2), "best_model", {"seed": 2, "data_last_date": "2026-11-17"})
    registry.promote(cfg, v1, "first champion")
    assert registry.current_champion(cfg)["version"] == v1
    registry.promote(cfg, v2, "challenger won")
    assert registry.current_champion(cfg)["version"] == v2
    old = registry.client(cfg).get_model_version(cfg["registry"]["model_name"], v1).tags
    assert old["replaced_by"] == v2 and "retired_at" in old
    data = registry.export_registry(cfg)
    assert [v["is_champion"] for v in data["versions"]] == [False, True]
    assert [e["version"] for e in data["events"] if e["type"] == "promotion"] == [v1, v2]


def test_retrain_round_promotes_only_a_better_challenger(sandbox, raw, monkeypatch):
    cfg = sandbox
    # Force the gate open so the test exercises the comparison and promotion plumbing; the gate
    # decision itself is covered in test_quality_gate.py.
    monkeypatch.setattr(retrain, "decide", lambda m, g: GateResult(True, [], {**m, "random_beaten_pct": 100.0}))

    # Round 1: no champion yet -> the passing challenger becomes champion.
    meta = retrain.prepare(cfg, raw=raw, drill=True)
    # DEF-14: the experiment exists before any (parallel) training task starts
    assert registry.client(cfg).get_experiment_by_name(retrain.EXPERIMENT) is not None
    out1 = retrain.evaluate_and_promote(meta["retrain_id"], {7: retrain.train(meta["retrain_id"], 7, cfg, 512)}, cfg)
    assert out1["promoted"] and out1["result"] == "champion_replaced"
    champ1 = registry.current_champion(cfg)["version"]
    assert champ1 == out1["new_champion_version"]

    # Round 2: a new challenger must beat the champion on the same validation window.
    meta2 = retrain.prepare(cfg, raw=raw, drill=True)
    out2 = retrain.evaluate_and_promote(meta2["retrain_id"], {8: retrain.train(meta2["retrain_id"], 8, cfg, 512)}, cfg)
    ch = out2["challengers"][0]
    expected = ch["val_return_pct"] > out2["champion_val_return_pct"]
    assert out2["promoted"] is expected
    assert registry.current_champion(cfg)["version"] == (out2["new_champion_version"] if expected else champ1)

    events = json.loads(registry.REGISTRY_JSON.read_text())["events"]
    assert [e["type"] for e in events].count("retrain") == 2
    assert all(e["drill"] for e in events if e["type"] == "retrain")


def test_smoke_test_never_registers_or_promotes(sandbox, raw, monkeypatch):
    """A wiring check of the retrain branch must not change the registry, even if the gate would pass."""
    cfg = sandbox
    monkeypatch.setattr(retrain, "decide", lambda m, g: GateResult(True, [], {**m, "random_beaten_pct": 100.0}))
    meta = retrain.prepare(cfg, raw=raw, smoke_test=True)
    out = retrain.evaluate_and_promote(meta["retrain_id"], {9: retrain.train(meta["retrain_id"], 9, cfg, 512)}, cfg)
    assert out["smoke_test"] and not out["promoted"] and "smoke test" in out["reason"]
    assert registry.current_champion(cfg) is None
    assert registry.client(cfg).search_model_versions(f"name='{cfg['registry']['model_name']}'") == []
