"""
src/ml/drl/registry.py
─────────────────────────────────────────────────────────────────────────────
MLflow Model Registry helpers for the closed loop.

  * Models that pass the quality gate are registered as versions of
    `energy_trading_ppo`; the approved model carries the alias `champion`
    (aliases replace the deprecated Staging/Production stages).
  * Each version is tagged with its seed, run ID, data window and gate result.
  * Every decision (gate result, registration, promotion, retrain attempt) is
    appended to the history in reports/live/model_registry.json, which the
    dashboard reads (GitHub Actions cannot see the local MLflow store).

Champion rule (choose_champion): among candidates that passed the gate, the one
with the highest validation return; ties go to the smaller drawdown. Test-split
results are never used to choose.

Champion/challenger rule (challenger_beats_champion): a retrained challenger
replaces the champion only if it passed the gate AND its return on the new
validation window is higher than the champion's return on the same window.

Model files are found from the run directory under ./mlruns rather than the
run's absolute artifact URI, so the same registry works on Windows and inside
the Airflow containers (where ./mlruns is mounted).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
LIVE_DIR = ROOT / "reports" / "live"
REGISTRY_JSON = LIVE_DIR / "model_registry.json"


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def tracking_uri(cfg: dict) -> str:
    uri = cfg.get("registry", {}).get("tracking_uri", "./mlruns")
    return (ROOT / uri[2:]).resolve().as_uri() if uri.startswith("./") else uri


def client(cfg: dict):
    from mlflow import MlflowClient
    uri = tracking_uri(cfg)
    return MlflowClient(tracking_uri=uri, registry_uri=uri)


# ─────────────────────────────────────────────────────────────────────────────
# Run artefacts
# ─────────────────────────────────────────────────────────────────────────────

def run_dir(run_id: str, mlruns: Path | None = None) -> Path:
    mlruns = mlruns or ROOT / "mlruns"
    hits = [p for p in mlruns.glob(f"*/{run_id}") if p.is_dir()]
    if len(hits) != 1:
        raise FileNotFoundError(f"run {run_id} not found under {mlruns} ({len(hits)} matches)")
    return hits[0]


def model_path(run_id: str, which: str = "best_model") -> Path:
    """which: 'best_model' (EvalCallback checkpoint) or 'model' (final model)."""
    name = "best_model.zip" if which == "best_model" else "ppo_final.zip"
    return run_dir(run_id) / "artifacts" / which / name


def load_agent(run_id: str, which: str = "best_model"):
    from stable_baselines3 import PPO
    return PPO.load(str(model_path(run_id, which)), device="cpu")


def save_preprocessing(cfg: dict, run_id: str, prep: dict) -> None:
    """Frozen preprocessing (normaliser and outlier statistics from the training split),
    stored with the run so deployment uses exactly what the model was trained with."""
    client(cfg).log_text(run_id, json.dumps(prep, indent=2), "preprocessing/preprocessing.json")


def load_preprocessing(run_id: str) -> dict:
    return json.loads((run_dir(run_id) / "artifacts" / "preprocessing" / "preprocessing.json").read_text())


# ─────────────────────────────────────────────────────────────────────────────
# Decision rules (pure functions, unit-tested)
# ─────────────────────────────────────────────────────────────────────────────

def choose_champion(candidates: list[dict]) -> dict | None:
    """candidates: dicts with 'passed', 'val_return_pct', 'max_drawdown_pct'.
    Highest validation return among those that passed; ties -> smaller drawdown."""
    passed = [c for c in candidates if c["passed"]]
    if not passed:
        return None
    return max(passed, key=lambda c: (c["val_return_pct"], c["max_drawdown_pct"]))


def challenger_beats_champion(challenger: dict, champion_return_pct: float | None) -> tuple[bool, str]:
    """challenger: dict with 'passed' and 'val_return_pct' on the new validation window.
    champion_return_pct: the current champion's return on that same window (None = no champion)."""
    if not challenger["passed"]:
        return False, "challenger failed the quality gate"
    if champion_return_pct is None:
        return True, "no current champion"
    if challenger["val_return_pct"] > champion_return_pct:
        return True, (f"challenger {challenger['val_return_pct']:+.2f}% > champion "
                      f"{champion_return_pct:+.2f}% on the same validation window")
    return False, (f"challenger {challenger['val_return_pct']:+.2f}% <= champion "
                   f"{champion_return_pct:+.2f}% on the same validation window")


# ─────────────────────────────────────────────────────────────────────────────
# Registry operations
# ─────────────────────────────────────────────────────────────────────────────

def register(cfg: dict, run_id: str, which: str, tags: dict) -> str:
    """Create a new version of the registered model from a run's model files."""
    from mlflow.exceptions import MlflowException
    c, name = client(cfg), cfg["registry"]["model_name"]
    try:
        c.create_registered_model(
            name, description="PPO agent trading daily Henry Hub natural gas (EIA RNGWHHD). "
                              "Versions are registered only after passing the quality gate.")
    except MlflowException:
        pass  # already exists
    mv = c.create_model_version(name, source=f"runs:/{run_id}/{which}", run_id=run_id,
                                tags={k: str(v) for k, v in tags.items()})
    return str(mv.version)


def current_champion(cfg: dict) -> dict | None:
    from mlflow.exceptions import MlflowException
    c, reg = client(cfg), cfg["registry"]
    try:
        mv = c.get_model_version_by_alias(reg["model_name"], reg["champion_alias"])
    except MlflowException:
        return None
    return {"version": str(mv.version), "run_id": mv.run_id, "tags": dict(mv.tags)}


def promote(cfg: dict, version: str, reason: str) -> None:
    """Move the champion alias to `version`, tagging old and new versions with dates."""
    c, reg = client(cfg), cfg["registry"]
    name, alias = reg["model_name"], reg["champion_alias"]
    old = current_champion(cfg)
    now = _now()
    if old and old["version"] != version:
        c.set_model_version_tag(name, old["version"], "retired_at", now)
        c.set_model_version_tag(name, old["version"], "replaced_by", version)
    c.set_registered_model_alias(name, alias, version)
    c.set_model_version_tag(name, version, "promoted_at", now)
    c.set_model_version_tag(name, version, "promotion_reason", reason)
    record_event({"type": "promotion", "version": version, "replaced": old["version"] if old else None,
                  "reason": reason})


# ─────────────────────────────────────────────────────────────────────────────
# History and export for the dashboard
# ─────────────────────────────────────────────────────────────────────────────

def _load_json() -> dict:
    if REGISTRY_JSON.exists():
        return json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))
    return {"events": []}


def record_event(event: dict) -> None:
    data = _load_json()
    data.setdefault("events", []).append({"time": _now(), **event})
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_JSON.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def export_registry(cfg: dict) -> dict:
    """Rewrite reports/live/model_registry.json from the MLflow registry, keeping the event history."""
    c, reg = client(cfg), cfg["registry"]
    data = _load_json()
    champ = current_champion(cfg)
    versions = []
    for mv in sorted(c.search_model_versions(f"name='{reg['model_name']}'"), key=lambda m: int(m.version)):
        t = dict(mv.tags)
        versions.append({
            "version": str(mv.version), "run_id": mv.run_id, "source": mv.source,
            "registered_at": datetime.fromtimestamp(mv.creation_timestamp / 1000, timezone.utc)
            .strftime("%Y-%m-%dT%H:%M:%SZ"),
            "is_champion": bool(champ and champ["version"] == str(mv.version)),
            **{k: t[k] for k in sorted(t)},
        })
    data.update({
        "model_name": reg["model_name"],
        "champion_alias": reg["champion_alias"],
        "champion": next((v for v in versions if v["is_champion"]), None),
        "versions": versions,
        "gate_thresholds": cfg["quality_gate"],
        "exported_at": _now(),
        "note": "Registry: MLflow Model Registry (file store ./mlruns). Paper trading is simulated; "
                "no real money is involved.",
    })
    LIVE_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY_JSON.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return data
