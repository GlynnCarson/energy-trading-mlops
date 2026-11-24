"""
src/ml/drl/monitor.py
─────────────────────────────────────────────────────────────────────────────
Health check of the champion after each paper-trading update (simulated
trading, no real money) and the retrain trigger.

Metrics, computed over the days the CURRENT champion has traded:
  gap_since_deploy_pp   champion return minus Buy & Hold return since the
                        champion became active (percentage points)
  rolling_gap_pp        the same gap over the last `rolling_window_days`
  drawdown_from_peak_pct how far the paper account is now below its highest
                        value since deployment

Thresholds come from the `monitoring:` section of config/drl.yaml.
Status:
  warming_up        fewer than `min_days` trading days: too little to judge
  retrain_required  any metric beyond its threshold
  watch             any metric beyond `watch_fraction` x its threshold
  healthy           otherwise
A retrain is triggered when the status is retrain_required, unless a retrain
that failed to replace the champion happened less than
`retrain_cooldown_days` trading days ago (otherwise a struggling champion
would be retrained every day). force=True triggers a retrain drill.

Output: reports/live/monitor_status.json (latest status, thresholds, history).

    python -m src.ml.drl.monitor
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from src.ml.drl import registry
from src.ml.drl.paper_trade import LABEL, LEDGER
from src.ml.drl.train_drl import load_drl_config

STATUS_FILE = registry.LIVE_DIR / "monitor_status.json"
LABELS = {"warming_up": "Warming up", "healthy": "Healthy", "watch": "Watch", "retrain_required": "Retrain required"}


def assess(ledger: pd.DataFrame, th: dict) -> dict:
    """Pure health assessment of the champion's rows of the ledger (unit-tested)."""
    n = len(ledger)
    out = {"days": n, "metrics": {}, "breaches": [], "watch": []}
    if n == 0:
        out["status"] = "warming_up"
        return out
    v, bh = ledger["portfolio_value"].to_numpy(float), ledger["buy_hold_value"].to_numpy(float)
    ret = (v[-1] / v[0] - 1) * 100
    bh_ret = (bh[-1] / bh[0] - 1) * 100
    w = int(th["rolling_window_days"])
    m = {
        "return_since_deploy_pct": ret,
        "buy_hold_since_deploy_pct": bh_ret,
        "gap_since_deploy_pp": ret - bh_ret,
        "rolling_gap_pp": None,
        "drawdown_from_peak_pct": (v[-1] / v.max() - 1) * 100,
        "max_drawdown_pct": float(((v / pd.Series(v).cummax().to_numpy()) - 1).min() * 100),
    }
    if n > w:  # w daily changes need w + 1 rows
        m["rolling_gap_pp"] = ((v[-1] / v[-w - 1]) - (bh[-1] / bh[-w - 1])) * 100
    out["metrics"] = m

    checks = [
        ("gap_since_deploy_pp", th["max_gap_since_deploy_pp"]),
        ("rolling_gap_pp", th["max_rolling_gap_pp"]),
        ("drawdown_from_peak_pct", th["max_drawdown_from_peak_pct"]),
    ]
    frac = float(th["watch_fraction"])
    for name, limit in checks:
        val = m[name]
        if val is None:
            continue
        if val < limit:
            out["breaches"].append(f"{name} {val:+.2f} below {limit:+.0f}")
        elif val < limit * frac:
            out["watch"].append(f"{name} {val:+.2f} below {limit * frac:+.1f}")
    if n < int(th["min_days"]):
        out["status"] = "warming_up"
    elif out["breaches"]:
        out["status"] = "retrain_required"
    elif out["watch"]:
        out["status"] = "watch"
    else:
        out["status"] = "healthy"
    return out


def retrain_decision(
    status: str, days_since_failed_retrain: int | None, cooldown: int, force: bool = False
) -> tuple[bool, str]:
    """Whether to start a retrain now (unit-tested)."""
    if force:
        return True, "manual retrain drill (forced)"
    if status != "retrain_required":
        return False, f"champion status is {LABELS.get(status, status).lower()}"
    if days_since_failed_retrain is not None and days_since_failed_retrain < cooldown:
        return False, (
            f"cool-down: last retrain failed to replace the champion "
            f"{days_since_failed_retrain} trading days ago (< {cooldown})"
        )
    return True, "champion breached a monitoring threshold"


def run_monitor(cfg: dict | None = None, force: bool = False) -> dict:
    cfg = cfg or load_drl_config()
    th = cfg["monitoring"]
    champ = registry.current_champion(cfg)
    if champ is None:
        raise RuntimeError("No champion registered.")
    ledger = pd.read_csv(LEDGER, dtype={"model_version": str, "seed": str}) if LEDGER.exists() else pd.DataFrame()
    mine = ledger[ledger["model_version"] == champ["version"]] if len(ledger) else ledger
    res = assess(mine, th)

    # trading days since the last retrain that did not replace the champion
    events = registry._load_json().get("events", [])
    failed = [e for e in events if e.get("type") == "retrain" and not e.get("promoted")]
    days_since = None
    if failed and len(ledger):
        last = pd.Timestamp(failed[-1]["data_last_date"])
        days_since = int((pd.to_datetime(ledger["date"]) > last).sum())
    trigger, reason = retrain_decision(res["status"], days_since, int(th["retrain_cooldown_days"]), force)

    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    status = {
        "label": LABEL,
        "champion_version": champ["version"],
        "champion_seed": champ["tags"].get("seed"),
        "deployed_from": mine["date"].iloc[0] if len(mine) else None,
        "last_data_date": ledger["date"].iloc[-1] if len(ledger) else None,
        "status": res["status"],
        "status_label": LABELS[res["status"]],
        "days": res["days"],
        "metrics": {k: (round(v, 4) if v is not None else None) for k, v in res["metrics"].items()},
        "breaches": res["breaches"],
        "watch": res["watch"],
        "thresholds": th,
        "retrain": {"triggered": trigger, "reason": reason, "forced": force},
        "updated_at": now,
    }
    prev = json.loads(STATUS_FILE.read_text()) if STATUS_FILE.exists() else {}
    hist = [
        h
        for h in prev.get("history", [])
        if h["last_data_date"] != status["last_data_date"] or h["champion_version"] != status["champion_version"]
    ]
    hist.append(
        {k: status[k] for k in ("last_data_date", "champion_version", "status", "days")}
        | {
            "gap_since_deploy_pp": status["metrics"].get("gap_since_deploy_pp"),
            "retrain_triggered": trigger,
            "reason": reason,
            "updated_at": now,
        }
    )
    status["history"] = hist
    registry.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(json.dumps(status, indent=2, default=str), encoding="utf-8")
    if trigger:
        registry.record_event(
            {
                "type": "retrain_required" if not force else "retrain_drill",
                "version": champ["version"],
                "reason": reason,
                "breaches": res["breaches"],
                "last_data_date": status["last_data_date"],
            }
        )
    return status


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="retrain drill")
    s = run_monitor(force=ap.parse_args().force)
    print(json.dumps({k: s[k] for k in ("status_label", "days", "metrics", "breaches", "retrain")}, indent=2))
