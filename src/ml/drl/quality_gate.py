"""
src/ml/drl/quality_gate.py
─────────────────────────────────────────────────────────────────────────────
Quality gate that controls model promotion (proposal objective iv, Chapter 6
recommendation vi). A candidate model is judged on the VALIDATION split only;
the test split is never used for a promotion decision.

A candidate passes when all three checks pass:
  (a) beats_buy_and_hold   its validation return is higher than Buy & Hold's
                           over the same validation days;
  (b) beats_random_agents  it beats at least `min_random_beaten_pct` % of
                           `n_random_agents` seeded Random Agents (seeds 0..N-1);
  (c) drawdown_floor       its maximum drawdown on validation is no worse than
                           `max_drawdown_floor_pct`.

Thresholds come from the `quality_gate:` section of config/drl.yaml. The
decision and the reason for each check are logged to the candidate's MLflow
run as tags (gate.*) and metrics (gate_*).

Normalisation: the agent's observations use the normaliser fitted on the
TRAINING split (norm_stats), so no validation row is scaled with statistics
that include later validation rows (DEF-08 fix on this path).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.ml.drl.backtest import run_backtest, run_buy_and_hold, run_random_agent


@dataclass
class GateCheck:
    name: str
    passed: bool
    value: float
    threshold: float
    reason: str


@dataclass
class GateResult:
    passed: bool
    checks: list[GateCheck] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)

    @property
    def reasons(self) -> str:
        return "; ".join(c.reason for c in self.checks)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "reasons": self.reasons,
                "checks": [asdict(c) for c in self.checks], "metrics": self.metrics}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on the validation window
# ─────────────────────────────────────────────────────────────────────────────

def window_cfg(drl_cfg: dict, window: pd.DataFrame) -> dict:
    """Copy of the config whose episode covers the whole window (len - 1 trading steps)."""
    c = json.loads(json.dumps(drl_cfg))
    c["environment"]["episode_length"] = int(len(window))
    return c


def max_drawdown_pct(values, initial_cash: float) -> float:
    """Worst peak-to-trough fall in %, counting the starting cash as the first peak."""
    pv = np.r_[initial_cash, np.asarray(values, dtype=float)]
    peak = np.maximum.accumulate(pv)
    return float(((pv - peak) / peak * 100).min())


def evaluate_on_window(agent, window: pd.DataFrame, drl_cfg: dict, norm_stats: dict | None,
                       n_random_agents: int = 30) -> dict:
    """Run the agent, Buy & Hold and seeded Random Agents over the same window."""
    c = window_cfg(drl_cfg, window)
    initial = c["environment"]["initial_cash"]
    bt = run_backtest(agent, window, c, norm_stats=norm_stats)
    bh = run_buy_and_hold(window, c)
    rand = [run_random_agent(window, c, seed=s)["metrics"]["total_return_pct"] for s in range(n_random_agents)]
    return {
        "window_start": str(window.index[0].date()),
        "window_end": str(window.index[-1].date()),
        "window_days": int(len(window)),
        "return_pct": float(bt["metrics"]["total_return_pct"]),
        "max_drawdown_pct": max_drawdown_pct(bt["portfolio_values"], initial),
        "sharpe": float(bt["metrics"]["sharpe"]),
        "n_steps": int(bt["metrics"]["n_steps"]),
        "stopped_out": bool(bt["metrics"]["n_steps"] < len(window) - 1),
        "buy_and_hold_return_pct": float(bh["metrics"]["total_return_pct"]),
        "buy_and_hold_max_drawdown_pct": max_drawdown_pct(bh["portfolio_values"], initial),
        "random_returns_pct": [float(r) for r in rand],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decision (pure function, unit-tested)
# ─────────────────────────────────────────────────────────────────────────────

def decide(metrics: dict, gate_cfg: dict) -> GateResult:
    """Apply checks (a)-(c) to the metrics produced by evaluate_on_window."""
    ret, bh = metrics["return_pct"], metrics["buy_and_hold_return_pct"]
    rand = np.asarray(metrics["random_returns_pct"], dtype=float)
    beaten_pct = float(np.mean(rand < ret) * 100) if len(rand) else 0.0
    min_beaten = float(gate_cfg["min_random_beaten_pct"])
    dd, floor = metrics["max_drawdown_pct"], float(gate_cfg["max_drawdown_floor_pct"])

    a = ret > bh
    b = beaten_pct >= min_beaten
    c = dd >= floor
    checks = [
        GateCheck("beats_buy_and_hold", a, ret, bh,
                  f"(a) validation return {ret:+.2f}% {'>' if a else '<='} Buy & Hold {bh:+.2f}%"),
        GateCheck("beats_random_agents", b, beaten_pct, min_beaten,
                  f"(b) beats {int(round(beaten_pct * len(rand) / 100))}/{len(rand)} Random Agents "
                  f"({beaten_pct:.0f}%, needs {min_beaten:.0f}%)"),
        GateCheck("drawdown_floor", c, dd, floor,
                  f"(c) max drawdown {dd:.2f}% {'within' if c else 'below'} floor {floor:.0f}%"),
    ]
    summary = {k: v for k, v in metrics.items() if k != "random_returns_pct"}
    summary["random_beaten_pct"] = beaten_pct
    summary["random_mean_return_pct"] = float(rand.mean()) if len(rand) else None
    return GateResult(passed=bool(a and b and c), checks=checks, metrics=summary)


# ─────────────────────────────────────────────────────────────────────────────
# MLflow logging
# ─────────────────────────────────────────────────────────────────────────────

def log_gate_to_mlflow(run_id: str, result: GateResult, gate_cfg: dict, tracking_uri: str,
                       context: str = "promotion") -> None:
    """Record the decision on the candidate's own run (tags gate.* and metrics gate_*)."""
    from mlflow import MlflowClient
    client = MlflowClient(tracking_uri=tracking_uri)
    tags = {
        "gate.passed": str(result.passed).lower(),
        "gate.decision": "PASS" if result.passed else "FAIL",
        "gate.reasons": result.reasons,
        "gate.context": context,
        "gate.window": f"{result.metrics['window_start']} to {result.metrics['window_end']}",
        "gate.thresholds": json.dumps(gate_cfg, sort_keys=True),
        "gate.evaluated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    for c in result.checks:
        tags[f"gate.check.{c.name}"] = "pass" if c.passed else "fail"
    for k, v in tags.items():
        client.set_tag(run_id, k, v)
    m = result.metrics
    for k in ["return_pct", "buy_and_hold_return_pct", "random_beaten_pct", "max_drawdown_pct", "sharpe"]:
        if m.get(k) is not None:
            client.log_metric(run_id, f"gate_val_{k}", float(m[k]))
