"""
tests/test_monitor.py
Monitoring thresholds and the retrain trigger.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.drl.monitor import assess, retrain_decision

TH = {
    "min_days": 20,
    "max_gap_since_deploy_pp": -10.0,
    "rolling_window_days": 60,
    "max_rolling_gap_pp": -10.0,
    "max_drawdown_from_peak_pct": -30.0,
    "watch_fraction": 0.5,
    "retrain_cooldown_days": 20,
}


def ledger(agent, bh=None):
    agent = np.asarray(agent, float)
    bh = np.full(len(agent), 100_000.0) if bh is None else np.asarray(bh, float)
    return pd.DataFrame({"portfolio_value": agent, "buy_hold_value": bh})


def path(*legs):
    """Piecewise-linear value path: legs of (n_days, end_value) starting from 100,000."""
    out, v = [100_000.0], 100_000.0
    for n, end in legs:
        out += list(np.linspace(v, end, n + 1)[1:])
        v = end
    return out


def test_healthy_when_tracking_buy_and_hold():
    r = assess(ledger(path((80, 101_000))), TH)
    assert r["status"] == "healthy" and not r["breaches"]
    assert r["metrics"]["gap_since_deploy_pp"] == pytest.approx(1.0)


def test_warming_up_before_min_days_even_if_a_threshold_is_breached():
    r = assess(ledger(path((10, 50_000))), TH)
    assert r["status"] == "warming_up" and r["breaches"]
    assert assess(ledger([]), TH)["status"] == "warming_up"


def test_gap_since_deployment_breach_and_watch():
    assert assess(ledger(path((30, 89_000))), TH)["status"] == "retrain_required"  # -11 pp
    r = assess(ledger(path((30, 94_000))), TH)  # -6 pp: past half (-5)
    assert r["status"] == "watch" and r["watch"]
    assert assess(ledger(path((30, 90_000))), TH)["status"] == "watch"  # -10 pp is not below -10


def test_rolling_gap_needs_a_full_window_and_catches_recent_decay():
    short = assess(ledger(path((40, 130_000))), TH)
    assert short["metrics"]["rolling_gap_pp"] is None  # only 41 rows
    # +40 % early, then a 12 % slide over the last 60 days: since-deployment gap still positive
    r = assess(ledger(path((40, 140_000), (60, 123_000))), TH)
    assert r["metrics"]["gap_since_deploy_pp"] > 0
    assert r["metrics"]["rolling_gap_pp"] < -10 and r["status"] == "retrain_required"


def test_drawdown_from_peak_breach_even_when_ahead_of_buy_and_hold():
    bh = path((25, 60_000), (25, 50_000))
    r = assess(ledger(path((25, 150_000), (25, 100_000)), bh), TH)  # -33 % from peak
    assert r["metrics"]["gap_since_deploy_pp"] > 0
    assert any("drawdown" in b for b in r["breaches"]) and r["status"] == "retrain_required"


@pytest.mark.parametrize(
    "status, since, force, expected",
    [
        ("retrain_required", None, False, True),
        ("retrain_required", 5, False, False),  # cool-down after a failed retrain
        ("retrain_required", 20, False, True),  # cool-down over
        ("healthy", None, False, False),
        ("watch", None, False, False),
        ("healthy", None, True, True),  # drill
    ],
)
def test_retrain_trigger(status, since, force, expected):
    ok, reason = retrain_decision(status, since, TH["retrain_cooldown_days"], force)
    assert ok is expected and reason
