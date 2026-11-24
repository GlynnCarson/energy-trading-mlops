"""
tests/test_quality_gate.py
Quality gate (pass/fail cases and boundaries), the frozen training-split normaliser (DEF-08 on the
closed-loop path), the champion rule and the champion/challenger promotion rule.
"""

import numpy as np
import pytest

from src.ingestion.ingest import SyntheticEnergyGenerator
from src.ml.drl.env import EnergyTradingEnv
from src.ml.drl.quality_gate import decide, evaluate_on_window, max_drawdown_pct
from src.ml.drl.registry import challenger_beats_champion, choose_champion
from src.ml.drl.train_drl import load_drl_config
from src.transformation.transform import clean_raw, engineer_features, temporal_split

GATE = {"n_random_agents": 10, "min_random_beaten_pct": 80.0, "max_drawdown_floor_pct": -40.0}


def metrics(ret=20.0, bh=5.0, rand=None, dd=-15.0):
    rand = list(range(-5, 5)) if rand is None else rand  # 10 agents, all below 5 %
    return {
        "return_pct": ret,
        "buy_and_hold_return_pct": bh,
        "random_returns_pct": rand,
        "max_drawdown_pct": dd,
        "window_start": "2024-01-01",
        "window_end": "2024-12-31",
    }


# ── Gate decision ────────────────────────────────────────────────────────────


def test_gate_passes_when_all_three_checks_pass():
    r = decide(metrics(), GATE)
    assert r.passed and all(c.passed for c in r.checks)
    assert r.metrics["random_beaten_pct"] == 100.0


def test_gate_fails_when_not_beating_buy_and_hold():
    r = decide(metrics(ret=5.0, bh=5.0), GATE)  # equal is not "beats"
    assert not r.passed
    assert [c.name for c in r.checks if not c.passed] == ["beats_buy_and_hold"]


def test_gate_random_agent_threshold_boundary():
    rand = [0.0] * 8 + [50.0] * 2  # candidate (20 %) beats 8/10 = 80 %
    assert decide(metrics(rand=rand), GATE).passed
    rand = [0.0] * 7 + [50.0] * 3  # 7/10 = 70 %
    r = decide(metrics(rand=rand), GATE)
    assert not r.passed and "7/10" in r.reasons


def test_gate_drawdown_floor_boundary():
    assert decide(metrics(dd=-40.0), GATE).passed  # exactly on the floor passes
    r = decide(metrics(dd=-40.01), GATE)
    assert not r.passed
    assert [c.name for c in r.checks if not c.passed] == ["drawdown_floor"]


def test_gate_reports_every_failed_reason():
    r = decide(metrics(ret=-50.0, bh=5.0, dd=-60.0), GATE)
    assert not r.passed and not any(c.passed for c in r.checks)
    assert r.reasons.count("(") >= 3


def test_max_drawdown_counts_starting_cash_as_first_peak():
    assert max_drawdown_pct([90.0, 95.0, 100.0], 100.0) == pytest.approx(-10.0)
    assert max_drawdown_pct([110.0, 99.0, 120.0], 100.0) == pytest.approx(-10.0)


# ── Window evaluation and the frozen normaliser ──────────────────────────────


@pytest.fixture(scope="module")
def daily_splits():
    df = SyntheticEnergyGenerator(seed=7).generate(days=60)
    return temporal_split(engineer_features(clean_raw(df)))


class AlwaysLong:
    """Stand-in agent that always holds the maximum long position."""

    def predict(self, obs, deterministic=True):
        return np.array([1.0], dtype=np.float32), None


def test_always_long_agent_matches_buy_and_hold(daily_splits):
    train, val, _ = daily_splits
    cfg = load_drl_config()
    stats = EnergyTradingEnv(train, cfg, mode="test").norm.to_dict()
    m = evaluate_on_window(AlwaysLong(), val, cfg, stats, n_random_agents=3)
    assert m["return_pct"] == pytest.approx(m["buy_and_hold_return_pct"])
    assert m["window_days"] == len(val) and len(m["random_returns_pct"]) == 3


def test_frozen_normaliser_ignores_the_evaluation_window(daily_splits):
    train, val, _ = daily_splits
    cfg = load_drl_config()
    stats = EnergyTradingEnv(train, cfg, mode="test").norm.to_dict()
    env = EnergyTradingEnv(val, cfg, mode="test", norm_stats=stats)
    np.testing.assert_allclose(env.norm.mean, stats["mean"])
    # changing the evaluation data must not change the statistics used to scale it
    env2 = EnergyTradingEnv(val * 3.0, cfg, mode="test", norm_stats=stats)
    np.testing.assert_allclose(env2.norm.mean, env.norm.mean)
    with pytest.raises(ValueError):
        EnergyTradingEnv(val, cfg, mode="test", norm_stats={**stats, "mean": stats["mean"][:-1]})


# ── Champion rule and champion/challenger promotion ──────────────────────────


def test_champion_is_best_validation_return_among_passing_runs():
    cands = [
        {"name": "a", "passed": True, "val_return_pct": 50.0, "max_drawdown_pct": -20.0},
        {"name": "b", "passed": False, "val_return_pct": 90.0, "max_drawdown_pct": -10.0},
        {"name": "c", "passed": True, "val_return_pct": 60.0, "max_drawdown_pct": -30.0},
    ]
    assert choose_champion(cands)["name"] == "c"  # b is higher but failed the gate


def test_champion_tie_goes_to_smaller_drawdown_and_none_when_all_fail():
    cands = [
        {"name": "a", "passed": True, "val_return_pct": 50.0, "max_drawdown_pct": -30.0},
        {"name": "b", "passed": True, "val_return_pct": 50.0, "max_drawdown_pct": -10.0},
    ]
    assert choose_champion(cands)["name"] == "b"
    assert choose_champion([{**c, "passed": False} for c in cands]) is None


@pytest.mark.parametrize(
    "passed, ret, champ, expected",
    [
        (True, 30.0, 20.0, True),  # passed and better -> replaces the champion
        (True, 20.0, 20.0, False),  # equal is not better
        (True, 10.0, 20.0, False),  # passed the gate but worse than the champion
        (False, 99.0, 20.0, False),  # failed the gate, however good its return
        (True, 5.0, None, True),  # no current champion
    ],
)
def test_challenger_promotion_rule(passed, ret, champ, expected):
    ok, reason = challenger_beats_champion({"passed": passed, "val_return_pct": ret}, champ)
    assert ok is expected and reason
