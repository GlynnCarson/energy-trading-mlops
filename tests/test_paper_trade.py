"""
tests/test_paper_trade.py
Paper trading (simulated): the no-look-ahead rule and that its accounting matches the backtest.
"""

import numpy as np
import pandas as pd
import pytest

from src.ml.drl.backtest import run_backtest
from src.ml.drl.env import EnergyTradingEnv
from src.ml.drl.paper_trade import features_as_of, run_days
from src.ml.drl.train_drl import load_drl_config
from src.transformation.transform import clean_raw, engineer_features


@pytest.fixture(scope="module")
def raw():
    """Business-daily series shaped like the EIA dataset (gas target + oil exogenous)."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=420, tz="UTC")
    gas = np.clip(3 + np.cumsum(rng.normal(0, 0.06, len(idx))), 0.5, None)
    oil = np.clip(70 + np.cumsum(rng.normal(0, 0.8, len(idx))), 10, None)
    return pd.DataFrame({"price_usd": gas, "oil_price_usd": oil, "data_source": "eia_api"}, index=idx)


@pytest.fixture(scope="module")
def cfg():
    return load_drl_config()


@pytest.fixture(scope="module")
def prep(raw, cfg):
    train_raw = raw.iloc[:300]
    num = train_raw.select_dtypes("number")
    train = engineer_features(clean_raw(train_raw.copy()))
    env = EnergyTradingEnv(train, cfg, mode="test")
    return {
        "outlier_stats": {c: {"mean": float(num[c].mean()), "std": float(num[c].std())} for c in num},
        "feature_cols": env.feature_cols,
        "norm_stats": env.norm.to_dict(),
    }


class ObsPolicy:
    """Deterministic stand-in agent whose action depends on the observation (features and portfolio)."""

    def __call__(self, obs):
        return np.array([np.tanh(obs[0] - obs[-2])], dtype=np.float32)

    def predict(self, obs, deterministic=True):
        return self(obs), None


def with_altered_future(raw, t):
    """Same history up to t, wildly different afterwards (a 10x price spike)."""
    alt = raw.copy()
    alt.loc[alt.index > t, "price_usd"] *= 10.0
    alt.loc[alt.index > t, "oil_price_usd"] *= 0.1
    return alt


def test_features_for_day_t_use_no_later_rows(raw, prep):
    t = raw.index[350]
    a = features_as_of(raw, t, prep)
    b = features_as_of(with_altered_future(raw, t), t, prep)
    assert a is not None and a.name == t
    pd.testing.assert_series_equal(a, b)
    # and equal to computing on the full series with frozen statistics (so a batch frame is causal too)
    full = engineer_features(clean_raw(raw.copy(), outlier_stats=prep["outlier_stats"]))
    pd.testing.assert_series_equal(a, full.loc[t], check_names=False)


def test_decisions_up_to_day_t_do_not_change_when_the_future_changes(raw, cfg, prep):
    days = list(raw.index[300:360])
    t = days[-1]

    def model(d):
        return ("1", "0", prep, ObsPolicy())

    rows_a, state_a = run_days(raw, days, cfg, {}, model)
    rows_b, state_b = run_days(with_altered_future(raw, t), days, cfg, {}, model)
    assert rows_a == rows_b and state_a == state_b
    assert len(rows_a) == len(days)


def test_ledger_is_append_only_continuation(raw, cfg, prep):
    """Running 60 days in one go equals running 30 days and then the next 30 from the saved state."""
    days = list(raw.index[300:360])

    def model(d):
        return ("1", "0", prep, ObsPolicy())

    whole, _ = run_days(raw, days, cfg, {}, model)
    first, state = run_days(raw, days[:30], cfg, {}, model)
    second, _ = run_days(raw, days[30:], cfg, state, model)
    assert whole == first + second


def test_paper_trading_accounting_matches_the_backtest(raw, cfg, prep):
    days = list(raw.index[300:400])
    rows, _ = run_days(raw, days, cfg, {}, lambda d: ("1", "0", prep, ObsPolicy()))
    full = engineer_features(clean_raw(raw.copy(), outlier_stats=prep["outlier_stats"]))
    frame = full.loc[days]
    c = {**cfg, "environment": {**cfg["environment"], "episode_length": len(frame)}}
    bt = run_backtest(ObsPolicy(), frame, c, norm_stats=prep["norm_stats"])
    # backtest value after the trade on day i (marked at day i+1) == paper value at the close of day i+1
    paper = np.array([r["portfolio_value"] for r in rows[1:]])
    np.testing.assert_allclose(paper, bt["portfolio_values"][: len(paper)], rtol=0, atol=0.01)
    assert rows[0]["portfolio_value"] == cfg["environment"]["initial_cash"]
    assert all(r["model_version"] == "1" for r in rows)
