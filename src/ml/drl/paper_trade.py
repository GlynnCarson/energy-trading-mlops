"""
src/ml/drl/paper_trade.py
─────────────────────────────────────────────────────────────────────────────
PAPER TRADING (SIMULATED). No real money is involved and no orders are sent
anywhere: the champion's decisions are applied to a simulated $100,000
account, using the same environment code, costs (0.1 %), position limit
(25,000 MMBtu) and stop-loss (50 %) as the backtest, and compared with a
simulated Buy & Hold account over the same days.

How a day t is processed (no look-ahead):
  1. take the EIA rows dated up to and including t, nothing later;
  2. clean them with the outlier statistics frozen from the model's training
     data, and engineer features (all lags / rolling windows are causal);
  3. scale the features with the normaliser frozen from the training split
     (both stored with the model run as preprocessing/preprocessing.json);
  4. mark the account to market at t's price, then let the model choose the
     target position, which is traded at t's price.
So the decision for day t can only use information available at t's close.
The ledger is append-only: days already logged are never recomputed.

Which model trades on day t: the champion active on t. A model becomes active
on the first day after the last date of the data it saw (for the retroactive
champion, 2026-07-07). When the champion is replaced the same account
continues with the new model; each row records the model version.

Outputs (reports/live/):
  paper_trading.csv   date, price, position, action, portfolio value,
                      Buy & Hold value, model version (one row per trading day)
  paper_state.json    account state needed to continue the next day

    python -m src.ml.drl.paper_trade            # update the ledger to the latest EIA data
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from loguru import logger

from src.ml.drl import registry
from src.ml.drl.env import EnergyTradingEnv
from src.ml.drl.train_drl import load_drl_config
from src.transformation.transform import clean_raw, engineer_features

LEDGER = registry.LIVE_DIR / "paper_trading.csv"
STATE = registry.LIVE_DIR / "paper_state.json"
LABEL = "paper trading (simulated)"
COLUMNS = ["date", "price", "action", "position_mmbtu", "portfolio_value", "buy_hold_value",
           "model_version", "seed", "status"]


# ─────────────────────────────────────────────────────────────────────────────
# Causal features
# ─────────────────────────────────────────────────────────────────────────────

def features_as_of(raw: pd.DataFrame, t: pd.Timestamp, prep: dict) -> pd.Series | None:
    """Feature row for day t computed from raw rows dated <= t only. None if t has no row."""
    upto = raw[raw.index <= t]
    if upto.empty or upto.index[-1] != t:
        return None
    feats = engineer_features(clean_raw(upto.copy(), outlier_stats=prep["outlier_stats"]))
    if feats.empty or feats.index[-1] != t:
        return None
    return feats.iloc[-1]


# ─────────────────────────────────────────────────────────────────────────────
# Account (reuses the environment's own trading and observation code)
# ─────────────────────────────────────────────────────────────────────────────

class Account:
    """A simulated account driven through EnergyTradingEnv's _get_obs / _interpret_action /
    _execute_trade, so costs, clamps and the observation match the backtest exactly."""

    FIELDS = ("cash", "position_mwh", "avg_entry_price", "total_trades")

    def __init__(self, cfg: dict, state: dict | None = None):
        self.cfg = cfg
        e = cfg["environment"]
        self.initial = float(e["initial_cash"])
        self.state = state or {"cash": self.initial, "position_mwh": 0.0, "avg_entry_price": 0.0,
                               "total_trades": 0}

    def _env(self, row: pd.Series, feature_cols: list[str], norm_stats: dict | None) -> EnergyTradingEnv:
        df = pd.DataFrame([row[["price_usd", *feature_cols]].astype(float)])
        env = EnergyTradingEnv(df, self.cfg, mode="test", norm_stats=norm_stats)
        env.reset()
        for k in self.FIELDS:
            setattr(env, k, self.state[k])
        return env

    def value(self, price: float) -> float:
        return self.state["cash"] + self.state["position_mwh"] * price

    def act(self, row, feature_cols, norm_stats, policy) -> float:
        """policy(obs) -> action array. Trades at the row's price; returns the action."""
        env = self._env(row, feature_cols, norm_stats)
        action = np.asarray(policy(env._get_obs()), dtype=np.float32).reshape(-1)
        env._execute_trade(env._interpret_action(action))
        self.state = {k: (int(getattr(env, k)) if k == "total_trades" else float(getattr(env, k)))
                      for k in self.FIELDS}
        return float(action[0])

    def liquidate(self, row, feature_cols):
        self.act(row, feature_cols, None, lambda obs: np.array([-1.0]))


# ─────────────────────────────────────────────────────────────────────────────
# Champion schedule
# ─────────────────────────────────────────────────────────────────────────────

def champion_periods(cfg: dict) -> list[dict]:
    """Ordered champion periods from the promotion history. A model is active from the first
    day after its data ended (never before the paper-trading start date)."""
    reg = registry._load_json()
    c, name = registry.client(cfg), cfg["registry"]["model_name"]
    start = pd.Timestamp(cfg["paper_trading"]["start_date"], tz="UTC")
    periods = []
    for ev in reg.get("events", []):
        if ev.get("type") != "promotion":
            continue
        mv = c.get_model_version(name, ev["version"])
        t = dict(mv.tags)
        active = max(start, pd.Timestamp(t["data_last_date"], tz="UTC") + pd.Timedelta(days=1))
        periods.append({"version": str(mv.version), "run_id": mv.run_id, "seed": t.get("seed"),
                        "model_file": t.get("model_file", "best_model"), "active_from": active,
                        "promoted_at": ev["time"]})
    return periods


def active_period(periods: list[dict], t: pd.Timestamp) -> dict | None:
    """Latest promoted model whose active_from <= t."""
    live = [p for p in periods if p["active_from"] <= t]
    return live[-1] if live else None


# ─────────────────────────────────────────────────────────────────────────────
# Ledger update
# ─────────────────────────────────────────────────────────────────────────────

def run_days(raw: pd.DataFrame, days, cfg: dict, state: dict, model_for_day) -> tuple[list[dict], dict]:
    """Process trading days in order. model_for_day(t) -> (version, seed, prep, policy) or None.
    Pure apart from the model loader, so tests can drive it with synthetic data."""
    agent_acc = Account(cfg, state.get("agent"))
    bh_acc = Account(cfg, state.get("buy_hold"))
    stopped = bool(state.get("stopped", False))
    stop = float(cfg["environment"]["stop_loss_pct"])
    rows = []
    for t in days:
        m = model_for_day(t)
        if m is None:
            continue
        version, seed, prep, policy = m
        row = features_as_of(raw, t, prep)
        if row is None:
            continue
        price = float(row["price_usd"])
        value, bh_value = agent_acc.value(price), bh_acc.value(price)
        status = "stopped out" if stopped else "trading"
        action = float("nan")
        if not stopped and value <= agent_acc.initial * (1 - stop):
            agent_acc.liquidate(row, prep["feature_cols"])     # same 50 % stop-loss as the backtest
            stopped, status = True, "stop-loss hit: position closed"
        elif not stopped:
            action = agent_acc.act(row, prep["feature_cols"], prep["norm_stats"], policy)
        bh_acc.act(row, prep["feature_cols"], None, lambda obs: np.array([1.0]))   # Buy & Hold
        rows.append({"date": t.strftime("%Y-%m-%d"), "price": round(price, 4),
                     "action": round(action, 4) if action == action else "",
                     "position_mmbtu": round(agent_acc.state["position_mwh"], 2),
                     "portfolio_value": round(value, 2), "buy_hold_value": round(bh_value, 2),
                     "model_version": version, "seed": seed, "status": status})
    new_state = {"agent": agent_acc.state, "buy_hold": bh_acc.state, "stopped": stopped,
                 "last_date": rows[-1]["date"] if rows else state.get("last_date")}
    return rows, new_state


def update(cfg: dict | None = None, raw: pd.DataFrame | None = None) -> dict:
    """Append every new trading day since the last logged one. Returns a summary."""
    cfg = cfg or load_drl_config()
    if raw is None:
        from src.ingestion.ingest import EIAIngester
        raw = EIAIngester().build_dataset(start="2010-01-01", commodity="gas")
    raw = raw.copy()
    if raw.index.tz is None:
        raw.index = raw.index.tz_localize("UTC")
    raw = raw.sort_index()

    periods = champion_periods(cfg)
    if not periods:
        raise RuntimeError("No champion has been promoted; nothing to paper trade.")
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    start = pd.Timestamp(cfg["paper_trading"]["start_date"], tz="UTC")
    after = pd.Timestamp(state["last_date"], tz="UTC") if state.get("last_date") else start - pd.Timedelta(days=1)
    days = [t for t in raw.index if t > after and t >= start]

    cache = {}

    def model_for_day(t):
        p = active_period(periods, t)
        if p is None:
            return None
        if p["version"] not in cache:
            agent = registry.load_agent(p["run_id"], p["model_file"])
            prep = registry.load_preprocessing(p["run_id"])
            cache[p["version"]] = (p["version"], p["seed"], prep,
                                   lambda obs, a=agent: a.predict(obs, deterministic=True)[0])
        return cache[p["version"]]

    logger.disable("src.transformation.transform")
    try:
        rows, new_state = run_days(raw, days, cfg, state, model_for_day)
    finally:
        logger.enable("src.transformation.transform")

    registry.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    if rows:
        new = pd.DataFrame(rows, columns=COLUMNS)
        if LEDGER.exists():
            new = pd.concat([pd.read_csv(LEDGER, dtype={"model_version": str, "seed": str}), new])
        new.to_csv(LEDGER, index=False)
    new_state.update({"label": LABEL, "data_last_date": str(raw.index[-1].date()),
                      "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")})
    STATE.write_text(json.dumps(new_state, indent=2), encoding="utf-8")
    summary = {"label": LABEL, "new_days": len(rows), "last_date": new_state["last_date"],
               "data_last_date": new_state["data_last_date"]}
    logger.info(f"[PAPER] {summary}")
    return summary


if __name__ == "__main__":
    print(json.dumps(update(), indent=2))
