"""
src/ml/drl/env.py
─────────────────────────────────────────────────────────────────────────────
Custom Gymnasium environment for energy price trading via DRL.

STATE  (observation):
    48 engineered market features (lags, rolling stats, RSI, MACD, Bollinger…)
    + portfolio state vector [cash_ratio, position_ratio, unrealised_pnl_norm]
    = up to 51-dimensional normalised observation

ACTION:
    Continuous Box([-1, 1])  →  target position as fraction of max_position_mwh
        +1.0  = go fully long (buy max MWh)
        -1.0  = fully exit / short  (sell all / short)
         0.0  = hold current position
    -- OR --
    Discrete(3): 0=Hold  1=Buy(+trade_size)  2=Sell(-trade_size)

REWARD:
    Realised + unrealised PnL change minus transaction costs, scaled to ~[-1,1].
    Optionally replaced by rolling Sharpe ratio for risk-adjusted learning.

EPISODE:
    A contiguous window of `episode_length` hourly timesteps sampled randomly
    from the dataset.  Done when window ends OR stop-loss triggers.
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from loguru import logger
from typing import Any

warnings.filterwarnings("ignore", category=FutureWarning)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: running statistics for online normalisation
# ─────────────────────────────────────────────────────────────────────────────

class RunningNorm:
    """Welford's online mean/variance for observation normalisation."""

    def __init__(self, shape: tuple, clip: float = 5.0, epsilon: float = 1e-8):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 0
        self.clip = clip
        self.epsilon = epsilon

    def update(self, x: np.ndarray):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.var += (delta * delta2 - self.var) / self.count

    def normalise(self, x: np.ndarray) -> np.ndarray:
        normed = (x - self.mean) / (np.sqrt(self.var) + self.epsilon)
        return np.clip(normed, -self.clip, self.clip).astype(np.float32)

    def to_dict(self) -> dict:
        """Frozen statistics, saved next to a registered model so that deployment uses
        exactly the normalisation the model was trained with."""
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": int(self.count)}

    def load_dict(self, stats: dict):
        self.mean = np.asarray(stats["mean"], dtype=np.float32)
        self.var = np.asarray(stats["var"], dtype=np.float32)
        self.count = int(stats["count"])


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class EnergyTradingEnv(gym.Env):
    """
    Gymnasium environment wrapping an energy price dataset so a DRL agent
    can learn to trade electricity (buy/sell/hold) to maximise portfolio value.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame from the transformation pipeline.
        Must contain 'price_usd' column.  All other numeric columns are used
        as the observation features.
    config : dict
        DRL config dict loaded from config/drl.yaml  (env + action sections).
    mode : str
        'train' or 'test'.  In test mode random_start is disabled and the
        full dataset is walked from the beginning.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        df: pd.DataFrame,
        config: dict,
        mode: str = "train",
        norm_stats: dict | None = None,
    ):
        super().__init__()

        self.df = df.reset_index(drop=True)
        self.cfg = config["environment"]
        self.act_cfg = config["action"]
        self.mode = mode

        # ── Core config ────────────────────────────────────────────────────
        self.initial_cash = float(self.cfg["initial_cash"])
        self.max_position = float(self.cfg["max_position_mwh"])
        self.tx_cost_pct = float(self.cfg["transaction_cost_pct"])
        self.min_trade = float(self.cfg["min_trade_mwh"])
        self.stop_loss_pct = float(self.cfg["stop_loss_pct"])
        self.reward_scaling = float(self.cfg["reward_scaling"])
        self.holding_penalty = float(self.cfg["holding_penalty"])
        self.use_sharpe = bool(self.cfg["use_sharpe_reward"])
        self.episode_length = int(self.cfg["episode_length"])
        self.random_start = bool(self.cfg["random_start"]) and mode == "train"
        self.normalise = bool(self.cfg["normalise_obs"])
        self.include_portfolio = bool(self.cfg["include_portfolio_state"])

        # ── Feature columns (everything except price target) ───────────────
        self.price_col = "price_usd"
        self.feature_cols = [
            c for c in df.select_dtypes(include=np.number).columns
            if c != self.price_col
        ]
        self.n_features = len(self.feature_cols)

        # Observation dimension: features + optional portfolio state (3)
        self.portfolio_state_dim = 3 if self.include_portfolio else 0
        self.obs_dim = self.n_features + self.portfolio_state_dim

        # ── Online normaliser (pre-fit on training data) ───────────────────
        # norm_stats (from RunningNorm.to_dict on the TRAINING split) freezes the normaliser, so an
        # evaluation or paper-trading environment never sees statistics of its own future rows (DEF-08).
        # Without it the environment fits on its own data, as all Chapter 5 evaluations did.
        self.norm = RunningNorm((self.n_features,))
        if self.normalise:
            if norm_stats is not None:
                if len(norm_stats["mean"]) != self.n_features:
                    raise ValueError(f"norm_stats has {len(norm_stats['mean'])} features, data has {self.n_features}")
                self.norm.load_dict(norm_stats)
            else:
                self._fit_normaliser()

        # ── Spaces ─────────────────────────────────────────────────────────
        obs_low = np.full(self.obs_dim, -np.inf, dtype=np.float32)
        obs_high = np.full(self.obs_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        if self.act_cfg["type"] == "continuous":
            # Target position fraction in [-1, 1]
            self.action_space = spaces.Box(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            # Discrete: 0=Hold, 1=Buy, 2=Sell
            self.action_space = spaces.Discrete(3)

        # ── Internal state (reset each episode) ───────────────────────────
        self._reset_portfolio()
        self.current_step = 0
        self.episode_start = 0

        # For Sharpe reward
        self._reward_history: list[float] = []

        logger.info(
            f"[ENV] EnergyTradingEnv ready | mode={mode} | "
            f"obs_dim={self.obs_dim} | action={self.act_cfg['type']} | "
            f"rows={len(self.df)} | episode_len={self.episode_length}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Normaliser pre-fitting
    # ─────────────────────────────────────────────────────────────────────

    def _fit_normaliser(self):
        """One-pass Welford update over the entire feature matrix."""
        feat_matrix = self.df[self.feature_cols].values.astype(np.float32)
        for row in feat_matrix:
            self.norm.update(row)

    # ─────────────────────────────────────────────────────────────────────
    # Portfolio state management
    # ─────────────────────────────────────────────────────────────────────

    def _reset_portfolio(self):
        self.cash = self.initial_cash
        self.position_mwh = 0.0          # MWh currently held
        self.avg_entry_price = 0.0       # Average cost basis
        self.portfolio_value = self.initial_cash
        self.prev_portfolio_value = self.initial_cash
        self.total_trades = 0
        self.trade_log: list[dict] = []
        self._reward_history = []

    @property
    def current_price(self) -> float:
        idx = self.episode_start + self.current_step
        return float(self.df.loc[idx, self.price_col])

    @property
    def unrealised_pnl(self) -> float:
        if self.position_mwh == 0:
            return 0.0
        return self.position_mwh * (self.current_price - self.avg_entry_price)

    @property
    def total_portfolio_value(self) -> float:
        return self.cash + self.position_mwh * self.current_price

    # ─────────────────────────────────────────────────────────────────────
    # Trade execution
    # ─────────────────────────────────────────────────────────────────────

    def _execute_trade(self, delta_mwh: float) -> float:
        """
        Execute a buy (delta > 0) or sell (delta < 0) of delta_mwh.
        Returns the transaction cost incurred.
        """
        price = self.current_price
        if price <= 0:
            return 0.0

        # Clamp to affordable / owned quantity
        if delta_mwh > 0:
            # Buy: limited by cash
            max_buyable = self.cash / (price * (1 + self.tx_cost_pct))
            delta_mwh = min(delta_mwh, max_buyable, self.max_position - self.position_mwh)
        else:
            # Sell: limited by position held
            delta_mwh = max(delta_mwh, -self.position_mwh)

        if abs(delta_mwh) < self.min_trade:
            return 0.0

        trade_value = abs(delta_mwh) * price
        tx_cost = trade_value * self.tx_cost_pct

        if delta_mwh > 0:
            # Buying energy
            self.cash -= (delta_mwh * price + tx_cost)
            # Update average entry price (weighted average)
            total_mwh = self.position_mwh + delta_mwh
            if total_mwh > 0:
                self.avg_entry_price = (
                    (self.position_mwh * self.avg_entry_price + delta_mwh * price)
                    / total_mwh
                )
            self.position_mwh += delta_mwh
        else:
            # Selling energy
            self.cash += (-delta_mwh * price - tx_cost)
            self.position_mwh += delta_mwh  # delta is negative
            if self.position_mwh <= 0:
                self.position_mwh = 0.0
                self.avg_entry_price = 0.0

        self.cash = max(self.cash, 0.0)
        self.position_mwh = max(self.position_mwh, 0.0)
        self.total_trades += 1

        self.trade_log.append({
            "step": self.current_step,
            "action": "BUY" if delta_mwh > 0 else "SELL",
            "delta_mwh": delta_mwh,
            "price": price,
            "tx_cost": tx_cost,
            "portfolio_value": self.total_portfolio_value,
        })

        return tx_cost

    # ─────────────────────────────────────────────────────────────────────
    # Action interpretation
    # ─────────────────────────────────────────────────────────────────────

    def _interpret_action(self, action) -> float:
        """Convert raw action to delta_mwh to trade."""
        if self.act_cfg["type"] == "continuous":
            # action ∈ [-1, 1] → target position as fraction of max
            target_fraction = float(np.clip(action[0], -1.0, 1.0))
            # Map [-1,1] → [0, max_position]  (no shorting by default)
            target_position = ((target_fraction + 1.0) / 2.0) * self.max_position
            delta_mwh = target_position - self.position_mwh
        else:
            # Discrete: 0=Hold, 1=Buy, 2=Sell
            trade_size = float(self.act_cfg["discrete_trade_size_mwh"])
            if int(action) == 0:
                delta_mwh = 0.0
            elif int(action) == 1:
                delta_mwh = trade_size
            else:
                delta_mwh = -trade_size

        return delta_mwh

    # ─────────────────────────────────────────────────────────────────────
    # Reward calculation
    # ─────────────────────────────────────────────────────────────────────

    def _compute_reward(self, tx_cost: float) -> float:
        """
        Compute the step reward.
        Default: scaled portfolio PnL minus transaction costs.
        Optional: rolling Sharpe ratio (encourages risk-adjusted returns).
        """
        new_value = self.total_portfolio_value
        raw_pnl = new_value - self.prev_portfolio_value

        if self.use_sharpe:
            # Use rolling Sharpe as the reward signal
            self._reward_history.append(raw_pnl)
            if len(self._reward_history) >= 10:
                hist = np.array(self._reward_history[-50:])
                sharpe = hist.mean() / (hist.std() + 1e-8) * np.sqrt(24)  # hourly → daily
                reward = float(sharpe)
            else:
                reward = raw_pnl * self.reward_scaling
        else:
            reward = raw_pnl * self.reward_scaling

        # Small penalty for holding a large position (risk management)
        if self.holding_penalty > 0:
            reward -= self.holding_penalty * (self.position_mwh / self.max_position)

        return float(reward)

    # ─────────────────────────────────────────────────────────────────────
    # Observation builder
    # ─────────────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        idx = self.episode_start + self.current_step
        feat_vec = self.df.loc[idx, self.feature_cols].values.astype(np.float32)

        # Handle any NaN (shouldn't exist after transform pipeline, but defensive)
        feat_vec = np.nan_to_num(feat_vec, nan=0.0, posinf=0.0, neginf=0.0)

        if self.normalise:
            feat_vec = self.norm.normalise(feat_vec)

        if self.include_portfolio:
            total = self.total_portfolio_value
            portfolio_vec = np.array([
                self.cash / (self.initial_cash + 1e-8),           # cash ratio
                self.position_mwh / (self.max_position + 1e-8),   # position ratio
                self.unrealised_pnl / (self.initial_cash + 1e-8), # normalised PnL
            ], dtype=np.float32)
            obs = np.concatenate([feat_vec, portfolio_vec])
        else:
            obs = feat_vec

        return obs.astype(np.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self._reset_portfolio()
        self.current_step = 0

        # Choose episode start
        max_start = len(self.df) - self.episode_length - 1
        if max_start <= 0:
            self.episode_start = 0
        elif self.random_start:
            self.episode_start = int(self.np_random.integers(0, max_start))
        else:
            self.episode_start = 0

        obs = self._get_obs()
        info = {
            "episode_start": self.episode_start,
            "portfolio_value": self.portfolio_value,
            "position_mwh": self.position_mwh,
        }
        return obs, info

    def step(self, action) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Returns: obs, reward, terminated, truncated, info
        """
        self.prev_portfolio_value = self.total_portfolio_value

        # Execute trade at today's price
        delta_mwh = self._interpret_action(action)
        tx_cost = self._execute_trade(delta_mwh)

        # Advance time BEFORE valuing the portfolio, so the reward includes the
        # profit or loss of the position over the next price move (DEF-01 fix:
        # previously both values were taken at the same price, so the reward
        # contained only transaction costs).
        self.current_step += 1

        # Portfolio value at the next price, and the reward from that change
        self.portfolio_value = self.total_portfolio_value
        reward = self._compute_reward(tx_cost)

        # Termination conditions
        terminated = False
        truncated = False

        # 1. Natural end of episode window
        if self.current_step >= self.episode_length:
            truncated = True

        # 2. Stop loss
        portfolio_drop = (self.initial_cash - self.portfolio_value) / self.initial_cash
        if portfolio_drop >= self.stop_loss_pct:
            terminated = True
            reward -= 1.0  # Penalty for triggering stop loss

        # 3. Out of data
        if self.episode_start + self.current_step >= len(self.df) - 1:
            truncated = True

        obs = self._get_obs() if not (terminated or truncated) else self._get_final_obs()

        info = {
            "step": self.current_step,
            "portfolio_value": self.portfolio_value,
            "cash": self.cash,
            "position_mwh": self.position_mwh,
            "current_price": self.current_price,
            "unrealised_pnl": self.unrealised_pnl,
            "total_trades": self.total_trades,
            "total_return_pct": (self.portfolio_value / self.initial_cash - 1.0) * 100,
        }
        return obs, reward, terminated, truncated, info

    def _get_final_obs(self) -> np.ndarray:
        """Return last valid observation (clamp to dataset end)."""
        idx = min(self.episode_start + self.current_step, len(self.df) - 1)
        feat_vec = self.df.loc[idx, self.feature_cols].values.astype(np.float32)
        feat_vec = np.nan_to_num(feat_vec, nan=0.0)
        if self.normalise:
            feat_vec = self.norm.normalise(feat_vec)
        if self.include_portfolio:
            portfolio_vec = np.array([
                self.cash / (self.initial_cash + 1e-8),
                self.position_mwh / (self.max_position + 1e-8),
                self.unrealised_pnl / (self.initial_cash + 1e-8),
            ], dtype=np.float32)
            return np.concatenate([feat_vec, portfolio_vec]).astype(np.float32)
        return feat_vec.astype(np.float32)

    def render(self, mode: str = "human"):
        print(
            f"Step {self.current_step:4d} | "
            f"Price: ${self.current_price:7.2f} | "
            f"Pos: {self.position_mwh:6.1f} MWh | "
            f"Cash: ${self.cash:10,.0f} | "
            f"Portfolio: ${self.total_portfolio_value:10,.0f} | "
            f"Return: {(self.total_portfolio_value / self.initial_cash - 1) * 100:+.2f}%"
        )

    def close(self):
        pass

    # ─────────────────────────────────────────────────────────────────────
    # Episode summary helpers (called after episode ends)
    # ─────────────────────────────────────────────────────────────────────

    def get_episode_summary(self) -> dict:
        returns = [t["portfolio_value"] for t in self.trade_log]
        return {
            "final_portfolio_value": self.total_portfolio_value,
            "total_return_pct": (self.total_portfolio_value / self.initial_cash - 1) * 100,
            "total_trades": self.total_trades,
            "n_buys": sum(1 for t in self.trade_log if t["action"] == "BUY"),
            "n_sells": sum(1 for t in self.trade_log if t["action"] == "SELL"),
            "episode_length": self.current_step,
        }
