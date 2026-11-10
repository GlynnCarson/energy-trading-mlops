"""
tests/test_drl.py
Unit tests for the DRL environment, training loop, and backtest.
"""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

from src.ingestion.ingest import SyntheticEnergyGenerator
from src.transformation.transform import clean_raw, engineer_features, temporal_split
from src.ml.drl.env import EnergyTradingEnv, RunningNorm
from src.ml.drl.train_drl import load_drl_config, build_agent, evaluate_agent
from src.ml.drl.backtest import (
    run_backtest,
    run_buy_and_hold,
    run_random_agent,
    _compute_metrics,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def featured_df():
    gen = SyntheticEnergyGenerator(seed=42)
    df = gen.generate(days=90)
    df = clean_raw(df)
    return engineer_features(df)


@pytest.fixture(scope="module")
def splits(featured_df):
    return temporal_split(featured_df)


@pytest.fixture(scope="module")
def drl_cfg():
    cfg = load_drl_config()
    cfg["training"]["total_timesteps"] = 1_000   # tiny for tests
    cfg["training"]["n_envs"] = 1
    cfg["training"]["eval_freq"] = 500
    cfg["training"]["save_freq"] = 500
    cfg["environment"]["episode_length"] = 100
    cfg["mlflow"]["tracking_uri"] = "./mlruns"
    return cfg


@pytest.fixture(scope="module")
def cont_env(featured_df, drl_cfg):
    cfg = {**drl_cfg, "action": {"type": "continuous", "discrete_trade_size_mwh": 10.0}}
    return EnergyTradingEnv(featured_df, cfg, mode="train")


@pytest.fixture(scope="module")
def disc_env(featured_df, drl_cfg):
    cfg = {**drl_cfg, "action": {"type": "discrete", "discrete_trade_size_mwh": 10.0}}
    return EnergyTradingEnv(featured_df, cfg, mode="train")


# ─────────────────────────────────────────────────────────────────────────────
# RunningNorm tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRunningNorm:
    def test_init(self):
        n = RunningNorm((10,))
        assert n.mean.shape == (10,)
        assert n.var.shape == (10,)

    def test_update_shifts_mean(self):
        n = RunningNorm((1,))
        for _ in range(1000):
            n.update(np.array([5.0]))
        assert abs(n.mean[0] - 5.0) < 0.1

    def test_normalise_clips(self):
        n = RunningNorm((3,), clip=5.0)
        n.update(np.array([0.0, 0.0, 0.0]))
        out = n.normalise(np.array([100.0, -100.0, 0.0]))
        assert out.max() <= 5.0
        assert out.min() >= -5.0


# ─────────────────────────────────────────────────────────────────────────────
# Environment tests
# ─────────────────────────────────────────────────────────────────────────────

class TestEnergyTradingEnvContinuous:

    def test_obs_space_shape(self, cont_env):
        obs, _ = cont_env.reset()
        assert obs.shape == cont_env.observation_space.shape

    def test_action_space_is_box(self, cont_env):
        assert isinstance(cont_env.action_space, gym.spaces.Box)
        assert cont_env.action_space.shape == (1,)

    def test_step_returns_correct_tuple(self, cont_env):
        cont_env.reset()
        obs, reward, terminated, truncated, info = cont_env.step(np.array([0.5]))
        assert obs.shape == cont_env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_info_keys(self, cont_env):
        cont_env.reset()
        _, _, _, _, info = cont_env.step(np.array([0.0]))
        for key in ["portfolio_value", "cash", "position_mwh", "current_price", "total_trades"]:
            assert key in info, f"Missing info key: {key}"

    def test_portfolio_starts_at_initial_cash(self, cont_env):
        cont_env.reset()
        assert abs(cont_env.total_portfolio_value - cont_env.initial_cash) < 1.0

    def test_buy_increases_position(self, cont_env):
        cont_env.reset()
        cont_env.step(np.array([1.0]))  # Full buy
        assert cont_env.position_mwh > 0

    def test_sell_decreases_position(self, cont_env):
        cont_env.reset()
        cont_env.step(np.array([1.0]))  # Buy
        pos_after_buy = cont_env.position_mwh
        cont_env.step(np.array([-1.0]))  # Sell
        assert cont_env.position_mwh < pos_after_buy

    def test_position_never_negative(self, cont_env):
        cont_env.reset()
        for _ in range(20):
            cont_env.step(np.array([-1.0]))  # Keep selling
        assert cont_env.position_mwh >= 0.0

    def test_cash_never_negative(self, cont_env):
        cont_env.reset()
        for _ in range(50):
            cont_env.step(np.array([1.0]))  # Keep buying
        assert cont_env.cash >= 0.0

    def test_episode_terminates(self, cont_env):
        cont_env.reset()
        done = False
        steps = 0
        while not done and steps < 10_000:
            _, _, terminated, truncated, _ = cont_env.step(np.array([0.0]))
            done = terminated or truncated
            steps += 1
        assert done, "Episode should terminate"

    def test_gymnasium_check_env(self, cont_env):
        """Gymnasium's official env checker must pass."""
        try:
            check_env(cont_env, warn=True, skip_render_check=True)
        except Exception as e:
            pytest.fail(f"check_env failed: {e}")


class TestEnergyTradingEnvDiscrete:

    def test_action_space_is_discrete(self, disc_env):
        assert isinstance(disc_env.action_space, gym.spaces.Discrete)
        assert disc_env.action_space.n == 3

    def test_hold_action(self, disc_env):
        disc_env.reset()
        pos_before = disc_env.position_mwh
        disc_env.step(0)  # Hold
        assert disc_env.position_mwh == pos_before

    def test_buy_action(self, disc_env):
        disc_env.reset()
        disc_env.step(1)  # Buy
        assert disc_env.position_mwh > 0

    def test_sell_action_after_buy(self, disc_env):
        disc_env.reset()
        disc_env.step(1)  # Buy
        pos = disc_env.position_mwh
        disc_env.step(2)  # Sell
        assert disc_env.position_mwh < pos


# ─────────────────────────────────────────────────────────────────────────────
# Training smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDRLTraining:

    def test_build_ppo_agent(self, cont_env, drl_cfg):
        from stable_baselines3 import PPO
        agent = build_agent("PPO", cont_env, drl_cfg, seed=0)
        assert isinstance(agent, PPO)

    def test_build_a2c_agent(self, cont_env, drl_cfg):
        from stable_baselines3 import A2C
        agent = build_agent("A2C", cont_env, drl_cfg, seed=0)
        assert isinstance(agent, A2C)

    def test_invalid_algo_raises(self, cont_env, drl_cfg):
        with pytest.raises(ValueError):
            build_agent("DREAMER_V3", cont_env, drl_cfg)

    def test_agent_can_learn(self, featured_df, splits, drl_cfg):
        import mlflow
        mlflow.set_tracking_uri("./mlruns")
        train, val, _ = splits
        cfg = {**drl_cfg}
        cfg["training"]["total_timesteps"] = 500
        cfg["training"]["eval_freq"] = 200
        cfg["training"]["save_freq"] = 300
        cfg["environment"]["episode_length"] = 50

        from src.ml.drl.train_drl import run_drl_training
        run_id = run_drl_training(train, val, cfg)
        assert run_id is not None

    def test_evaluate_agent_returns_dict(self, featured_df, splits, drl_cfg):
        train, val, test = splits
        cfg = {**drl_cfg}
        cfg["environment"]["episode_length"] = 50
        env = EnergyTradingEnv(test, cfg, mode="test")
        agent = build_agent("PPO", env, cfg)
        metrics = evaluate_agent(agent, env, n_episodes=3)
        for key in ["mean_return_pct", "mean_sharpe", "win_rate"]:
            assert key in metrics


# ─────────────────────────────────────────────────────────────────────────────
# Backtest tests
# ─────────────────────────────────────────────────────────────────────────────

class TestBacktest:

    def test_compute_metrics_keys(self):
        pv = np.linspace(100_000, 110_000, 200)
        prices = np.random.uniform(40, 60, 200)
        metrics = _compute_metrics(pv, prices, 100_000.0)
        for key in ["total_return_pct", "sharpe", "max_drawdown_pct", "calmar_ratio", "win_rate_pct", "var_95_pct", "profit_factor"]:
            assert key in metrics, f"Missing metric: {key}"

    def test_positive_return_gives_positive_total(self):
        pv = np.linspace(100_000, 120_000, 200)
        metrics = _compute_metrics(pv, np.ones(200) * 50, 100_000)
        assert metrics["total_return_pct"] > 0

    def test_buy_and_hold_runs(self, featured_df, splits, drl_cfg):
        _, _, test = splits
        cfg = {**drl_cfg}
        cfg["environment"]["episode_length"] = len(test) - 1
        result = run_buy_and_hold(test, cfg)
        assert "metrics" in result
        assert "portfolio_values" in result

    def test_random_agent_runs(self, featured_df, splits, drl_cfg):
        _, _, test = splits
        cfg = {**drl_cfg}
        cfg["environment"]["episode_length"] = len(test) - 1
        result = run_random_agent(test, cfg, seed=42)
        assert "metrics" in result
