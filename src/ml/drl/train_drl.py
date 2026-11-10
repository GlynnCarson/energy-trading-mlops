"""
src/ml/drl/train_drl.py
─────────────────────────────────────────────────────────────────────────────
Trains a DRL agent (PPO / SAC / TD3 / A2C) on the energy trading environment
using Stable-Baselines3.  All runs are tracked in MLflow.

Pipeline flow:
    Data (transform output)
        ↓
    EnergyTradingEnv  ×  n_envs  (vectorised)
        ↓
    SB3 algorithm (PPO default)
        ↓
    EvalCallback  →  saves best model checkpoint
        ↓
    MLflow  →  logs hyperparams, metrics, model artifact
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import json
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from pathlib import Path
from loguru import logger
from datetime import datetime

import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from stable_baselines3 import PPO, SAC, TD3, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    CallbackList,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed

from src.ml.drl.env import EnergyTradingEnv
from src.utils.helpers import load_pipeline_config

# ── Config ───────────────────────────────────────────────────────────────────

def load_drl_config() -> dict:
    import yaml
    from pathlib import Path
    cfg_path = Path(__file__).resolve().parents[3] / "config" / "drl.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


SB3_ALGOS = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "A2C": A2C}


# ─────────────────────────────────────────────────────────────────────────────
# Custom MLflow callback
# ─────────────────────────────────────────────────────────────────────────────

class MLflowCallback(BaseCallback):
    """
    Logs SB3 training metrics to MLflow at each rollout.
    Captures: mean_reward, policy_gradient_loss, value_loss, entropy_loss,
              explained_variance, learning_rate, total_timesteps.
    """

    def __init__(self, log_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            metrics = {}

            # Collect from SB3 logger
            for key, val in self.logger.name_to_value.items():
                if isinstance(val, (int, float)):
                    clean_key = key.replace("/", "_")
                    metrics[clean_key] = float(val)

            if metrics:
                mlflow.log_metrics(metrics, step=self.num_timesteps)

        return True


class PortfolioMetricsCallback(BaseCallback):
    """
    Tracks portfolio performance (return, Sharpe, max drawdown)
    across evaluation episodes and logs to MLflow.
    """

    def __init__(self, eval_env: EnergyTradingEnv, eval_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.episode_returns: list[float] = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            returns = self._run_eval_episode()
            if returns:
                sharpe = self._sharpe(returns)
                max_dd = self._max_drawdown(returns)
                final_ret = returns[-1]
                logger.info(
                    f"[PORTFOLIO EVAL @ {self.num_timesteps}] "
                    f"Return={final_ret:+.2f}% | Sharpe={sharpe:.3f} | MaxDD={max_dd:.2f}%"
                )
                mlflow.log_metrics({
                    "portfolio_return_pct": final_ret,
                    "portfolio_sharpe": sharpe,
                    "portfolio_max_drawdown_pct": max_dd,
                }, step=self.num_timesteps)
        return True

    def _run_eval_episode(self) -> list[float]:
        """Run one greedy episode and return portfolio return history."""
        obs, _ = self.eval_env.reset()
        returns = []
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, info = self.eval_env.step(action)
            returns.append(info.get("total_return_pct", 0.0))
            done = terminated or truncated
        return returns

    @staticmethod
    def _sharpe(returns: list[float], risk_free: float = 0.0) -> float:
        r = np.diff(returns)
        if r.std() < 1e-8:
            return 0.0
        return float((r.mean() - risk_free) / r.std() * np.sqrt(24 * 365))

    @staticmethod
    def _max_drawdown(returns: list[float]) -> float:
        arr = np.array(returns)
        peak = np.maximum.accumulate(arr)
        dd = (arr - peak) / (np.abs(peak) + 1e-8) * 100
        return float(dd.min())


# ─────────────────────────────────────────────────────────────────────────────
# Environment factory
# ─────────────────────────────────────────────────────────────────────────────

def make_env_fn(df: pd.DataFrame, drl_cfg: dict, mode: str = "train", seed: int = 0):
    """Return a Gymnasium env factory function for vectorised envs."""
    def _init():
        env = EnergyTradingEnv(df, drl_cfg, mode=mode)
        env = Monitor(env)
        return env
    set_random_seed(seed)
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# Policy network architecture builder
# ─────────────────────────────────────────────────────────────────────────────

def build_policy_kwargs(algo_cfg: dict) -> dict:
    """Build SB3 policy_kwargs from config net_arch."""
    net_arch = algo_cfg.get("net_arch", [256, 256, 128])
    return {"net_arch": net_arch}


# ─────────────────────────────────────────────────────────────────────────────
# Algorithm factory
# ─────────────────────────────────────────────────────────────────────────────

def build_agent(
    algorithm: str,
    env,
    drl_cfg: dict,
    seed: int = 42,
):
    """
    Instantiate the SB3 agent with hyperparameters from config.
    Supports: PPO, SAC, TD3, A2C.
    """
    AlgoClass = SB3_ALGOS.get(algorithm)
    if AlgoClass is None:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Choose from {list(SB3_ALGOS.keys())}")

    algo_cfg = drl_cfg.get(algorithm, {})
    policy_kwargs = build_policy_kwargs(algo_cfg)
    device = drl_cfg["training"].get("device", "cpu")

    common_kwargs = dict(
        policy=algo_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(algo_cfg.get("learning_rate", 3e-4)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        policy_kwargs=policy_kwargs,
        device=device,
        seed=seed,
        verbose=1,
    )

    if algorithm == "PPO":
        agent = AlgoClass(
            **common_kwargs,
            n_steps=int(algo_cfg.get("n_steps", 2048)),
            batch_size=int(algo_cfg.get("batch_size", 64)),
            n_epochs=int(algo_cfg.get("n_epochs", 10)),
            gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
            clip_range=float(algo_cfg.get("clip_range", 0.2)),
            normalize_advantage=bool(algo_cfg.get("normalize_advantage", True)),
            ent_coef=float(algo_cfg.get("ent_coef", 0.01)),
            vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(algo_cfg.get("max_grad_norm", 0.5)),
        )

    elif algorithm in ("SAC", "TD3"):
        agent = AlgoClass(
            **common_kwargs,
            buffer_size=int(algo_cfg.get("buffer_size", 1_000_000)),
            learning_starts=int(algo_cfg.get("learning_starts", 10_000)),
            batch_size=int(algo_cfg.get("batch_size", 256)),
            tau=float(algo_cfg.get("tau", 0.005)),
            train_freq=int(algo_cfg.get("train_freq", 1)),
            gradient_steps=int(algo_cfg.get("gradient_steps", 1)),
        )

    elif algorithm == "A2C":
        agent = AlgoClass(
            **common_kwargs,
            n_steps=int(algo_cfg.get("n_steps", 5)),
            gae_lambda=float(algo_cfg.get("gae_lambda", 0.95)),
            ent_coef=float(algo_cfg.get("ent_coef", 0.01)),
            vf_coef=float(algo_cfg.get("vf_coef", 0.5)),
            max_grad_norm=float(algo_cfg.get("max_grad_norm", 0.5)),
        )

    logger.info(f"[AGENT] {algorithm} agent built | obs_dim={env.observation_space.shape} | action={env.action_space}")
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────────────────────

def run_drl_training(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    drl_cfg: dict = None,
) -> str:
    """
    End-to-end DRL training run.
    Returns the MLflow run ID.

    Parameters
    ----------
    train_df : pd.DataFrame
        Feature-engineered training split from the transform pipeline.
    val_df : pd.DataFrame
        Validation split for EvalCallback.
    drl_cfg : dict, optional
        Override config dict. Loads from config/drl.yaml if None.
    """
    if drl_cfg is None:
        drl_cfg = load_drl_config()

    tr_cfg = drl_cfg["training"]
    algorithm = tr_cfg["algorithm"]
    total_timesteps = int(tr_cfg["total_timesteps"])
    n_envs = int(tr_cfg.get("n_envs", 1))
    seed = int(tr_cfg.get("seed", 42))
    eval_freq = int(tr_cfg.get("eval_freq", 10_000))
    save_freq = int(tr_cfg.get("save_freq", 50_000))

    checkpoint_dir = Path(tr_cfg.get("checkpoint_dir", "./checkpoints/drl"))
    best_model_path = str(tr_cfg.get("best_model_path", checkpoint_dir / "best_model"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"[DRL TRAIN] Algorithm={algorithm} | timesteps={total_timesteps:,} | "
        f"n_envs={n_envs} | train_rows={len(train_df)} | val_rows={len(val_df)}"
    )

    # ── Validate the environment first ────────────────────────────────────
    logger.info("[DRL] Validating environment with check_env()...")
    _check_env = EnergyTradingEnv(train_df, drl_cfg, mode="train")
    try:
        check_env(_check_env, warn=True, skip_render_check=True)
        logger.success("[DRL] Environment validation passed.")
    except Exception as e:
        logger.warning(f"[DRL] check_env warning: {e}")
    finally:
        _check_env.close()

    # ── Vectorised training environments ─────────────────────────────────
    # PPO/A2C support multiple parallel envs; SAC/TD3 use single env
    actual_n_envs = n_envs if algorithm in ("PPO", "A2C") else 1
    train_env = make_vec_env(
        make_env_fn(train_df, drl_cfg, mode="train", seed=seed),
        n_envs=actual_n_envs,
        seed=seed,
        vec_env_cls=DummyVecEnv,
    )

    # ── Single eval environment ───────────────────────────────────────────
    eval_env_raw = EnergyTradingEnv(val_df, drl_cfg, mode="test")
    eval_env = Monitor(eval_env_raw)

    # ── Build agent ───────────────────────────────────────────────────────
    agent = build_agent(algorithm, train_env, drl_cfg, seed=seed)

    # ── Callbacks ─────────────────────────────────────────────────────────
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=str(checkpoint_dir / "eval_logs"),
        eval_freq=max(eval_freq // actual_n_envs, 1),
        n_eval_episodes=int(tr_cfg.get("n_eval_episodes", 5)),
        deterministic=True,
        render=False,
        verbose=1,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(save_freq // actual_n_envs, 1),
        save_path=str(checkpoint_dir / "checkpoints"),
        name_prefix=f"{algorithm.lower()}_energy",
        verbose=1,
    )

    mlflow_callback = MLflowCallback(log_freq=1000)

    portfolio_callback = PortfolioMetricsCallback(
        eval_env=EnergyTradingEnv(val_df, drl_cfg, mode="test"),
        eval_freq=eval_freq // actual_n_envs,
    )

    callbacks = CallbackList([
        eval_callback,
        checkpoint_callback,
        mlflow_callback,
        portfolio_callback,
    ])

    # ── MLflow tracking ───────────────────────────────────────────────────
    mlflow_cfg = drl_cfg.get("mlflow", {})
    mlflow.set_tracking_uri(mlflow_cfg.get("tracking_uri", "http://mlflow-server:5000"))
    mlflow.set_experiment(mlflow_cfg.get("experiment_name", "energy-drl-trading"))

    with mlflow.start_run(run_name=f"{algorithm}_drl_{datetime.utcnow().strftime('%Y%m%d_%H%M')}") as run:
        run_id = run.info.run_id
        logger.info(f"[MLFLOW] DRL Run ID: {run_id}")

        # Log all config params
        mlflow.log_params({
            "algorithm": algorithm,
            "total_timesteps": total_timesteps,
            "n_envs": actual_n_envs,
            "action_type": drl_cfg["action"]["type"],
            "obs_dim": agent.observation_space.shape[0],
            "initial_cash": drl_cfg["environment"]["initial_cash"],
            "episode_length": drl_cfg["environment"]["episode_length"],
            "train_rows": len(train_df),
            "val_rows": len(val_df),
            "seed": seed,
            **{f"algo_{k}": v
               for k, v in drl_cfg.get(algorithm, {}).items()
               if not isinstance(v, (list, dict))},
        })

        # Log config files as artifacts
        import yaml, tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(drl_cfg, f)
            mlflow.log_artifact(f.name, "config")

        # ── TRAIN ──────────────────────────────────────────────────────────
        logger.info(f"[DRL TRAIN] Starting training for {total_timesteps:,} timesteps...")
        agent.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            log_interval=int(tr_cfg.get("log_interval", 1)),
            reset_num_timesteps=True,
        )
        logger.success("[DRL TRAIN] Training complete.")

        # ── Save final model ───────────────────────────────────────────────
        final_model_path = str(checkpoint_dir / f"{algorithm.lower()}_final")
        agent.save(final_model_path)
        mlflow.log_artifact(f"{final_model_path}.zip", "model")
        logger.info(f"[DRL] Final model saved → {final_model_path}.zip")

        # ── Log best model ─────────────────────────────────────────────────
        best_path = f"{best_model_path}/best_model.zip"
        if os.path.exists(best_path):
            mlflow.log_artifact(best_path, "best_model")

        # ── Final evaluation ───────────────────────────────────────────────
        final_eval_env = EnergyTradingEnv(val_df, drl_cfg, mode="test")
        eval_metrics = evaluate_agent(agent, final_eval_env, n_episodes=10)
        mlflow.log_metrics({f"final_{k}": v for k, v in eval_metrics.items()})
        logger.info(f"[DRL] Final eval: {eval_metrics}")

        mlflow.set_tag("model_type", "drl")
        mlflow.set_tag("algorithm", algorithm)
        mlflow.set_tag("action_type", drl_cfg["action"]["type"])

    train_env.close()
    eval_env.close()

    return run_id


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helper
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_agent(
    agent,
    env: EnergyTradingEnv,
    n_episodes: int = 10,
    deterministic: bool = True,
) -> dict:
    """
    Run n_episodes greedy episodes and return aggregate metrics.
    """
    all_returns = []
    all_trades = []
    all_sharpes = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        done = False
        portfolio_history = [env.initial_cash]

        while not done:
            action, _ = agent.predict(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            portfolio_history.append(info["portfolio_value"])
            done = terminated or truncated

        final_return = (portfolio_history[-1] / env.initial_cash - 1.0) * 100
        all_returns.append(final_return)
        all_trades.append(info["total_trades"])

        # Sharpe for this episode
        ph = np.array(portfolio_history)
        step_returns = np.diff(ph) / (ph[:-1] + 1e-8)
        if step_returns.std() > 1e-8:
            sharpe = step_returns.mean() / step_returns.std() * np.sqrt(24 * 365)
        else:
            sharpe = 0.0
        all_sharpes.append(sharpe)

    return {
        "mean_return_pct": round(float(np.mean(all_returns)), 4),
        "std_return_pct": round(float(np.std(all_returns)), 4),
        "mean_sharpe": round(float(np.mean(all_sharpes)), 4),
        "mean_trades": round(float(np.mean(all_trades)), 2),
        "win_rate": round(float(np.mean(np.array(all_returns) > 0)), 4),
        "min_return_pct": round(float(np.min(all_returns)), 4),
        "max_return_pct": round(float(np.max(all_returns)), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from src.ingestion.ingest import SyntheticEnergyGenerator
    from src.transformation.transform import clean_raw, engineer_features, temporal_split

    print("Generating data...")
    gen = SyntheticEnergyGenerator(seed=42)
    df = gen.generate(days=365)
    df = clean_raw(df)
    df = engineer_features(df)
    train, val, test = temporal_split(df)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    drl_cfg = load_drl_config()
    # Reduce for quick local test
    drl_cfg["training"]["total_timesteps"] = 20_000
    drl_cfg["training"]["n_envs"] = 1
    drl_cfg["training"]["eval_freq"] = 5_000
    drl_cfg["training"]["save_freq"] = 10_000
    drl_cfg["mlflow"]["tracking_uri"] = "./mlruns"

    import mlflow
    mlflow.set_tracking_uri("./mlruns")

    run_id = run_drl_training(train, val, drl_cfg)
    print(f"\nTraining complete. MLflow run: {run_id}")
