"""
src/ml/drl/backtest.py
─────────────────────────────────────────────────────────────────────────────
Backtests a trained DRL agent on the held-out test set and generates a
full trading performance report with comparison against benchmarks.

Metrics computed:
    ─ Total Return %
    ─ Annualised Sharpe Ratio
    ─ Maximum Drawdown %
    ─ Calmar Ratio  (Return / Max Drawdown)
    ─ Win Rate  (% of hours with positive PnL)
    ─ Average Trade Size (MWh)
    ─ Trade Frequency (trades/day)
    ─ Profit Factor  (gross profits / gross losses)
    ─ Value at Risk (95% VaR)

Benchmarks compared:
    ─ Buy & Hold  (buy at t=0, hold until end)
    ─ Random Agent  (random actions)
─────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from loguru import logger
from stable_baselines3.common.base_class import BaseAlgorithm

from src.ml.drl.env import EnergyTradingEnv


# ── Dark theme ────────────────────────────────────────────────────────────────
DARK = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor": "#161b22",
    "axes.edgecolor": "#30363d",
    "axes.labelcolor": "#c9d1d9",
    "xtick.color": "#8b949e",
    "ytick.color": "#8b949e",
    "text.color": "#c9d1d9",
    "grid.color": "#21262d",
    "grid.linestyle": "--",
    "grid.alpha": 0.6,
    "legend.facecolor": "#21262d",
    "legend.edgecolor": "#30363d",
}

COLOURS = {
    "agent":       "#58a6ff",   # blue
    "buy_hold":    "#3fb950",   # green
    "random":      "#f78166",   # red/orange
    "price":       "#e3b341",   # yellow
    "buy_marker":  "#3fb950",
    "sell_marker": "#f85149",
    "hold_marker": "#8b949e",
    "drawdown":    "#f85149",
    "portfolio":   "#58a6ff",
}


# ─────────────────────────────────────────────────────────────────────────────
# Core runner
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    agent: BaseAlgorithm,
    test_df: pd.DataFrame,
    drl_cfg: dict,
    deterministic: bool = True,
) -> dict:
    """
    Walk the agent through the entire test set step-by-step (no random starts).
    Returns a results dict with time-series arrays and aggregate metrics.
    """
    env = EnergyTradingEnv(test_df, drl_cfg, mode="test")
    obs, _ = env.reset()

    prices = []
    portfolio_values = []
    positions = []
    actions_taken = []
    rewards = []
    cash_history = []
    done = False

    logger.info(f"[BACKTEST] Running agent on {len(test_df)} test steps...")

    while not done:
        action, _ = agent.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)

        prices.append(info["current_price"])
        portfolio_values.append(info["portfolio_value"])
        positions.append(info["position_mwh"])
        rewards.append(reward)
        cash_history.append(info["cash"])

        if drl_cfg["action"]["type"] == "continuous":
            actions_taken.append(float(action[0]))
        else:
            actions_taken.append(int(action))

        done = terminated or truncated

    env.close()

    prices = np.array(prices)
    portfolio_values = np.array(portfolio_values)
    positions = np.array(positions)
    actions_arr = np.array(actions_taken)
    initial = drl_cfg["environment"]["initial_cash"]

    bt = drl_cfg.get("backtest", {})
    ppy = bt.get("periods_per_year", 8760)
    spd = bt.get("steps_per_day", 24)
    metrics = _compute_metrics(portfolio_values, prices, initial, env,
                               periods_per_year=ppy, steps_per_day=spd)
    metrics["algorithm"] = drl_cfg["training"]["algorithm"]

    logger.success(
        f"[BACKTEST] Done. Return={metrics['total_return_pct']:+.2f}% | "
        f"Sharpe={metrics['sharpe']:.3f} | MaxDD={metrics['max_drawdown_pct']:.2f}%"
    )

    return {
        "metrics": metrics,
        "prices": prices,
        "portfolio_values": portfolio_values,
        "positions": positions,
        "actions": actions_arr,
        "rewards": rewards,
        "cash": cash_history,
        "trade_log": env.trade_log,
        "initial_cash": initial,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────────

def run_buy_and_hold(test_df: pd.DataFrame, drl_cfg: dict) -> dict:
    """Buy maximum position at t=0, hold until end."""
    env = EnergyTradingEnv(test_df, drl_cfg, mode="test")
    obs, _ = env.reset()

    portfolio_values = []
    prices_out = []
    done = False
    first_step = True

    while not done:
        if first_step:
            # Buy: full long (+1 for continuous or 1 for discrete)
            if drl_cfg["action"]["type"] == "continuous":
                action = np.array([1.0], dtype=np.float32)
            else:
                action = 1  # Buy
            first_step = False
        else:
            # Hold
            if drl_cfg["action"]["type"] == "continuous":
                action = np.array([1.0], dtype=np.float32)  # Keep full position
            else:
                action = 0  # Hold

        obs, _, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        prices_out.append(info["current_price"])
        done = terminated or truncated

    env.close()

    pv = np.array(portfolio_values)
    bt = drl_cfg.get("backtest", {})
    metrics = _compute_metrics(pv, np.array(prices_out), drl_cfg["environment"]["initial_cash"],
                               periods_per_year=bt.get("periods_per_year", 8760),
                               steps_per_day=bt.get("steps_per_day", 24))
    metrics["algorithm"] = "Buy & Hold"
    return {"metrics": metrics, "portfolio_values": pv}


def run_random_agent(test_df: pd.DataFrame, drl_cfg: dict, seed: int = 0) -> dict:
    """Uniformly random actions as a baseline."""
    env = EnergyTradingEnv(test_df, drl_cfg, mode="test")
    env.reset(seed=seed)

    portfolio_values = []
    done = False
    prices_out = []

    while not done:
        action = env.action_space.sample()
        _, _, terminated, truncated, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])
        prices_out.append(info["current_price"])
        done = terminated or truncated

    env.close()

    pv = np.array(portfolio_values)
    bt = drl_cfg.get("backtest", {})
    metrics = _compute_metrics(pv, np.array(prices_out), drl_cfg["environment"]["initial_cash"],
                               periods_per_year=bt.get("periods_per_year", 8760),
                               steps_per_day=bt.get("steps_per_day", 24))
    metrics["algorithm"] = "Random Agent"
    return {"metrics": metrics, "portfolio_values": pv}


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def _compute_metrics(
    portfolio_values: np.ndarray,
    prices: np.ndarray,
    initial_cash: float,
    env: EnergyTradingEnv = None,
    periods_per_year: float = 8760,
    steps_per_day: float = 24,
) -> dict:
    """Compute full suite of financial performance metrics.

    periods_per_year / steps_per_day describe the data frequency so that
    annualised metrics are correct: hourly -> (8760, 24), daily -> (252, 1).
    """
    pv = portfolio_values
    n = len(pv)

    # Returns
    total_return = (pv[-1] / initial_cash - 1.0) * 100
    step_returns = np.diff(pv) / (pv[:-1] + 1e-8)

    # Annualised Sharpe (scaled by the data's periods-per-year)
    if step_returns.std() > 1e-8:
        sharpe = float(step_returns.mean() / step_returns.std() * np.sqrt(periods_per_year))
    else:
        sharpe = 0.0

    # Maximum Drawdown
    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / (running_max + 1e-8) * 100
    max_dd = float(drawdowns.min())

    # Calmar Ratio
    calmar = total_return / (abs(max_dd) + 1e-8)

    # Win rate (% of steps with positive return)
    win_rate = float(np.mean(step_returns > 0)) * 100

    # VaR 95%
    var_95 = float(np.percentile(step_returns, 5)) * 100

    # Profit factor
    gains = step_returns[step_returns > 0].sum()
    losses = abs(step_returns[step_returns < 0].sum())
    profit_factor = float(gains / (losses + 1e-8))

    # Annualised return
    years = n / periods_per_year
    annualised_return = ((1 + total_return / 100) ** (1 / max(years, 0.001)) - 1) * 100

    metrics = {
        "total_return_pct": round(total_return, 4),
        "annualised_return_pct": round(annualised_return, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd, 4),
        "calmar_ratio": round(calmar, 4),
        "win_rate_pct": round(win_rate, 4),
        "var_95_pct": round(var_95, 4),
        "profit_factor": round(profit_factor, 4),
        "final_portfolio_value": round(float(pv[-1]), 2),
        "n_steps": n,
    }

    if env is not None:
        metrics["total_trades"] = env.total_trades
        metrics["trade_freq_per_day"] = round(env.total_trades / max(n / steps_per_day, 1), 4)

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_backtest_report(
    agent_results: dict,
    buy_hold_results: dict,
    random_results: dict,
    output_dir: str = "./reports/backtest",
) -> str:
    """
    Generate a comprehensive 6-panel backtest dashboard.
    Returns path to the saved PNG.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(output_dir) / "backtest_report.png")

    pv_agent = agent_results["portfolio_values"]
    pv_bh = buy_hold_results["portfolio_values"]
    pv_rand = random_results["portfolio_values"]
    prices = agent_results["prices"]
    positions = agent_results["positions"]
    actions = agent_results["actions"]
    initial = agent_results["initial_cash"]
    algo = agent_results["metrics"]["algorithm"]

    # Normalise to % return for comparison
    n = min(len(pv_agent), len(pv_bh), len(pv_rand))
    ret_agent = (pv_agent[:n] / initial - 1) * 100
    ret_bh = (pv_bh[:n] / initial - 1) * 100
    ret_rand = (pv_rand[:n] / initial - 1) * 100
    steps = np.arange(n)

    with plt.rc_context(DARK):
        fig = plt.figure(figsize=(18, 14), constrained_layout=True)
        fig.suptitle(
            f"⚡  Energy Trading DRL Backtest  —  {algo}",
            fontsize=17, fontweight="bold", color="#e6edf3", y=1.01,
        )

        gs = gridspec.GridSpec(3, 3, figure=fig)

        # ── Panel 1: Portfolio Returns ─────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(steps, ret_agent, label=f"{algo}", color=COLOURS["agent"], linewidth=1.8)
        ax1.plot(steps, ret_bh[:n], label="Buy & Hold", color=COLOURS["buy_hold"], linewidth=1.2, alpha=0.8)
        ax1.plot(steps, ret_rand[:n], label="Random Agent", color=COLOURS["random"], linewidth=1.0, alpha=0.6, linestyle="--")
        ax1.axhline(0, color="#c9d1d9", linewidth=0.5, linestyle=":")
        ax1.fill_between(steps, ret_agent, 0, alpha=0.12, color=COLOURS["agent"])
        ax1.set_title("Portfolio Return (%)", fontweight="bold")
        ax1.set_ylabel("Return (%)")
        ax1.set_xlabel("Timestep (hours)")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        # ── Panel 2: Metrics summary card ─────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis("off")
        m = agent_results["metrics"]
        bh_m = buy_hold_results["metrics"]
        metrics_text = (
            f"  {'Metric':<22} {'Agent':>10}  {'B&H':>10}\n"
            f"  {'─'*46}\n"
            f"  {'Total Return %':<22} {m['total_return_pct']:>+10.2f}  {bh_m['total_return_pct']:>+10.2f}\n"
            f"  {'Sharpe Ratio':<22} {m['sharpe']:>10.3f}  {bh_m['sharpe']:>10.3f}\n"
            f"  {'Max Drawdown %':<22} {m['max_drawdown_pct']:>10.2f}  {bh_m['max_drawdown_pct']:>10.2f}\n"
            f"  {'Calmar Ratio':<22} {m['calmar_ratio']:>10.3f}  {bh_m['calmar_ratio']:>10.3f}\n"
            f"  {'Win Rate %':<22} {m['win_rate_pct']:>10.2f}  {bh_m['win_rate_pct']:>10.2f}\n"
            f"  {'VaR 95% %':<22} {m['var_95_pct']:>10.3f}  {bh_m['var_95_pct']:>10.3f}\n"
            f"  {'Profit Factor':<22} {m['profit_factor']:>10.3f}  {bh_m['profit_factor']:>10.3f}\n"
        )
        if "total_trades" in m:
            metrics_text += f"  {'Total Trades':<22} {m['total_trades']:>10}\n"
        ax2.text(
            0.02, 0.95, metrics_text,
            transform=ax2.transAxes,
            fontfamily="monospace", fontsize=9,
            verticalalignment="top", color="#e6edf3",
            bbox=dict(facecolor="#21262d", edgecolor="#30363d", boxstyle="round,pad=0.5"),
        )
        ax2.set_title("Performance Summary", fontweight="bold")

        # ── Panel 3: Energy Price ──────────────────────────────────────────
        ax3 = fig.add_subplot(gs[1, :2])
        ax3.plot(steps, prices[:n], color=COLOURS["price"], linewidth=1.0, alpha=0.9)
        ax3.set_title("Energy Spot Price (USD/MWh)", fontweight="bold")
        ax3.set_ylabel("Price (USD/MWh)")
        ax3.grid(True)

        # Overlay buy/sell markers from trade log
        if agent_results.get("trade_log"):
            buy_steps = [t["step"] for t in agent_results["trade_log"] if t["action"] == "BUY" and t["step"] < n]
            sell_steps = [t["step"] for t in agent_results["trade_log"] if t["action"] == "SELL" and t["step"] < n]
            if buy_steps:
                ax3.scatter(buy_steps, prices[buy_steps], marker="^", color=COLOURS["buy_marker"], s=40, zorder=5, label="Buy", alpha=0.8)
            if sell_steps:
                ax3.scatter(sell_steps, prices[sell_steps], marker="v", color=COLOURS["sell_marker"], s=40, zorder=5, label="Sell", alpha=0.8)
            ax3.legend(loc="upper right", fontsize=8)

        # ── Panel 4: Position held ─────────────────────────────────────────
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.fill_between(steps, positions[:n], alpha=0.5, color=COLOURS["agent"], step="pre")
        ax4.plot(steps, positions[:n], color=COLOURS["agent"], linewidth=0.8, alpha=0.7)
        ax4.set_title("Position Held (MWh)", fontweight="bold")
        ax4.set_ylabel("MWh")
        ax4.set_xlabel("Timestep")
        ax4.grid(True)

        # ── Panel 5: Drawdown ──────────────────────────────────────────────
        ax5 = fig.add_subplot(gs[2, :2])
        running_max = np.maximum.accumulate(pv_agent[:n])
        drawdown = (pv_agent[:n] - running_max) / (running_max + 1e-8) * 100
        ax5.fill_between(steps, drawdown, 0, alpha=0.5, color=COLOURS["drawdown"])
        ax5.plot(steps, drawdown, color=COLOURS["drawdown"], linewidth=0.8)
        ax5.set_title("Drawdown (%)", fontweight="bold")
        ax5.set_ylabel("Drawdown (%)")
        ax5.set_xlabel("Timestep (hours)")
        ax5.grid(True)

        # ── Panel 6: Action distribution ──────────────────────────────────
        ax6 = fig.add_subplot(gs[2, 2])
        if drl_cfg_ref := agent_results.get("_drl_cfg"):
            pass

        # For continuous actions, plot histogram of position targets
        if actions.dtype == np.float64 or (actions.ndim > 0 and actions.dtype.kind == 'f'):
            ax6.hist(actions[:n], bins=50, color=COLOURS["agent"], alpha=0.8, edgecolor="#30363d")
            ax6.set_xlabel("Action Value")
            ax6.set_title("Action Distribution", fontweight="bold")
        else:
            # Discrete
            labels = ["Hold", "Buy", "Sell"]
            counts = [np.sum(actions[:n] == i) for i in range(3)]
            bars = ax6.bar(labels, counts, color=[COLOURS["hold_marker"], COLOURS["buy_marker"], COLOURS["sell_marker"]], alpha=0.85)
            for bar, count in zip(bars, counts):
                ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         str(count), ha="center", va="bottom", fontsize=10)
            ax6.set_title("Action Counts", fontweight="bold")

        ax6.set_ylabel("Count")
        ax6.grid(True, axis="y")

        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
        plt.close()

    logger.success(f"[BACKTEST] Dashboard saved → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Full backtest pipeline (called by Airflow or standalone)
# ─────────────────────────────────────────────────────────────────────────────

def run_full_backtest(
    agent: BaseAlgorithm,
    test_df: pd.DataFrame,
    drl_cfg: dict,
    output_dir: str = "./reports/backtest",
) -> dict:
    """
    Run agent + benchmarks, generate report, return all metrics.
    """
    logger.info("[BACKTEST] Running full backtest suite...")

    agent_results = run_backtest(agent, test_df, drl_cfg)
    bh_results = run_buy_and_hold(test_df, drl_cfg)
    rand_results = run_random_agent(test_df, drl_cfg)

    plot_path = plot_backtest_report(agent_results, bh_results, rand_results, output_dir)

    summary = {
        "agent": agent_results["metrics"],
        "buy_and_hold": bh_results["metrics"],
        "random": rand_results["metrics"],
        "report_path": plot_path,
    }

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"  BACKTEST RESULTS - {agent_results['metrics']['algorithm']}")
    print("=" * 60)
    for metric, label in [
        ("total_return_pct", "Total Return (%)"),
        ("sharpe", "Sharpe Ratio"),
        ("max_drawdown_pct", "Max Drawdown (%)"),
        ("calmar_ratio", "Calmar Ratio"),
        ("win_rate_pct", "Win Rate (%)"),
    ]:
        print(f"  {label:<25} Agent: {summary['agent'][metric]:>+8.3f}   "
              f"B&H: {summary['buy_and_hold'][metric]:>+8.3f}   "
              f"Random: {summary['random'][metric]:>+8.3f}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    from src.ingestion.ingest import SyntheticEnergyGenerator
    from src.transformation.transform import clean_raw, engineer_features, temporal_split
    from src.ml.drl.train_drl import load_drl_config, run_drl_training, evaluate_agent
    from stable_baselines3 import PPO
    from pathlib import Path

    gen = SyntheticEnergyGenerator(seed=42)
    df = gen.generate(days=365)
    df = clean_raw(df)
    df = engineer_features(df)
    train, val, test = temporal_split(df)

    drl_cfg = load_drl_config()
    drl_cfg["training"]["total_timesteps"] = 20_000
    drl_cfg["training"]["n_envs"] = 1
    drl_cfg["mlflow"]["tracking_uri"] = "./mlruns"

    import mlflow
    mlflow.set_tracking_uri("./mlruns")

    run_drl_training(train, val, drl_cfg)

    # Load best model and backtest
    best_path = Path(drl_cfg["training"]["best_model_path"]) / "best_model"
    if best_path.with_suffix(".zip").exists():
        agent = PPO.load(str(best_path))
        summary = run_full_backtest(agent, test, drl_cfg, output_dir="./reports/backtest")
        print("\nReport saved to:", summary["report_path"])
