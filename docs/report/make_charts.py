"""
Generates the Chapter 5 result figures from recorded evidence only:
  - reports/evaluation/*.csv / evaluation_summary.json  (evaluate_eia_runs.py)
  - mlruns/  (MLflow metric histories)

    python docs/report/make_charts.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "reports" / "evaluation"
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#1a1a1a", "#666666", "#e5e5e5"
C_PPO300, C_PPO10, C_BH, C_PRICE = "#1f4e79", "#7f7f7f", "#2e7d32", "#1a1a1a"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": "#999999", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
})


def fig_price_series():
    summ = json.load(open(EVAL / "evaluation_summary.json"))
    df = pd.read_csv(EVAL / "eia_price_series.csv", parse_dates=["timestamp"]).set_index("timestamp")
    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=220)
    shades = {"train": "#f3f3f3", "val": "#dfe8f1", "test": "#e3efe4"}
    labels = {"train": "Training", "val": "Validation", "test": "Test"}
    for k, sp in summ["splits"].items():
        a, b = pd.Timestamp(sp["start"], tz="UTC"), pd.Timestamp(sp["end"], tz="UTC")
        ax.axvspan(a, b, color=shades[k], zorder=0)
        ax.text(a + (b - a) / 2, 11.9, f"{labels[k]}\n{sp['rows']:,} days", ha="center", va="top", fontsize=8.5,
                color=MUTED)
    ax.plot(df.index, df["price_usd"], color=C_PRICE, lw=0.9, label="Henry Hub spot ($/MMBtu, target)")
    ax.set_ylabel("Henry Hub ($/MMBtu)")
    ax.set_ylim(0, 12.2)
    ax2 = ax.twinx()
    ax2.plot(df.index, df["oil_price_usd"], color="#b0b0b0", lw=0.8, label="WTI crude ($/bbl, exogenous feature)")
    ax2.set_ylabel("WTI ($/bbl)", color=MUTED)
    ax2.grid(False)
    ax2.spines["right"].set_visible(True)
    ax2.set_ylim(-60, 160)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8.5)
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_1_price_series.png", facecolor="white")
    plt.close(fig)


def fig_training_curves():
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    c = MlflowClient()
    rid = "dec62fade7684182a8bf58ab7fc13a0b"
    val = pd.DataFrame([(m.step, m.value) for m in c.get_metric_history(rid, "portfolio_return_pct")],
                       columns=["step", "v"]).sort_values("step")
    ent = pd.DataFrame([(m.step, m.value) for m in c.get_metric_history(rid, "train_entropy_loss")],
                       columns=["step", "v"]).sort_values("step")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), dpi=220)
    a1.axhline(0, color="#999999", lw=0.8)
    a1.plot(val.step / 1000, val.v, color=C_PPO300, lw=1.6, marker="o", ms=3)
    a1.set_title("Validation-episode portfolio return", fontsize=10.5, color=INK, loc="left")
    a1.set_xlabel("Training timesteps (thousands)")
    a1.set_ylabel("Return over validation window (%)")
    a2.plot(ent.step / 1000, ent.v, color=INK, lw=1.6, marker="o", ms=3)
    a2.set_title("PPO entropy loss", fontsize=10.5, color=INK, loc="left")
    a2.set_xlabel("Training timesteps (thousands)")
    a2.set_ylabel("Entropy loss (rises as policy\nbecomes deterministic)")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_2_training_curves.png", facecolor="white")
    plt.close(fig)


def fig_equity_curves():
    t3 = pd.read_csv(EVAL / "PPO-300k_full_test_trajectory.csv", parse_dates=["date"])
    t1 = pd.read_csv(EVAL / "PPO-10k_full_test_trajectory.csv", parse_dates=["date"])
    init = 100_000
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.2), dpi=220, sharex=True,
                                 gridspec_kw={"height_ratios": [1.6, 1]})
    a1.axhline(0, color="#999999", lw=0.8)
    a1.plot(t3.date, (t3.agent_value / init - 1) * 100, color=C_PPO300, lw=1.8, label="PPO-300k (best checkpoint)")
    a1.plot(t1.date, (t1.agent_value / init - 1) * 100, color=C_PPO10, lw=1.4, label="PPO-10k (best checkpoint)")
    a1.plot(t3.date, (t3.buy_hold_value / init - 1) * 100, color=C_BH, lw=1.2, ls="--", label="Buy & Hold")
    a1.set_ylabel("Cumulative return (%)")
    a1.legend(loc="upper right", fontsize=8.8)
    a1.annotate("17-20 Jan 2025 price spike\n(\\$4.30 to \\$9.86 per MMBtu)", xy=(pd.Timestamp("2025-01-20"), 124),
                xytext=(pd.Timestamp("2025-04-15"), 150), fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    a1.set_ylim(-60, 175)
    a2b = a2.twinx()
    a2.fill_between(t3.date, t3.position, step="post", color=C_PPO300, alpha=0.25, lw=0)
    a2.set_ylabel("PPO-300k position\n(units held)", color=C_PPO300)
    a2b.plot(t3.date, t3.price, color=INK, lw=0.9)
    a2b.set_ylabel("Henry Hub ($/MMBtu)", color=MUTED)
    a2b.grid(False)
    a2b.spines["right"].set_visible(True)
    a2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_3_test_equity.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_price_series()
    fig_training_curves()
    fig_equity_curves()
    print("Charts written to", OUT)
