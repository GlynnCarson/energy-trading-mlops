"""
Generates the Chapter 5 result figures from recorded evidence only:
  - reports/evaluation/*.csv / evaluation_summary.json  (evaluate_eia_runs.py)
  - mlruns/  (MLflow metric histories)
  - reports/evaluation_reward_fix/  (retrain_fixed_reward.py: fixed-reward runs, Figures 5.3 and 5.4)

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


# Fixed-reward runs (DEF-01 fix, retrain_fixed_reward.py) and the July cost-only run, for comparison
FIX = ROOT / "reports" / "evaluation_reward_fix"
JULY_300K = "dec62fade7684182a8bf58ab7fc13a0b"
SEEDS = [42, 1, 2, 3, 4]
C_FIX, C_FIX_LIGHT, C_JULY, C_BH = "#2a78d6", "#9ec5f4", "#1baf7a", "#eb6834"   # validated categorical slots 1-3


def _fixed_run_ids():
    return {s: json.load(open(FIX / "runs" / f"PPO-300k-fixed-s{s}.json"))["run_id"] for s in SEEDS}


def _history(client, rid, key):
    h = client.get_metric_history(rid, key)
    return pd.DataFrame([(m.step, m.value) for m in h], columns=["step", "v"]).sort_values("step")


def fig_training_curves():
    """Figure 5.3: validation return and entropy loss during training, July cost-only run vs the five
    fixed-reward runs."""
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    c = MlflowClient()
    runs = _fixed_run_ids()
    val_bh = json.load(open(FIX / "runs" / "PPO-300k-fixed-s42.json"))["val_buy_and_hold_return_pct"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), dpi=220)
    for ax, key in [(a1, "portfolio_return_pct"), (a2, "train_entropy_loss")]:
        for s in SEEDS[1:]:
            h = _history(c, runs[s], key)
            ax.plot(h.step / 1000, h.v, color=C_FIX_LIGHT, lw=1.1)
        h = _history(c, runs[42], key)
        ax.plot(h.step / 1000, h.v, color=C_FIX, lw=2.0)
        h = _history(c, JULY_300K, key)
        ax.plot(h.step / 1000, h.v, color=C_JULY, lw=1.8, ls=":")
        ax.set_xlabel("Training timesteps (thousands)")
        ax.set_xlim(0, 345)
    a1.axhline(val_bh, color=C_BH, lw=1.4, ls="--")
    a1.axhline(0, color="#999999", lw=0.8)
    a1.text(305, val_bh - 2.5, f"Buy & Hold\n({val_bh:+.1f} %)", fontsize=8, color=INK, va="top")
    a1.set_title("Validation-episode portfolio return", fontsize=10.5, color=INK, loc="left")
    a1.set_ylabel("Return over validation window (%)")
    a2.set_title("PPO entropy loss", fontsize=10.5, color=INK, loc="left")
    a2.set_ylabel("Entropy loss (rises as policy\nbecomes deterministic)")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=C_FIX, lw=2.0, label="Fixed reward, seed 42"),
               Line2D([], [], color=C_FIX_LIGHT, lw=1.1, label="Fixed reward, seeds 1-4"),
               Line2D([], [], color=C_JULY, lw=1.8, ls=":", label="Cost-only reward (July run)"),
               Line2D([], [], color=C_BH, lw=1.4, ls="--", label="Buy & Hold on validation")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "fig5_2_training_curves.png", facecolor="white")
    plt.close(fig)


def fig_equity_curves():
    """Figure 5.4: full-test cumulative return of the five fixed-reward agents, the July cost-only agent and
    Buy & Hold, with the Henry Hub price and the seed-42 position in separate panels (no dual axes)."""
    init = 100_000
    t = {s: pd.read_csv(FIX / f"PPO-300k-fixed-s{s}_full_test_trajectory.csv", parse_dates=["date"]) for s in SEEDS}
    july = pd.read_csv(EVAL / "PPO-300k_full_test_trajectory.csv", parse_dates=["date"])
    t42 = t[42]
    ret = lambda v: (v / init - 1) * 100
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 8.4), dpi=220, sharex=True,
                                     gridspec_kw={"height_ratios": [2.0, 1, 1]})
    spikes = [("2025-01-15", "2025-01-24", "Jan 2025 spike"), ("2026-01-20", "2026-02-06", "Jan 2026 spike")]
    for ax in (a1, a2, a3):
        for a, b, _ in spikes:
            ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#efefef", zorder=0, lw=0)
    a1.axhline(0, color="#999999", lw=0.8)
    for s in SEEDS[1:]:
        a1.plot(t[s].date, ret(t[s].agent_value), color=C_FIX_LIGHT, lw=1.1)
    a1.plot(t42.date, ret(t42.agent_value), color=C_FIX, lw=2.0)
    a1.plot(july.date, ret(july.agent_value), color=C_JULY, lw=1.8, ls=":")
    a1.plot(t42.date, ret(t42.buy_hold_value), color=C_BH, lw=1.4, ls="--")
    vals = [ret(t[s].agent_value) for s in SEEDS] + [ret(july.agent_value), ret(t42.buy_hold_value)]
    lo, hi = min(v.min() for v in vals), max(v.max() for v in vals)
    a1.set_ylim(lo - 10, hi + 45)
    for a, b, lab in spikes:   # label the spikes in the price panel, clear of the legend
        a2.text(pd.Timestamp(b) + pd.Timedelta(days=5), 10.6, lab, ha="left", va="top", fontsize=7.8, color=MUTED)
    end = t42.date.iloc[-1] + pd.Timedelta(days=6)
    a1.text(end, ret(t42.agent_value).iloc[-1] - 9, "Fixed, seed 42", fontsize=8, color=INK, va="center")
    a1.text(end, ret(july.agent_value).iloc[-1] + 12, "Cost-only (July)", fontsize=8, color=INK, va="center")
    a1.text(end, ret(t42.buy_hold_value).iloc[-1], "Buy & Hold", fontsize=8, color=INK, va="center")
    a1.text(end, max(ret(t[s].agent_value).iloc[-1] for s in SEEDS[1:]), "Fixed, seeds 1-4", fontsize=8,
            color=INK, va="center")
    a1.set_ylabel("Cumulative return (%)")
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], color=C_FIX, lw=2.0, label="PPO-300k, fixed reward, seed 42"),
                       Line2D([], [], color=C_FIX_LIGHT, lw=1.1, label="PPO-300k, fixed reward, seeds 1-4"),
                       Line2D([], [], color=C_JULY, lw=1.8, ls=":", label="PPO-300k, cost-only reward (July)"),
                       Line2D([], [], color=C_BH, lw=1.4, ls="--", label="Buy & Hold")],
              loc="upper left", fontsize=8.3, ncol=2)
    a2.plot(t42.date, t42.price, color=INK, lw=0.9)
    a2.set_ylabel("Henry Hub\n($/MMBtu)")
    a2.set_ylim(0, 11)
    a3.fill_between(t42.date, t42.position, step="post", color=C_FIX, alpha=0.35, lw=0)
    a3.set_ylabel("Seed-42 position\n(units held)")
    a3.set_ylim(0, 26_000)
    a3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a3.set_xlim(t42.date.iloc[0], t42.date.iloc[-1] + pd.Timedelta(days=75))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_3_test_equity.png", facecolor="white")
    plt.close(fig)


def fig_paper_trading():
    """Figure 5.5: paper trading (simulated) of the champion against Buy & Hold since 7 July 2026, with the
    Henry Hub price and the champion's position in separate panels (no dual axes).
    Data: reports/live/paper_trading.csv (written by src/ml/drl/paper_trade.py)."""
    df = pd.read_csv(ROOT / "reports" / "live" / "paper_trading.csv", parse_dates=["date"])
    ret = lambda v: (v / v.iloc[0] - 1) * 100
    champ, bh = ret(df.portfolio_value), ret(df.buy_hold_value)
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(10, 7.2), dpi=220, sharex=True,
                                     gridspec_kw={"height_ratios": [2.0, 1, 1]})
    a1.axhline(0, color="#999999", lw=0.8)
    a1.plot(df.date, champ, color=C_FIX, lw=2.0)
    a1.plot(df.date, bh, color=C_BH, lw=1.4, ls="--")
    lo, hi = min(champ.min(), bh.min()), max(champ.max(), bh.max())
    a1.set_ylim(lo - 2, hi + 4)
    end = df.date.iloc[-1] + pd.Timedelta(days=2)
    a1.text(end, champ.iloc[-1] + 0.9, f"Champion v{df.model_version.iloc[-1]} ({champ.iloc[-1]:+.2f} %)",
            fontsize=8, color=INK, va="center")
    a1.text(end, bh.iloc[-1] - 0.9, f"Buy & Hold ({bh.iloc[-1]:+.2f} %)", fontsize=8, color=INK, va="center")
    a1.set_ylabel("Return since 7 Jul 2026 (%)")
    from matplotlib.lines import Line2D
    a1.legend(handles=[Line2D([], [], color=C_FIX, lw=2.0, label="Champion (energy_trading_ppo v5, seed 4), "
                                                                   "paper trading (simulated)"),
                       Line2D([], [], color=C_BH, lw=1.4, ls="--", label="Buy & Hold")],
              loc="upper left", fontsize=8.3, ncol=2)
    a2.plot(df.date, df.price, color=INK, lw=0.9)
    a2.set_ylabel("Henry Hub\n($/MMBtu)")
    a2.set_ylim(df.price.min() - 0.2, df.price.max() + 0.2)
    a3.fill_between(df.date, df.position_mmbtu, step="post", color=C_FIX, alpha=0.35, lw=0)
    a3.set_ylabel("Champion position\n(MMBtu, after trade)")
    a3.set_ylim(0, 26_000)
    a3.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a3.xaxis.set_major_locator(mdates.MonthLocator())
    a3.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    a3.set_xlim(df.date.iloc[0], df.date.iloc[-1] + pd.Timedelta(days=24))
    fig.tight_layout()
    fig.savefig(OUT / "fig5_5_paper_trading.png", facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    fig_price_series()
    fig_training_curves()
    fig_equity_curves()
    fig_paper_trading()
    print("Charts written to", OUT)
