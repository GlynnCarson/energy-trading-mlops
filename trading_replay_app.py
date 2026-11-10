"""
trading_replay_app.py
─────────────────────────────────────────────────────────────────────────────
Native desktop replay of the trained PPO energy-trading agent stepping through
the held-out test episode. Pure-Python GUI (Tkinter + matplotlib) — no browser,
no server, runs fully offline.

    python trading_replay_app.py

Reads reports/animation/trajectory.json (produced by the training/backtest run).
Controls: Play/Pause · speed 1×/2×/4×/8× · scrub slider · Reset.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec
import tkinter as tk

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "reports" / "animation" / "trajectory.json"

# ── Palette (matches backtest.py / the web replay) ───────────────────────────
BG, PANEL, GRID = "#0d1117", "#161b22", "#21262d"
TEXT, MUTED, BRIGHT = "#c9d1d9", "#8b949e", "#e6edf3"
AGENT, BH, RAND, PRICE = "#58a6ff", "#3fb950", "#f78166", "#e3b341"
BUY, SELL, POS = "#3fb950", "#f85149", "#58a6ff"
UP, DOWN = "#3fb950", "#f85149"


def load_data():
    if not DATA_PATH.exists():
        print(f"ERROR: {DATA_PATH} not found.\n"
              f"Run the training/backtest first so the trajectory is generated.")
        sys.exit(1)
    return json.loads(DATA_PATH.read_text())


class ReplayApp:
    def __init__(self, root, D):
        self.root = root
        self.D = D
        self.N = D["n"]
        self.x = list(range(self.N))
        self.frame = 0
        self.playing = False
        self.speed = 1                # steps advanced per tick
        self.tick_ms = 33             # ~30 fps

        # trade steps split by side
        self.buy_steps = sorted(t["step"] for t in D["trades"] if t["action"] == "BUY")
        self.sell_steps = sorted(t["step"] for t in D["trades"] if t["action"] == "SELL")
        self.all_trade_steps = sorted(t["step"] for t in D["trades"])

        root.title("⚡ PPO Agent — Live Trading Replay")
        root.configure(bg=BG)
        root.geometry("1180x900")
        root.minsize(900, 720)

        self._build_header()
        self._build_figure()
        self._build_controls()

        self.render(0)
        self._set_playing(True)
        self.root.after(self.tick_ms, self._tick)

    # ── Header + KPI tiles ───────────────────────────────────────────────────
    def _build_header(self):
        top = tk.Frame(self.root, bg=BG)
        top.pack(fill="x", padx=18, pady=(14, 4))
        tk.Label(top, text="⚡  PPO Agent — Live Trading Replay", bg=BG, fg=BRIGHT,
                 font=("Segoe UI Semibold", 15)).pack(side="left")
        self.clock = tk.StringVar(value="HOUR 000 / %d" % self.N)
        tk.Label(top, textvariable=self.clock, bg=BG, fg=MUTED,
                 font=("Consolas", 12)).pack(side="right")

        row = tk.Frame(self.root, bg=BG)
        row.pack(fill="x", padx=14, pady=(6, 2))
        self.kpi = {}
        specs = [("Hour", "hour"), ("Spot Price", "price"), ("Position MWh", "pos"),
                 ("Agent Return", "ret"), ("Lead vs B&H", "lead"), ("Trades", "trades")]
        for label, key in specs:
            card = tk.Frame(row, bg=PANEL, bd=0, highlightthickness=1,
                            highlightbackground=GRID)
            card.pack(side="left", expand=True, fill="both", padx=4)
            tk.Label(card, text=label.upper(), bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 8)).pack(anchor="w", padx=11, pady=(8, 0))
            var = tk.StringVar(value="—")
            lbl = tk.Label(card, textvariable=var, bg=PANEL, fg=BRIGHT,
                           font=("Consolas", 17, "bold"))
            lbl.pack(anchor="w", padx=11, pady=(0, 9))
            self.kpi[key] = (var, lbl)

    # ── Matplotlib figure (price / position / race) ──────────────────────────
    def _build_figure(self):
        self.fig = plt.figure(figsize=(11.4, 6.4), facecolor=BG)
        gs = GridSpec(3, 1, height_ratios=[3.1, 1.0, 2.1], hspace=0.42, figure=self.fig)
        self.ax_p = self.fig.add_subplot(gs[0])
        self.ax_pos = self.fig.add_subplot(gs[1])
        self.ax_r = self.fig.add_subplot(gs[2])

        D = self.D
        for ax in (self.ax_p, self.ax_pos, self.ax_r):
            ax.set_facecolor(PANEL)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.tick_params(colors=MUTED, labelsize=8)
            ax.grid(True, color=GRID, linewidth=0.6, alpha=0.7)
            ax.set_xlim(0, self.N - 1)

        # scales
        pmin, pmax = min(D["price"]), max(D["price"])
        allret = D["agent_ret"] + D["bh_ret"] + D["rand_ret"]
        rmin, rmax = min(allret), max(allret)
        posmax = max(100.0, max(D["position"]))
        self.ax_p.set_ylim(pmin - (pmax - pmin) * .08, pmax + (pmax - pmin) * .08)
        self.ax_pos.set_ylim(0, posmax * 1.08)
        self.ax_r.set_ylim(rmin - (rmax - rmin) * .08, rmax + (rmax - rmin) * .08)

        # titles
        self.ax_p.set_title("Energy Spot Price · USD/MWh", color=MUTED, fontsize=10,
                            loc="left", pad=6)
        self.ax_pos.set_title("Position Held · MWh", color=MUTED, fontsize=10,
                              loc="left", pad=4)
        self.ax_r.set_title("Cumulative Return · Agent vs Buy&Hold vs Random",
                            color=MUTED, fontsize=10, loc="left", pad=6)

        # ── static ghosts ──
        self.ax_p.plot(self.x, D["price"], color=PRICE, lw=1, alpha=0.10)
        self.ax_pos.plot(self.x, D["position"], color=POS, lw=1, alpha=0.12)
        self.ax_pos.axhline(100, color=POS, lw=1, ls="--", alpha=0.25)
        if rmin < 0 < rmax:
            self.ax_r.axhline(0, color=TEXT, lw=0.8, ls=":", alpha=0.5)

        # ── dynamic artists (updated via set_data) ──
        (self.l_price,) = self.ax_p.plot([], [], color=PRICE, lw=1.7)
        (self.m_buy,) = self.ax_p.plot([], [], linestyle="none", marker="^",
                                       color=BUY, ms=5, alpha=0.7)
        (self.m_sell,) = self.ax_p.plot([], [], linestyle="none", marker="v",
                                        color=SELL, ms=5, alpha=0.7)
        self.cursor = self.ax_p.axvline(0, color=AGENT, lw=1, alpha=0.35)
        (self.dot_p,) = self.ax_p.plot([], [], marker="o", color=PRICE, ms=6,
                                       markeredgecolor=BG)

        (self.l_pos,) = self.ax_pos.plot([], [], color=POS, lw=1.4)

        (self.l_rand,) = self.ax_r.plot([], [], color=RAND, lw=1.1, label="Random")
        (self.l_bh,) = self.ax_r.plot([], [], color=BH, lw=1.4, label="Buy & Hold")
        (self.l_agent,) = self.ax_r.plot([], [], color=AGENT, lw=2.1, label="PPO Agent")
        leg = self.ax_r.legend(loc="upper left", fontsize=8, facecolor=PANEL,
                               edgecolor=GRID, labelcolor=TEXT)
        (self.dot_a,) = self.ax_r.plot([], [], marker="o", color=AGENT, ms=5,
                                       markeredgecolor=BG)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill="both", expand=True, padx=12, pady=6)

    # ── Controls ─────────────────────────────────────────────────────────────
    def _build_controls(self):
        bar = tk.Frame(self.root, bg=BG)
        bar.pack(fill="x", padx=18, pady=(2, 14))

        self.play_btn = tk.Button(bar, text="▶  Play", width=10, command=self.toggle,
                                  bg=PANEL, fg=BRIGHT, activebackground="#232a31",
                                  activeforeground=BRIGHT, relief="flat",
                                  font=("Segoe UI", 10, "bold"), cursor="hand2")
        self.play_btn.pack(side="left")

        tk.Button(bar, text="↻  Reset", command=self.reset, bg=PANEL, fg=TEXT,
                  activebackground="#232a31", relief="flat",
                  font=("Segoe UI", 10), cursor="hand2").pack(side="left", padx=(8, 16))

        tk.Label(bar, text="SPEED", bg=BG, fg=MUTED,
                 font=("Segoe UI", 8)).pack(side="left", padx=(0, 6))
        self.speed_btns = {}
        for sp in (1, 2, 4, 8):
            b = tk.Button(bar, text=f"{sp}×", width=3, relief="flat",
                          command=lambda s=sp: self.set_speed(s),
                          bg=PANEL, fg=MUTED, activebackground="#232a31",
                          font=("Consolas", 10), cursor="hand2")
            b.pack(side="left", padx=2)
            self.speed_btns[sp] = b
        self._highlight_speed()

        self.slider = tk.Scale(bar, from_=0, to=self.N - 1, orient="horizontal",
                               showvalue=False, command=self.on_scrub,
                               bg=BG, fg=TEXT, troughcolor=GRID, highlightthickness=0,
                               activebackground=AGENT, sliderrelief="flat", length=380)
        self.slider.pack(side="left", fill="x", expand=True, padx=(18, 0))

    # ── Rendering ────────────────────────────────────────────────────────────
    def render(self, f):
        f = int(f)
        D, x = self.D, self.x
        self.l_price.set_data(x[:f + 1], D["price"][:f + 1])
        self.l_pos.set_data(x[:f + 1], D["position"][:f + 1])
        self.l_rand.set_data(x[:f + 1], D["rand_ret"][:f + 1])
        self.l_bh.set_data(x[:f + 1], D["bh_ret"][:f + 1])
        self.l_agent.set_data(x[:f + 1], D["agent_ret"][:f + 1])

        bs = [s for s in self.buy_steps if s <= f]
        ss = [s for s in self.sell_steps if s <= f]
        self.m_buy.set_data(bs, [D["price"][s] for s in bs])
        self.m_sell.set_data(ss, [D["price"][s] for s in ss])

        self.cursor.set_xdata([f, f])
        self.dot_p.set_data([f], [D["price"][f]])
        self.dot_a.set_data([f], [D["agent_ret"][f]])

        self._update_kpis(f)
        self.slider.set(f)
        self.canvas.draw_idle()

    def _update_kpis(self, f):
        D = self.D
        ar, br = D["agent_ret"][f], D["bh_ret"][f]
        lead = ar - br
        n_tr = sum(1 for s in self.all_trade_steps if s <= f)

        def col(v):
            return UP if v > 0.02 else (DOWN if v < -0.02 else BRIGHT)

        self.kpi["hour"][0].set(f"{f + 1}")
        self.kpi["price"][0].set(f"${D['price'][f]:.2f}")
        self.kpi["pos"][0].set(f"{D['position'][f]:.1f}")
        self.kpi["ret"][0].set(f"{ar:+.2f}%");  self.kpi["ret"][1].config(fg=col(ar))
        self.kpi["lead"][0].set(f"{lead:+.2f}%"); self.kpi["lead"][1].config(fg=col(lead))
        self.kpi["trades"][0].set(f"{n_tr}")
        self.clock.set(f"HOUR {f:03d} / {self.N}")

    # ── Playback engine ──────────────────────────────────────────────────────
    def _tick(self):
        if self.playing:
            self.frame += self.speed
            if self.frame >= self.N - 1:
                self.frame = self.N - 1
                self.render(self.frame)
                self._set_playing(False)
            else:
                self.render(self.frame)
        self.root.after(self.tick_ms, self._tick)

    def _set_playing(self, p):
        self.playing = p
        if p:
            self.play_btn.config(text="❚❚  Pause", fg=PRICE)
        else:
            self.play_btn.config(
                text="↻  Replay" if self.frame >= self.N - 1 else "▶  Play", fg=BRIGHT)

    def toggle(self):
        if self.frame >= self.N - 1:
            self.frame = 0
        self._set_playing(not self.playing)

    def reset(self):
        self.frame = 0
        self.render(0)
        self._set_playing(False)

    def set_speed(self, s):
        self.speed = s
        self._highlight_speed()

    def _highlight_speed(self):
        for sp, b in self.speed_btns.items():
            if sp == self.speed:
                b.config(bg=AGENT, fg="#04101f", font=("Consolas", 10, "bold"))
            else:
                b.config(bg=PANEL, fg=MUTED, font=("Consolas", 10))

    def on_scrub(self, val):
        f = int(float(val))
        if f != self.frame:
            self.frame = f
            self._set_playing(False)
            self.render(f)


def main():
    D = load_data()
    root = tk.Tk()
    ReplayApp(root, D)
    root.mainloop()


if __name__ == "__main__":
    main()
