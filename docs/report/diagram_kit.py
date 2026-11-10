"""
Small matplotlib drawing kit used to produce the Chapter 4 analysis and design
diagrams in the same monochrome style as the proposal figures.
Coordinates are in "diagram units": x 0..100, y 0..H (top = H).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, Rectangle, Circle, Polygon
from matplotlib.lines import Line2D

BLACK, DARK, MID, LIGHT, PALE, WHITE = "#000000", "#333333", "#666666", "#DDDDDD", "#F2F2F2", "#FFFFFF"
FONT = "DejaVu Sans"
plt.rcParams["font.family"] = FONT


def canvas(title, subtitle="Energy Trading MLOps Platform", w=14, h=9.5, H=None):
    """Create a figure with the black title band used across the proposal diagrams."""
    H = H or 100 * h / w
    fig = plt.figure(figsize=(w, h), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.add_patch(Rectangle((0, H - 6.2), 100, 6.2, color=BLACK, zorder=1))
    ax.text(50, H - 2.5, title.upper(), ha="center", va="center", color=WHITE,
            fontsize=17, fontweight="bold", zorder=2)
    ax.text(50, H - 4.9, subtitle, ha="center", va="center", color="#CCCCCC", fontsize=10.5, zorder=2)
    return fig, ax, H


def save(fig, path):
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text="", sub=None, fill=WHITE, edge=BLACK, lw=1.6, fs=10.5, bold=True,
        round_=True, text_color=BLACK, sub_fs=8.8, dashed=False, z=3):
    """Rounded box with a centred title and optional sub-text. (x, y) is the lower-left corner."""
    style = "round,pad=0,rounding_size=1.0" if round_ else "square,pad=0"
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style, fc=fill, ec=edge, lw=lw,
                                ls="--" if dashed else "-", zorder=z))
    if sub:
        ax.text(x + w / 2, y + h * 0.62, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=text_color, zorder=z + 1)
        ax.text(x + w / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=sub_fs,
                color=MID, zorder=z + 1, linespacing=1.25)
    else:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color=text_color, zorder=z + 1, linespacing=1.3)
    return (x, y, w, h)


def header_box(ax, x, y, w, title, rows=None, stereo=None, head_fill=BLACK, body_fill=WHITE, fs=9.2,
               row_h=2.3, head_h=None, sections=None):
    """UML-style class / entity box: dark header + one or more compartments of text rows.
    `sections` is a list of row lists; if None, `rows` is a single compartment.
    Returns (x, y_bottom, w, total_h)."""
    sections = sections or [rows]
    head_h = head_h or (4.4 if stereo else 3.4)
    body_h = sum(len(s) * row_h + 1.2 for s in sections)
    total = head_h + body_h
    yb = y - total
    ax.add_patch(FancyBboxPatch((x, yb), w, total, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=body_fill, ec=BLACK, lw=1.5, zorder=3))
    ax.add_patch(FancyBboxPatch((x, y - head_h), w, head_h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc=head_fill, ec=BLACK, lw=1.5, zorder=4))
    ax.add_patch(Rectangle((x, y - head_h), w, head_h * 0.4, fc=head_fill, ec="none", zorder=4))
    tc = WHITE if head_fill in (BLACK, DARK, MID) else BLACK
    if stereo:
        ax.text(x + w / 2, y - 1.3, f"«{stereo}»", ha="center", va="center", fontsize=8, style="italic",
                color="#DDDDDD" if tc == WHITE else MID, zorder=5)
        ax.text(x + w / 2, y - 3.2, title, ha="center", va="center", fontsize=10.5, fontweight="bold",
                color=tc, zorder=5)
    else:
        ax.text(x + w / 2, y - head_h / 2, title, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=tc, zorder=5)
    cy = y - head_h - 0.6
    for i, sec in enumerate(sections):
        if i > 0:
            ax.add_line(Line2D([x, x + w], [cy + 0.6, cy + 0.6], color=BLACK, lw=1.0, zorder=5))
            cy -= 0.6
        for r in sec:
            ax.text(x + 1.0, cy - row_h / 2, r, ha="left", va="center", fontsize=fs, color=BLACK, zorder=5,
                    family="DejaVu Sans")
            cy -= row_h
        cy -= 0.6
    return (x, yb, w, total)


def arrow(ax, p1, p2, label=None, dashed=False, color=DARK, lw=1.3, head=True, lpos=0.5, loff=(0, 0.9),
          fs=8.5, style="-|>", ms=12, label_bg=True, z=2, open_head=False, rad=0.0):
    ls = (0, (4, 3)) if dashed else "-"
    if head:
        ax.annotate("", xy=p2, xytext=p1, zorder=z,
                    arrowprops=dict(arrowstyle=("->" if open_head else style), color=color, lw=lw, ls=ls,
                                    mutation_scale=ms, shrinkA=0, shrinkB=0,
                                    connectionstyle=f"arc3,rad={rad}"))
    else:
        ax.add_line(Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=lw, ls=ls, zorder=z))
    if label:
        lx = p1[0] + (p2[0] - p1[0]) * lpos + loff[0]
        ly = p1[1] + (p2[1] - p1[1]) * lpos + loff[1]
        ax.text(lx, ly, label, ha="center", va="center", fontsize=fs, color=DARK, zorder=z + 6,
                bbox=dict(fc="white", ec="none", pad=0.6) if label_bg else None, linespacing=1.2)


def poly_arrow(ax, pts, label=None, dashed=False, color=DARK, lw=1.3, lidx=0, loff=(0, 0.9), fs=8.5,
               open_head=False, label_bg=True):
    ls = (0, (4, 3)) if dashed else "-"
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, ls=ls, zorder=2))
    arrow(ax, pts[-2], pts[-1], dashed=dashed, color=color, lw=lw, open_head=open_head)
    if label:
        a, b = pts[lidx], pts[lidx + 1]
        ax.text((a[0] + b[0]) / 2 + loff[0], (a[1] + b[1]) / 2 + loff[1], label, ha="center", va="center",
                fontsize=fs, color=DARK, zorder=8, bbox=dict(fc="white", ec="none", pad=0.6) if label_bg else None)


def actor(ax, x, y, name, sub=None, s=1.0):
    """UML stick-figure actor centred at x with feet at y."""
    hr = 1.6 * s
    ax.add_patch(Circle((x, y + 9.2 * s), hr, fc="white", ec=BLACK, lw=1.8, zorder=4))
    ax.add_line(Line2D([x, x], [y + 7.6 * s, y + 3.6 * s], color=BLACK, lw=1.8, zorder=4))
    ax.add_line(Line2D([x - 2.4 * s, x + 2.4 * s], [y + 6.3 * s, y + 6.3 * s], color=BLACK, lw=1.8, zorder=4))
    ax.add_line(Line2D([x, x - 2.0 * s], [y + 3.6 * s, y], color=BLACK, lw=1.8, zorder=4))
    ax.add_line(Line2D([x, x + 2.0 * s], [y + 3.6 * s, y], color=BLACK, lw=1.8, zorder=4))
    ax.text(x, y - 1.8, name, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    if sub:
        ax.text(x, y - 3.8, sub, ha="center", va="center", fontsize=8.8, color=MID, zorder=4)


def system_actor(ax, x, y, name, sub=None, w=9, h=7):
    """Boxed «system» actor (external system)."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                fc="white", ec=BLACK, lw=1.8, zorder=4))
    for i in range(3):
        yy = y + h * (0.25 + 0.25 * i)
        ax.add_patch(FancyBboxPatch((x - w * 0.32, yy - 0.6), w * 0.64, 1.2,
                                    boxstyle="round,pad=0,rounding_size=0.3", fc="white", ec=BLACK, lw=1.0, zorder=5))
        ax.add_patch(Circle((x - w * 0.22, yy), 0.28, fc=BLACK, zorder=6))
    ax.text(x, y - 1.3, "«system»", ha="center", va="center", fontsize=8.5, style="italic", color=MID, zorder=4)
    ax.text(x, y - 3.1, name, ha="center", va="center", fontsize=11, fontweight="bold", zorder=4)
    if sub:
        ax.text(x, y - 5.0, sub, ha="center", va="center", fontsize=8.8, color=MID, zorder=4)


def usecase(ax, cx, cy, text, w=22, h=5.2, fill=WHITE):
    ax.add_patch(Ellipse((cx, cy), w, h, fc=fill, ec=BLACK, lw=1.5, zorder=3))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=10, zorder=4, linespacing=1.15)


def datastore(ax, x, y, w, h, ident, text, sub=None, dup=False):
    """Gane-Sarson style data store (open-ended rectangle with ID cell).
    dup=True adds the extra vertical line that marks a duplicated store."""
    ax.add_patch(Rectangle((x, y), w, h, fc=PALE, ec="none", zorder=3))
    ax.add_line(Line2D([x, x + w], [y + h, y + h], color=BLACK, lw=1.6, zorder=4))
    ax.add_line(Line2D([x, x + w], [y, y], color=BLACK, lw=1.6, zorder=4))
    ax.add_line(Line2D([x, x], [y, y + h], color=BLACK, lw=1.6, zorder=4))
    ax.add_line(Line2D([x + 5, x + 5], [y, y + h], color=BLACK, lw=1.2, zorder=4))
    if dup:
        ax.add_line(Line2D([x + 0.9, x + 0.9], [y, y + h], color=BLACK, lw=1.2, zorder=4))
    ax.text(x + 2.5, y + h / 2, ident, ha="center", va="center", fontsize=10, fontweight="bold", zorder=5)
    if sub:
        ax.text(x + 5 + (w - 5) / 2, y + h * 0.64, text, ha="center", va="center", fontsize=9.6,
                fontweight="bold", zorder=5)
        ax.text(x + 5 + (w - 5) / 2, y + h * 0.30, sub, ha="center", va="center", fontsize=8.2, color=MID, zorder=5)
    else:
        ax.text(x + 5 + (w - 5) / 2, y + h / 2, text, ha="center", va="center", fontsize=9.6,
                fontweight="bold", zorder=5)


def process(ax, x, y, w, h, num, text, sub=None):
    """Gane-Sarson process: rounded box with a numbered header band."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0", fc=WHITE,
                                ec=BLACK, lw=1.6, zorder=3))
    ax.add_line(Line2D([x, x + w], [y + h - 3.0, y + h - 3.0], color=BLACK, lw=1.2, zorder=4))
    ax.text(x + w / 2, y + h - 1.5, num, ha="center", va="center", fontsize=9.5, fontweight="bold", zorder=5)
    ax.text(x + w / 2, y + (h - 3.0) * (0.62 if sub else 0.5), text, ha="center", va="center", fontsize=10,
            fontweight="bold", zorder=5, linespacing=1.2)
    if sub:
        ax.text(x + w / 2, y + (h - 3.0) * 0.25, sub, ha="center", va="center", fontsize=8.2, color=MID, zorder=5)


def legend(ax, items, y, x0=None, gap=3.0):
    """items: list of (kind, label) where kind in {'box_white','box_pale','box_mid','dash','solid'}."""
    widths = [len(l) * 0.75 + 4.5 for _, l in items]
    total = sum(widths) + gap * (len(items) - 1)
    x = x0 if x0 is not None else 50 - total / 2
    ax.add_line(Line2D([4, 96], [y + 2.6, y + 2.6], color=LIGHT, lw=1.0))
    for (kind, label), wdt in zip(items, widths):
        if kind.startswith("box"):
            fc = {"box_white": WHITE, "box_pale": PALE, "box_mid": LIGHT, "box_dark": DARK}[kind]
            ax.add_patch(FancyBboxPatch((x, y - 0.8), 2.2, 1.6, boxstyle="round,pad=0,rounding_size=0.3",
                                        fc=fc, ec=BLACK, lw=1.2))
        else:
            ax.add_line(Line2D([x, x + 2.6], [y, y], color=DARK, lw=1.3, ls=(0, (4, 3)) if kind == "dash" else "-"))
        ax.text(x + 3.4, y, label, ha="left", va="center", fontsize=9.2, color=MID)
        x += wdt + gap
