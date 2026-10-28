"""
plot_demo_experience.py
-----------------------
2x2 panel figure summarising the four AI-demonstration experience items
(POST only, demo group = students who attended >= 1 demonstration, n=59
after the honesty-item rule in load_research_data.load_data):

    [0,0] Demos attended (1-9 count histogram)
    [0,1] Demos helped understanding (5-point Likert)
    [1,0] Engagement with demos (5-point ordinal)
    [1,1] Demos changed how I use AI (5-point ordinal)

Y-axis is percentage of the demo group. The three ordinal panels are colour-
graded along the diverging Likert palette (negative -> positive). Colours and
fonts come from plot_style.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_research_data import load_data
from plot_style import (
    apply_style,
    PALETTE_LIKERT,
    COLOUR_ACCENT,
    FONT_SIZES,
    FIGURE,
)

# ── Ordinal scales for each item (low → high) ─────────────────────────────────
HELPED_ORDER = [
    "Strongly Disagree", "Disagree", "Neither Agree nor Disagree",
    "Agree", "Strongly Agree",
]
ENGAGE_ORDER = [
    "Not at all engaged", "Slightly engaged", "Moderately engaged",
    "Very engaged", "Extremely engaged",
]
CHANGED_ORDER = [
    "Not at all", "A little", "Somewhat", "Quite a bit", "A great deal",
]

# Short tick labels (the full labels are too long for an axis)
ENGAGE_SHORT  = ["Not at all", "Slightly", "Moderately", "Very", "Extremely"]
CHANGED_SHORT = ["Not at all", "A little", "Somewhat", "Quite a bit", "A great deal"]
HELPED_SHORT  = ["Strongly\nDisagree", "Disagree", "Neither", "Agree",
                 "Strongly\nAgree"]

# The 5-step diverging palette, reused as a low→high gradient for all ordinals
GRADIENT5 = list(PALETTE_LIKERT.values())


def _pct(series: pd.Series, order: list[str]) -> np.ndarray:
    """Percentage of valid responses in each ordinal category."""
    s = series.dropna()
    total = len(s)
    counts = s.value_counts()
    return np.array([counts.get(c, 0) / total * 100 for c in order])


def _draw_ordinal(ax, series, order, short_labels, title):
    """Draw one colour-graded ordinal bar panel."""
    pct = _pct(series, order)
    n   = series.notna().sum()
    x   = np.arange(len(order))
    ax.bar(x, pct, width=0.78, color=GRADIENT5, linewidth=0, zorder=2)

    # Count labels above bars
    counts = series.dropna().value_counts()
    for xi, cat in zip(x, order):
        c = int(counts.get(cat, 0))
        if c > 0:
            ax.text(xi, pct[xi] + 1.0, str(c), ha="center", va="bottom",
                    fontsize=FONT_SIZES["bar_label"] - 1, color="#444444")

    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=FONT_SIZES["tick"] - 2)
    ax.set_ylim(0, pct.max() * 1.22)   # headroom so count labels clear the title
    ax.set_title(f"{title}  (n={n})", fontsize=FONT_SIZES["tick"] + 1,
                 fontweight="semibold", loc="left", color="#222222", pad=6)
    ax.set_ylabel("% of demo group", fontsize=FONT_SIZES["axis_label"] - 1)
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def plot_demo_experience(
    data_path: str,
    title: str = "AI demonstration experience (demo attendees only)",
    out_path: str | None = None,
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    apply_style()

    data = load_data(data_path)
    df   = data["all"]

    # Demo group = attended >= 1
    attended_all = pd.to_numeric(df["ai_demos_attended_post"], errors="coerce")
    demo_mask    = attended_all >= 1
    demo         = df[demo_mask]
    n_demo       = demo_mask.sum()

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 7.5))

    # ── [0,0] Demos attended (count histogram, 1-9) ───────────────────────────
    ax = axes[0, 0]
    attended = pd.to_numeric(demo["ai_demos_attended_post"], errors="coerce").dropna()
    values   = list(range(1, 10))
    total    = len(attended)
    pct      = np.array([(attended == v).sum() / total * 100 for v in values])
    ax.bar(values, pct, width=0.8, color=COLOUR_ACCENT, linewidth=0, zorder=2)
    for v, p in zip(values, pct):
        c = int((attended == v).sum())
        if c > 0:
            ax.text(v, p + 1.0, str(c), ha="center", va="bottom",
                    fontsize=FONT_SIZES["bar_label"] - 1, color="#444444")
    ax.set_xticks(values)
    ax.set_ylim(0, pct.max() * 1.22)   # headroom so count labels clear the title
    ax.set_xlabel("Number of demos attended", fontsize=FONT_SIZES["axis_label"] - 1)
    ax.set_ylabel("% of demo group", fontsize=FONT_SIZES["axis_label"] - 1)
    ax.set_title(f"Demos attended  (n={total})",
                 fontsize=FONT_SIZES["tick"] + 1, fontweight="semibold",
                 loc="left", color="#222222", pad=6)
    ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)

    # ── [0,1] Engagement ──────────────────────────────────────────────────────
    _draw_ordinal(axes[0, 1], demo["ai_demo_engagement_post"],
                  ENGAGE_ORDER, ENGAGE_SHORT,
                  "Engagement with demos")

    # ── [1,0] Helped understanding (Likert) ───────────────────────────────────
    _draw_ordinal(axes[1, 0], demo["ai_demo_helped_understanding_post"],
                  HELPED_ORDER, HELPED_SHORT,
                  "Demos helped my understanding")

    # ── [1,1] Changed how I use AI ────────────────────────────────────────────
    _draw_ordinal(axes[1, 1], demo["ai_demo_changed_use_post"],
                  CHANGED_ORDER, CHANGED_SHORT,
                  "Demos changed how I use AI")

    # Figure title
    fig.suptitle(f"{title}", fontsize=FONT_SIZES["title"],
                 fontweight="bold", x=0.5, y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


if __name__ == "__main__":
    from load_research_data import data_path, QUANT_FILE
    plot_demo_experience(
        data_path=str(data_path(QUANT_FILE)),
        out_path="desc_page5_demos.pdf",
    )
