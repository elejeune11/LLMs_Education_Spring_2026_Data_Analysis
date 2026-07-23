"""
plot_grade_quartile_freq.py
---------------------------
2x2 panel figure: AI use frequency broken down by grade quartile.
Rows = domain (engineering mechanics / general), columns = timepoint (pre/post).
Within each panel, one horizontal stacked bar per quartile shows the % of
students in that quartile falling into each frequency category. Quartiles are
bins of the average midterm z-score pooled across all sections; Q4 is best.

Restricted to students with paired pre+post on BOTH frequency measures and a
valid grade quartile. Colours/fonts from plot_style.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_research_data import load_data, FREQ_ORDER, GRADE_QUARTILE_ORDER
from plot_style import (
    apply_style,
    PALETTE_FREQ,
    FONT_SIZES,
    FIGURE,
    freq_legend_handles,
)

# Minimum segment width (%) to print a count label inside it
SEG_LABEL_THRESHOLD = 4


def _quartile_pcts(sub: pd.DataFrame, col: str) -> tuple[np.ndarray, np.ndarray]:
    """
    For one grade-quartile subset and one frequency column, return:
      pct   - percentage in each FREQ_ORDER category (sums to 100)
      count - raw count in each category
    """
    s = sub[col].dropna()
    total = len(s)
    counts = s.value_counts()
    count_arr = np.array([counts.get(c, 0) for c in FREQ_ORDER])
    pct_arr   = count_arr / total * 100 if total else np.zeros(len(FREQ_ORDER))
    return pct_arr, count_arr


def _draw_panel(ax, df, freq_col, panel_label, show_quartile_labels):
    """Draw one panel: stacked freq bars by grade quartile."""
    quartiles = GRADE_QUARTILE_ORDER
    y_positions = list(range(len(quartiles)))[::-1]   # Q4 at top

    dark_cats = {"1-3 times per month", "1-5 times per week",
                 "Daily or almost daily"}

    ytick_labels = []
    for y, quartile in zip(y_positions, quartiles):
        sub = df[df["grade_quartile"] == quartile]
        n_q = sub[freq_col].notna().sum()
        pct, count = _quartile_pcts(sub, freq_col)

        left = 0.0
        for i, cat in enumerate(FREQ_ORDER):
            w = pct[i]
            if w == 0:
                continue
            ax.barh(y, w, 0.72, left=left, color=PALETTE_FREQ[cat],
                    linewidth=0, zorder=2)
            if w >= SEG_LABEL_THRESHOLD:
                txt_color = "white" if cat in dark_cats else "#333333"
                ax.text(left + w / 2, y, str(int(count[i])),
                        ha="center", va="center",
                        fontsize=FONT_SIZES["bar_label"] - 1,
                        fontweight="semibold", color=txt_color, zorder=4)
            left += w

        # Quartile label includes n; on right-column panels we show just (n=..)
        if show_quartile_labels:
            ytick_labels.append(f"{quartile}  (n={n_q})")
        else:
            ytick_labels.append(f"(n={n_q})")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(ytick_labels, fontsize=FONT_SIZES["tick"] - 1)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"],
                       fontsize=FONT_SIZES["tick"] - 1)
    ax.set_xlabel("% of students within grade quartile",
                  fontsize=FONT_SIZES["axis_label"] - 1)
    ax.set_title(panel_label, fontsize=FONT_SIZES["tick"] + 1,
                 fontweight="bold", loc="left", color="#222222", pad=6)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)


def plot_grade_quartile_freq(
    data_path: str,
    title: str = "AI use frequency by grade quartile",
    out_path: str | None = None,
    fig_width: float = FIGURE["width"] + 3,
) -> plt.Figure:
    apply_style()

    data = load_data(data_path)
    df   = data["all"]

    # Restrict to paired pre+post on BOTH measures + valid grade tier
    freq_cols = ["ai_use_freq_general_pre", "ai_use_freq_general_post",
                 "ai_use_freq_engmech_pre", "ai_use_freq_engmech_post"]
    mask = df[freq_cols].notna().all(axis=1) & df["grade_quartile"].notna()
    sub  = df[mask].copy()
    n    = len(sub)

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 8.5))

    # Panel definitions: (axes, freq column, label, show quartile labels (left col))
    panels = [
        (axes[0, 0], "ai_use_freq_general_pre",  "General (any purpose) — PRE",                   True),
        (axes[0, 1], "ai_use_freq_general_post", "General (any purpose) — POST",                  False),
        (axes[1, 0], "ai_use_freq_engmech_pre",  "For engineering mechanics specifically — PRE",  True),
        (axes[1, 1], "ai_use_freq_engmech_post", "For engineering mechanics specifically — POST", False),
    ]
    for ax, col, label, show_labels in panels:
        _draw_panel(ax, sub, col, label, show_labels)

    # Shared y-axis label for the left column
    axes[0, 0].set_ylabel("Grade quartile (Q4 is best)",
                          fontsize=FONT_SIZES["axis_label"] - 1)
    axes[1, 0].set_ylabel("Grade quartile (Q4 is best)",
                          fontsize=FONT_SIZES["axis_label"] - 1)

    # Title + subtitle
    fig.suptitle(
        f"{title} — domain (rows) × timepoint (columns)",
        fontsize=FONT_SIZES["title"], fontweight="bold", x=0.5, y=0.99,
    )
    fig.text(
        0.5, 0.945,
        f"All sections pooled, restricted to students with paired pre+post "
        f"on both measures (n={n})",
        ha="center", va="top", fontsize=FONT_SIZES["subtitle"],
        fontweight="bold", color="#333333",
    )

    # Legend at bottom
    fig.legend(
        handles=freq_legend_handles(),
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=len(PALETTE_FREQ),
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.4,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.93])

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


if __name__ == "__main__":
    from load_research_data import data_path, QUANT_FILE

    plot_grade_quartile_freq(
        data_path=str(data_path(QUANT_FILE)),
        out_path="figures/ai_use_freq_with_grades.pdf",
    )
