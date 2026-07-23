"""
plot_demo_comparison.py
-----------------------
Descriptive demo-section vs. other-sections comparisons:

    plot_use_freq_by_demo : AI use frequency (general / engineering mechanics
                            x pre / post), one panel per combination, with a
                            Demo-section row and an Other-sections row per
                            panel. Same visual language as
                            plot_grade_quartile_freq (counts inside segments,
                            %-of-group x axis, shared legend).
    plot_grade_by_demo    : grade-quartile composition of each section group
                            (quartiles are midterm z-score bins pooled across
                            ALL sections; Q4 is best).

Both figures are descriptive. Sections were assigned by ordinary
registration and differ in composition; the grade figure exists partly to
make that composition difference visible.

All colours, fonts, and spacing come from plot_style.py; data loading and
category orders come from load_research_data.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from load_research_data import load_data, FREQ_ORDER, GRADE_QUARTILE_ORDER
from plot_style import (
    apply_style,
    PALETTE_FREQ,
    FONT_SIZES,
    FIGURE,
    freq_legend_handles,
)

# ── Layout constants (match plot_stacked_freq.py conventions) ─────────────────
BAR_H = 0.35
COUNT_LABEL_MIN_PCT = 4      # label a segment only when >= this % of the bar

# Section-group rows, top to bottom
GROUPS = [(True, "Demo section"), (False, "Other sections")]

# Sequential blues for grade quartiles, best (Q4) darkest; single-hue ramp is
# CVD-safe and consistent with PALETTE_FREQ.
PALETTE_QUARTILE: dict[str, str] = {
    "Q4": "#08306b",
    "Q3": "#2171b5",
    "Q2": "#6baed6",
    "Q1": "#c6dbef",
}
_QUARTILE_DARK = {"Q4", "Q3"}

DARK_FREQ = {"1-3 times per month", "1-5 times per week",
             "Daily or almost daily"}


def _draw_group_bar(ax, y, series, order, palette, dark_cats):
    """One left-aligned 100% stacked bar with in-segment count labels."""
    counts = series.value_counts(dropna=True)
    n = counts.sum()
    left = 0.0
    for cat in order:
        c = int(counts.get(cat, 0))
        if c == 0:
            continue
        w = 100.0 * c / n
        ax.barh(y, w, BAR_H, left=left, color=palette[cat],
                linewidth=0, zorder=2)
        if w >= COUNT_LABEL_MIN_PCT:
            ax.text(left + w / 2, y, str(c),
                    ha="center", va="center",
                    fontsize=FONT_SIZES["bar_label"] - 1,
                    fontweight="semibold",
                    color="white" if cat in dark_cats else "#333333",
                    zorder=4)
        left += w


def _pct_axis(ax):
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def plot_use_freq_by_demo(
    data_path: str,
    out_path: str | None = None,
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    """AI use frequency by section group — domain (rows) x timepoint (cols)."""
    apply_style()
    df = load_data(data_path)["all"]

    panels = [
        ("ai_use_freq_general_pre",  "General (any purpose) — PRE"),
        ("ai_use_freq_general_post", "General (any purpose) — POST"),
        ("ai_use_freq_engmech_pre",  "For engineering mechanics specifically — PRE"),
        ("ai_use_freq_engmech_post", "For engineering mechanics specifically — POST"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(fig_width, 5.6))
    for ax, (col, ptitle) in zip(axes.flat, panels):
        yticklabels = []
        for y, (flag, label) in enumerate(reversed(GROUPS)):
            series = df.loc[df["demo_section"] == flag, col].dropna()
            _draw_group_bar(ax, y, series, FREQ_ORDER, PALETTE_FREQ, DARK_FREQ)
            yticklabels.append(f"{label}\n(n={len(series)})")
        ax.set_yticks(range(len(GROUPS)))
        ax.set_yticklabels(yticklabels)
        ax.set_ylim(-0.55, len(GROUPS) - 0.45)
        _pct_axis(ax)
        ax.set_title(ptitle, fontsize=FONT_SIZES["tick"] + 1,
                     fontweight="semibold", loc="left", color="#222222",
                     pad=6)

    for ax in axes[1]:
        ax.set_xlabel("% of students within section group",
                      fontsize=FONT_SIZES["axis_label"])

    fig.suptitle("AI use frequency by section group — domain (rows) × timepoint (columns)",
                 fontsize=FONT_SIZES["title"], fontweight="bold", y=1.03)
    fig.text(0.5, 0.965,
             "All consenting respondents; descriptive comparison "
             "(sections assigned by registration)",
             ha="center", fontsize=FONT_SIZES["subtitle"] - 1, color="#555555")

    fig.legend(handles=freq_legend_handles(),
               loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.05),
               fontsize=FONT_SIZES["legend"], frameon=False,
               handlelength=1.2, handletextpad=0.5, columnspacing=1.0)

    fig.tight_layout(rect=[0, 0.02, 1, 0.95])

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


def plot_grade_by_demo(
    data_path: str,
    out_path: str | None = None,
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    """Grade-quartile composition of each section group."""
    apply_style()
    df = load_data(data_path)["all"]

    fig, ax = plt.subplots(figsize=(fig_width, 2.4))
    yticklabels = []
    for y, (flag, label) in enumerate(reversed(GROUPS)):
        series = df.loc[df["demo_section"] == flag, "grade_quartile"].dropna()
        _draw_group_bar(ax, y, series, GRADE_QUARTILE_ORDER,
                        PALETTE_QUARTILE, _QUARTILE_DARK)
        yticklabels.append(f"{label}\n(n={len(series)})")
    ax.set_yticks(range(len(GROUPS)))
    ax.set_yticklabels(yticklabels)
    ax.set_ylim(-0.55, len(GROUPS) - 0.45)
    _pct_axis(ax)
    ax.set_xlabel("% of students within section group",
                  fontsize=FONT_SIZES["axis_label"])
    ax.set_title("Grade quartile composition by section group\n"
                 "(quartiles of midterm z-scores pooled across all sections; "
                 "Q4 is best)",
                 fontsize=FONT_SIZES["title"] - 1, fontweight="bold",
                 loc="left", pad=10)

    handles = [mpatches.Patch(color=PALETTE_QUARTILE[q],
                              label=q + (" (best)" if q == "Q4" else ""))
               for q in GRADE_QUARTILE_ORDER]
    ax.legend(handles=handles, loc="upper center", ncol=4,
              bbox_to_anchor=(0.5, -0.55),
              fontsize=FONT_SIZES["legend"], frameon=False,
              handlelength=1.2, handletextpad=0.5, columnspacing=1.0)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


if __name__ == "__main__":
    plot_use_freq_by_demo(
        data_path="Spring_2026_quantitative.xlsx",
        out_path="ai_use_freq_by_demo_section.pdf",
    )
    plot_grade_by_demo(
        data_path="Spring_2026_quantitative.xlsx",
        out_path="grade_quartile_by_demo_section.pdf",
    )
