"""
plot_stacked_freq.py
--------------------
Stacked frequency bar chart for ordinal frequency items, with pre/post rows
per question. Bars are left-aligned at 0%, x-axis runs 0-100%, and percentage
labels appear inside segments that are wide enough to fit them.

Single-axes approach: all bars drawn on one axes with manual y positioning,
same structural pattern as plot_diverging_likert.py.

All colours, fonts, and spacing come from plot_style.py.
"""

import textwrap
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_research_data import load_data, FREQ_ORDER
from question_metadata import STEM_LABELS
from plot_style import (
    apply_style,
    PALETTE_FREQ,
    FONT_SIZES,
    FIGURE,
    freq_legend_handles,
)

# ── Layout constants (data units) ─────────────────────────────────────────────
BAR_H           = 0.35   # bar thickness
PRE_OFFSET      = 0.65   # y offset of pre bar above group baseline
POST_OFFSET     = 0.0    # y offset of post bar above group baseline
GROUP_H         = 1.9    # total y height per question group
LABEL_PAD       = 0.14   # gap between top bar and panel title
LABEL_WRAP_COLS = 95     # wrap panel titles longer than this
LABEL_LINE_H    = 0.28   # extra height per additional wrapped line

PCT_LABEL_THRESHOLD = 5  # show % label only when segment ≥ this many percent


def _pcts(series: pd.Series) -> dict[str, float]:
    counts = series.value_counts(dropna=True)
    total  = counts.sum()
    return {cat: counts.get(cat, 0) / total * 100 for cat in FREQ_ORDER}


def _draw_bar(ax, y_center, pcts, bar_h):
    """Draw one left-aligned stacked frequency bar with inline % labels."""
    left = 0.0
    for cat in FREQ_ORDER:
        w = pcts[cat]
        if w == 0:
            continue
        ax.barh(y_center, w, bar_h, left=left,
                color=PALETTE_FREQ[cat], linewidth=0, zorder=2)

        # Inline percentage label, only if segment is wide enough
        if w >= PCT_LABEL_THRESHOLD:
            # White text on darker segments, dark text on lighter ones
            dark_cats = {"1-3 times per month", "1-5 times per week",
                         "Daily or almost daily"}
            text_color = "white" if cat in dark_cats else "#333333"
            ax.text(
                left + w / 2, y_center,
                f"{int(round(w))}%",
                ha="center", va="center",
                fontsize=FONT_SIZES["bar_label"],
                fontweight="semibold",
                color=text_color,
                zorder=4,
            )
        left += w


def plot_stacked_freq(
    data_path: str,
    col_stems: list[str],
    title: str,
    out_path: str | None = None,
    panel_titles: list[str] | None = None,
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    """
    Parameters
    ----------
    data_path     : path to the xlsx file
    col_stems     : column stem names (without _pre/_post)
    title         : figure title
    out_path      : save path (optional)
    panel_titles  : per-question panel titles; defaults to STEM_LABELS if None
    fig_width     : figure width in inches
    """
    apply_style()

    if panel_titles is None:
        panel_titles = [STEM_LABELS.get(s, s) for s in col_stems]

    # Wrap long panel titles
    panel_titles = [
        textwrap.fill(t, width=LABEL_WRAP_COLS) for t in panel_titles
    ]

    assert len(col_stems) == len(panel_titles)

    data = load_data(data_path)
    df   = data["all"]
    n_q  = len(col_stems)

    # Per-group heights expand for wrapped titles
    group_heights = [
        GROUP_H + max(0, t.count("\n")) * LABEL_LINE_H
        for t in panel_titles
    ]
    total_y = sum(group_heights)

    HEADER_H    = 0.7
    fig_h       = total_y + HEADER_H
    header_frac = 1.0 - HEADER_H / fig_h

    fig, ax = plt.subplots(figsize=(fig_width, fig_h))

    xtick_positions = [0, 25, 50, 75, 100]
    xtick_labels    = ["0", "25%", "50%", "75%", "100%"]

    # Top-down layout
    y_cursor = total_y
    for i, (stem, ptitle) in enumerate(zip(col_stems, panel_titles)):
        gh     = group_heights[i]
        y_base = y_cursor - gh
        y_cursor = y_base

        pre_y  = y_base + PRE_OFFSET  + BAR_H / 2 + 0.2
        post_y = y_base + POST_OFFSET + BAR_H / 2 + 0.2

        # ── Bars ──────────────────────────────────────────────────────────────
        for col, y_c in [(stem + "_pre", pre_y), (stem + "_post", post_y)]:
            series = df[col].dropna()
            _draw_bar(ax, y_c, _pcts(series), BAR_H)

        # ── Row labels (Pre/Post with n) ──────────────────────────────────────
        for col, y_c, tag in [
            (stem + "_pre",  pre_y,  "Pre"),
            (stem + "_post", post_y, "Post"),
        ]:
            n = len(df[col].dropna())
            ax.text(
                -1, y_c, f"{tag} (n={n})",
                ha="right", va="center",
                fontsize=FONT_SIZES["tick"],
            )

        # ── Panel title above pre bar ─────────────────────────────────────────
        ax.text(
            0, pre_y + BAR_H / 2 + LABEL_PAD,
            ptitle,
            ha="left", va="bottom",
            fontsize=FONT_SIZES["tick"] + 1,
            fontweight="semibold",
            color="#222222",
            zorder=5,
        )

        # ── X-axis tick marks and labels at bottom of each group ─────────────
        tick_y = post_y - BAR_H / 2 - 0.18
        for xval, xlabel in zip(xtick_positions, xtick_labels):
            ax.plot([xval, xval], [tick_y, tick_y + 0.08],
                    color="#aaaaaa", linewidth=0.6, zorder=1)
            ax.text(xval, tick_y - 0.05, xlabel,
                    ha="center", va="top",
                    fontsize=FONT_SIZES["tick"] - 1,
                    color="#555555")
        ax.plot([0, 100], [tick_y, tick_y],
                color="#aaaaaa", linewidth=0.6, zorder=1)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(-22, 102)   # extra left room for row labels
    ax.set_ylim(0, total_y + 0.3)
    ax.axis("off")

    # ── Title ─────────────────────────────────────────────────────────────────
    fig.text(
        0.5, 0.995,
        title,
        ha="center", va="top",
        fontsize=FONT_SIZES["title"],
        fontweight="bold",
        transform=fig.transFigure,
    )

    # ── Legend below title ────────────────────────────────────────────────────
    legend_frac = header_frac + (1.0 - header_frac) * 0.35
    fig.legend(
        handles=freq_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.55, legend_frac),
        ncol=len(PALETTE_FREQ),
        fontsize=FONT_SIZES["legend"],
        frameon=False,
        handlelength=1.2,
        handletextpad=0.5,
        columnspacing=1.0,
    )

    fig.subplots_adjust(left=0.01, right=0.99, top=header_frac, bottom=0.02)

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


if __name__ == "__main__":
    from load_research_data import data_path, QUANT_FILE
    plot_stacked_freq(
        data_path=str(data_path(QUANT_FILE)),
        col_stems=["ai_use_freq_general", "ai_use_freq_engmech"],
        panel_titles=[
            "AI use frequency — general (any purpose)",
            "AI use frequency — for engineering mechanics specifically",
        ],
        title="AI use frequency: pre vs. post",
        out_path="desc_page2_frequency.pdf",
    )
