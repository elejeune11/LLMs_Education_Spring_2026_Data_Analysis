"""
plot_diverging_likert.py
------------------------
Diverging stacked bar chart (Likert-style) with pre/post rows per question.

Single-axes approach: all bars drawn on one axes with manual y positioning.
X-axis tick labels repeated at top and bottom of each question group via
annotation. Centre line drawn only within each question group's span.

All colours, fonts, and spacing come from plot_style.py.
"""

import textwrap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from load_research_data import load_data, LIKERT_ORDER
from question_metadata import STEM_LABELS
from plot_style import (
    apply_style,
    PALETTE_LIKERT,
    FONT_SIZES,
    FIGURE,
    likert_legend_handles,
)

# ── Layout constants (all in data / figure units) ─────────────────────────────
BAR_H           = 0.35    # bar thickness in data units
PRE_OFFSET      = 0.65    # y offset of pre bar above group baseline
POST_OFFSET     = 0.0     # y offset of post bar above group baseline
GROUP_H         = 1.9     # total y height allocated per question group
LABEL_PAD       = 0.14    # gap between top bar and question label
LABEL_WRAP_COLS = 95      # wrap question labels longer than this many characters
LABEL_LINE_H    = 0.28    # extra data-unit height per additional wrapped line


def _pcts(series: pd.Series) -> dict[str, float]:
    counts = series.value_counts(dropna=True)
    total  = counts.sum()
    return {cat: counts.get(cat, 0) / total * 100 for cat in LIKERT_ORDER}


def _draw_bar(ax, y_center, pcts, bar_h):
    """Draw one diverging bar centred at y_center."""
    neg_total = (sum(pcts[c] for c in ["Strongly Disagree", "Disagree"])
                 + pcts["Neither Agree nor Disagree"] / 2)
    left = -neg_total
    for cat in LIKERT_ORDER:
        w = pcts[cat]
        if w == 0:
            continue
        if cat == "Neither Agree nor Disagree":
            ax.barh(y_center, w / 2, bar_h, left=left,
                    color=PALETTE_LIKERT[cat], linewidth=0, zorder=2)
            ax.barh(y_center, w / 2, bar_h, left=left + w / 2,
                    color=PALETTE_LIKERT[cat], linewidth=0, zorder=2)
        else:
            ax.barh(y_center, w, bar_h, left=left,
                    color=PALETTE_LIKERT[cat], linewidth=0, zorder=2)
        left += w


def plot_diverging_likert(
    data_path: str,
    col_stems: list[str],
    title: str,
    out_path: str | None = None,
    question_labels: list[str] | None = None,
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    """
    Parameters
    ----------
    data_path       : path to the xlsx file
    col_stems       : column stem names (without _pre/_post)
    title           : chart title
    out_path        : save path (optional)
    question_labels : overrides STEM_LABELS if provided
    fig_width       : figure width in inches
    """
    apply_style()

    if question_labels is None:
        question_labels = [STEM_LABELS.get(s, s) for s in col_stems]

    # Wrap labels that exceed LABEL_WRAP_COLS characters
    question_labels = [
        textwrap.fill(lbl, width=LABEL_WRAP_COLS) for lbl in question_labels
    ]

    assert len(col_stems) == len(question_labels)

    data = load_data(data_path)
    df   = data["all"]
    n_q  = len(col_stems)

    # Group height expands for wrapped labels (each extra line adds LABEL_LINE_H)
    group_heights = [
        GROUP_H + max(0, lbl.count("\n")) * LABEL_LINE_H
        for lbl in question_labels
    ]
    total_y = sum(group_heights)

    # Fixed header height in inches (title + legend); scales subplots_adjust.
    HEADER_H    = 0.7
    fig_h       = total_y + HEADER_H
    header_frac = 1.0 - HEADER_H / fig_h

    fig, ax = plt.subplots(figsize=(fig_width, fig_h))

    # Y runs top-to-bottom: first question at top. The cursor starts at the
    # correct total height (sum of per-group heights, which already accounts
    # for taller groups whose labels wrapped) and walks downward.
    xtick_positions = [-100, -50, 0, 50, 100]
    xtick_labels    = ["100%", "50%", "0", "50%", "100%"]

    # Cumulative y baseline from top down
    y_cursor = total_y
    for i, (stem, qlabel) in enumerate(zip(col_stems, question_labels)):
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
                -101, y_c, f"{tag} (n={n})",
                ha="right", va="center",
                fontsize=FONT_SIZES["tick"],
            )

        # ── Question label above pre bar ──────────────────────────────────────
        ax.text(
            -100, pre_y + BAR_H / 2 + LABEL_PAD,
            qlabel,
            ha="left", va="bottom",
            fontsize=FONT_SIZES["tick"] + 1,
            fontweight="semibold",
            color="#222222",
            zorder=5,
        )

        # ── Centre line scoped to this group only ─────────────────────────────
        group_bottom = post_y - BAR_H / 2 - 0.05
        group_top    = pre_y  + BAR_H / 2 + 0.05
        ax.plot([0, 0], [group_bottom, group_top],
                color="#444444", linewidth=0.8, zorder=3)

        # ── X-axis tick marks and labels at bottom of each group ─────────────
        tick_y = post_y - BAR_H / 2 - 0.18
        for xval, xlabel in zip(xtick_positions, xtick_labels):
            ax.plot([xval, xval], [tick_y, tick_y + 0.08],
                    color="#aaaaaa", linewidth=0.6, zorder=1)
            ax.text(xval, tick_y - 0.05, xlabel,
                    ha="center", va="top",
                    fontsize=FONT_SIZES["tick"] - 1,
                    color="#555555")
        # Horizontal axis line
        ax.plot([-100, 100], [tick_y, tick_y],
                color="#aaaaaa", linewidth=0.6, zorder=1)

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(-130, 105)
    ax.set_ylim(-0.35, total_y + 0.3)
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

    # ── Legend below title — position scales with figure height ───────────────
    legend_frac = header_frac + (1.0 - header_frac) * 0.35
    fig.legend(
        handles=likert_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.55, legend_frac),
        ncol=len(PALETTE_LIKERT),
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
    plot_diverging_likert(
        data_path=str(data_path(QUANT_FILE)),
        col_stems=[
            "understand_apply_principles",
            "understand_identify_mistakes",
            "understand_until_practice",
            "understand_exams_reflect",
        ],
        title="Self-rated understanding of engineering mechanics",
        out_path="desc_page1_understanding.pdf",
    )
