"""
plot_paired_histogram.py
------------------------
Paired pre/post histogram for a numeric 0-N rating item (e.g. the 0-10
"impact of AI tools" rating). Two stacked panels share a common x-axis:
pre on top, post below. Y-axis is percentage of valid responses (so the
differing pre/post n are comparable). A dashed mean marker is drawn on each
panel.

Single-figure, two-axes (stacked) layout. Colours/fonts from plot_style.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from load_research_data import load_data
from question_metadata import STEM_LABELS
from plot_style import (
    apply_style,
    COLOUR_PRE,
    COLOUR_POST,
    FONT_SIZES,
    FIGURE,
)


def _pct_by_value(series: pd.Series, values: list[int]) -> np.ndarray:
    """Return percentage of valid responses at each integer value."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    total = len(s)
    counts = s.round().astype(int).value_counts()
    return np.array([counts.get(v, 0) / total * 100 for v in values])


def plot_paired_histogram(
    data_path: str,
    stem: str,
    title: str,
    out_path: str | None = None,
    scale_min: int = 0,
    scale_max: int = 10,
    panel_title: str | None = None,
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    """
    Parameters
    ----------
    data_path    : path to the xlsx file
    stem         : column stem (without _pre/_post); columns must be numeric
    title        : figure title
    out_path     : save path (optional)
    scale_min    : minimum rating value (default 0)
    scale_max    : maximum rating value (default 10)
    panel_title  : question text shown above the panels; defaults to STEM_LABELS
    fig_width    : figure width in inches
    """
    apply_style()

    if panel_title is None:
        panel_title = STEM_LABELS.get(stem, stem)

    data = load_data(data_path)
    df   = data["all"]

    pre_col, post_col = stem + "_pre", stem + "_post"
    values = list(range(scale_min, scale_max + 1))

    pre_series  = pd.to_numeric(df[pre_col],  errors="coerce").dropna()
    post_series = pd.to_numeric(df[post_col], errors="coerce").dropna()

    pre_pct  = _pct_by_value(df[pre_col],  values)
    post_pct = _pct_by_value(df[post_col], values)

    n_pre,  n_post  = len(pre_series), len(post_series)
    mean_pre, mean_post = pre_series.mean(), post_series.mean()

    # Shared y max with a little headroom
    y_max = max(pre_pct.max(), post_pct.max()) * 1.18

    fig, (ax_pre, ax_post) = plt.subplots(
        2, 1, figsize=(fig_width, 5.0), sharex=True
    )

    for ax, pct, colour, tag, n, mean_val in [
        (ax_pre,  pre_pct,  COLOUR_PRE,  "Pre",  n_pre,  mean_pre),
        (ax_post, post_pct, COLOUR_POST, "Post", n_post, mean_post),
    ]:
        ax.bar(values, pct, width=0.8, color=colour, linewidth=0, zorder=2)

        # Mean marker — dashed vertical line + label
        ax.axvline(mean_val, color="#d62728", linestyle="--",
                   linewidth=1.4, zorder=4)
        ax.text(
            mean_val, y_max * 0.92,
            f"mean = {mean_val:.2f}",
            ha="left" if mean_val < scale_max * 0.7 else "right",
            va="top",
            fontsize=FONT_SIZES["annotation"],
            color="#d62728",
            fontweight="semibold",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.8),
        )

        # Panel row label (Pre/Post with n)
        ax.text(
            0.005, 0.92, f"{tag} (n={n})",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=FONT_SIZES["tick"],
            fontweight="semibold",
        )

        ax.set_ylim(0, y_max)
        ax.set_ylabel("% of responses", fontsize=FONT_SIZES["axis_label"])
        ax.yaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)

    # X axis (shared)
    ax_post.set_xticks(values)
    ax_post.set_xlim(scale_min - 0.6, scale_max + 0.6)
    ax_post.set_xlabel(
        "Rating  (0 = very negative   →   10 = very positive)",
        fontsize=FONT_SIZES["axis_label"],
    )

    # Panel title above the top axes
    ax_pre.set_title(panel_title, fontsize=FONT_SIZES["tick"] + 1,
                     fontweight="semibold", loc="left", color="#222222",
                     pad=8)

    # Figure title
    fig.suptitle(title, fontsize=FONT_SIZES["title"], fontweight="bold",
                 x=0.5, y=0.99)

    fig.tight_layout(rect=[0, 0, 1, 0.96])

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


if __name__ == "__main__":
    from load_research_data import data_path, QUANT_FILE
    plot_paired_histogram(
        data_path=str(data_path(QUANT_FILE)),
        stem="ai_impact_rating",
        title="Perceived impact of AI tools on learning",
        out_path="desc_page4_impact.pdf",
    )
