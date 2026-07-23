"""
plot_qualitative_codes.py
-------------------------
Visualise coded free-text data as a horizontal bar chart of code frequencies.
One bar per code, grouped and coloured by category (Likes / Dislikes /
Suggestions / Uncodable). Magnitude is percent of responses (a code's bar =
how many responses received that code, as a % of all coded responses).

Counts all mentions (primary + secondary + tertiary).
Colours/fonts from plot_style.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

from load_qualitative import load_qualitative
from plot_style import (
    apply_style,
    get_category_colors,
    FONT_SIZES,
    FIGURE,
)

# Min bar width (%) to print the count label inside rather than outside
INSIDE_LABEL_THRESHOLD = 6


def plot_qualitative_codes(
    base: str,
    title: str,
    out_path: str | None = None,
    search_dir: str = ".",
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    """
    Parameters
    ----------
    base       : item base name, e.g. "txt_ai_demo_feedback_post"
    title      : figure title
    out_path   : save path (optional)
    search_dir : where to find the codebook/coded CSVs
    fig_width  : figure width in inches
    """
    apply_style()

    q = load_qualitative(base, search_dir=search_dir)
    long = q["long"]
    n    = q["n_responses"]

    counts = long.groupby(["category", "code"]).size().to_dict()

    categories = q["category_order"]
    codes_by_cat = {cat: [] for cat in categories}
    for cat, code in q["code_order"]:
        if cat in codes_by_cat:
            codes_by_cat[cat].append(code)

    # Colourblind-safe colour per category (semantic where known, else cycled)
    cat_colors = get_category_colors(categories)

    # Assign y positions, grouped by category with a small gap between groups.
    # Categories are walked in reverse so the first category ends up on top.
    ytick_pos, ytick_lab = [], []
    GROUP_GAP = 0.6

    positions = []
    group_tops = {}     # category -> y of its topmost (last-plotted) bar
    cursor = 0.0
    for cat in reversed(categories):
        codes = [code for code in codes_by_cat[cat] if counts.get((cat, code), 0) > 0]
        cat_ys = []
        for code in reversed(codes):
            positions.append((cursor, cat, code))
            cat_ys.append(cursor)
            cursor += 1.0
        if cat_ys:
            group_tops[cat] = max(cat_ys)   # topmost bar of this group
        cursor += GROUP_GAP   # gap between category groups

    fig_h = 0.42 * len(positions) + 1.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))

    for y, cat, code in positions:
        c   = counts.get((cat, code), 0)
        pct = c / n * 100
        colour = cat_colors.get(cat, "#888888")
        ax.barh(y, pct, 0.72, color=colour, linewidth=0, zorder=2)

        # Count label: inside if the bar is wide enough, else just past the end
        if pct >= INSIDE_LABEL_THRESHOLD:
            ax.text(pct - 0.6, y, str(c), ha="right", va="center",
                    fontsize=FONT_SIZES["bar_label"] - 1, color="white",
                    zorder=4)
        else:
            ax.text(pct + 0.6, y, str(c), ha="left", va="center",
                    fontsize=FONT_SIZES["bar_label"] - 1, color="#444444",
                    zorder=4)

        ytick_pos.append(y)
        ytick_lab.append(code)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=FONT_SIZES["tick"] - 1)
    ax.set_ylim(-0.7, max(p[0] for p in positions) + 1.3)

    # ── Bold category headers above each group, colour-matched ────────────────
    for cat, y_top in group_tops.items():
        ax.text(
            -0.5, y_top + 0.7, cat.upper(),
            ha="left", va="center",
            fontsize=FONT_SIZES["tick"],
            fontweight="bold",
            color=cat_colors.get(cat, "#333333"),
            transform=ax.get_yaxis_transform(),
            clip_on=False,
        )

    ax.set_xlabel(f"% of responses receiving each code\n"
                  f"(n={n}; responses may receive multiple codes)",
                  fontsize=FONT_SIZES["axis_label"], loc="left")
    max_pct = max(counts.get((c, code), 0) / n * 100
                  for c, code in q["code_order"] if counts.get((c, code), 0) > 0)
    ax.set_xlim(0, max_pct + 6)
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)

    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold",
                 loc="left", pad=10)

    fig.tight_layout()

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")

    return fig


if __name__ == "__main__":
    plot_qualitative_codes(
        base="txt_ai_demo_feedback_post",
        title="AI demonstration feedback — coded themes",
        out_path="qual_demo_feedback.pdf",
    )
