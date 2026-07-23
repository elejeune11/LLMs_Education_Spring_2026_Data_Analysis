"""
plot_qualitative_pre_post.py
----------------------------
Pre/post comparison plot for FLAT (multi-select, non-tiered) qualitative
items that were asked at both timepoints (e.g. txt_ai_verify_correctness).

Layout mirrors plot_tier2_drivers in plot_qualitative_tiered.py: one group of
two horizontal bars (pre = grey, post = category colour) per code, grouped by
codebook category with bold colour-matched headers. Magnitude is percent of
responses at that timepoint; count labels sit past the bar ends.

Expects the two flat bases "{stem}_pre" and "{stem}_post" to exist in the
load_qualitative.py format ({base}_qualitative_codebook.csv, {base}_coded.csv);
export_for_plots.py produces these. Colours/fonts from plot_style.py.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from load_qualitative import load_qualitative
from plot_style import (
    apply_style,
    get_category_colors,
    COLOUR_PRE,
    FONT_SIZES,
    FIGURE,
)


def plot_qualitative_pre_post(
    stem: str,
    title: str,
    out_path: str | None = None,
    search_dir: str = ".",
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    """
    Parameters
    ----------
    stem       : item stem WITHOUT the _pre/_post suffix,
                 e.g. "qualitative_coding/txt_ai_verify_correctness"
    title      : figure title
    out_path   : save path (optional)
    search_dir : where to find the codebook/coded CSVs
    fig_width  : figure width in inches
    """
    apply_style()

    q_pre = load_qualitative(f"{stem}_pre", search_dir=search_dir)
    q_post = load_qualitative(f"{stem}_post", search_dir=search_dir)

    n_pre, n_post = q_pre["n_responses"], q_post["n_responses"]
    pre_counts = q_pre["long"].groupby(["category", "code"]).size().to_dict()
    post_counts = q_post["long"].groupby(["category", "code"]).size().to_dict()

    # Codebook order and categories come from the (shared) pre codebook
    categories = q_pre["category_order"]
    code_order = q_pre["code_order"]
    cat_colors = get_category_colors(categories)

    # Keep only codes used at either timepoint, in codebook order
    used = [(cat, code) for cat, code in code_order
            if pre_counts.get((cat, code), 0) > 0
            or post_counts.get((cat, code), 0) > 0]

    # Layout: two bars (pre/post) per code, grouped by category, first
    # category on top (built bottom-up)
    positions = []
    cursor = 0.0
    GROUP_GAP = 0.8
    group_tops = {}
    for cat in reversed(categories):
        cat_codes = [code for (cc, code) in used if cc == cat]
        ys = []
        for code in reversed(cat_codes):
            positions.append((cursor, cat, code))
            ys.append(cursor)
            cursor += 1.0
        if ys:
            group_tops[cat] = max(ys)
        cursor += GROUP_GAP

    fig_h = 0.55 * len(positions) + 1.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))

    bar_h = 0.36
    ytick_pos, ytick_lab = [], []
    for y, cat, code in positions:
        c_pre = pre_counts.get((cat, code), 0)
        c_post = post_counts.get((cat, code), 0)
        pre_pct = c_pre / n_pre * 100
        post_pct = c_post / n_post * 100
        ax.barh(y + bar_h / 2 + 0.02, pre_pct, bar_h, color=COLOUR_PRE,
                linewidth=0, zorder=2)
        ax.barh(y - bar_h / 2 - 0.02, post_pct, bar_h,
                color=cat_colors.get(cat, "#888888"), linewidth=0, zorder=2)
        ax.text(pre_pct + 0.5, y + bar_h / 2 + 0.02, str(c_pre),
                ha="left", va="center", fontsize=FONT_SIZES["bar_label"] - 2,
                color="#666666")
        ax.text(post_pct + 0.5, y - bar_h / 2 - 0.02, str(c_post),
                ha="left", va="center", fontsize=FONT_SIZES["bar_label"] - 2,
                color="#666666")
        ytick_pos.append(y)
        ytick_lab.append(code)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=FONT_SIZES["tick"] - 1)
    ax.set_ylim(-0.8, max(p[0] for p in positions) + 1.2)

    for cat, y_top in group_tops.items():
        ax.text(-0.5, y_top + 0.75, cat.upper(), ha="left", va="center",
                fontsize=FONT_SIZES["tick"] - 1, fontweight="bold",
                color=cat_colors.get(cat, "#333333"),
                transform=ax.get_yaxis_transform(), clip_on=False)

    max_pct = max(
        max(pre_counts.get((cat, c), 0) / n_pre * 100,
            post_counts.get((cat, c), 0) / n_post * 100)
        for cat, c in used
    )
    ax.set_xlim(0, max_pct + 6)
    ax.set_xlabel("% of responses (responses may receive multiple codes)",
                  fontsize=FONT_SIZES["axis_label"], loc="left")
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold",
                 loc="left", pad=10)

    handles = [
        mpatches.Patch(color=COLOUR_PRE, label=f"Pre (n={n_pre})"),
        mpatches.Patch(facecolor="white", edgecolor="#666666",
                       label=f"Post (n={n_post}) — category color"),
    ]
    ax.legend(handles=handles, loc="lower right",
              fontsize=FONT_SIZES["legend"] - 1, frameon=False,
              handlelength=1.2, handletextpad=0.5, title="Timepoint",
              title_fontsize=FONT_SIZES["legend"] - 1)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


if __name__ == "__main__":
    plot_qualitative_pre_post(
        stem="qualitative_coding/txt_ai_verify_correctness",
        title="How students check AI-generated responses — coded, pre vs. post",
        out_path="qual_verify_pre_post.pdf",
    )
