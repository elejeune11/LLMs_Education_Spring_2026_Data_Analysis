"""
plot_qualitative_tiered.py
--------------------------
Two plot functions for the tiered (Tier 1 / Tier 2) pre/post qualitative format
produced by load_qualitative_tiered.py:

  plot_tier1_sentiment(...)  → 100% stacked horizontal bars (pre vs post).
      Tier 1 is mutually exclusive (one code per response), so the stack sums
      to 100% and shows the sentiment shift.

  plot_tier2_drivers(...)    → grouped horizontal bars (pre vs post per code).
      Tier 2 is select-all-that-apply, so a stack would mislead; grouped bars
      show each driver's prevalence at both timepoints.

Both use percent of responses. Colours/fonts from plot_style.py.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches

from load_qualitative_tiered import load_qualitative_tiered
from plot_style import (
    apply_style,
    PALETTE_SENTIMENT,
    COLOUR_PRE,
    COLOUR_POST,
    COLOUR_POST_DARK,
    get_category_colors,
    FONT_SIZES,
    FIGURE,
)


# ══════════════════════════════════════════════════════════════════════════════
# TIER 1 — 100% stacked sentiment (pre vs post)
# ══════════════════════════════════════════════════════════════════════════════

def plot_tier1_sentiment(
    stem: str,
    title: str,
    out_path: str | None = None,
    search_dir: str = ".",
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    apply_style()
    q = load_qualitative_tiered(stem, search_dir=search_dir)
    order = q["tier1_order"]   # sentiment codes, codebook order

    n_pre, n_post = q["pre"]["n"], q["post"]["n"]
    pre_c, post_c = q["pre"]["tier1_counts"], q["post"]["tier1_counts"]

    # One pre + one post bar per sentiment code (codebook order, top-down)
    positions = list(range(len(order)))[::-1]

    fig_h = 0.7 * len(order) + 1.8
    fig, ax = plt.subplots(figsize=(fig_width, fig_h))

    bar_h = 0.36
    ytick_pos, ytick_lab = [], []
    for y, code in zip(positions, order):
        pre_pct  = pre_c.get(code, 0)  / n_pre  * 100
        post_pct = post_c.get(code, 0) / n_post * 100
        # Consistent pre/post colours (grey vs blue) — sentiment is labelled
        # on the y-axis, so per-bar colour would only add visual noise.
        ax.barh(y + bar_h / 2 + 0.02, pre_pct,  bar_h, color=COLOUR_PRE,
                linewidth=0, zorder=2)
        ax.barh(y - bar_h / 2 - 0.02, post_pct, bar_h, color=COLOUR_POST,
                linewidth=0, zorder=2)
        ax.text(pre_pct + 0.5, y + bar_h / 2 + 0.02, f"{pre_c.get(code,0)}",
                ha="left", va="center", fontsize=FONT_SIZES["bar_label"] - 2,
                color="#666666")
        ax.text(post_pct + 0.5, y - bar_h / 2 - 0.02, f"{post_c.get(code,0)}",
                ha="left", va="center", fontsize=FONT_SIZES["bar_label"] - 2,
                color="#666666")
        ytick_pos.append(y)
        ytick_lab.append(code)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=FONT_SIZES["tick"])
    ax.set_ylim(-0.8, max(positions) + 0.8)

    max_pct = max(
        max(pre_c.get(c, 0) / n_pre * 100, post_c.get(c, 0) / n_post * 100)
        for c in order
    )
    ax.set_xlim(0, max_pct + 6)
    ax.set_xlabel("% of responses (each response has exactly one sentiment)",
                  fontsize=FONT_SIZES["axis_label"], loc="left")
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold",
                 loc="left", pad=10)

    # Pre/Post legend
    handles = [
        mpatches.Patch(color=COLOUR_PRE,  label=f"Pre (n={n_pre})"),
        mpatches.Patch(color=COLOUR_POST, label=f"Post (n={n_post})"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=FONT_SIZES["legend"],
              frameon=False, handlelength=1.2, handletextpad=0.5,
              title="Timepoint", title_fontsize=FONT_SIZES["legend"])

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# TIER 2 — grouped pre/post driver prevalence
# ══════════════════════════════════════════════════════════════════════════════

def plot_tier2_drivers(
    stem: str,
    title: str,
    out_path: str | None = None,
    search_dir: str = ".",
    fig_width: float = FIGURE["width"],
) -> plt.Figure:
    apply_style()
    q = load_qualitative_tiered(stem, search_dir=search_dir)

    tier2_order = q["tier2_order"]            # list of (category, code)
    categories  = q["tier2_categories"]
    cat_colors  = get_category_colors(categories)

    n_pre, n_post = q["pre"]["n"], q["post"]["n"]
    pre_c, post_c = q["pre"]["tier2_counts"], q["post"]["tier2_counts"]

    # Keep only codes used at either timepoint, in codebook order
    codes = [(cat, code) for cat, code in tier2_order
             if pre_c.get(code, 0) > 0 or post_c.get(code, 0) > 0]

    # Layout: two bars (pre/post) per code, grouped by Tier 2 category
    positions = []   # (y, category, code)
    cursor = 0.0
    GROUP_GAP = 0.8
    group_tops = {}
    for cat in reversed(categories):
        cat_codes = [c for (cc, c) in codes if cc == cat]
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
        pre_pct  = pre_c.get(code, 0)  / n_pre  * 100
        post_pct = post_c.get(code, 0) / n_post * 100
        # Pre = lighter (grey), Post = category colour
        ax.barh(y + bar_h / 2 + 0.02, pre_pct,  bar_h, color=COLOUR_PRE,
                linewidth=0, zorder=2)
        ax.barh(y - bar_h / 2 - 0.02, post_pct, bar_h,
                color=cat_colors.get(cat, "#888888"), linewidth=0, zorder=2)
        # Count labels just past bar ends
        ax.text(pre_pct + 0.5, y + bar_h / 2 + 0.02, str(pre_c.get(code, 0)),
                ha="left", va="center", fontsize=FONT_SIZES["bar_label"] - 2,
                color="#666666")
        ax.text(post_pct + 0.5, y - bar_h / 2 - 0.02, str(post_c.get(code, 0)),
                ha="left", va="center", fontsize=FONT_SIZES["bar_label"] - 2,
                color="#666666")
        ytick_pos.append(y)
        ytick_lab.append(code)

    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lab, fontsize=FONT_SIZES["tick"] - 1)
    ax.set_ylim(-0.8, max(p[0] for p in positions) + 1.2)

    # Category headers
    for cat, y_top in group_tops.items():
        ax.text(-0.5, y_top + 0.75, cat.upper(), ha="left", va="center",
                fontsize=FONT_SIZES["tick"] - 1, fontweight="bold",
                color=cat_colors.get(cat, "#333333"),
                transform=ax.get_yaxis_transform(), clip_on=False)

    max_pct = max(
        max(pre_c.get(c, 0) / n_pre * 100, post_c.get(c, 0) / n_post * 100)
        for _, c in codes
    )
    ax.set_xlim(0, max_pct + 6)
    ax.set_xlabel("% of responses (responses may have multiple drivers)",
                  fontsize=FONT_SIZES["axis_label"], loc="left")
    ax.xaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for sp in ["top", "right", "left"]:
        ax.spines[sp].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold",
                 loc="left", pad=10)

    # Pre/Post legend
    handles = [
        mpatches.Patch(color=COLOUR_PRE,  label=f"Pre (n={n_pre})"),
        mpatches.Patch(color=COLOUR_POST, label=f"Post (n={n_post})"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=FONT_SIZES["legend"],
              frameon=False, handlelength=1.2, handletextpad=0.5,
              title="Timepoint  (Post bars use category colour)",
              title_fontsize=FONT_SIZES["legend"] - 1)

    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# COMBINED — Tier 1 sentiment (top) + Tier 2 drivers (bottom) in one figure
# ══════════════════════════════════════════════════════════════════════════════

def plot_tiered_combined(
    stem: str,
    out_path: str | None = None,
    search_dir: str = ".",
    fig_width: float = FIGURE["width"],
    suptitle: str = (
        "How would you describe the impact of AI tools (positive and negative)\n"
        "on your learning in this course?  (open-ended, pre vs. post)"
    ),
    panel1_title: str = "Overall sentiment  (each response coded with exactly one)",
    panel2_title: str = "Drivers of sentiment",
    panel2_xlabel: str = "% of responses (responses may reflect multiple drivers)",
) -> plt.Figure:
    """
    Single figure with two stacked panels for a tiered pre/post item:
      Top    — Tier 1 overall sentiment (grey = pre, dark = post)
      Bottom — Tier 2 drivers of sentiment, grouped by category
               (grey = pre, category colour = post)
    Both panels share the same x-axis (% of responses) and pre/post encoding.
    """
    apply_style()
    q = load_qualitative_tiered(stem, search_dir=search_dir)

    n_pre, n_post = q["pre"]["n"], q["post"]["n"]

    # ── Tier 1 data ───────────────────────────────────────────────────────────
    t1_order  = q["tier1_order"]
    t1_pre    = q["pre"]["tier1_counts"]
    t1_post   = q["post"]["tier1_counts"]

    # ── Tier 2 data (used codes only, codebook order, grouped by category) ────
    t2_order   = q["tier2_order"]
    t2_cats    = q["tier2_categories"]
    cat_colors = get_category_colors(t2_cats)
    t2_pre     = q["pre"]["tier2_counts"]
    t2_post    = q["post"]["tier2_counts"]
    t2_used    = [(cat, code) for cat, code in t2_order
                  if t2_pre.get(code, 0) > 0 or t2_post.get(code, 0) > 0]

    # ── Figure layout: height ratio scales with number of bar groups ──────────
    n_t1 = len(t1_order)
    n_t2 = len(t2_used)
    n_t2_groups = len(t2_cats)
    # Approx vertical "slots" each panel needs (codes + a bit for headers)
    t1_slots = n_t1
    t2_slots = n_t2 + 0.6 * n_t2_groups
    fig_h = 0.55 * (t1_slots + t2_slots) + 2.6

    fig = plt.figure(figsize=(fig_width, fig_h))
    gs = gridspec.GridSpec(
        2, 1, figure=fig,
        height_ratios=[t1_slots, t2_slots],
        hspace=0.28,
        top=0.86, bottom=0.10, left=0.30, right=0.97,
    )
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    bar_h = 0.36

    # Shared x max across both panels for visual comparability
    t1_max = max(max(t1_pre.get(c, 0) / n_pre * 100,
                     t1_post.get(c, 0) / n_post * 100) for c in t1_order)
    t2_max = max(max(t2_pre.get(c, 0) / n_pre * 100,
                     t2_post.get(c, 0) / n_post * 100) for _, c in t2_used)
    x_max = max(t1_max, t2_max) + 6

    # ── Panel 1: Tier 1 sentiment (grey/dark) ─────────────────────────────────
    positions1 = list(range(n_t1))[::-1]
    yt1, yl1 = [], []
    for y, code in zip(positions1, t1_order):
        pre_pct  = t1_pre.get(code, 0)  / n_pre  * 100
        post_pct = t1_post.get(code, 0) / n_post * 100
        ax1.barh(y + bar_h/2 + 0.02, pre_pct,  bar_h, color=COLOUR_PRE,
                 linewidth=0, zorder=2)
        ax1.barh(y - bar_h/2 - 0.02, post_pct, bar_h, color=COLOUR_POST_DARK,
                 linewidth=0, zorder=2)
        ax1.text(pre_pct + 0.5, y + bar_h/2 + 0.02, f"{t1_pre.get(code,0)}",
                 ha="left", va="center", fontsize=FONT_SIZES["bar_label"]-2, color="#666")
        ax1.text(post_pct + 0.5, y - bar_h/2 - 0.02, f"{t1_post.get(code,0)}",
                 ha="left", va="center", fontsize=FONT_SIZES["bar_label"]-2, color="#666")
        yt1.append(y); yl1.append(code)
    ax1.set_yticks(yt1); ax1.set_yticklabels(yl1, fontsize=FONT_SIZES["tick"]-1)
    ax1.set_ylim(-0.7, max(positions1) + 0.7)
    ax1.set_xlim(0, x_max)
    ax1.set_title(panel1_title,
                  fontsize=FONT_SIZES["tick"]+1, fontweight="semibold",
                  loc="left", color="#222", pad=6)
    ax1.xaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax1.set_axisbelow(True)
    for sp in ["top","right","left"]: ax1.spines[sp].set_visible(False)
    ax1.tick_params(axis="y", length=0)

    # Tier 1 pre/post legend (grey/dark)
    h1 = [mpatches.Patch(color=COLOUR_PRE, label=f"Pre (n={n_pre})"),
          mpatches.Patch(color=COLOUR_POST_DARK, label=f"Post (n={n_post})")]
    ax1.legend(handles=h1, loc="lower right", fontsize=FONT_SIZES["legend"]-1,
               frameon=False, handlelength=1.2, handletextpad=0.5, title="Timepoint",
               title_fontsize=FONT_SIZES["legend"]-1)

    # ── Panel 2: Tier 2 drivers (grey/category colour), grouped ───────────────
    positions2 = []
    cursor = 0.0
    GROUP_GAP = 0.8
    group_tops = {}
    for cat in reversed(t2_cats):
        cat_codes = [c for (cc, c) in t2_used if cc == cat]
        ys = []
        for code in reversed(cat_codes):
            positions2.append((cursor, cat, code)); ys.append(cursor); cursor += 1.0
        if ys: group_tops[cat] = max(ys)
        cursor += GROUP_GAP

    yt2, yl2 = [], []
    for y, cat, code in positions2:
        pre_pct  = t2_pre.get(code, 0)  / n_pre  * 100
        post_pct = t2_post.get(code, 0) / n_post * 100
        ax2.barh(y + bar_h/2 + 0.02, pre_pct,  bar_h, color=COLOUR_PRE,
                 linewidth=0, zorder=2)
        ax2.barh(y - bar_h/2 - 0.02, post_pct, bar_h,
                 color=cat_colors.get(cat, "#888"), linewidth=0, zorder=2)
        ax2.text(pre_pct + 0.5, y + bar_h/2 + 0.02, f"{t2_pre.get(code,0)}",
                 ha="left", va="center", fontsize=FONT_SIZES["bar_label"]-2, color="#666")
        ax2.text(post_pct + 0.5, y - bar_h/2 - 0.02, f"{t2_post.get(code,0)}",
                 ha="left", va="center", fontsize=FONT_SIZES["bar_label"]-2, color="#666")
        yt2.append(y); yl2.append(code)
    ax2.set_yticks(yt2); ax2.set_yticklabels(yl2, fontsize=FONT_SIZES["tick"]-1)
    ax2.set_ylim(-0.8, max(p[0] for p in positions2) + 1.0)
    ax2.set_xlim(0, x_max)

    for cat, y_top in group_tops.items():
        ax2.text(-0.5, y_top + 0.72, cat.upper(), ha="left", va="center",
                 fontsize=FONT_SIZES["tick"]-2, fontweight="bold",
                 color=cat_colors.get(cat, "#333"),
                 transform=ax2.get_yaxis_transform(), clip_on=False)

    ax2.set_title(panel2_title,
                  fontsize=FONT_SIZES["tick"]+1, fontweight="semibold",
                  loc="left", color="#222", pad=6)
    ax2.set_xlabel(panel2_xlabel,
                   fontsize=FONT_SIZES["axis_label"], loc="left")
    ax2.xaxis.grid(True, color="#e8e8e8", linewidth=0.6, zorder=0)
    ax2.set_axisbelow(True)
    for sp in ["top","right","left"]: ax2.spines[sp].set_visible(False)
    ax2.tick_params(axis="y", length=0)

    # Pre/post legend for panel 2 (grey vs neutral dark swatch, since post
    # colour varies by category)
    h2 = [mpatches.Patch(color=COLOUR_PRE, label=f"Pre (n={n_pre})"),
          mpatches.Patch(facecolor="white", edgecolor="#666",
                         label=f"Post (n={n_post}) — category color")]
    ax2.legend(handles=h2, loc="lower right", fontsize=FONT_SIZES["legend"]-1,
               frameon=False, handlelength=1.2, handletextpad=0.5, title="Timepoint",
               title_fontsize=FONT_SIZES["legend"]-1)

    # ── Figure title ──────────────────────────────────────────────────────────
    fig.suptitle(
        suptitle,
        fontsize=FONT_SIZES["title"], fontweight="bold", x=0.5, y=0.975,
    )

    if out_path:
        fig.savefig(out_path, bbox_inches="tight")
        print(f"Saved: {out_path}")
    return fig


# Backwards-compatible alias: existing callers use plot_impact_combined
def plot_impact_combined(*args, **kwargs):
    return plot_tiered_combined(*args, **kwargs)


if __name__ == "__main__":
    plot_impact_combined(
        stem="txt_ai_impact_describe",
        out_path="qual_impact_combined.pdf",
    )
