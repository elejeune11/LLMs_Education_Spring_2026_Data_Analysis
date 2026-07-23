"""
plot_style.py
-------------
Central style configuration for all EK301 survey figures.

Every plotting script imports from here. To update the look of all figures
at once, edit this file only — no need to touch individual plot scripts.

STRUCTURE
---------
  1. Typography
  2. Layout / spacing defaults
  3. Colour palettes  ← most likely things to tweak
  4. rcParams application
  5. Convenience helpers

COLOURBLIND SAFETY
------------------
All palettes are designed to be distinguishable under the three most common
forms of colour vision deficiency (deuteranopia, protanopia, tritanopia).

  - Diverging Likert  : blue ↔ orange, grey neutral (avoids red/green axis)
  - Frequency         : single-hue blue sequential ramp (safe for all CVD)
  - Qualitative codes : Okabe-Ito cycle, with fixed semantic colours where known
  - Tier 1 sentiment  : diverging blue ↔ orange, grey neutral

To preview palettes: run this file directly  →  python plot_style.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# 1. TYPOGRAPHY
#    Change font family or sizes here; all scripts inherit automatically.
# ══════════════════════════════════════════════════════════════════════════════

FONT_FAMILY   = "DejaVu Sans"   # swap to e.g. "Helvetica" if available

FONT_SIZES = {
    "title":       14,   # figure / panel title
    "subtitle":    12,   # secondary title / annotation
    "axis_label":  12,   # x/y axis labels
    "tick":        12,   # tick labels
    "legend":      11,   # legend text
    "bar_label":   11,   # in-bar / on-bar count or % labels
    "annotation":  11,   # misc annotations
}


# ══════════════════════════════════════════════════════════════════════════════
# 2. LAYOUT & SPACING DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

FIGURE = {
    "dpi":        150,
    "width":       11,   # wider, squat figure matching reference (~11x6 for 4 questions)
}

SPINE_COLOUR = "#dddddd"   # applied via rcParams in apply_style()


# ══════════════════════════════════════════════════════════════════════════════
# 3. COLOUR PALETTES
#    Each palette is a dict or list mapping category → hex colour.
#    Edit hex values here to restyle all plots that use a palette.
# ══════════════════════════════════════════════════════════════════════════════

# ── 3a. Diverging Likert (5-point agree/disagree) ─────────────────────────────
# Blue ↔ orange diverging; safe for deuteranopia & protanopia.
# Order matches LIKERT_ORDER in load_research_data.py.
PALETTE_LIKERT: dict[str, str] = {
    "Strongly Disagree":          "#b35806",  # dark orange
    "Disagree":                   "#f1a340",  # light orange
    "Neither Agree nor Disagree": "#d8d8d8",  # neutral grey
    "Agree":                      "#74add1",  # light blue
    "Strongly Agree":             "#2166ac",  # dark blue
}

# Short legend labels for Likert palette (same key order)
LIKERT_LEGEND_LABELS: dict[str, str] = {
    "Strongly Disagree":          "Strongly Disagree",
    "Disagree":                   "Disagree",
    "Neither Agree nor Disagree": "Neither Agree",
    "Agree":                      "Agree",
    "Strongly Agree":             "Strongly Agree",
}

# ── 3b. Sequential frequency (5-level, low → high) ───────────────────────────
# Single-hue blue ramp; safe for all CVD types.
# Order matches FREQ_ORDER in load_research_data.py.
PALETTE_FREQ: dict[str, str] = {
    "Never used":               "#d0d0d0",  # light grey (anchors "none")
    "Less than once per month": "#c6dbef",  # very light blue
    "1-3 times per month":      "#6baed6",  # medium blue
    "1-5 times per week":       "#2171b5",  # dark blue
    "Daily or almost daily":    "#08306b",  # very dark blue
}

FREQ_LEGEND_LABELS: dict[str, str] = {
    "Never used":               "Never used",
    "Less than once per month": "Less than once per month",
    "1-3 times per month":      "1-3 times per month",
    "1-5 times per week":       "1-5 times per week",
    "Daily or almost daily":    "Daily or almost daily",
}

# ── 3c. Single accent colour (for histograms, single-series bar charts) ───────
COLOUR_ACCENT = "#4393c3"   # mid blue

# ── 3d. Pre/Post pair colours (for paired histograms, slope plots, etc.) ──────
COLOUR_PRE   = "#bdbdbd"   # neutral grey — "before"
COLOUR_POST  = "#2171b5"   # blue — "after"
COLOUR_POST_DARK = "#2b2b2b"   # near-black "after" for grey/black schemes

# ── 3e. Qualitative coding category colours (Likes/Dislikes/Suggestions/etc) ──
# Colourblind-safe under deuteranopia/protanopia/tritanopia: blue, orange,
# purple, and grey are mutually distinguishable (green was avoided because it
# collides with orange under red-green CVD).
PALETTE_QUAL_CATEGORY: dict[str, str] = {
    "Likes":       "#2166ac",   # blue (positive)
    "Dislikes":    "#b35806",   # orange (negative)
    "Suggestions": "#762a83",   # purple (constructive) — CVD-safe vs orange
    "Uncodable":   "#999999",   # grey (no signal)
}

# Okabe-Ito colourblind-safe qualitative palette (minus black & pale yellow,
# which read poorly as bars on white). Used for coded items whose categories
# aren't in the semantic map above.
_CVD_CYCLE = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#999999",  # grey
]


# ── 3f. Overall-sentiment palette (Tier 1 of tiered qualitative coding) ───────
# Diverging: blue (positive) → grey (neutral/uncodable) → orange (negative),
# colourblind-safe. Keyed by the Tier 1 sentiment codes.
PALETTE_SENTIMENT: dict[str, str] = {
    "Positive Impact":     "#2166ac",   # blue
    "Mixed Impact":        "#9aaecc",   # muted blue-grey (pro + con)
    "Neutral / No Impact": "#cccccc",   # light grey
    "Negative Impact":     "#b35806",   # orange
    "Irrelevant / Noise":  "#999999",   # darker grey (uncodable)
}


def get_category_colors(categories: list[str]) -> dict[str, str]:
    """
    Map an arbitrary list of coding categories to colourblind-safe colours.

    Categories present in PALETTE_QUAL_CATEGORY keep their fixed semantic
    colour (so the Likes/Dislikes item is unchanged); any others are assigned
    in order from the Okabe-Ito CVD-safe cycle, skipping colours already taken
    by the semantic assignments to avoid collisions.
    """
    colors: dict[str, str] = {}
    used: set[str] = set()

    # First pass: fixed semantic colours
    for cat in categories:
        if cat in PALETTE_QUAL_CATEGORY:
            colors[cat] = PALETTE_QUAL_CATEGORY[cat]
            used.add(colors[cat])

    # Second pass: assign remaining categories from the CVD cycle
    cycle = [c for c in _CVD_CYCLE if c not in used]
    i = 0
    for cat in categories:
        if cat not in colors:
            colors[cat] = cycle[i % len(cycle)]
            i += 1

    return colors


# ══════════════════════════════════════════════════════════════════════════════
# 4. APPLY RCPARAMS
#    Call apply_style() once at the top of each plotting script.
# ══════════════════════════════════════════════════════════════════════════════

def apply_style() -> None:
    """
    Push all style settings into matplotlib's global rcParams.
    Call once at the start of each plotting script, before any plt calls.
    """
    mpl.rcParams.update({
        # Font
        "font.family":            FONT_FAMILY,
        "font.size":              FONT_SIZES["tick"],

        # Axes
        "axes.titlesize":         FONT_SIZES["title"],
        "axes.titleweight":       "bold",
        "axes.titlepad":          10,
        "axes.labelsize":         FONT_SIZES["axis_label"],
        "axes.spines.top":        False,
        "axes.spines.right":      False,
        "axes.spines.left":       True,
        "axes.spines.bottom":     True,
        "axes.edgecolor":         SPINE_COLOUR,
        "axes.linewidth":         0.8,
        "axes.grid":              False,   # grids added per-plot where needed

        # Ticks
        "xtick.labelsize":        FONT_SIZES["tick"],
        "ytick.labelsize":        FONT_SIZES["tick"],
        "xtick.major.size":       3,
        "ytick.major.size":       0,       # suppress y tick marks by default
        "xtick.major.width":      0.8,
        "xtick.color":            "#555555",
        "ytick.color":            "#555555",

        # Legend
        "legend.fontsize":        FONT_SIZES["legend"],
        "legend.frameon":         False,
        "legend.handlelength":    1.2,
        "legend.handletextpad":   0.5,
        "legend.columnspacing":   0.8,

        # Figure
        "figure.dpi":             FIGURE["dpi"],
        "figure.facecolor":       "white",
        "savefig.bbox":           "tight",
        "savefig.dpi":            FIGURE["dpi"],
    })


# ══════════════════════════════════════════════════════════════════════════════
# 5. CONVENIENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def likert_legend_handles() -> list[mpatches.Patch]:
    """Return legend handles for the diverging Likert palette."""
    return [
        mpatches.Patch(color=PALETTE_LIKERT[cat],
                       label=LIKERT_LEGEND_LABELS[cat])
        for cat in PALETTE_LIKERT
    ]


def freq_legend_handles() -> list[mpatches.Patch]:
    """Return legend handles for the frequency palette."""
    return [
        mpatches.Patch(color=PALETTE_FREQ[cat],
                       label=FREQ_LEGEND_LABELS[cat])
        for cat in PALETTE_FREQ
    ]



# ── Preview when run directly ─────────────────────────────────────────────────

if __name__ == "__main__":
    apply_style()

    fig, axes = plt.subplots(1, 3, figsize=(12, 2.5))
    fig.suptitle("plot_style.py — palette preview", fontsize=10, fontweight="bold")

    # Likert
    ax = axes[0]
    ax.set_title("PALETTE_LIKERT", fontsize=9)
    for i, (cat, col) in enumerate(PALETTE_LIKERT.items()):
        ax.barh(i, 1, color=col)
        ax.text(0.5, i, LIKERT_LEGEND_LABELS[cat],
                ha="center", va="center", fontsize=7, color="white" if i != 2 else "#444")
    ax.set_yticks([])
    ax.set_xticks([])

    # Frequency
    ax = axes[1]
    ax.set_title("PALETTE_FREQ", fontsize=9)
    for i, (cat, col) in enumerate(PALETTE_FREQ.items()):
        ax.barh(i, 1, color=col)
        ax.text(0.5, i, FREQ_LEGEND_LABELS[cat],
                ha="center", va="center", fontsize=7,
                color="white" if i >= 2 else "#444")
    ax.set_yticks([])
    ax.set_xticks([])

    # Tier 1 sentiment (tiered qualitative coding)
    ax = axes[2]
    ax.set_title("PALETTE_SENTIMENT", fontsize=9)
    for i, (label, col) in enumerate(PALETTE_SENTIMENT.items()):
        ax.barh(i, 1, color=col)
        ax.text(0.5, i, label, ha="center", va="center", fontsize=7,
                color="white" if i in (0, 3) else "#444")
    ax.set_yticks([])
    ax.set_xticks([])

    plt.tight_layout()
    plt.savefig("plot_style_preview.pdf")
    print("Saved: plot_style_preview.pdf")
    print("\nFont sizes:", FONT_SIZES)
    print("Figure defaults:", FIGURE)
