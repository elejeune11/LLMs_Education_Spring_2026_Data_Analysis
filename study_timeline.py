"""
EK301 Spring 2026 study timeline (multi-tier swimlane).

Four tiers, top to bottom:
    1. Study administration   (IRB, DUA, data assembly / deposit)
    2. Study delivery         (consent, surveys, AI demonstrations)
    3. Course                 (exams, labs, project deliverables)
    4. Institutional          (BU registrar dates: add/drop, holidays, finals)

How to use:
    - Edit the EVENTS dict below. Each tier holds a list of events.
    - A "point" event needs a single date.
    - A "span" event needs start and end.
    - Leave date / start / end as None for anything not yet known. The script
      skips it on the figure and prints it in a "still to fill in" list.

Layout note:
    Labels no longer cycle through fixed heights. Each label's rendered width
    is measured, and labels are stacked onto as many levels as needed so they
    never overlap horizontally. Lane heights and the figure height are then
    sized to fit. This means you can raise LABEL_FONTSIZE freely; the layout
    expands instead of colliding.

Run:
    python study_timeline.py
Outputs ek301_study_timeline.png and .pdf into figures/, which is created
automatically if it does not exist.

Requires: matplotlib (no other dependencies).
"""

from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# ----------------------------------------------------------------------------
# 1. EVENT DATA  (edit freely)
# ----------------------------------------------------------------------------
# point event:  {"label": str, "date": date or None}
# span event:   {"label": str, "start": date or None, "end": date or None}

EVENTS = {
    "Study administration": [
        {"label": "IRB protocol submission", "date": date(2025, 10, 27)},
        {"label": "IRB exemption (Protocol #8256X)", "date": date(2025, 12, 17)},
        {"label": "Course records released to team", "date": date(2026, 5, 15)},
        {"label": "Survey-to-records linkage", "date": date(2026, 5, 18)},
        {"label": "Merged dataset de-identified", "date": date(2026, 5, 29)},
        {"label": "OpenBU dataset deposit", "date": date(2026, 7, 23)},
    ],
    "Study delivery": [
        {"label": "Consent administration", "date": date(2026, 1, 27)},
        {"label": "Pre-semester survey", "start": date(2026, 1, 27), "end": date(2026, 1, 30)},
        # 9 AI demonstrations, ~weekly. Demo 1 was tied to the buckling lab.
        # Start is anchored to the buckling lab; end is an ESTIMATE -> verify.
        {"label": "9 AI demonstrations (~weekly, approx.)",
         "start": date(2026, 2, 3), "end": date(2026, 4, 21)},
        {"label": "Post-semester survey", "start": date(2026, 4, 28), "end": date(2026, 5, 4)},
    ],
    "Course": [
        {"label": "Buckling lab", "start": date(2026, 2, 9), "end": date(2026, 2, 20)},
        {"label": "Midterm 1", "date": date(2026, 2, 24)},
        {"label": "Midterm 2", "date": date(2026, 4, 2)},
        {"label": "Prelim. design report", "date": date(2026, 4, 10)},
        {"label": "Final design report", "date": date(2026, 4, 24)},
        {"label": "Truss testing day", "date": date(2026, 4, 25)},
        {"label": "EK301 final exam", "date": date(2026, 5, 5)},
        {"label": "Grades submitted", "date": date(2026, 5, 9)},
    ],
    "Institutional": [
        {"label": "Classes begin", "date": date(2026, 1, 20)},
        {"label": "Add deadline", "date": date(2026, 2, 2)},
        {"label": "Drop, no W", "date": date(2026, 2, 24)},
        {"label": "Drop w/ W", "date": date(2026, 4, 3)},
        {"label": "Withdrawal deadline", "date": date(2026, 4, 15)},
        {"label": "Last day of classes", "date": date(2026, 4, 30)},
        {"label": "Final exam period", "start": date(2026, 5, 4), "end": date(2026, 5, 8)},
    ],
}

# Full-height context bands drawn behind everything (e.g. breaks).
BACKGROUND_BANDS = [
    {"label": "Spring Recess", "start": date(2026, 3, 7), "end": date(2026, 3, 15)},
]

# Optional vertical "as of" marker. Set to None to hide.
TODAY_LINE = None  # e.g. date(2026, 6, 1)

TITLE = "EK301 Spring 2026: AI Demonstrations Study Timeline"

# Created automatically if missing.
OUTPUT_DIR = Path("figures")


# ----------------------------------------------------------------------------
# 2. STYLE  (all the knobs you'll want)
# ----------------------------------------------------------------------------
TIER_ORDER = ["Study administration", "Study delivery", "Course", "Institutional"]

COLORS = {
    "Study administration": "#2A6F77",   # teal
    "Study delivery":       "#C46A1B",   # amber
    "Course":               "#3D5A98",   # blue
    "Institutional":        "#6B6B6B",   # neutral gray
}

# Fonts (raise these freely; layout expands to avoid overlap)
# LABEL_FONTSIZE = 12      # event labels
# TIER_FONTSIZE = 14       # tier names on the left
# TICK_FONTSIZE = 12       # month axis
# TITLE_FONTSIZE = 18
val = 5
LABEL_FONTSIZE = 12 + val      # event labels
TIER_FONTSIZE = 14 + val       # tier names on the left
TICK_FONTSIZE = 12 + val       # month axis
TITLE_FONTSIZE = 18 + val

# Vertical layout (data units; figure height is derived from these)
BASE_OFFSET = 1.0        # height of the first label level above a baseline
LEVEL_DY = 1.0           # vertical step between stacked label levels
LABEL_H = 0.95           # approx height of one label line
INTER_TIER_GAP = 2.4     # blank space between a tier's tallest label and the
                         #   next tier's baseline
SPAN_HALF_HEIGHT = 0.34  # half-thickness of span bars
LABEL_PAD_DAYS = 2.0     # min horizontal gap between neighboring labels (days)
MARKER_SIZE = 9          # point marker diameter (points)

VERT_INCH_PER_UNIT = 0.34  # taller -> more breathing room per level


# ----------------------------------------------------------------------------
# 3. INTERNALS
# ----------------------------------------------------------------------------
def _collect_dates():
    ds = []
    for evs in EVENTS.values():
        for e in evs:
            for k in ("date", "start", "end"):
                if e.get(k):
                    ds.append(e[k])
    for b in BACKGROUND_BANDS:
        ds += [b["start"], b["end"]]
    if TODAY_LINE:
        ds.append(TODAY_LINE)
    return ds


def _drawable_labels(tier):
    """Return drawable items for a tier as dicts with center x (datenum),
    kind, and the underlying event. Also return the list of missing items."""
    items, missing = [], []
    for e in EVENTS[tier]:
        if "start" in e:  # span
            s, en = e.get("start"), e.get("end")
            if not (s and en):
                missing.append(f"[{tier}] {e['label']} (span dates)")
                continue
            cx = (mdates.date2num(s) + mdates.date2num(en)) / 2
            items.append({"kind": "span", "cx": cx, "s": s, "en": en, "label": e["label"]})
        else:  # point
            d = e.get("date")
            if not d:
                missing.append(f"[{tier}] {e['label']}")
                continue
            items.append({"kind": "point", "cx": mdates.date2num(d), "d": d, "label": e["label"]})
    return items, missing


def _half_width_datax(ax, fig, renderer, text, fontsize):
    """Half-width of a rendered label, expressed in x-axis data units."""
    t = ax.text(0.5, 0.5, text, fontsize=fontsize, ha="center",
                transform=ax.transAxes)
    bb = t.get_window_extent(renderer)
    t.remove()
    p0 = ax.transData.transform((0, 0))[0]
    p1 = ax.transData.transform((1, 0))[0]
    px_per_unit = (p1 - p0)
    return (bb.width / px_per_unit) / 2.0


def _assign_levels(items):
    """Greedy left-to-right stacking. Each item gets the lowest level whose
    last occupant ends to its left (with padding). Returns max level used."""
    order = sorted(range(len(items)), key=lambda i: items[i]["cx"])
    level_right = []  # rightmost occupied x per level
    max_level = 0
    for i in order:
        left = items[i]["cx"] - items[i]["half"] - LABEL_PAD_DAYS
        right = items[i]["cx"] + items[i]["half"]
        placed = False
        for lvl in range(len(level_right)):
            if left >= level_right[lvl]:
                items[i]["level"] = lvl
                level_right[lvl] = right
                placed = True
                break
        if not placed:
            items[i]["level"] = len(level_right)
            level_right.append(right)
        max_level = max(max_level, items[i]["level"])
    return max_level


# ----------------------------------------------------------------------------
# 4. BUILD
# ----------------------------------------------------------------------------
def build():
    all_dates = _collect_dates()
    xmin, xmax = min(all_dates), max(all_dates)
    pad = (xmax - xmin).days * 0.04

    # Provisional figure; height is corrected after we know the level counts.
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.92, bottom=0.07)
    ax.set_xlim(mdates.date2num(xmin) - pad, mdates.date2num(xmax) + pad)
    ax.set_ylim(0, 100)  # temporary
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Gather drawables + measure widths + assign levels per tier.
    tier_items, missing = {}, []
    tier_maxlevel = {}
    for tier in TIER_ORDER:
        items, miss = _drawable_labels(tier)
        missing += miss
        for it in items:
            it["half"] = _half_width_datax(ax, fig, renderer, it["label"], LABEL_FONTSIZE)
        tier_maxlevel[tier] = _assign_levels(items) if items else 0
        tier_items[tier] = items

    # Baselines bottom-up so each tier clears the labels of the tier below.
    baseline = {}
    y = 0.0
    for tier in reversed(TIER_ORDER):  # institutional first (lowest)
        baseline[tier] = y
        stack_top = y + BASE_OFFSET + tier_maxlevel[tier] * LEVEL_DY + LABEL_H
        y = stack_top + INTER_TIER_GAP
    top_tier = TIER_ORDER[0]
    y_top = baseline[top_tier] + BASE_OFFSET + tier_maxlevel[top_tier] * LEVEL_DY + LABEL_H

    # Resize the figure vertically to give each level real breathing room.
    total_units = y_top + 3.0
    data_frac = 0.92 - 0.07  # top - bottom from subplots_adjust
    fig_h = max(9.0, total_units * VERT_INCH_PER_UNIT / data_frac)
    fig.set_size_inches(16, fig_h)
    fig.canvas.draw()  # geometry refresh (x scale unchanged: same width & xlim)

    # ----- draw lane backgrounds + baselines -----
    for i, tier in enumerate(TIER_ORDER):
        yb = baseline[tier]
        top = yb + BASE_OFFSET + tier_maxlevel[tier] * LEVEL_DY + LABEL_H
        if i % 2 == 0:
            ax.axhspan(yb - 1.3, top, color="#000000", alpha=0.025, zorder=0)
        ax.axhline(yb, color=COLORS[tier], lw=1.3, alpha=0.55, zorder=1)

    # ----- full-height context bands -----
    for b in BACKGROUND_BANDS:
        if b["start"] and b["end"]:
            ax.axvspan(b["start"], b["end"], color="#000000", alpha=0.05, zorder=0)
            mid = (mdates.date2num(b["start"]) + mdates.date2num(b["end"])) / 2
            ax.text(mid, y_top + 1.1, b["label"], ha="center", va="bottom",
                    fontsize=LABEL_FONTSIZE - 2, style="italic",
                    color="#888888", zorder=2)

    # ----- events -----
    for tier in TIER_ORDER:
        yb = baseline[tier]
        color = COLORS[tier]
        for it in tier_items[tier]:
            label_y = yb + BASE_OFFSET + it["level"] * LEVEL_DY
            if it["kind"] == "span":
                x0 = mdates.date2num(it["s"])
                width = mdates.date2num(it["en"]) - x0
                ax.add_patch(Rectangle((x0, yb - SPAN_HALF_HEIGHT), width,
                                       2 * SPAN_HALF_HEIGHT, facecolor=color,
                                       alpha=0.32, edgecolor=color, lw=1.3, zorder=3))
                ax.plot([it["cx"], it["cx"]], [yb + SPAN_HALF_HEIGHT, label_y - 0.15],
                        color=color, lw=0.8, alpha=0.5, zorder=2)
                ax.text(it["cx"], label_y, it["label"], ha="center", va="bottom",
                        fontsize=LABEL_FONTSIZE, color=color, fontweight="medium",
                        zorder=5)
            else:
                ax.plot([it["cx"]], [yb], marker="o", markersize=MARKER_SIZE,
                        color=color, markeredgecolor="white", markeredgewidth=1.1,
                        zorder=4)
                ax.plot([it["cx"], it["cx"]], [yb, label_y - 0.15],
                        color=color, lw=0.8, alpha=0.55, zorder=2)
                ax.text(it["cx"], label_y, it["label"], ha="center", va="bottom",
                        fontsize=LABEL_FONTSIZE, color="#222222", zorder=5)

    # ----- today / as-of marker -----
    if TODAY_LINE:
        ax.axvline(TODAY_LINE, color="#B03030", lw=1.3, ls="--", alpha=0.8, zorder=2)
        ax.text(TODAY_LINE, baseline[TIER_ORDER[-1]] - 1.0, "as of",
                ha="center", va="top", fontsize=LABEL_FONTSIZE - 2, color="#B03030")

    # ----- axes cosmetics -----
    ax.set_yticks([baseline[t] for t in TIER_ORDER])
    ax.set_yticklabels(TIER_ORDER, fontsize=TIER_FONTSIZE, fontweight="bold")
    for tick, t in zip(ax.get_yticklabels(), TIER_ORDER):
        tick.set_color(COLORS[t])

    ax.set_ylim(baseline[TIER_ORDER[-1]] - 1.8, y_top + 2.2)

    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.WeekdayLocator(byweekday=mdates.MO))
    ax.tick_params(axis="x", which="major", labelsize=TICK_FONTSIZE)
    ax.grid(axis="x", which="major", color="#000000", alpha=0.08, zorder=0)
    ax.grid(axis="x", which="minor", color="#000000", alpha=0.035, zorder=0)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_alpha(0.3)

    ax.set_title(TITLE, fontsize=TITLE_FONTSIZE, fontweight="bold", pad=18, loc="left")
    return fig, missing


def main():
    fig, missing = build()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in ("png", "pdf"):
        fig.savefig(OUTPUT_DIR / f"ek301_study_timeline.{fmt}",
                    dpi=300, bbox_inches="tight")
    print("Saved ek301_study_timeline.png and ek301_study_timeline.pdf")
    if missing:
        print("\nStill to fill in (currently None, not drawn):")
        for m in missing:
            print(f"  - {m}")
    else:
        print("\nAll events have dates.")


if __name__ == "__main__":
    main()
