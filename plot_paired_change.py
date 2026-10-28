"""
plot_paired_change.py
---------------------
Matched (within-student) pre/post summaries for the students with valid
responses to both surveys (data["paired"] from load_research_data, n = 93
after the honesty-item rule). Purely descriptive: no inferential tests.

Produces
  figures/paired_change_matched.pdf
      (a) transition heatmap, general AI-use frequency (pre x post counts)
      (b) transition heatmap, engineering-mechanics AI-use frequency
      (c) histogram of within-student change in the 0-10 impact rating
  report/paired_frequency_transitions.csv   long-format transition counts
  report/paired_impact_change.csv           per-difference counts + summary
  report/paired_likert_summary.csv          item-wise paired summary
  report/table_paired_likert.tex            LaTeX version of the summary

Blank cells: every survey item except the 0-10 impact rating required a
response, so a blank on a Section C/D attitude item (the items that offered
"Not Applicable") records a "Not Applicable" answer. The summary reports, for
each item, the matched students who answered "Not Applicable" at either
survey. The impact rating was not required; one matched student left the
early-semester rating blank, so panel (c) has 92 pairs.

Run:  python plot_paired_change.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from load_research_data import FREQ_ORDER, LIKERT_ORDER, load_data
from plot_style import FONT_SIZES, apply_style

OUTPUT_DIR = "figures"
REPORT_DIR = "report"

FREQ_ITEMS = [
    ("ai_use_freq_general", "(a) General AI use"),
    ("ai_use_freq_engmech", "(b) AI use for engineering mechanics"),
]
FREQ_SHORT = ["Never", "< monthly", "1-3/month", "1-5/week", "Daily"]

# The 11 Likert items shown in Figs 2 and 7, with short labels for the table.
LIKERT_ITEMS = [
    ("understand_apply_principles", "A", "Confident applying principles to new problems"),
    ("understand_identify_mistakes", "A", "Can identify and correct own mistakes"),
    ("understand_until_practice", "A", "Understand concepts until working examples alone"),
    ("understand_exams_reflect", "A", "Exam scores reflect understanding"),
    ("ai_useful_coursework", "C", "AI useful for university coursework"),
    ("ai_helped_understand_engmech", "C", "AI helped me understand engineering mechanics"),
    ("ai_more_confusing", "C", "AI makes mechanics problems more confusing"),
    ("ai_can_evaluate_correctness", "C", "Can evaluate whether AI solution is correct"),
    ("ai_accuracy_concerns", "C", "Concerns about AI accuracy for mechanics"),
    ("ai_vs_textbooks", "D", "AI more helpful than textbooks"),
    ("ai_vs_discussion", "D", "AI more helpful than discussion section"),
]


def _transition_matrix(paired: pd.DataFrame, stem: str) -> tuple[np.ndarray, int]:
    pre = paired[f"{stem}_pre"]
    post = paired[f"{stem}_post"]
    ok = pre.notna() & post.notna()
    mat = np.zeros((len(FREQ_ORDER), len(FREQ_ORDER)), dtype=int)
    for a, b in zip(pre[ok], post[ok]):
        mat[FREQ_ORDER.index(a), FREQ_ORDER.index(b)] += 1
    return mat, int(ok.sum())


def _draw_heatmap(ax, mat: np.ndarray, n: int, title: str) -> None:
    ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(mat.max(), 1), origin="lower")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center",
                        fontsize=FONT_SIZES["bar_label"] - 1,
                        color="white" if v > mat.max() * 0.55 else "#222222")
    # outline the diagonal (no change)
    for k in range(mat.shape[0]):
        ax.add_patch(plt.Rectangle((k - 0.5, k - 0.5), 1, 1, fill=False,
                                   edgecolor="#555555", linewidth=1.0))
    ax.set_xticks(range(len(FREQ_SHORT)))
    ax.set_xticklabels(FREQ_SHORT, rotation=35, ha="right",
                       fontsize=FONT_SIZES["tick"] - 2)
    ax.set_yticks(range(len(FREQ_SHORT)))
    ax.set_yticklabels(FREQ_SHORT, fontsize=FONT_SIZES["tick"] - 2)
    ax.set_xlabel("Post-semester", fontsize=FONT_SIZES["axis_label"] - 1)
    ax.set_ylabel("Early-semester", fontsize=FONT_SIZES["axis_label"] - 1)
    up = int(np.triu(mat, 1).sum())
    same = int(np.trace(mat))
    down = int(np.tril(mat, -1).sum())
    ax.set_title(f"{title}\n(n = {n}: {up} more, {same} same, {down} less)",
                 fontsize=FONT_SIZES["subtitle"] - 1)


def main() -> None:
    apply_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    paired = load_data()["paired"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.9),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})

    rows = []
    for ax, (stem, title) in zip(axes[:2], FREQ_ITEMS):
        mat, n = _transition_matrix(paired, stem)
        _draw_heatmap(ax, mat, n, title)
        for i, a in enumerate(FREQ_ORDER):
            for j, b in enumerate(FREQ_ORDER):
                rows.append({"item": stem, "pre": a, "post": b, "count": int(mat[i, j])})
    pd.DataFrame(rows).to_csv(os.path.join(REPORT_DIR, "paired_frequency_transitions.csv"),
                              index=False)

    # (c) within-student change in the 0-10 impact rating
    pre = pd.to_numeric(paired["ai_impact_rating_pre"], errors="coerce")
    post = pd.to_numeric(paired["ai_impact_rating_post"], errors="coerce")
    ok = pre.notna() & post.notna()
    diff = (post[ok] - pre[ok]).astype(int)
    lo, hi = int(diff.min()), int(diff.max())
    values = np.arange(lo, hi + 1)
    counts = np.array([(diff == v).sum() for v in values])
    ax = axes[2]
    colours = ["#f1a340" if v < 0 else "#d8d8d8" if v == 0 else "#74add1" for v in values]
    ax.bar(values, counts, width=0.8, color=colours, linewidth=0, zorder=2)
    for v, c in zip(values, counts):
        if c:
            ax.text(v, c + 0.3, str(c), ha="center", va="bottom",
                    fontsize=FONT_SIZES["bar_label"] - 2, color="#444444")
    ax.axvline(diff.mean(), color="#d62728", linestyle="--", linewidth=1.4, zorder=3)
    ax.set_xlabel("Change in rating (post minus early-semester)",
                  fontsize=FONT_SIZES["axis_label"] - 1)
    ax.set_ylabel("Number of students", fontsize=FONT_SIZES["axis_label"] - 1)
    ax.set_xticks(values)
    ax.tick_params(axis="x", labelsize=FONT_SIZES["tick"] - 3)
    n_up, n_same, n_down = int((diff > 0).sum()), int((diff == 0).sum()), int((diff < 0).sum())
    ax.set_title(f"(c) Change in AI impact rating (0-10)\n(n = {len(diff)}: "
                 f"{n_up} higher, {n_same} same, {n_down} lower; "
                 f"mean {diff.mean():+.2f}, median {diff.median():+.0f})",
                 fontsize=FONT_SIZES["subtitle"] - 1)
    ax.grid(axis="y", color="#eeeeee", zorder=0)

    fig.suptitle("Within-student change, matched sample", fontweight="bold",
                 fontsize=FONT_SIZES["title"])
    fig.tight_layout()
    out = os.path.join(OUTPUT_DIR, "paired_change_matched.pdf")
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")

    imp = pd.DataFrame({"change": values, "count": counts})
    imp.to_csv(os.path.join(REPORT_DIR, "paired_impact_change.csv"), index=False)
    print(f"Impact rating: n={len(diff)} mean {diff.mean():+.3f} median {diff.median():+.1f} "
          f"higher {n_up} same {n_same} lower {n_down}")

    # Item-wise paired summary for the 11 Likert items
    order = {lvl.lower(): k for k, lvl in enumerate(LIKERT_ORDER)}
    summ = []
    for stem, section, label in LIKERT_ITEMS:
        a = paired[f"{stem}_pre"].astype("string").str.lower().map(order)
        b = paired[f"{stem}_post"].astype("string").str.lower().map(order)
        both = a.notna() & b.notna()
        d = (b[both] - a[both])
        summ.append({
            "item": stem, "section": section, "label": label,
            "matched_students": len(paired),
            "paired_n": int(both.sum()),
            "increased": int((d > 0).sum()),
            "same": int((d == 0).sum()),
            "decreased": int((d < 0).sum()),
            "na_either": int((~both).sum()),
        })
    summ = pd.DataFrame(summ)
    summ.to_csv(os.path.join(REPORT_DIR, "paired_likert_summary.csv"), index=False)

    lines = [
        r"% Auto-generated by plot_paired_change.py",
        r"\begin{longtable}{p{6.2cm}cccccc}",
        r"\caption{Item-wise paired summary of the Likert items, matched sample. "
        r"Paired $n$ = matched students with a response at both surveys; Increased / Same / "
        r"Decreased = change toward agreement of at least one scale point, no change, or "
        r"change toward disagreement; N/A = matched students who answered ``Not Applicable'' "
        r"at either survey (all items required a response; Section A items had no "
        r"``Not Applicable'' option).}"
        r"\label{tab:paired_likert} \\",
        r"\toprule",
        r"Item & Section & Paired $n$ & Increased & Same & Decreased & N/A \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Item & Section & Paired $n$ & Increased & Same & Decreased & N/A \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endlastfoot",
    ]
    for _, r in summ.iterrows():
        lines.append(f"{r.label} & {r.section} & {r.paired_n} & {r.increased} & "
                     f"{r.same} & {r.decreased} & {r.na_either} \\\\")
    lines.append(r"\end{longtable}")
    with open(os.path.join(REPORT_DIR, "table_paired_likert.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(summ[["label", "paired_n", "increased", "same", "decreased", "na_either"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
