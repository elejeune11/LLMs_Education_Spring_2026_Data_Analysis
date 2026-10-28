"""
export_figure_counts.py
-----------------------
Writes machine-readable tables of the counts plotted in every manuscript
figure (report/figure_counts/*.csv). Counts are computed with the same loaders
as the figures, so the honesty-item rule in load_research_data.load_data is
applied. Figure numbers refer to the revised manuscript; the matched-sample
figure's counts are written by plot_paired_change.py.

Structured items: a blank cell on a Section C/D item is reported as
"Not Applicable" (every item except the 0-10 impact rating required a response).

Run:  python export_figure_counts.py
"""

import json
import os

import pandas as pd

from load_research_data import FREQ_ORDER, GRADE_QUARTILE_ORDER, LIKERT_ORDER, load_data

OUT = os.path.join("report", "figure_counts")

UNDERSTANDING = ["understand_apply_principles", "understand_identify_mistakes",
                 "understand_until_practice", "understand_exams_reflect"]
ATTITUDES = ["ai_useful_coursework", "ai_helped_understand_engmech", "ai_more_confusing",
             "ai_can_evaluate_correctness", "ai_accuracy_concerns", "ai_vs_textbooks",
             "ai_vs_discussion"]
FREQ = ["ai_use_freq_general", "ai_use_freq_engmech"]
DEMO = ["ai_demos_attended_post", "ai_demo_helped_understanding_post",
        "ai_demo_engagement_post", "ai_demo_changed_use_post"]


def _took(df: pd.DataFrame, wave: str) -> pd.Series:
    cols = [f"{s}_{wave}" for s in UNDERSTANDING + FREQ]
    return df[cols].notna().any(axis=1)


def _item_counts(df, stems, levels, na_label=None):
    rows = []
    for stem in stems:
        for wave in ("pre", "post"):
            took = _took(df, wave)
            s = df.loc[took, f"{stem}_{wave}"]
            lv = {str(x).strip().lower(): x for x in levels}
            norm = s.astype("string").str.strip().str.lower().map(lv)
            for level in levels:
                rows.append({"item": stem, "timepoint": wave, "response": level,
                             "count": int((norm == level).sum()), "n_took_survey": int(took.sum())})
            if na_label:
                rows.append({"item": stem, "timepoint": wave, "response": na_label,
                             "count": int(s.isna().sum()), "n_took_survey": int(took.sum())})
    return pd.DataFrame(rows)


def _code_counts(columns):
    fin = pd.read_csv(os.path.join("results", "final_codes.csv")).fillna("")
    books = json.load(open("codebooks.json"))
    rows = []
    for col in columns:
        book = next(b for b in books.values() if col in b["columns"])
        sub = fin[fin["column"] == col]
        n = len(sub)
        for code in book["codes"]:
            c = sub["final_codes"].apply(lambda s: code["id"] in str(s).split(";")).sum()
            rows.append({"column": col, "code_id": code["id"], "tier": code.get("tier", ""),
                         "category": code["category"], "code": code["name"],
                         "count": int(c), "n_responses": n})
    return pd.DataFrame(rows)


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    df = load_data()["all"]

    _item_counts(df, ATTITUDES, LIKERT_ORDER, "Not Applicable").to_csv(
        os.path.join(OUT, "fig02_attitudes.csv"), index=False)
    _code_counts(["txt_ai_workflow_steps_pre", "txt_ai_workflow_steps_post"]).to_csv(
        os.path.join(OUT, "fig03_workflow_codes.csv"), index=False)

    rows = []
    for wave in ("pre", "post"):
        s = pd.to_numeric(df[f"ai_impact_rating_{wave}"], errors="coerce").dropna().astype(int)
        for v in range(11):
            rows.append({"timepoint": wave, "rating": v, "count": int((s == v).sum()), "n": len(s)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig04_impact_rating.csv"), index=False)
    _code_counts(["txt_ai_impact_describe_pre", "txt_ai_impact_describe_post"]).to_csv(
        os.path.join(OUT, "fig04_impact_codes.csv"), index=False)

    _item_counts(df, FREQ, FREQ_ORDER).to_csv(
        os.path.join(OUT, "fig05_use_frequency.csv"), index=False)
    _code_counts(["txt_ai_use_change_post"]).to_csv(
        os.path.join(OUT, "fig05_use_change_codes.csv"), index=False)
    _code_counts(["txt_ai_verify_correctness_pre", "txt_ai_verify_correctness_post"]).to_csv(
        os.path.join(OUT, "fig06_verification_codes.csv"), index=False)
    _item_counts(df, UNDERSTANDING, LIKERT_ORDER).to_csv(
        os.path.join(OUT, "fig07_understanding.csv"), index=False)

    # Fig 8: frequency by grade quartile, students with both frequency items at both waves
    fc = [f"{s}_{w}" for s in FREQ for w in ("pre", "post")]
    p = df[df[fc].notna().all(axis=1) & df["grade_quartile"].notna()]
    rows = []
    for s in FREQ:
        for w in ("pre", "post"):
            for q in GRADE_QUARTILE_ORDER:
                sq = p[p["grade_quartile"] == q][f"{s}_{w}"]
                for level in FREQ_ORDER:
                    rows.append({"item": s, "timepoint": w, "grade_quartile": q,
                                 "response": level, "count": int((sq == level).sum()),
                                 "n_quartile": len(sq)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig08_frequency_by_quartile.csv"), index=False)

    # Fig 9: demo experience (demo-section students who attended >= 1) and feedback codes
    att = pd.to_numeric(df["ai_demos_attended_post"], errors="coerce")
    grp = df[(df["demo_section"] == True) & (att >= 1)]  # noqa: E712
    rows = []
    for c in DEMO:
        vc = grp[c].astype("string").value_counts()
        for level, n in vc.items():
            rows.append({"item": c, "response": level, "count": int(n), "n_demo_group": len(grp)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig09_demo_experience.csv"), index=False)
    _code_counts(["txt_ai_demo_feedback_post"]).to_csv(
        os.path.join(OUT, "fig09_demo_feedback_codes.csv"), index=False)

    # Fig 10: frequency and quartile composition by section group (all respondents)
    rows = []
    for flag, name in ((True, "Demo section"), (False, "Other sections")):
        g = df[df["demo_section"] == flag]
        for s in FREQ:
            for w in ("pre", "post"):
                sv = g[f"{s}_{w}"].dropna()
                for level in FREQ_ORDER:
                    rows.append({"section_group": name, "item": s, "timepoint": w,
                                 "response": level, "count": int((sv == level).sum()),
                                 "n_group": len(sv)})
        gq = g["grade_quartile"].dropna()
        for q in GRADE_QUARTILE_ORDER:
            rows.append({"section_group": name, "item": "grade_quartile", "timepoint": "",
                         "response": q, "count": int((gq == q).sum()), "n_group": len(gq)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "fig10_section_comparison.csv"), index=False)

    print(f"Wrote {len(os.listdir(OUT))} files to {OUT}/")


if __name__ == "__main__":
    main()
