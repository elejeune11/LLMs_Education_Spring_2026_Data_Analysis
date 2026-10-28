"""
report_stats.py
---------------
Publication-ready reliability reporting for the LLM coding pipeline. Reads
the coding caches (results/*.jsonl), codebooks.json, and (if present)
final_codes.csv, and writes to --report_dir:

  reliability_summary.csv      one row per question: n, coder verbosity,
                               identical-set %, mean pairwise Jaccard,
                               median Fleiss kappa over reportable codes,
                               % of reportable codes with kappa >= 0.61,
                               pairwise model agreement (mean Jaccard)
  reliability_by_code.csv      per code: name, per-model prevalence, final
                               (adjudicated) prevalence, Fleiss kappa,
                               percent agreement
  table_reliability.tex        LaTeX (booktabs) version of the summary table
  table_reliability_by_code.tex  LaTeX appendix table, one panel per question
  methods_boilerplate.tex      a drafted methods/results paragraph with all
                               numbers, model IDs, and adjudication stats
                               filled in (edit before use)

"Reportable codes" = prevalence >= --min_prev (default 0.05) among items
coded by all models; kappa is unstable below that and is reported as "--"
in tables (prevalence still shown).

Run:  python report_stats.py --outdir results --codebooks codebooks.json \
          --report_dir report
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from aggregate_results import load_records, fleiss_kappa_binary, jaccard

QUESTION_LABELS = {
    "txt_ai_workflow_steps_pre": "Workflow steps (pre)",
    "txt_ai_workflow_steps_post": "Workflow steps (post)",
    "txt_ai_verify_correctness_pre": "Verification behavior (pre)",
    "txt_ai_verify_correctness_post": "Verification behavior (post)",
    "txt_ai_impact_describe_pre": "Impact on learning (pre)",
    "txt_ai_impact_describe_post": "Impact on learning (post)",
    "txt_ai_demo_feedback_post": "Demo feedback (post)",
    "txt_ai_use_change_post": "Change in AI use (post)",
}


def tex_escape(s):
    return (str(s).replace("&", "\\&").replace("%", "\\%").replace("_", "\\_")
            .replace("#", "\\#"))


def fmt(x, nd=2, dash_if_nan=True):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--" if dash_if_nan else ""
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--codebooks", default="codebooks.json")
    ap.add_argument("--report_dir", default="report")
    ap.add_argument("--min_prev", type=float, default=0.05,
                    help="min mean prevalence for a code's kappa to be reportable")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    rep = Path(args.report_dir)
    rep.mkdir(exist_ok=True)
    books = json.loads(Path(args.codebooks).read_text())
    col_to_book = {c: b for b in books.values() for c in b["columns"]}
    id_to_name = {b_key: {c["id"]: c["name"] for c in b["codes"]}
                  for b_key, b in books.items()}
    col_to_bkey = {c: k for k, b in books.items() for c in b["columns"]}

    df = load_records(outdir)  # deduped, latest successful per item/model
    ok = df[df["error"].isin([None, ""]) | df["error"].isna()].copy()
    models = sorted(ok["model_key"].unique())
    model_ids = {m: sorted(ok[ok.model_key == m]["model"].unique()) for m in models}

    fin = None
    fin_path = outdir / "final_codes.csv"
    if fin_path.exists():
        fin = pd.read_csv(fin_path).fillna("")

    summary_rows, code_rows = [], []
    for col in [c for c in QUESTION_LABELS if c in set(ok["column"])]:
        grp = ok[ok["column"] == col]
        col_models = sorted(grp["model_key"].unique())
        pivot = (grp.pivot_table(index="QualID", columns="model_key",
                                 values="codes", aggfunc="first")
                 .reindex(columns=col_models).dropna())
        n_items = len(pivot)
        code_ids = [c["id"] for c in col_to_book[col]["codes"]]

        # item-level agreement
        jacs, ident = [], 0
        pair_jacs = {p: [] for p in itertools.combinations(col_models, 2)}
        for q in pivot.index:
            sets = {m: set(pivot.loc[q, m]) for m in col_models}
            pj = [jaccard(sets[a], sets[b]) for a, b in pair_jacs]
            for (p, v) in zip(pair_jacs, pj):
                pair_jacs[p].append(v)
            jacs.append(np.mean(pj))
            ident += int(len({frozenset(v) for v in sets.values()}) == 1)

        # per-code kappa/prevalence
        kappas_reportable = []
        n_reportable = n_pass = 0
        for cid in code_ids:
            mat = np.array([[int(cid in pivot.loc[q, m]) for m in col_models]
                            for q in pivot.index])
            # Codes no first-pass model applied are still listed (zero
            # prevalence, no kappa) so the table covers every codebook code.
            prev = {m: mat[:, i].mean() for i, m in enumerate(col_models)}
            mean_prev = float(np.mean(list(prev.values())))
            kappa = (fleiss_kappa_binary(mat)
                     if len(col_models) >= 2 and mat.sum() > 0 else np.nan)
            pct = np.mean([np.mean(mat[:, i] == mat[:, j])
                           for i, j in itertools.combinations(range(len(col_models)), 2)])
            reportable = mean_prev >= args.min_prev
            if reportable and kappa == kappa:
                kappas_reportable.append(kappa)
                n_reportable += 1
                n_pass += int(kappa >= 0.61)
            fin_prev = np.nan
            if fin is not None:
                sub = fin[fin["column"] == col]
                if len(sub):
                    fin_prev = sub["final_codes"].apply(
                        lambda s: cid in [x for x in str(s).split(";") if x]).mean()
            code_rows.append({
                "column": col, "question": QUESTION_LABELS[col], "code_id": cid,
                "code": id_to_name[col_to_bkey[col]][cid],
                "n_items": n_items, "mean_prevalence": round(mean_prev, 3),
                "final_prevalence": (round(float(fin_prev), 3)
                                     if fin_prev == fin_prev else np.nan),
                "fleiss_kappa": (round(float(kappa), 3) if kappa == kappa else np.nan),
                "pct_agreement": round(float(pct), 3),
                "reportable": reportable,
            })

        row = {
            "column": col, "question": QUESTION_LABELS[col], "n_items": n_items,
            "mean_codes_per_response": round(float(np.mean(
                [len(pivot.loc[q, m]) for q in pivot.index for m in col_models])), 2),
            "identical_set_pct": round(100 * ident / n_items, 1) if n_items else np.nan,
            "mean_pairwise_jaccard": round(float(np.mean(jacs)), 3) if jacs else np.nan,
            "median_kappa_reportable": (round(float(np.median(kappas_reportable)), 3)
                                        if kappas_reportable else np.nan),
            "n_reportable_codes": n_reportable,
            "pct_reportable_kappa_ge_061": (round(100 * n_pass / n_reportable, 1)
                                            if n_reportable else np.nan),
        }
        for (a, b), v in pair_jacs.items():
            row[f"jaccard_{a}_vs_{b}"] = round(float(np.mean(v)), 3)
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    codes_df = pd.DataFrame(code_rows)
    summary.to_csv(rep / "reliability_summary.csv", index=False)
    codes_df.to_csv(rep / "reliability_by_code.csv", index=False)

    # ---------------------------------------------------------- LaTeX tables
    lines = [
        "% Auto-generated by report_stats.py; requires \\usepackage{booktabs}",
        "\\begin{table}[t]", "\\centering",
        "\\caption{Inter-model reliability of the LLM first-pass coding. "
        "Identical = percent of responses receiving identical code sets from "
        "all three models; $J$ = mean pairwise Jaccard similarity of code "
        "sets; $\\tilde{\\kappa}$ = median Fleiss' $\\kappa$ across codes "
        f"with mean prevalence $\\geq$ {args.min_prev:.0%}".replace("%", "\\%")
        + "; $\\kappa \\geq .61$ = percent of those codes reaching "
        "substantial agreement.}",
        "\\label{tab:llm_reliability}",
        "\\begin{tabular}{lrrrrr}", "\\toprule",
        "Question & $n$ & Identical (\\%) & $J$ & $\\tilde{\\kappa}$ & "
        "$\\kappa \\geq .61$ (\\%) \\\\", "\\midrule",
    ]
    for r in summary.itertuples():
        lines.append(
            f"{tex_escape(r.question)} & {r.n_items} & "
            f"{fmt(r.identical_set_pct, 1)} & {fmt(r.mean_pairwise_jaccard)} & "
            f"{fmt(r.median_kappa_reportable)} & "
            f"{fmt(r.pct_reportable_kappa_ge_061, 1)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    (rep / "table_reliability.tex").write_text("\n".join(lines))

    lines = [
        "% Auto-generated by report_stats.py; per-code appendix table",
        "\\begin{table}[p]", "\\centering", "\\footnotesize",
        "\\caption{Per-code inter-model reliability. Prev.\\ = mean prevalence "
        "across models; Final = prevalence in the adjudicated coding; "
        "$\\kappa$ = Fleiss' $\\kappa$ (shown when prevalence $\\geq$ "
        + f"{args.min_prev:.0%}".replace("%", "\\%")
        + " before rounding); Agr.\\ = mean pairwise percent agreement. "
        "Every codebook code is listed, including codes no model applied.}",
        "\\label{tab:llm_reliability_by_code}",
        "\\begin{tabular}{llrrrr}", "\\toprule",
        "Question & Code & Prev. & Final & $\\kappa$ & Agr. \\\\", "\\midrule",
    ]
    for col, sub in codes_df.groupby("column", sort=False):
        for r in sub.itertuples():
            k = fmt(r.fleiss_kappa) if r.reportable else "--"
            lines.append(
                f"{tex_escape(r.question)} & {tex_escape(r.code)} & "
                f"{fmt(r.mean_prevalence)} & {fmt(r.final_prevalence)} & "
                f"{k} & {fmt(r.pct_agreement)} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines += ["\\end{tabular}", "\\end{table}", ""]
    (rep / "table_reliability_by_code.tex").write_text("\n".join(lines))

    # ------------------------------------------------------------ boilerplate
    n_total = int(summary.n_items.sum())
    ident_overall = float(np.average(summary.identical_set_pct,
                                     weights=summary.n_items))
    jac_overall = float(np.average(summary.mean_pairwise_jaccard,
                                   weights=summary.n_items))
    all_k = codes_df[codes_df.reportable].fleiss_kappa.dropna()
    model_str = "; ".join(f"{m}: {', '.join(model_ids[m])}" for m in models)
    adj_str = ""
    if fin is not None:
        src = fin.source.value_counts()
        adj_str = (
            f"Responses on which all three models produced identical code sets "
            f"({src.get('unanimous', 0)} of {len(fin)}) were accepted directly; the "
            f"remaining {src.get('adjudicated', 0) + src.get('majority-fallback', 0)} "
            f"were resolved by an adjudication call to a designated model that "
            f"received the response, the codebook, and all three codings with "
            f"rationales, and ruled strictly against the codebook"
            + (f" ({src.get('majority-fallback', 0)} items fell back to the "
               f"majority coding after adjudication-call failures)"
               if src.get('majority-fallback', 0) else "") + ". ")
    text = f"""% Drafted by report_stats.py -- verify all numbers and edit wording before use.
Each free-text response was coded independently by three large language models
({tex_escape(model_str)}) against the study codebooks, with structured JSON
output validated for code validity, tier cardinality, and exclusivity rules.
Across {n_total} response-question pairs coded by all three models, the models
assigned identical code sets to {ident_overall:.0f}\\% of responses, with a mean
pairwise Jaccard similarity of {jac_overall:.2f}. Treating each code as a binary
apply/not-apply decision, the median Fleiss' $\\kappa$ across the
{len(all_k)} codes with mean prevalence $\\geq$ {args.min_prev:.0%} was
{np.median(all_k):.2f} (IQR {np.percentile(all_k, 25):.2f}--{np.percentile(all_k, 75):.2f}),
and {100 * (all_k >= 0.61).mean():.0f}\\% of these codes reached the conventional
threshold for substantial agreement ($\\kappa \\geq .61$).
{adj_str}Inter-model agreement establishes the reliability of the machine
coding, not its validity; [describe human validation here, or note it as a
limitation].
""".replace("5%", "5\\%")
    (rep / "methods_boilerplate.tex").write_text(text)

    print(summary[["question", "n_items", "identical_set_pct",
                   "mean_pairwise_jaccard", "median_kappa_reportable",
                   "pct_reportable_kappa_ge_061"]].to_string(index=False))
    print(f"\nWrote reliability_summary.csv, reliability_by_code.csv, "
          f"table_reliability.tex, table_reliability_by_code.tex, "
          f"methods_boilerplate.tex -> {rep}/")


if __name__ == "__main__":
    main()
