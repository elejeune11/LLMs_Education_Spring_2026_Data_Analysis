"""
aggregate_results.py

Combines the per-model jsonl caches produced by run_llm_coding.py into:

1. coded_long.csv        one row per (QualID, column, model, code)
2. coded_binary.csv      one row per (QualID, column, code) with a 0/1
                         column per model, n_models_applied, and consensus
                         (applied by >= --consensus_k of the models that
                         successfully coded the item)
3. reliability.csv       per (column, code): prevalence per model, Fleiss'
                         kappa across models on the binary apply/not-apply
                         decision, and pairwise percent agreement
4. item_agreement.csv    per (QualID, column): mean pairwise Jaccard
                         similarity of the models' code sets
5. review_queue.csv      every item where model code sets are not identical,
                         any model set flag_for_review, or any model errored,
                         sorted most-disagreement-first, with each model's
                         codes and rationales side by side for human
                         adjudication
6. human_sample.csv      (optional, --sample_n) stratified random sample per
                         column for independent human coding to validate the
                         LLM first pass; codes hidden.

Run:  python aggregate_results.py --outdir results --codebooks codebooks.json
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_records(outdir):
    recs = []
    for p in sorted(Path(outdir).glob("*__*.jsonl")):
        # adjudication__<model>.jsonl is the adjudicator's own cache, not a
        # coder file; adjudicate_llm.py reads it separately
        if p.name.startswith("adjudication__"):
            continue
        with open(p) as f:
            for line in f:
                if line.strip():
                    recs.append(json.loads(line))
    if not recs:
        raise SystemExit(f"no jsonl results found in {outdir}")
    df = pd.DataFrame(recs)
    # Deduplicate: reruns append to the jsonl caches, so an item can have both
    # an old failed record and a newer successful one for the same model. Keep
    # one record per (QualID, column, model): successful preferred, then latest.
    df["_ok"] = (df["error"].isin([None, ""]) | df["error"].isna()).astype(int)
    df = (df.sort_values(["_ok", "ts"])
            .drop_duplicates(["QualID", "column", "model_key"], keep="last")
            .drop(columns="_ok")
            .reset_index(drop=True))
    return df


def fleiss_kappa_binary(mat):
    """mat: items x raters, values 0/1. Returns kappa (nan if undefined)."""
    n_items, n_raters = mat.shape
    if n_items == 0 or n_raters < 2:
        return np.nan
    counts = np.stack([mat.sum(axis=1), n_raters - mat.sum(axis=1)], axis=1)  # [n1, n0]
    p_i = (np.sum(counts ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    p_bar = p_i.mean()
    p_j = counts.sum(axis=0) / (n_items * n_raters)
    p_e = np.sum(p_j ** 2)
    if np.isclose(1 - p_e, 0):
        return np.nan
    return (p_bar - p_e) / (1 - p_e)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--codebooks", default="codebooks.json")
    ap.add_argument("--consensus_k", type=int, default=2,
                    help="min models applying a code for consensus")
    ap.add_argument("--sample_n", type=int, default=0,
                    help="if >0, draw a human validation sample of this size per column")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    books = json.loads(Path(args.codebooks).read_text())
    col_to_book = {col: b for b in books.values() for col in b["columns"]}

    df = load_records(outdir)
    models = sorted(df["model_key"].unique())
    ok = df[df["error"].isin([None, ""]) | df["error"].isna()].copy()

    # surface failed calls prominently: a model whose calls all errored would
    # otherwise silently vanish from the agreement stats
    err_df = df[~(df["error"].isin([None, ""]) | df["error"].isna())]
    if len(err_df):
        print("\n*** WARNING: some API calls failed and are excluded from agreement stats ***")
        for m, g in err_df.groupby("model_key"):
            top_err = g["error"].value_counts().index[0]
            print(f"  {m}: {len(g)} failed calls "
                  f"({len(df[df.model_key == m])} total). Most common error:\n"
                  f"    {str(top_err)[:300]}")
        print("  Failed items are retried automatically: fix the underlying issue if persistent, "
              "then rerun run_llm_coding.py (no cache deletion needed) and re-aggregate.\n")

    # 1. long format -------------------------------------------------------
    long_rows = []
    for _, r in ok.iterrows():
        for c in r["codes"]:
            long_rows.append({"QualID": r["QualID"], "column": r["column"],
                              "model": r["model_key"], "code": c})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(outdir / "coded_long.csv", index=False)

    # 2. binary matrix + consensus ----------------------------------------
    bin_rows = []
    for (qid, col), grp in ok.groupby(["QualID", "column"]):
        code_ids = [c["id"] for c in col_to_book[col]["codes"]]
        model_codes = {m: set() for m in models}
        for _, r in grp.iterrows():
            model_codes[r["model_key"]] = set(r["codes"])
        n_ok = grp["model_key"].nunique()
        for cid in code_ids:
            applied = {m: int(cid in model_codes[m]) for m in models
                       if m in set(grp["model_key"])}
            n_applied = sum(applied.values())
            if n_applied == 0:
                continue
            row = {"QualID": qid, "column": col, "code": cid,
                   "n_models_coded": n_ok, "n_models_applied": n_applied,
                   "consensus": int(n_applied >= args.consensus_k)}
            for m in models:
                row[f"applied_{m}"] = applied.get(m, np.nan)
            bin_rows.append(row)
    bin_df = pd.DataFrame(bin_rows)
    bin_df.to_csv(outdir / "coded_binary.csv", index=False)

    # 3. per-code reliability ----------------------------------------------
    rel_rows = []
    for col, grp in ok.groupby("column"):
        # models that actually produced valid codings for THIS column; a model
        # whose calls all failed (or wasn't run) is absent from the pivot
        col_models = sorted(grp["model_key"].unique())
        pivot = grp.pivot_table(index="QualID", columns="model_key",
                                values="codes", aggfunc="first")
        pivot = pivot.reindex(columns=col_models)
        # keep items coded by all available models for kappa
        complete = pivot.dropna()
        code_ids = [c["id"] for c in col_to_book[col]["codes"]]
        for cid in code_ids:
            mat = np.array([[int(cid in complete.loc[q, m]) for m in col_models]
                            for q in complete.index]) if len(complete) else np.zeros((0, len(col_models)))
            prev = {m: (mat[:, i].mean() if len(mat) else np.nan)
                    for i, m in enumerate(col_models)}
            if mat.size and mat.sum() == 0:
                continue  # code never applied by anyone
            kappa = fleiss_kappa_binary(mat) if len(mat) and len(col_models) >= 2 else np.nan
            pct_agree = np.nan
            if len(mat) and len(col_models) >= 2:
                pairs = list(itertools.combinations(range(len(col_models)), 2))
                pct_agree = np.mean([np.mean(mat[:, i] == mat[:, j]) for i, j in pairs])
            row = {"column": col, "code": cid, "n_items": len(mat),
                   "n_models": len(col_models),
                   "fleiss_kappa": round(float(kappa), 3) if kappa == kappa else np.nan,
                   "pct_agreement": round(float(pct_agree), 3) if pct_agree == pct_agree else np.nan}
            for m in models:
                p = prev.get(m, np.nan)
                row[f"prevalence_{m}"] = round(float(p), 3) if p == p else np.nan
            rel_rows.append(row)
    pd.DataFrame(rel_rows).to_csv(outdir / "reliability.csv", index=False)

    # 4. item-level agreement ----------------------------------------------
    item_rows = []
    for (qid, col), grp in ok.groupby(["QualID", "column"]):
        sets = {r["model_key"]: r["codes"] for _, r in grp.iterrows()}
        ms = sorted(sets)
        if len(ms) >= 2:
            jac = np.mean([jaccard(sets[a], sets[b])
                           for a, b in itertools.combinations(ms, 2)])
        else:
            jac = np.nan
        item_rows.append({"QualID": qid, "column": col,
                          "n_models": len(ms),
                          "mean_pairwise_jaccard": round(float(jac), 3) if jac == jac else np.nan,
                          "identical_code_sets": int(len({frozenset(v) for v in sets.values()}) == 1)})
    item_df = pd.DataFrame(item_rows)
    item_df.to_csv(outdir / "item_agreement.csv", index=False)

    # 5. review queue -------------------------------------------------------
    review_rows = []
    for (qid, col), grp in df.groupby(["QualID", "column"]):
        is_err = ~(grp["error"].isin([None, ""]) | grp["error"].isna())
        sets = {r["model_key"]: frozenset(r["codes"]) for _, r in grp[~is_err].iterrows()}
        errored = grp[is_err]
        flagged = grp["flag_for_review"].any()
        disagree = len(set(sets.values())) > 1
        if not (disagree or flagged or len(errored)):
            continue
        row = {"QualID": qid, "column": col,
               "reason": ";".join(x for x, y in
                                  [("disagreement", disagree), ("model_flagged", flagged),
                                   ("model_error", len(errored) > 0)] if y),
               "text": grp["text"].iloc[0],
               # majority = applied by >= 2 of the models that coded the item
               # (matches the consensus definition in coded_binary.csv);
               # unanimous = applied by all of them
               "majority_codes": sorted({c for c in set.union(*map(set, sets.values()))
                                         if sum(c in s for s in sets.values()) >= 2}) if sets else [],
               "unanimous_codes": sorted(set.intersection(*map(set, sets.values()))) if sets else [],
               "union_codes": sorted(set.union(*map(set, sets.values()))) if sets else []}
        for _, r in grp.iterrows():
            row[f"codes_{r['model_key']}"] = ",".join(r["codes"])
            row[f"rationale_{r['model_key']}"] = r["rationale"]
            row[f"notes_{r['model_key']}"] = r["notes"]
        jac = item_df[(item_df.QualID == qid) & (item_df.column == col)]["mean_pairwise_jaccard"]
        row["mean_pairwise_jaccard"] = jac.iloc[0] if len(jac) else np.nan
        row["human_codes"] = ""     # to be filled during adjudication
        row["human_notes"] = ""
        review_rows.append(row)
    review_cols = ["QualID", "column", "reason", "text", "majority_codes",
                   "unanimous_codes", "union_codes", "mean_pairwise_jaccard",
                   "human_codes", "human_notes"]
    review = pd.DataFrame(review_rows)
    if len(review):
        review = review.sort_values(
            ["mean_pairwise_jaccard", "column"], na_position="first")
    else:
        review = pd.DataFrame(columns=review_cols)
    review.to_csv(outdir / "review_queue.csv", index=False)

    # 6. human validation sample --------------------------------------------
    if args.sample_n > 0:
        rng = np.random.default_rng(args.seed)
        samples = []
        for col, grp in ok.groupby("column"):
            qids = grp["QualID"].unique()
            take = rng.choice(qids, size=min(args.sample_n, len(qids)), replace=False)
            sub = grp[grp["QualID"].isin(take)][["QualID", "column", "text"]].drop_duplicates("QualID")
            samples.append(sub)
        hs = pd.concat(samples)
        hs["human_codes"] = ""
        hs["human_notes"] = ""
        hs.to_csv(outdir / "human_sample.csv", index=False)

    n_items = item_df.shape[0]
    n_identical = int(item_df["identical_code_sets"].sum())
    print(f"Items coded: {n_items}; identical code sets across models: "
          f"{n_identical} ({n_identical / max(n_items,1):.0%})")
    print(f"Review queue: {len(review)} items -> {outdir/'review_queue.csv'}")
    print("Wrote coded_long.csv, coded_binary.csv, reliability.csv, item_agreement.csv")


if __name__ == "__main__":
    main()
