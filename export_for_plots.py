"""
export_for_plots.py
-------------------
Bridge between the LLM coding pipeline and the manuscript plotting
infrastructure. Reads final_codes.csv (from adjudicate_llm.py) and
codebooks.json, and writes the files that load_qualitative.py and
load_qualitative_tiered.py expect, into --plot_dir (default
qualitative_coding/):

FLAT items (multi-select; rank columns hold code applications):
    txt_ai_demo_feedback_post, txt_ai_use_change_post,
    txt_ai_verify_correctness_pre, txt_ai_verify_correctness_post
  ->  {base}_qualitative_codebook.csv   (Category, Code, Description)
      {base}_coded.csv                  (StudyID, {base},
                                         Primary/Secondary/Tertiary/... Category+Code)
  Rank order follows codebook order and carries NO importance meaning; the
  LLM coding is unordered multi-label and the plots count all ranks equally.

TIERED items (Tier 1 exactly-one + Tier 2 select-all, pre/post):
    txt_ai_impact_describe, txt_ai_workflow_steps
  ->  {stem}_pre_post_qualitative_codebook.csv  (Tier, Category, Code, Description)
      {stem}_pre_coded.csv / {stem}_post_coded.csv
        (StudyID, {stem}_{tp}, Tier_1_Category, Tier_1_Code, Tier_2_Codes)
  Tier_2_Codes is comma-separated code NAMES (asserted comma-free).

Run after adjudication:
    python export_for_plots.py --outdir results --codebooks codebooks.json \
        --plot_dir qualitative_coding
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RANK_NAMES = ["Primary", "Secondary", "Tertiary", "Quaternary", "Quinary",
              "Senary", "Septenary", "Octonary"]

FLAT_COLUMNS = ["txt_ai_demo_feedback_post", "txt_ai_use_change_post",
                "txt_ai_verify_correctness_pre", "txt_ai_verify_correctness_post"]
TIERED_STEMS = {"txt_ai_impact_describe": ["txt_ai_impact_describe_pre",
                                           "txt_ai_impact_describe_post"],
                "txt_ai_workflow_steps": ["txt_ai_workflow_steps_pre",
                                          "txt_ai_workflow_steps_post"]}


def code_maps(books):
    """column -> (ordered code ids, id -> {tier, category, name, description})"""
    out = {}
    for b in books.values():
        info = {c["id"]: c for c in b["codes"]}
        for col in b["columns"]:
            out[col] = ([c["id"] for c in b["codes"]], info)
    return out


def export_flat(fin, col, order, info, plot_dir):
    # codebook
    cb = pd.DataFrame([{"Category": info[i]["category"], "Code": info[i]["name"],
                        "Description": info[i]["description"]} for i in order])
    cb.to_csv(plot_dir / f"{col}_qualitative_codebook.csv", index=False)

    sub = fin[fin["column"] == col]
    max_codes = 1
    rows = []
    for r in sub.itertuples():
        ids = [c for c in str(r.final_codes).split(";") if c]
        ids = [i for i in order if i in ids]           # codebook order
        max_codes = max(max_codes, len(ids))
        row = {"StudyID": r.QualID, col: r.text}
        for k, cid in enumerate(ids):
            row[f"{RANK_NAMES[k]} Category"] = info[cid]["category"]
            row[f"{RANK_NAMES[k]} Code"] = info[cid]["name"]
        rows.append(row)
    cols = ["StudyID", col]
    for k in range(max_codes):
        cols += [f"{RANK_NAMES[k]} Category", f"{RANK_NAMES[k]} Code"]
    coded = pd.DataFrame(rows).reindex(columns=cols)
    coded.to_csv(plot_dir / f"{col}_coded.csv", index=False)
    print(f"  {col}: {len(coded)} responses, up to {max_codes} codes "
          f"-> {col}_coded.csv")


def export_tiered(fin, stem, columns, order, info, plot_dir):
    # codebook (shared across pre/post; same codebook by construction)
    cb = pd.DataFrame([{"Tier": info[i]["tier_label"], "Category": info[i]["category"],
                        "Code": info[i]["name"], "Description": info[i]["description"]}
                       for i in order])
    cb.to_csv(plot_dir / f"{stem}_pre_post_qualitative_codebook.csv", index=False)

    t2_names = [info[i]["name"] for i in order if info[i]["tier"] == "T2"]
    bad = [n for n in t2_names if "," in n]
    assert not bad, f"Tier 2 code names contain commas (breaks the loader): {bad}"

    for col in columns:
        tp = "pre" if col.endswith("_pre") else "post"
        sub = fin[fin["column"] == col]
        rows = []
        for r in sub.itertuples():
            ids = [c for c in str(r.final_codes).split(";") if c]
            ids = [i for i in order if i in ids]
            t1 = [i for i in ids if info[i]["tier"] == "T1"]
            t2 = [i for i in ids if info[i]["tier"] == "T2"]
            if len(t1) != 1:
                print(f"  WARNING: {col} StudyID {r.QualID} has {len(t1)} Tier 1 "
                      f"codes ({t1}); exporting "
                      f"{'first' if t1 else 'blank'} (source={r.source})")
            rows.append({
                "StudyID": r.QualID,
                col: r.text,
                "Tier_1_Category": info[t1[0]]["category"] if t1 else "",
                "Tier_1_Code": info[t1[0]]["name"] if t1 else "",
                "Tier_2_Codes": ", ".join(info[i]["name"] for i in t2),
            })
        coded = pd.DataFrame(rows, columns=["StudyID", col, "Tier_1_Category",
                                            "Tier_1_Code", "Tier_2_Codes"])
        coded.to_csv(plot_dir / f"{stem}_{tp}_coded.csv", index=False)
        print(f"  {col}: {len(coded)} responses -> {stem}_{tp}_coded.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--codebooks", default="codebooks.json")
    ap.add_argument("--plot_dir", default="qualitative_coding")
    args = ap.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(exist_ok=True)

    fin = pd.read_csv(Path(args.outdir) / "final_codes.csv").fillna("")
    if "text" not in fin.columns:
        raise SystemExit("final_codes.csv has no 'text' column; rerun "
                         "adjudicate_llm.py with the current version.")
    books = json.loads(Path(args.codebooks).read_text())
    cmap = code_maps(books)

    # tier labels for tiered codebook export come from the CSV Tier strings,
    # which live in codebooks.json as T1/T2 plus the codebook file; rebuild a
    # human label from the tier + selection rules
    for b in books.values():
        for c in b["codes"]:
            c["tier_label"] = ""
            if b["selection"] == "tiered":
                rule = b["tier_rules"][c["tier"]]
                pick = ("select exactly one" if rule["max"] == 1
                        else "select all that apply")
                c["tier_label"] = f"Tier {c['tier'][-1]}: {c['category'] if False else ''}".strip()
                c["tier_label"] = f"Tier {c['tier'][-1]} ({pick})"

    print("Flat items:")
    for col in FLAT_COLUMNS:
        order, info = cmap[col]
        export_flat(fin, col, order, info, plot_dir)

    print("Tiered items:")
    for stem, columns in TIERED_STEMS.items():
        order, info = cmap[columns[0]]
        export_tiered(fin, stem, columns, order, info, plot_dir)

    print(f"\nDone. Plot-ready files in '{plot_dir}/'.")


if __name__ == "__main__":
    main()
