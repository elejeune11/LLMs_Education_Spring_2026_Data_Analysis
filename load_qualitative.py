"""
load_qualitative.py
-------------------
Loader for qualitative codebooks and coded free-text data. The coding format
is shared across all open-ended (txt_*) items, so this loader is generic: pass
the item's base name and it locates and parses both files.

Expected files for a given base name (e.g. "txt_ai_demo_feedback_post"):
    {base}_qualitative_codebook.csv   columns: Category, Code, Description
    {base}_coded.csv                  columns: StudyID, {base},
                                               Primary Category, Primary Code,
                                               Secondary Category, Secondary Code,
                                               Tertiary Category, Tertiary Code

Usage:
    from load_qualitative import load_qualitative

    q = load_qualitative("txt_ai_demo_feedback_post")
    q["codebook"]    # DataFrame: Category, Code, Description
    q["coded"]       # DataFrame: one row per response, raw coding columns
    q["long"]        # tidy DataFrame: one row per code APPLICATION
                     #   (StudyID, rank, category, code)
    q["n_responses"] # total number of coded responses
    q["category_order"]  # categories in codebook order
    q["code_order"]      # codes in codebook order (list of (category, code))
"""

import os
import pandas as pd

RANKS = ["Primary", "Secondary", "Tertiary"]   # legacy default; see _detect_ranks
NONE_TOKENS = {"None", "none", "", "nan"}


def _detect_ranks(coded) -> list[str]:
    """Rank names present in the coded file, in column order.

    Historically fixed to Primary/Secondary/Tertiary; LLM-coded exports may
    add further ranks (Quaternary, ...) when a response carries more codes.
    Any '<X> Code' column with a matching '<X> Category' column counts.
    """
    ranks = []
    for col in coded.columns:
        if col.endswith(" Code"):
            stem = col[: -len(" Code")]
            if f"{stem} Category" in coded.columns:
                ranks.append(stem)
    return ranks or RANKS


def _resolve(path_or_base: str, suffix: str, search_dir: str) -> str:
    """Find a file given a base name and suffix, trying a few conventions."""
    candidates = [
        os.path.join(search_dir, f"{path_or_base}{suffix}"),
        f"{path_or_base}{suffix}",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Could not find '{path_or_base}{suffix}' in '{search_dir}' or cwd."
    )


def load_qualitative(base: str, search_dir: str = ".") -> dict:
    """
    Load codebook + coded data for one free-text item.

    Parameters
    ----------
    base       : item base name, e.g. "txt_ai_demo_feedback_post"
    search_dir : directory to look in (default current directory)
    """
    codebook_path = _resolve(base, "_qualitative_codebook.csv", search_dir)
    coded_path    = _resolve(base, "_coded.csv", search_dir)

    codebook = pd.read_csv(codebook_path)
    coded    = pd.read_csv(coded_path)

    # Codebook-defined ordering (used to keep plots in a sensible order)
    category_order = list(dict.fromkeys(codebook["Category"].tolist()))
    code_order     = list(zip(codebook["Category"], codebook["Code"]))

    # Build tidy long form: one row per code application
    ranks = _detect_ranks(coded)
    records = []
    for _, row in coded.iterrows():
        for rank in ranks:
            cat  = row.get(f"{rank} Category")
            code = row.get(f"{rank} Code")
            if pd.isna(code) or str(code).strip() in NONE_TOKENS:
                continue
            records.append({
                "StudyID":  row["StudyID"],
                "rank":     rank,
                "category": str(cat).strip(),
                "code":     str(code).strip(),
            })
    long = pd.DataFrame(records)

    return {
        "base":           base,
        "codebook":       codebook,
        "coded":          coded,
        "long":           long,
        "n_responses":    len(coded),
        "category_order": category_order,
        "code_order":     code_order,
    }


if __name__ == "__main__":
    import sys
    base = sys.argv[1] if len(sys.argv) > 1 else "txt_ai_demo_feedback_post"
    q = load_qualitative(base)
    print(f"Item: {q['base']}")
    print(f"Responses: {q['n_responses']}")
    print(f"Code applications: {len(q['long'])}")
    print(f"\nCategories (codebook order): {q['category_order']}")
    print("\nCode frequency (all ranks):")
    counts = q["long"].groupby(["category", "code"]).size().sort_values(ascending=False)
    for (cat, code), c in counts.items():
        print(f"   {cat:<12} {code:<40} {c}")
