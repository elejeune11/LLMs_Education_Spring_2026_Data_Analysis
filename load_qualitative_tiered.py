"""
load_qualitative_tiered.py
--------------------------
Loader for the *tiered* qualitative coding format used by pre/post free-text
items (e.g. txt_ai_impact_describe). This format differs from the primary/
secondary/tertiary format handled by load_qualitative.py:

  - Tier 1: exactly one code per response (mutually exclusive sentiment)
  - Tier 2: select-all-that-apply, stored comma-separated in one cell

Expected files for a given base stem (e.g. "txt_ai_impact_describe"):
    {stem}_pre_post_qualitative_codebook.csv
        columns: Tier, Category, Code, Description
    {stem}_pre_coded.csv   and   {stem}_post_coded.csv
        columns: StudyID, {stem}_{tp}, Tier_1_Category, Tier_1_Code, Tier_2_Codes

Usage:
    from load_qualitative_tiered import load_qualitative_tiered

    q = load_qualitative_tiered("txt_ai_impact_describe")
    q["codebook"]          # full codebook DataFrame
    q["tier1_order"]       # Tier 1 codes in codebook order
    q["tier2_order"]       # list of (category, code) for Tier 2, codebook order
    q["tier2_categories"]  # Tier 2 categories in codebook order
    q["pre"], q["post"]    # per-timepoint dicts (see below)

Each timepoint dict contains:
    n           : number of responses
    tier1_counts: dict {tier1_code: count}      (sums to n)
    tier2_counts: dict {tier2_code: count}      (responses, multi-select)
"""

import os
import pandas as pd

NONE_TOKENS = {"None", "none", "", "nan", "N/A", "n/a", "NA", "na"}


def _resolve(name: str, search_dir: str) -> str:
    for c in (os.path.join(search_dir, name), name):
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Could not find '{name}' in '{search_dir}' or cwd.")


def _split_tier2(cell) -> list[str]:
    """Split a comma-separated Tier 2 cell into a clean list of codes."""
    if pd.isna(cell) or str(cell).strip() in NONE_TOKENS:
        return []
    return [part.strip() for part in str(cell).split(",")
            if part.strip() and part.strip() not in NONE_TOKENS]


def _summarise(coded: pd.DataFrame, tier1_order, tier2_order) -> dict:
    """Count Tier 1 (mutually exclusive) and Tier 2 (multi-select) codes."""
    n = len(coded)

    tier1_counts = {code: 0 for code in tier1_order}
    for v in coded["Tier_1_Code"].dropna():
        code = str(v).strip()
        if code in tier1_counts:
            tier1_counts[code] += 1

    tier2_codes = [code for _, code in tier2_order]
    tier2_counts = {code: 0 for code in tier2_codes}
    for cell in coded["Tier_2_Codes"]:
        for code in _split_tier2(cell):
            if code in tier2_counts:
                tier2_counts[code] += 1

    return {"n": n, "tier1_counts": tier1_counts, "tier2_counts": tier2_counts}


def load_qualitative_tiered(stem: str, search_dir: str = ".") -> dict:
    """
    Load codebook + pre/post coded data for one tiered free-text item.

    Parameters
    ----------
    stem       : item stem WITHOUT the _pre/_post suffix,
                 e.g. "txt_ai_impact_describe"
    search_dir : directory to look in (default current directory)
    """
    codebook_path = _resolve(f"{stem}_pre_post_qualitative_codebook.csv", search_dir)
    pre_path      = _resolve(f"{stem}_pre_coded.csv",  search_dir)
    post_path     = _resolve(f"{stem}_post_coded.csv", search_dir)

    codebook = pd.read_csv(codebook_path)
    pre_df   = pd.read_csv(pre_path)
    post_df  = pd.read_csv(post_path)

    # Split codebook into the two tiers by the "Tier" column
    is_tier1 = codebook["Tier"].str.startswith("Tier 1")
    tier1_book = codebook[is_tier1]
    tier2_book = codebook[~is_tier1]

    # Tier 1 codes in codebook order (includes the Uncodable row)
    tier1_order = list(tier1_book["Code"])

    # Tier 2 (category, code) in codebook order, plus category order
    tier2_order = list(zip(tier2_book["Category"], tier2_book["Code"]))
    tier2_categories = list(dict.fromkeys(tier2_book["Category"].tolist()))

    return {
        "stem":             stem,
        "codebook":         codebook,
        "tier1_order":      tier1_order,
        "tier2_order":      tier2_order,
        "tier2_categories": tier2_categories,
        "pre":              _summarise(pre_df,  tier1_order, tier2_order),
        "post":             _summarise(post_df, tier1_order, tier2_order),
    }


if __name__ == "__main__":
    import sys
    stem = sys.argv[1] if len(sys.argv) > 1 else "txt_ai_impact_describe"
    q = load_qualitative_tiered(stem)
    print(f"Item stem: {q['stem']}")
    print(f"\nTier 1 order: {q['tier1_order']}")
    print(f"Tier 2 categories: {q['tier2_categories']}")
    for tp in ("pre", "post"):
        d = q[tp]
        print(f"\n=== {tp.upper()} (n={d['n']}) ===")
        print("Tier 1:", d["tier1_counts"])
        print("Tier 2:", d["tier2_counts"])
