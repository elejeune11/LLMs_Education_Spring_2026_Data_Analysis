"""
Data loader for the EK301 AI survey study (Spring 2026 de-identified format).

The data ships as SEPARATE, INTENTIONALLY UNLINKED CSV files:

    data/Spring_2026_quantitative.csv  - Likert / frequency / numeric items + grade
    data/Spring_2026_qualitative.csv   - open-ended (txt_*) responses
    data/Spring_2026_columnIDs.csv     - data dictionary

The directory is resolved by DATA_DIR below (default "data/"), overridable with
the EK301_DATA_DIR environment variable. Passing an explicit path to any loader
still works and bypasses resolution entirely.

De-identification note
----------------------
The quantitative and qualitative files use DIFFERENT id columns (QuantID vs
QualID) whose row orders are randomised independently, and whose value ranges
are disjoint on purpose. There is NO key that links a student's quantitative
record to their qualitative record. Do not attempt to join them; `load_data`
below deliberately loads only the quantitative file. Open-ended text is handled
separately (see `load_qualitative_wide` here, and the coded-CSV loaders in
load_qualitative.py / load_qualitative_tiered.py, which are unaffected by this
change).

CSV migration notes (what changed from the .xlsx version)
---------------------------------------------------------
1. `pd.read_excel` -> `pd.read_csv` throughout; default paths now end in .csv.
2. All reads pass `encoding="utf-8-sig"`. Two of the three deposited files
   carry a UTF-8 BOM and one does not; utf-8-sig strips a BOM when present and
   is a no-op when absent, so this is safe for all three.
3. Excel's `_x000D_` carriage-return artifact no longer appears. The CSV stores
   real CRLF line breaks inside quoted free-text fields, so `_clean_text_col`
   now collapses real newlines. The old `_x000D_` rule is retained so the
   function still works if anyone points it at a legacy .xlsx export.
4. `ai_demo_helped_understanding_pre` was dropped from the deposited data (it
   was administered before any demonstration occurred and is uninterpretable).
   It is therefore removed from PAIRED_STEMS, and
   `ai_demo_helped_understanding_post` is declared in POST_ONLY_COLS instead.
   Leaving it in PAIRED_STEMS made `_validate_columns` raise on the new file.
5. `demo_section` reads back as a real bool from the CSV (values TRUE/FALSE),
   so no cast is needed.
6. New: `load_column_dictionary` and `response_levels` read the data dictionary
   so scale orderings can come from the deposit itself rather than only from
   the hardcoded constants below.

Public API (unchanged where it matters for the plotting scripts)
----------------------------------------------------------------
    from load_research_data import load_data, LIKERT_ORDER, FREQ_ORDER

    data = load_data()                 # resolves data/Spring_2026_quantitative.csv
    df          = data["all"]          # full cleaned quantitative DataFrame
    pre         = data["pre_likert"]   # pre-survey Likert columns only
    post        = data["post_likert"]  # post-survey Likert columns only
    paired      = data["paired"]       # rows with both pre & post, int-encoded
                                       # (after the honesty-item rule: 93 rows)

Every plotting script only ever reads data["all"], so no plotting script needs
to change other than pointing DATA_PATH at the quantitative CSV.

Course performance is reported as `grade_quartile` (values Q1-Q4) and ordered by
GRADE_QUARTILE_ORDER. The earlier `grade_tier` / GRADE_TIER_ORDER aliases have
been removed; use the quartile names.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

# Encoding used for every read. Handles files with or without a UTF-8 BOM.
CSV_ENCODING = "utf-8-sig"

# -- Data location -------------------------------------------------------------
# The three deposited files live in a data/ subdirectory. Override with the
# EK301_DATA_DIR environment variable, or pass an explicit path to any loader.
DATA_DIR = Path(os.environ.get("EK301_DATA_DIR", "data"))

QUANT_FILE = "Spring_2026_quantitative.csv"
QUAL_FILE  = "Spring_2026_qualitative.csv"
DICT_FILE  = "Spring_2026_columnIDs.csv"


def data_path(name: str) -> Path:
    """Resolve a deposited data file by name.

    Tried in order: DATA_DIR/name relative to the working directory, then the
    bare name (the old flat layout), then DATA_DIR/name relative to this
    module's directory so scripts still work when run from elsewhere. Falls
    back to DATA_DIR/name so the FileNotFoundError names the expected location.
    """
    here = Path(__file__).resolve().parent
    for cand in (DATA_DIR / name, Path(name), here / DATA_DIR / name):
        if cand.exists():
            return cand
    return DATA_DIR / name

# -- Column groups -------------------------------------------------------------

# Quantitative-file metadata. StudyID/consent_status/section/class_year are gone
# in the de-identified export; QuantID is the (quant-only) row id, demo_section
# is a bool (True = AI-demo-exposed section), grade_quartile is the coarse grade
# bin described below.
META_COLS = ["QuantID", "demo_section", "grade_quartile"]

# Qualitative-file metadata (used only by load_qualitative_wide).
QUAL_META_COLS = ["QualID", "demo_section"]

# Open-ended columns live in the QUALITATIVE file, not the quant file.
TEXT_COLS = [
    "txt_ai_workflow_steps_pre",
    "txt_ai_verify_correctness_pre",
    "txt_ai_impact_describe_pre",
    "txt_ai_demo_feedback_post",
    "txt_ai_workflow_steps_post",
    "txt_ai_verify_correctness_post",
    "txt_ai_impact_describe_post",
    "txt_ai_use_change_post",
]

# Shared stem -> (pre col, post col) for paired analysis. Pairing is WITHIN the
# quantitative file only (QuantID is stable across a student's pre/post columns
# because they sit on the same row); this is unaffected by de-identification.
#
# NOTE: `ai_demo_helped_understanding` used to appear here. Its _pre variant was
# removed from the deposited dataset, so it is now post-only (POST_ONLY_COLS).
PAIRED_STEMS = [
    "understand_apply_principles",
    "understand_identify_mistakes",
    "understand_until_practice",
    "understand_exams_reflect",
    "ai_use_freq_general",
    "ai_use_freq_engmech",
    "ai_useful_coursework",
    "ai_helped_understand_engmech",
    "ai_more_confusing",
    "ai_can_evaluate_correctness",
    "ai_accuracy_concerns",
    "ai_vs_textbooks",
    "ai_vs_discussion",
    "ai_impact_rating",
]

# Items that exist only at POST (no _pre counterpart in the deposited file).
# Validated for presence, but never paired.
POST_ONLY_COLS = [
    "ai_demos_attended_post",
    "ai_demo_helped_understanding_post",
    "ai_demo_engagement_post",
    "ai_demo_changed_use_post",
]

# Numeric rating columns (not Likert, not frequency). These survive as floats
# and are excluded from Likert string normalisation and encoding.
NUMERIC_RATING_COLS = {
    "ai_impact_rating_pre",
    "ai_impact_rating_post",
    "ai_demos_attended_post",   # 0-9 count, not Likert
}

# Frequency stems use FREQ_ORDER rather than LIKERT_ORDER
FREQ_STEMS = {"ai_use_freq_general", "ai_use_freq_engmech"}

# Columns to exclude from Likert pre/post subsets (admin/honesty checks, demo
# attendance counts, and demo-specific items that aren't standard Likert)
EXCLUDE_FROM_LIKERT = {
    "responded_honestly_pre",
    "responded_honestly_post",
    "ai_demos_attended_post",
    "ai_demo_engagement_post",
    "ai_demo_changed_use_post",
}

# Honesty-item rule (manuscript Appendix F, step 7.4 and the note beneath the
# processing-steps table). The honesty item asks whether the respondent answered
# THAT survey honestly and thoughtfully, so it is applied per survey wave: where
# the post-survey honesty item is answered with disagreement, all other
# post-survey structured responses for that row are set to missing. Pre-survey
# responses are kept. The rule is applied from the public column, so no
# respondent identifier is needed. It is not applied to the qualitative file,
# which cannot be linked to this one (IRB protocol).
HONESTY_POST_COL = "responded_honestly_post"
HONESTY_FAIL_VALUES = {"disagree", "strongly disagree"}

# Ordered Likert scale for ordinal encoding.
#
# The deposited data preserves capitalisation exactly as Qualtrics recorded it,
# which means some items store "Strongly agree" and others "Strongly Agree".
# `_normalise_likert` folds both onto these canonical forms at load time; the
# deposited files themselves are left untouched.
LIKERT_ORDER = [
    "Strongly Disagree",
    "Disagree",
    "Neither Agree nor Disagree",
    "Agree",
    "Strongly Agree",
]

FREQ_ORDER = [
    "Never used",
    "Less than once per month",
    "1-3 times per month",
    "1-5 times per week",
    "Daily or almost daily",
]

# Grade bins. The de-identified export replaces all raw grades (midterms,
# finals, GPAs, letter grades) with a single quartile of the average z-score of
# the two in-class midterms: Q1 = bottom quartile ... Q4 = top quartile.
# Ordered best -> worst, so Q4 plots at the top.
GRADE_QUARTILE_ORDER = ["Q4", "Q3", "Q2", "Q1"]

# Regex matching placeholder "none", "n/a", "na", "null" (optional trailing
# period), used to convert these to pd.NA in text columns.
_NULL_TEXT_RE = re.compile(r'^(none\.?|n/?a\.?|na\.?|null)$', re.IGNORECASE)


# -- Validation ----------------------------------------------------------------

def _validate_columns(df: pd.DataFrame) -> None:
    """
    Check that every expected quantitative column is present. Raises ValueError
    listing any missing names so problems surface immediately rather than
    silently producing empty subsets.
    """
    expected = set(META_COLS) | set(POST_ONLY_COLS)
    for stem in PAIRED_STEMS:
        expected.add(f"{stem}_pre")
        expected.add(f"{stem}_post")

    missing = sorted(expected - set(df.columns))
    if missing:
        raise ValueError(
            f"load_data: {len(missing)} expected column(s) not found in file:\n"
            + "\n".join(f"  - {c}" for c in missing)
        )


# -- Helpers -------------------------------------------------------------------

def _normalise_likert(series: pd.Series) -> pd.Series:
    """Normalise capitalisation variants to a single canonical form."""
    mapping = {
        "strongly agree":             "Strongly Agree",
        "agree":                      "Agree",
        "neither agree nor disagree": "Neither Agree nor Disagree",
        "disagree":                   "Disagree",
        "strongly disagree":          "Strongly Disagree",
    }
    return series.str.strip().str.lower().map(mapping).fillna(series)


def _encode_likert(series: pd.Series) -> pd.Series:
    """Convert Likert strings to ordered integers (1-5). NaN preserved."""
    order = {v: i + 1 for i, v in enumerate(LIKERT_ORDER)}
    return series.map(order)


def _encode_freq(series: pd.Series) -> pd.Series:
    """Convert frequency strings to ordered integers (1-5). NaN preserved."""
    order = {v: i + 1 for i, v in enumerate(FREQ_ORDER)}
    return series.map(order)


def _clean_text_col(series: pd.Series) -> pd.Series:
    """
    Clean a single open-ended text column:

      1. Cast to str so every value is workable.
      2. Collapse embedded line breaks to a single space. The CSV export stores
         real CRLF/LF inside quoted fields; the legacy .xlsx export surfaced
         them as the literal token `_x000D_` followed by a newline. Both forms
         are handled so this works against either source.
      3. Collapse any resulting runs of whitespace.
      4. Strip leading/trailing whitespace.
      5. Convert the bare string 'nan' (a truly-null cell cast to str) to pd.NA.
      6. Convert placeholder strings ("none", "n/a", "na", "null" and trailing-
         period variants) to pd.NA so they read as missing, not as responses.
    """
    cleaned = (
        series
        .astype(str)
        .str.replace(r'_x000D_\r?\n?', ' ', regex=True)   # legacy xlsx artifact
        .str.replace(r'[\r\n]+', ' ', regex=True)         # real CSV line breaks
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
        .replace('nan', pd.NA)
    )
    cleaned = cleaned.where(~cleaned.str.match(_NULL_TEXT_RE, na=False), pd.NA)
    return cleaned


# -- Main loader (quantitative) ------------------------------------------------

def load_data(path: str | None = None) -> dict[str, pd.DataFrame]:
    """
    Load and clean the EK301 QUANTITATIVE CSV.

    Returns a dict with keys:
        all          - full cleaned quantitative DataFrame
        pre_likert   - pre-survey Likert items (normalised strings) + meta
        post_likert  - post-survey Likert items (normalised strings) + meta
        paired       - rows with both pre & post Likert data, int-encoded

    Post-survey structured responses are set to missing for any row whose
    `responded_honestly_post` answer is Disagree or Strongly Disagree (the
    honesty-item rule; manuscript Appendix F). Pre-survey responses are kept.

    Note: `text` and `grades` subsets from the old single-file API are gone.
    Text lives in the separate qualitative file (see load_qualitative_wide);
    grades collapse to the single `grade_quartile` column inside `all`.
    """
    df = pd.read_csv(path or data_path(QUANT_FILE), encoding=CSV_ENCODING)

    # Name-based validation; raises immediately on any missing expected column.
    _validate_columns(df)

    # -- Honesty-item rule (see HONESTY_POST_COL above) ----------------------
    # Blank every post-survey structured response (except the honesty item
    # itself) for rows whose post-survey honesty answer is a disagreement.
    honesty_fail = (
        df[HONESTY_POST_COL].astype("string").str.strip().str.lower()
        .isin(HONESTY_FAIL_VALUES).fillna(False)
    )
    post_cols = [c for c in df.columns
                 if c.endswith("_post") and c != HONESTY_POST_COL]
    # np.nan (not pd.NA) so the existing string normalisation below, which
    # maps the text "nan" back to missing, treats these cells as missing.
    df.loc[honesty_fail, post_cols] = np.nan

    # -- Grade quartile passthrough -------------------------------------------
    # grade_quartile already exists in the file (values Q1-Q4). Blank or missing
    # grades become <NA>.
    df["grade_quartile"] = (
        df["grade_quartile"].astype("string").str.strip().replace("", pd.NA)
    )

    # -- Identify Likert columns dynamically by suffix -------------------------
    # Name-based (suffix _pre / _post). Excluded: txt_* (not in this file
    # anyway), honesty/demo items (EXCLUDE_FROM_LIKERT), and numeric 0-N ratings
    # (NUMERIC_RATING_COLS, kept as floats). Frequency items are kept here but
    # normalised differently below.
    pre_likert_cols = [
        c for c in df.columns
        if c.endswith("_pre")
        and not c.startswith("txt_")
        and c not in EXCLUDE_FROM_LIKERT
        and c not in NUMERIC_RATING_COLS
    ]
    post_likert_cols = [
        c for c in df.columns
        if c.endswith("_post")
        and not c.startswith("txt_")
        and c not in EXCLUDE_FROM_LIKERT
        and c not in NUMERIC_RATING_COLS
    ]

    # -- Normalise Likert string capitalisation -------------------------------
    # Maps variants like "Strongly agree" to the canonical LIKERT_ORDER forms.
    # Frequency items use FREQ_ORDER and are left alone here.
    for col in pre_likert_cols + post_likert_cols:
        stem = col.rsplit("_", 1)[0]
        if stem in FREQ_STEMS:
            continue
        df[col] = _normalise_likert(df[col].astype(str).replace("nan", pd.NA))

    # -- Paired pre/post subset ------------------------------------------------
    pre_present  = df[pre_likert_cols].notna().any(axis=1)
    post_present = df[post_likert_cols].notna().any(axis=1)
    paired_df = df[pre_present & post_present].copy()

    # Add integer-encoded columns (_enc suffix) for each paired stem so ordinal
    # stats (Wilcoxon, etc.) work directly. Frequency items use FREQ_ORDER;
    # numeric ratings pass through as-is; everything else is Likert.
    for stem in PAIRED_STEMS:
        pre_col  = f"{stem}_pre"
        post_col = f"{stem}_post"
        if pre_col not in df.columns or post_col not in df.columns:
            continue
        if pre_col in NUMERIC_RATING_COLS or post_col in NUMERIC_RATING_COLS:
            paired_df[f"{stem}_pre_enc"]  = pd.to_numeric(paired_df[pre_col],  errors="coerce")
            paired_df[f"{stem}_post_enc"] = pd.to_numeric(paired_df[post_col], errors="coerce")
        elif stem in FREQ_STEMS:
            paired_df[f"{stem}_pre_enc"]  = _encode_freq(paired_df[pre_col])
            paired_df[f"{stem}_post_enc"] = _encode_freq(paired_df[post_col])
        else:
            paired_df[f"{stem}_pre_enc"]  = _encode_likert(paired_df[pre_col])
            paired_df[f"{stem}_post_enc"] = _encode_likert(paired_df[post_col])

    return {
        "all":         df,
        "pre_likert":  df[META_COLS + pre_likert_cols],
        "post_likert": df[META_COLS + post_likert_cols],
        "paired":      paired_df,
    }


# -- Optional qualitative-text loader (kept separate; NOT linkable to quant) ----

def load_qualitative_wide(path: str | None = None) -> pd.DataFrame:
    """
    Load and clean the wide QUALITATIVE CSV of raw open-ended answers, keyed by
    QualID. This is the raw-text companion to load_data; the two CANNOT be
    joined (QualID and QuantID are independent). Returns a cleaned DataFrame
    with QUAL_META_COLS + TEXT_COLS.
    """
    df = pd.read_csv(path or data_path(QUAL_FILE), encoding=CSV_ENCODING)
    missing = sorted(set(QUAL_META_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"load_qualitative_wide: missing column(s): {missing}")
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = _clean_text_col(df[col])
    keep = QUAL_META_COLS + [c for c in TEXT_COLS if c in df.columns]
    return df[keep]


# -- Data dictionary loader ----------------------------------------------------

def load_column_dictionary(path: str | None = None) -> pd.DataFrame:
    """
    Load the data dictionary that documents both data files.

    Columns: timepoint, item_number, survey_question_or_description,
             column_name, demo_section_only, result_file,
             response_options, response_type

    `keep_default_na=False` is required: the four administrative rows use the
    literal string "N/A" in `timepoint` and `item_number` as a deliberate
    marker, and pandas would otherwise silently convert it to missing.
    """
    return pd.read_csv(path or data_path(DICT_FILE), encoding=CSV_ENCODING,
                       keep_default_na=False)


def response_levels(column_name: str,
                    dictionary: pd.DataFrame | None = None,
                    path: str | None = None) -> list[str]:
    """
    Return the ordered response options for one data column, taken from the
    dictionary rather than hardcoded. Useful for setting categorical levels:

        levels = response_levels("ai_use_freq_engmech_post")
        df[col] = pd.Categorical(df[col], categories=levels, ordered=True)

    Because the deposit preserves capitalisation as recorded, these levels match
    the raw file exactly, which the hardcoded LIKERT_ORDER does not for every
    item. Returns an empty list for free-text and identifier columns.
    """
    if dictionary is None:
        dictionary = load_column_dictionary(path)
    row = dictionary.loc[dictionary["column_name"] == column_name]
    if row.empty:
        raise KeyError(f"{column_name!r} is not listed in the data dictionary.")
    opts = str(row.iloc[0]["response_options"])
    if "|" not in opts:
        return []
    return [p.strip() for p in opts.split("|") if p.strip()]


# -- Quick summary when run directly -------------------------------------------

if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else data_path(QUANT_FILE)
    data = load_data(path)

    print("=" * 55)
    print(f"  EK301 Research Data (quantitative) - {path}")
    print("=" * 55)
    for name, d in data.items():
        print(f"  {name:<12}  {d.shape[0]:>3} rows x {d.shape[1]:>2} cols")
    print("=" * 55)

    print("\nDemo section:", data["all"]["demo_section"].value_counts().to_dict())
    print("Grade quartile:", data["all"]["grade_quartile"].value_counts().to_dict())
    print(f"\nPaired respondents (pre+post): {len(data['paired'])}")
    print(f"Pre/post-only respondents:     "
          f"{len(data['all']) - len(data['paired'])}")

    try:
        qual = load_qualitative_wide()
        print(f"\nQualitative file: {qual.shape[0]} rows x {qual.shape[1]} cols "
              f"(keyed by QualID, NOT linkable to the above)")
        print("Text column null counts (after cleaning):")
        for col in TEXT_COLS:
            if col in qual.columns:
                print(f"  {col}: {qual[col].isna().sum()} NA")
    except FileNotFoundError:
        print("\n(qualitative file not found in cwd; skipping)")

    try:
        cd = load_column_dictionary()
        print(f"\nData dictionary: {cd.shape[0]} entries")
        print("Example levels for ai_use_freq_engmech_post:")
        print("  ", response_levels("ai_use_freq_engmech_post", cd))
    except FileNotFoundError:
        print("\n(column dictionary not found in cwd; skipping)")
