"""
question_metadata.py
--------------------
Human-readable question labels for every survey column in the EK301 dataset.

Labels are DERIVED at import time from the deposited data dictionary
(data/Spring_2026_columnIDs.csv) rather than duplicated here. The dictionary is
the single source of truth for question wording; this module only strips the
shared instruction stems ("Please indicate how much you agree with each
statement. - ") so that plots show the meaningful part of each item.

Kept free of third-party dependencies on purpose: it reads the dictionary with
the stdlib csv module, so importing it costs nothing.

Three public objects are exported:

  QUESTION_LABELS : dict[str, str]
      Question text keyed by column name (includes the _pre/_post suffix).
      Use this when you need the exact wording, e.g. for axis labels or tables.

        QUESTION_LABELS["ai_useful_coursework_pre"]
        # → "Generative AI tools have been a useful resource for learning
        #    university coursework."

  STEM_LABELS : dict[str, str]
      Question text keyed by stem (without _pre/_post). Pre and post share the
      same wording for paired items, so this avoids repetition.

  AI_DEMO_COLS : frozenset[str]
      Column names restricted to the AI-demonstration section, taken from the
      dictionary's demo_section_only flag.

Run this module directly to print a summary and verify the derivation.
"""

import csv
import os
import re

# ── Locating the data dictionary ──────────────────────────────────────────────
# Mirrors load_research_data.data_path: data/ relative to the working directory,
# then the old flat layout, then data/ relative to this file. Kept independent
# of load_research_data so this module stays dependency-free.

DICT_FILE = "Spring_2026_columnIDs.csv"
_DATA_DIR = os.environ.get("EK301_DATA_DIR", "data")


def _dictionary_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(_DATA_DIR, DICT_FILE),
                 DICT_FILE,
                 os.path.join(here, _DATA_DIR, DICT_FILE)):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"question_metadata: could not find {DICT_FILE}. Looked in "
        f"{_DATA_DIR}/, the working directory, and {here}/{_DATA_DIR}/. "
        f"Set EK301_DATA_DIR if the data lives elsewhere."
    )


# ── Deriving item text from the dictionary ────────────────────────────────────
# Qualtrics exports a matrix item as "<shared stem> - <item text>". These are
# the four instruction stems used in the Spring 2026 instrument; anything
# matching one of them is stripped so only the item text remains. Add to this
# tuple if a future instrument introduces a new stem.

_INSTRUCTION_STEMS = (
    'Please indicate how much you agree with each statement. '
    'Select "Not Applicable" if the statement does not apply to you.',
    "Please indicate how much you agree with each statement.",
    "Please indicate how much you agree with this statement.",
    "Please indicate your generative AI tool use frequency.",
)

# Trailing sub-labels that carry no meaning on their own. Item 15 is a single
# question whose Qualtrics answer field is labelled "Estimate".
_SUBLABEL_SUFFIXES = (" - Estimate",)

# Columns that are dataset bookkeeping rather than survey items.
_ADMIN_COLUMNS = frozenset({"QualID", "QuantID", "demo_section", "grade_quartile"})


def _clean_label(text: str) -> str:
    """Strip a shared instruction stem and any trailing sub-label."""
    text = re.sub(r"\s+", " ", str(text)).strip()
    for stem in _INSTRUCTION_STEMS:
        prefix = stem + " - "
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
            break
    for suffix in _SUBLABEL_SUFFIXES:
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
            break
    return text


def _load() -> tuple[dict[str, str], frozenset[str]]:
    labels: dict[str, str] = {}
    demo_cols: set[str] = set()
    with open(_dictionary_path(), encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            col = (row.get("column_name") or "").strip()
            if not col or col in _ADMIN_COLUMNS:
                continue
            labels[col] = _clean_label(row.get("survey_question_or_description", ""))
            if (row.get("demo_section_only") or "").strip().lower() == "yes":
                demo_cols.add(col)
    return labels, frozenset(demo_cols)


QUESTION_LABELS, AI_DEMO_COLS = _load()


# ── Stem labels (pre/post share identical wording for paired items) ───────────

def _make_stem_labels() -> dict[str, str]:
    stems: dict[str, str] = {}
    for col, label in QUESTION_LABELS.items():
        for suffix in ("_pre", "_post"):
            if col.endswith(suffix):
                stem = col[: -len(suffix)]
                # First occurrence wins; pre and post are identical for paired stems
                stems.setdefault(stem, label)
                break
    return stems


STEM_LABELS: dict[str, str] = _make_stem_labels()


# ── Sanity check when run directly ───────────────────────────────────────────

if __name__ == "__main__":
    print(f"Source          : {_dictionary_path()}")
    print(f"QUESTION_LABELS : {len(QUESTION_LABELS)} entries")
    print(f"STEM_LABELS     : {len(STEM_LABELS)} entries")
    print(f"AI_DEMO_COLS    : {len(AI_DEMO_COLS)} entries")
    print()

    # Any label still containing " - " probably means an instruction stem is
    # missing from _INSTRUCTION_STEMS and is leaking into the plotted text.
    suspect = {c: t for c, t in QUESTION_LABELS.items() if " - " in t}
    if suspect:
        print("WARNING: labels still contain ' - '; add the stem to "
              "_INSTRUCTION_STEMS:")
        for c, t in suspect.items():
            print(f"  {c}: {t!r}")
        print()

    print("Sample STEM_LABELS:")
    for stem in list(STEM_LABELS)[:5]:
        print(f"  {stem!r:<40} -> {STEM_LABELS[stem]!r}")
