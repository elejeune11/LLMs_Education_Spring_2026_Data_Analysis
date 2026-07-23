"""
build_codebooks.py

Reads the five improved codebook CSVs (uniform format: Code ID, Tier,
Category, Code, Description, Do NOT apply when, Distinguish from, Example
responses) and writes codebooks.json for run_llm_coding.py. Selection
structure, exclusive codes, tier rules, and extra rules are defined in
RULES below; edit here if coding rules change.

Run:  python build_codebooks.py --codebook_dir codebooks --out codebooks.json
"""

import argparse
import json
from pathlib import Path

import pandas as pd

RULES = {
    "demo_feedback": {
        "file": "codebook_txt_ai_demo_feedback_post.csv",
        "columns": ["txt_ai_demo_feedback_post"],
        "question_text": ("What did you like/dislike about the AI demonstrations you "
                          "attended? If you have one suggestion to help us improve the "
                          "AI demonstrations, what would it be?"),
        "selection": "multi", "min_codes": 1,
        "exclusive_codes": ["DF16", "DF17"],
        "tier_rules": None,
        "extra_rules": [
            "A single response can contain likes, dislikes, and suggestions; apply one code per distinct point.",
            "DF15 (Did Not Attend) may co-occur with substantive codes only when partial attendance is explicitly given as the reason for limited feedback; otherwise it stands alone.",
            "Complaints and matching suggestions are separate codes: code both when both are stated (e.g., DF07 + DF13, DF06 + DF11).",
        ],
    },
    "impact_describe": {
        "file": "codebook_txt_ai_impact_describe_pre_post.csv",
        "columns": ["txt_ai_impact_describe_pre", "txt_ai_impact_describe_post"],
        "question_text": ("How would you describe the impact of AI tools (positive and "
                          "negative) on your learning in this course?"),
        "selection": "tiered", "min_codes": 1,
        "exclusive_codes": ["IM06"],
        "tier_rules": {"T1": {"min": 1, "max": 1}, "T2": {"min": 0, "max": None}},
        "extra_rules": [
            "Select exactly one Tier 1 code, then all Tier 2 driver codes that apply. An empty Tier 2 is valid when no specific driver is stated.",
            "Tier 1 net-direction rule: if the student states which way the impact nets out, code that direction and put caveats in Tier 2; Mixed (IM03) is reserved for unresolved trade-offs or explicit both.",
            "If Tier 1 is IM05 (Does Not Use AI), Tier 2 codes may be applied only for concrete stated reasons for non-use.",
            "If Tier 1 is IM06 (Irrelevant / Noise), select no Tier 2 codes.",
            "If IM14 (Other / Idiosyncratic) is applied, describe the novel driver in the notes field.",
        ],
    },
    "use_change": {
        "file": "codebook_txt_ai_use_change_post.csv",
        "columns": ["txt_ai_use_change_post"],
        "question_text": ("Please describe any changes in how you use AI tools for your "
                          "learning since the start of the semester."),
        "selection": "multi", "min_codes": 1,
        "exclusive_codes": ["UC01", "UC14", "UC15"],
        "tier_rules": None,
        "extra_rules": [
            "UC01 (No Substantive Change), UC14 (Irrelevant / Noise), and UC15 (Never Used AI) are standalone codes; never combine them with other codes.",
            "Direction rule: Manner of Use codes (UC05-UC07) describe the manner the student has adopted or increased, never a manner they report abandoning.",
            "UC03 (Decreased) co-occurs with UC12 when a non-AI replacement is named, and with UC10 when the reason is distrust or discovered limitations.",
        ],
    },
    "verify_correctness": {
        "file": "codebook_txt_ai_verify_correctness_pre_post.csv",
        "columns": ["txt_ai_verify_correctness_pre", "txt_ai_verify_correctness_post"],
        "question_text": ("What, if anything, do you do to check whether an AI-generated "
                          "response is correct?"),
        "selection": "multi", "min_codes": 1,
        "exclusive_codes": ["E1", "E2"],
        "tier_rules": None,
        "extra_rules": [
            "E1 (Does Not Use AI) and E2 (Uncodable) are standalone codes; never combine them with other codes.",
            "If any other code can be applied even at low confidence, apply it and set flag_for_review rather than defaulting to E2.",
        ],
    },
    "workflow_steps": {
        "file": "codebook_txt_ai_workflow_steps_pre_post.csv",
        "columns": ["txt_ai_workflow_steps_pre", "txt_ai_workflow_steps_post"],
        "question_text": ("When you use AI tools for this course, what steps do you usually "
                          "take from first question to final answer?"),
        "selection": "tiered", "min_codes": 1,
        "exclusive_codes": ["WF08"],
        "tier_rules": {"T1": {"min": 1, "max": 1}, "T2": {"min": 0, "max": None}},
        "extra_rules": [
            "Select exactly one Tier 1 code for the DOMINANT overall process structure, then all Tier 2 step codes that apply.",
            "Tie-break: if two structures are described, select the one covering the larger share of the described workflow and set flag_for_review.",
            "If Tier 1 is WF06 (Non-Use) select Tier 2 codes only when a limitation or distrust (WF18/WF19) is the stated reason for non-use; if Tier 1 is WF08 (Noise), select no Tier 2 codes.",
        ],
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codebook_dir", default="codebooks")
    ap.add_argument("--out", default="codebooks.json")
    args = ap.parse_args()
    d = Path(args.codebook_dir)

    books = {}
    for name, rules in RULES.items():
        df = pd.read_csv(d / rules["file"]).fillna("")
        codes = []
        for _, r in df.iterrows():
            tier = None
            if rules["selection"] == "tiered":
                tier = "T1" if "Tier 1" in str(r["Tier"]) else "T2"
            codes.append({
                "id": str(r["Code ID"]).strip(),
                "tier": tier,
                "category": str(r["Category"]).strip(),
                "name": str(r["Code"]).strip(),
                "description": str(r["Description"]).strip(),
                "do_not_apply_when": str(r["Do NOT apply when"]).strip(),
                "distinguish_from": str(r["Distinguish from"]).strip(),
                "examples": str(r["Example responses"]).strip(),
            })
        ids = [c["id"] for c in codes]
        assert len(ids) == len(set(ids)), f"duplicate ids in {name}"
        for e in rules["exclusive_codes"]:
            assert e in ids, f"exclusive code {e} not found in {name}"
        books[name] = {
            "codebook_file": rules["file"],
            "columns": rules["columns"],
            "question_text": rules["question_text"],
            "selection": rules["selection"],
            "min_codes": rules["min_codes"],
            "exclusive_codes": rules["exclusive_codes"],
            "tier_rules": rules["tier_rules"],
            "codes": codes,
            "extra_rules": rules["extra_rules"],
        }

    Path(args.out).write_text(json.dumps(books, indent=2))
    n = sum(len(b["codes"]) for b in books.values())
    print(f"Wrote {args.out}: {len(books)} codebooks, {n} codes total.")


if __name__ == "__main__":
    main()
