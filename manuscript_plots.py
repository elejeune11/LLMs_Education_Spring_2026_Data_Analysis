"""
manuscript_plots.py
-------------------
Master script to generate all manuscript figures for the EK301 AI study.
Run from the directory containing the data file and all supporting modules:

    python manuscript_plots.py

Output PDFs are written to the OUTPUT_DIR directory.

TO ADD A NEW PLOT: add a call at the bottom of main() following the existing
pattern. All style settings live in plot_style.py.
"""

import os
from plot_diverging_likert import plot_diverging_likert
from plot_stacked_freq    import plot_stacked_freq
from plot_paired_histogram import plot_paired_histogram
from plot_demo_experience  import plot_demo_experience
from plot_grade_quartile_freq import plot_grade_quartile_freq
from plot_qualitative_codes import plot_qualitative_codes
from plot_qualitative_tiered import plot_impact_combined, plot_tiered_combined
from plot_qualitative_pre_post import plot_qualitative_pre_post
from plot_demo_comparison import plot_use_freq_by_demo, plot_grade_by_demo
from load_research_data import data_path, QUANT_FILE

# ── Configuration ─────────────────────────────────────────────────────────────

# Resolved via load_research_data.DATA_DIR (default data/); set the
# EK301_DATA_DIR environment variable to point somewhere else.
DATA_PATH = str(data_path(QUANT_FILE))
OUTPUT_DIR = "figures"
QUAL_DIR   = "."   # directory holding {base}_coded.csv and _qualitative_codebook.csv

# ── Helpers ───────────────────────────────────────────────────────────────────

def out(filename: str) -> str:
    """Return full output path for a figure filename."""
    return os.path.join(OUTPUT_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
# PLOT DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

def fig_understanding():
    """Self-rated understanding of engineering mechanics."""
    plot_diverging_likert(
        data_path=DATA_PATH,
        col_stems=[
            "understand_apply_principles",
            "understand_identify_mistakes",
            "understand_until_practice",
            "understand_exams_reflect",
        ],
        title="Self-rated understanding of engineering mechanics",
        out_path=out("understanding_eng_mech.pdf"),
    )


def fig_ai_frequency():
    """Self-reported AI use frequency."""
    plot_stacked_freq(
        data_path=DATA_PATH,
        col_stems=["ai_use_freq_general", "ai_use_freq_engmech"],
        panel_titles=[
            "AI use frequency — general (any purpose)",
            "AI use frequency — for engineering mechanics specifically",
        ],
        title="AI use frequency: pre vs. post",
        out_path=out("ai_use_freq_pre_post.pdf"),
    )


def fig_ai_attitudes():
    """AI attitudes, AI relative to other learning resources."""
    plot_diverging_likert(
        data_path=DATA_PATH,
        col_stems=[
            "ai_useful_coursework",
            "ai_helped_understand_engmech",
            "ai_more_confusing",
            "ai_can_evaluate_correctness",
            "ai_accuracy_concerns",
            "ai_vs_textbooks",
            "ai_vs_discussion",
        ],
        title="AI attitudes and beliefs",
        out_path=out("ai_attitude_pre_post.pdf"),
    )


def fig_ai_impact():
    """Perceived impact of AI tools on learning (0-10 rating)."""
    plot_paired_histogram(
        data_path=DATA_PATH,
        stem="ai_impact_rating",
        title="Perceived impact of AI tools on learning",
        out_path=out("ai_impact_rating_pre_post.pdf"),
    )


def fig_demo_experience():
    """AI demonstration experience (2x2, demo attendees only)."""
    plot_demo_experience(
        data_path=DATA_PATH,
        out_path=out("ai_demo_experience.pdf"),
    )


def fig_grade_quartile():
    """AI use frequency by grade quartile (2x2, domain x timepoint)."""
    plot_grade_quartile_freq(
        data_path=DATA_PATH,
        out_path=out("ai_use_freq_with_grades.pdf"),
    )


def fig_use_freq_by_demo():
    """AI use frequency, demo section vs other sections (descriptive)."""
    plot_use_freq_by_demo(
        data_path=DATA_PATH,
        out_path=out("ai_use_freq_by_demo_section.pdf"),
    )


def fig_grade_by_demo():
    """Grade quartile composition by section group (descriptive)."""
    plot_grade_by_demo(
        data_path=DATA_PATH,
        out_path=out("grade_quartile_by_demo_section.pdf"),
    )


def fig_demo_feedback_qual():
    """Qualitative coded themes for AI demo feedback."""
    plot_qualitative_codes(
        base="qualitative_coding/txt_ai_demo_feedback_post",
        title="AI demonstration feedback — coded themes",
        out_path=out("txt_ai_demo_feedback_post.pdf"),
        search_dir=QUAL_DIR,
    )


def fig_use_change_qual():
    """Qualitative coded themes for changes in AI use."""
    plot_qualitative_codes(
        base="qualitative_coding/txt_ai_use_change_post",
        title="Changes in AI use over the semester — coded themes",
        out_path=out("txt_ai_use_change_post.pdf"),
        search_dir=QUAL_DIR,
    )


def fig_verify_correctness_qual():
    """Qualitative coded verification behaviors, pre vs post."""
    plot_qualitative_pre_post(
        stem="qualitative_coding/txt_ai_verify_correctness",
        title="How students check AI-generated responses — coded, pre vs. post",
        out_path=out("txt_ai_verify_correctness_pre_post.pdf"),
        search_dir=QUAL_DIR,
    )


def fig_workflow_steps_qual():
    """Tiered qualitative coding of AI workflow steps,
    process structure + specific steps, pre vs post, in one figure."""
    plot_tiered_combined(
        stem="qualitative_coding/txt_ai_workflow_steps",
        out_path=out("txt_ai_workflow_steps_pre_post.pdf"),
        search_dir=QUAL_DIR,
        suptitle=("When you use AI tools for this course, what steps do you usually\n"
                  "take from first question to final answer?  (open-ended, pre vs. post)"),
        panel1_title="Overall process structure  (each response coded with exactly one)",
        panel2_title="Specific process steps",
        panel2_xlabel="% of responses (responses may include multiple steps)",
    )


def fig_impact_describe_qual():
    """Tiered qualitative coding of described AI impact,
    sentiment + drivers, pre vs post, in one figure."""
    plot_impact_combined(
        stem="qualitative_coding/txt_ai_impact_describe",
        out_path=out("txt_ai_impact_describe_pre_post.pdf"),
        search_dir=QUAL_DIR,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating manuscript figures...")

    fig_understanding()
    fig_ai_frequency()
    fig_ai_attitudes()
    fig_ai_impact()
    fig_demo_experience()
    fig_grade_quartile()
    fig_use_freq_by_demo()
    fig_grade_by_demo()
    fig_demo_feedback_qual()
    fig_use_change_qual()
    fig_impact_describe_qual()
    fig_verify_correctness_qual()
    fig_workflow_steps_qual()

    print(f"\nDone. All figures written to '{OUTPUT_DIR}/'.")


if __name__ == "__main__":
    main()
