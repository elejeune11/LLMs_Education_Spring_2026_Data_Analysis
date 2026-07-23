# Protocol: Multi-LLM First-Pass Coding of Spring 2026 Qualitative Survey Data

## Purpose and scope

This protocol specifies a reproducible procedure for producing a first-pass coding of the eight free-text survey columns in `Spring_2026_qualitative.xlsx` (N = 105 respondents; 65 demo section, 40 control) using three independent LLMs, followed by agreement analysis, human adjudication of disagreements, and validation against an independent human-coded subsample. LLM output is treated as a first pass, not a final coding: consensus codes are provisional until the validation step passes, and all disagreements are adjudicated by a human coder.

## Column to codebook mapping

| Codebook | Columns coded | Structure |
|---|---|---|
| demo_feedback | txt_ai_demo_feedback_post | Flat, multi-select |
| impact_describe | txt_ai_impact_describe_pre, _post | Tier 1 exactly one, Tier 2 select-all |
| use_change | txt_ai_use_change_post | Flat, multi-select |
| verify_correctness | txt_ai_verify_correctness_pre, _post | Flat, multi-select |
| workflow_steps | txt_ai_workflow_steps_pre, _post | Tier 1 exactly one (convention, see issues log), Tier 2 select-all |

The five codebook CSVs live in the `codebooks/` folder and are compiled into `codebooks.json` by `build_codebooks.py`, which reads them and attaches the machine constraints (exclusive standalone codes, tier min/max, extra coding rules defined in its RULES dict). To revise a codebook, edit its CSV in `codebooks/`, rerun `build_codebooks.py`, delete the affected JSONL caches, and rerun the coding script. See codebook_changelog.md for what changed relative to the original codebooks.

## Unit of analysis and missing data

The coding unit is the full response to one question by one respondent (no sub-sentence unitization; multi-label coding captures multiple ideas within a response). Truly blank cells are excluded as item nonresponse. Typed non-answers ("NA", "yeah", "-") are coded, and should receive the relevant Uncodable/Noise code; the loader uses `keep_default_na=False` so pandas does not silently convert the literal string "NA" to a missing value (12 of the use_change responses are typed NA-like strings and would otherwise vanish).

## Coding procedure

Step 1. Build the machine-readable codebook: `python build_codebooks.py --codebook_dir codebooks --out codebooks.json`.

Step 2. Run the three models independently: `python run_llm_coding.py --data Spring_2026_qualitative.xlsx --codebooks codebooks.json --outdir results`. The models are Claude Sonnet 5 (`claude-sonnet-5`), GPT-5.5 (`gpt-5.5`), and Gemini 3.5 Flash (`gemini-3.5-flash`); API keys are read from a `.env` file in the working directory via python-dotenv using the FEM-bench key names (`CLAUDE_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`), and a missing key raises a clear error at call time. Install dependencies with `pip install pandas openpyxl python-dotenv anthropic openai google-genai`. Each (response, model) pair is one API call. The system prompt contains the survey question, the full codebook (IDs, apply-when descriptions, and, where available, do-not-apply criteria, distinguish-from notes, and verbatim examples), the exclusivity and tier rules, and a strict JSON output schema: `codes` (array of IDs), `rationale` (one sentence citing the response wording), `flag_for_review` (bool), `notes` (required for Other/Idiosyncratic/NEW codes). Sampling parameters are left at provider defaults because current frontier models reject or discourage overrides (Claude Sonnet 5 returns a 400 error for non-default temperature/top_p/top_k; GPT-5.5 reasoning models reject temperature); reproducibility rests on archiving all raw outputs and pinning exact model IDs. Models never see each other's output, other respondents' answers, respondent identity, or section assignment. Outputs are validated programmatically (known IDs only, minimum one code, exclusive codes standalone, tier cardinality); an invalid output triggers one repair retry with the validation error appended; persistent failures are logged and routed to the review queue. Results are cached per (column, model) as JSONL, so runs are resumable. Recommended smoke test before the full run: `--limit 5 --columns txt_ai_use_change_post`, then inspect the JSONL by hand.

Step 3. Aggregate: `python aggregate_results.py --outdir results --codebooks codebooks.json --sample_n 25`. This produces the binary code-by-model matrix, consensus codes (default: applied by at least 2 of 3 models), per-code Fleiss' kappa and percent agreement across models, per-item mean pairwise Jaccard similarity of code sets, a review queue, and a human validation sample.

Step 4a. Automated adjudication (no human required): `python adjudicate_llm.py --outdir results --codebooks codebooks.json`. Items where the three coders returned identical code sets and none flagged uncertainty pass through as final (source: unanimous). All other items get one adjudication call to a designated LLM (default Claude Sonnet 5; configurable) that sees the response, the codebook, and each coder's codes and rationales, and rules strictly against the codebook; validation and caching match the coding stage, with a majority-vote fallback if the call fails. The result, final_codes.csv / final_long.csv, is a complete first-pass coding with every item labeled unanimous, adjudicated, or majority-fallback, so the machine-resolved subset stays auditable. Note the self-preference caveat: the default adjudicator is also a rater; the adjudicated label preserves the ability to audit that subset. Inter-model agreement plus adjudication establishes reliability, not validity - step 5 remains the accuracy check and should be completed before publication, but can be deferred.

Step 4b. Human adjudication (optional while step 4a is in use). A human coder resolves every item in `review_queue.csv` (any inter-model disagreement, any model-set review flag, any API/validation failure), recording final codes in the `human_codes` column. Items with identical code sets across all three models pass through with consensus codes unless step 5 fails.

Step 5. Reporting: `python report_stats.py --outdir results --codebooks codebooks.json --report_dir report` produces reliability_summary.csv and reliability_by_code.csv, paste-ready LaTeX tables (table_reliability.tex for the main text, table_reliability_by_code.tex for the appendix), and methods_boilerplate.tex, a drafted methods paragraph with model IDs, agreement statistics, and adjudication counts filled in; verify the numbers and edit the wording before use. Kappa is reported only for codes with mean prevalence at or above 5 percent (configurable via --min_prev) because it is unstable for rare codes; prevalence and percent agreement are reported for all codes.

## Interpretation notes

Inter-model agreement is a reliability signal, not a validity guarantee: three models can agree on a wrong code. Kappa is unstable for very low-prevalence codes; interpret alongside prevalence and percent agreement. Consensus at 2-of-3 favors precision over recall for rare codes; if recall matters more for a given analysis (e.g., counting any mention of Technical Failure), use the union coding in `coded_binary.csv` (`n_models_applied >= 1`) and report both.
