"""
adjudicate_llm.py

Fully automated final-coding stage. For every (response, question):

- If all models that coded it returned IDENTICAL code sets and none flagged
  it, that set becomes the final coding (source = "unanimous").
- Otherwise, one adjudication call is made to a designated LLM that sees the
  response, the full codebook, and each model's codes + rationales, and
  returns a final code set under the same validation rules (source =
  "adjudicated"). If the adjudication call fails, the majority coding
  (codes applied by >= 2 models) is used (source = "majority-fallback").

Outputs (in --outdir):
  final_codes.csv   one row per (QualID, column): final codes, source, flags
  final_long.csv    one row per (QualID, column, code) for analysis

Adjudication calls are cached to adjudication__<model_key>.jsonl and are
resumable, same as the coding stage.

Caveat for the methods section: the default adjudicator (Claude Sonnet 5) is
also one of the three raters, which introduces a mild self-preference risk;
the adjudicator is instructed to judge strictly against the codebook, and
every adjudicated item remains marked in final_codes.csv so the subset can
be audited later. Inter-model agreement plus adjudication is a reliability
mechanism, not a validity guarantee; the human validation sample in the
protocol remains the check that the coding is ACCURATE, whenever bandwidth
allows.

Run:  python adjudicate_llm.py --outdir results --codebooks codebooks.json
      [--adjudicator anthropic]   (anthropic | openai | gemini)
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from run_llm_coding import (MODELS, PROVIDER_FN, build_system_prompt,
                            call_with_retries, extract_json, validate)

# adjudication rulings are few and high-stakes; allow more repair rounds for
# invalid JSON / tier-rule violations than the coding stage uses
ADJ_REPAIR_RETRIES = 2
from aggregate_results import load_records


def input_fingerprint(codings):
    """Stable hash of the coder inputs an adjudication was based on. A cached
    ruling is reused only while the coders' code sets are unchanged, so e.g.
    retried failed calls that add a third coder invalidate exactly the
    affected items rather than requiring a cache wipe."""
    canon = sorted((m, tuple(sorted(c["codes"]))) for m, c in codings.items())
    return hashlib.sha1(json.dumps(canon).encode()).hexdigest()[:16]


def build_adjudicator_prompt(book, text, codings):
    lines = [f'Student response:\n"""\n{text}\n"""\n',
             "Three independent coders (LLMs) applied the codebook to this response and disagreed or flagged uncertainty. Their codings:"]
    for m, c in codings.items():
        lines.append(f"- Coder {m}: codes = {sorted(c['codes'])}; rationale: {c['rationale'] or '(none)'}"
                     + (f"; notes: {c['notes']}" if c.get("notes") else ""))
    lines.append(
        "\nYou are the adjudicator. Judge strictly against the codebook definitions, "
        "not by majority vote and not by deference to any coder: a code belongs in the "
        "final set if and only if the response, as written, meets that code's criteria. "
        "You may keep, drop, or add codes relative to the coders' sets. "
        "Return ONLY the JSON object in the required schema.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--codebooks", default="codebooks.json")
    ap.add_argument("--adjudicator", default=None, choices=list(MODELS.keys()),
                    help="provider whose client to use; inferred from the model "
                         "name if omitted")
    ap.add_argument("--adjudicator_model", default="claude-fable-5",
                    help="exact model ID for adjudication (default: claude-fable-5, "
                         "outside the rater trio to avoid model-level self-preference)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    books = json.loads(Path(args.codebooks).read_text())
    col_to_book = {col: b for b in books.values() for col in b["columns"]}

    provider = args.adjudicator
    if provider is None:  # infer client from the model name
        m = args.adjudicator_model.lower()
        provider = ("anthropic" if m.startswith("claude")
                    else "openai" if m.startswith(("gpt", "o"))
                    else "gemini" if m.startswith("gemini")
                    else None)
        if provider is None:
            raise SystemExit(f"cannot infer provider for '{args.adjudicator_model}'; "
                             "pass --adjudicator anthropic|openai|gemini")
    cfg = {"provider": provider, "model": args.adjudicator_model}
    fn = PROVIDER_FN[provider]
    print(f"adjudicator: {cfg['model']} (via {provider} client)")

    df = load_records(outdir)  # deduped: latest successful record per item/model
    ok_mask = df["error"].isin([None, ""]) | df["error"].isna()
    ok = df[ok_mask]

    cache_path = outdir / f"adjudication__{cfg['model'].replace('.', '-')}.jsonl"
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if not rec.get("error"):
                        cache[(rec["QualID"], rec["column"])] = rec


    final_rows = []
    to_adjudicate = []
    for (qid, col), grp in ok.groupby(["QualID", "column"]):
        codings = {r["model_key"]: {"codes": r["codes"], "rationale": r["rationale"],
                                    "notes": r["notes"]}
                   for _, r in grp.iterrows()}
        sets = [frozenset(c["codes"]) for c in codings.values()]
        flagged = any(bool(r) for r in grp["flag_for_review"])
        unanimous = len(set(sets)) == 1 and len(sets) >= 2
        if unanimous and not flagged:
            final_rows.append({"QualID": qid, "column": col,
                               "text": grp["text"].iloc[0],
                               "final_codes": ";".join(sorted(sets[0])),
                               "source": "unanimous", "flag_for_review": False,
                               "notes": ""})
        else:
            to_adjudicate.append((qid, col, grp, codings))

    print(f"{len(final_rows)} items unanimous; {len(to_adjudicate)} to adjudicate "
          f"({len(cache)} already cached)")

    with open(cache_path, "a") as f:
        for qid, col, grp, codings in to_adjudicate:
            book = col_to_book[col]
            text = grp["text"].iloc[0]
            majority = sorted({c for s in codings.values() for c in s["codes"]
                               if sum(c in s2["codes"] for s2 in codings.values()) >= 2})
            fp = input_fingerprint(codings)
            cached = cache.get((qid, col))
            # reuse a cached ruling only if the coder inputs it was based on
            # are unchanged; legacy records without a fingerprint are trusted
            if cached is not None and cached.get("input_fp", fp) == fp:
                rec = cached
            else:
                system = build_system_prompt(book)
                user = build_adjudicator_prompt(book, text, codings)
                parsed, raw, err = None, "", None
                for attempt in range(ADJ_REPAIR_RETRIES + 1):
                    try:
                        raw = call_with_retries(fn, system, user, cfg["model"])
                        parsed = extract_json(raw)
                        err = validate(parsed, book)
                    except Exception as e:
                        err, parsed = str(e), None
                    if err is None:
                        break
                    user = (build_adjudicator_prompt(book, text, codings)
                            + f"\n\nYour previous output was invalid: {err}. "
                              "Return ONLY a corrected JSON object.")
                rec = {"QualID": qid, "column": col, "model": cfg["model"],
                       "input_fp": fp,
                       "text": text,
                       "codes": (parsed or {}).get("codes", []),
                       "rationale": (parsed or {}).get("rationale", ""),
                       "flag_for_review": (parsed or {}).get("flag_for_review", True),
                       "notes": (parsed or {}).get("notes", ""),
                       "error": err, "ts": time.strftime("%Y-%m-%dT%H:%M:%S")}
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if rec.get("error"):
                final_rows.append({"QualID": qid, "column": col,
                                   "text": text,
                                   "final_codes": ";".join(majority),
                                   "source": "majority-fallback",
                                   "flag_for_review": True,
                                   "notes": f"adjudication failed: {str(rec['error'])[:120]}"})
            else:
                final_rows.append({"QualID": qid, "column": col,
                                   "text": text,
                                   "final_codes": ";".join(sorted(rec["codes"])),
                                   "source": "adjudicated",
                                   "flag_for_review": bool(rec["flag_for_review"]),
                                   "notes": rec["notes"]})

    fin = pd.DataFrame(final_rows).sort_values(["column", "QualID"])
    fin.to_csv(outdir / "final_codes.csv", index=False)
    long_rows = [{"QualID": r.QualID, "column": r.column, "code": c}
                 for r in fin.itertuples() for c in str(r.final_codes).split(";") if c]
    pd.DataFrame(long_rows).to_csv(outdir / "final_long.csv", index=False)

    n = len(fin)
    src = fin.source.value_counts().to_dict()
    print(f"final_codes.csv: {n} items -> {src}")

    # Self-preference audit: on adjudicated items only, how similar is the
    # final ruling to each rater's original coding? A markedly higher value
    # for one rater (especially the adjudicator's sibling) suggests bias.
    adj_keys = {(r["QualID"], r["column"]) for r in final_rows
                if r["source"] == "adjudicated"}
    if adj_keys:
        from aggregate_results import jaccard
        fin_map = {(r["QualID"], r["column"]): set(str(r["final_codes"]).split(";")) - {""}
                   for r in final_rows}
        sims = {}
        for (qid, col), grp in ok.groupby(["QualID", "column"]):
            if (qid, col) not in adj_keys:
                continue
            for _, r in grp.iterrows():
                sims.setdefault(r["model_key"], []).append(
                    jaccard(fin_map[(qid, col)], set(r["codes"])))
        print("adjudicated-item agreement with each rater (mean Jaccard):")
        for m, v in sorted(sims.items()):
            print(f"  {m}: {sum(v) / len(v):.3f}  (n={len(v)})")
    print(f"flagged for eventual human eyes: {int(fin.flag_for_review.sum())}")


if __name__ == "__main__":
    main()
