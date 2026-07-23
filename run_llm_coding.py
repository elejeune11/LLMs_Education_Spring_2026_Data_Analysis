"""
run_llm_coding.py

First-pass LLM coding of Spring 2026 qualitative survey responses using
multiple independent LLMs (Anthropic, OpenAI, Google), one API call per
(response, model). Multi-label: models may assign multiple codes per entry,
subject to the constraints defined in codebooks.json (single-select tiers,
exclusive codes).

Design decisions:
- Sampling parameters are left at provider defaults: Claude Sonnet 5
  returns a 400 error if temperature/top_p/top_k are set to non-default
  values, and GPT-5.5 reasoning models likewise reject the temperature
  parameter. Reproducibility comes from archiving every raw output (JSONL)
  and pinning exact model IDs, not from greedy decoding.
- Each model codes independently; no model sees another model's output.
- Strict JSON output, validated against the codebook. One automatic repair
  retry with the validation error appended; persistent failures are recorded
  with codes=[] and error set, and surface in aggregation as items needing
  human coding.
- Results cached to results/<column>__<model_key>.jsonl keyed by QualID, so
  the script is resumable and reruns skip completed items.
- Blank cells are skipped (student skipped the question). Typed non-answers
  ("NA", "N/A", "nothing", ...) ARE sent to the models so they receive the
  codebook's Uncodable/Noise code; load uses keep_default_na=False so pandas
  does not silently convert the string "NA" to a missing value.
- The deposited data is now CSV rather than .xlsx. One behavioural note: the
  .xlsx reader surfaced in-cell line breaks as the literal token `_x000D_`,
  which was passed through to the models; the CSV reader yields clean
  newlines. The archived JSONL caches are keyed by QualID and are reused, so
  the published coding is unaffected. Only a cache wipe would re-code those
  responses, and their prompts would then differ slightly from the originals.

Setup (matches the FEM-bench convention: keys live in a .env file at the
project root and are loaded with python-dotenv; missing keys raise a clear
error at call time):
  pip install pandas python-dotenv anthropic openai google-genai
  # .env file:
  #   OPENAI_API_KEY=...
  #   GEMINI_API_KEY=...
  #   CLAUDE_API_KEY=...

Run:
  python run_llm_coding.py --data data/Spring_2026_qualitative.csv \
      --codebooks codebooks.json --outdir results
  Options: --models anthropic openai gemini   (default: all three)
           --columns txt_ai_use_change_post   (default: all mapped columns)
           --limit 5                          (smoke test on first 5 rows)

Data-sharing note: confirm that your IRB protocol and any data use
agreements permit sending free-text student responses to third-party API
providers, and use no-training / zero-retention API tiers where available.
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # reads .env from the working directory (FEM-bench convention)

# ---------------------------------------------------------------- config

# Model IDs verified July 2026. Override via env vars if needed.
MODELS = {
    "anthropic": {"provider": "anthropic", "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5")},
    "openai":    {"provider": "openai",    "model": os.environ.get("OPENAI_MODEL", "gpt-5.5")},
    "gemini":    {"provider": "gemini",    "model": os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")},
}

# .env key names follow the FEM-bench convention (CLAUDE_API_KEY for
# Anthropic); standard names are accepted as fallbacks.
KEY_ENV = {
    "anthropic": ["CLAUDE_API_KEY", "ANTHROPIC_API_KEY"],
    "openai":    ["OPENAI_API_KEY"],
    "gemini":    ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
}


def get_api_key(provider):
    for var in KEY_ENV[provider]:
        val = os.environ.get(var)
        if val:
            return val
    raise RuntimeError(
        f"Missing API key for {provider}: set {KEY_ENV[provider][0]} in your .env file")

MAX_RETRIES = 4          # network/rate-limit retries per call
REPAIR_RETRIES = 1       # extra attempt if the model returns invalid JSON/codes
SLEEP_BETWEEN_CALLS = 0.3

# ------------------------------------------------------------- providers


def call_anthropic(system, user, model):
    # No temperature: Claude Sonnet 5 returns 400 for non-default sampling
    # parameters. Adaptive thinking is on by default and thinking tokens
    # count toward max_tokens, hence the larger budget.
    import anthropic
    client = anthropic.Anthropic(api_key=get_api_key("anthropic"))
    msg = client.messages.create(
        model=model, max_tokens=8000,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return "".join(b.text for b in msg.content if b.type == "text")


def call_openai(system, user, model):
    # No temperature: GPT-5.x reasoning models reject the parameter.
    from openai import OpenAI
    client = OpenAI(api_key=get_api_key("openai"))
    resp = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content": user}],
    )
    return resp.choices[0].message.content


def call_gemini(system, user, model):
    # No temperature: Google recommends default sampling for Gemini 3.x.
    # Explicit max_output_tokens: thinking tokens count against the output
    # budget on Gemini 3.x, and an exhausted budget truncates the JSON.
    from google import genai
    client = genai.Client(api_key=get_api_key("gemini"))
    resp = client.models.generate_content(
        model=model,
        contents=user,
        config={"system_instruction": system,
                "response_mime_type": "application/json",
                "max_output_tokens": 16384},
    )
    text = resp.text
    finish = None
    try:
        finish = resp.candidates[0].finish_reason
    except Exception:
        pass
    if not text or (finish is not None and "STOP" not in str(finish).upper()):
        # empty or truncated (e.g. MAX_TOKENS) -> raise so the retry loop
        # makes a fresh call instead of feeding partial JSON to the parser
        raise RuntimeError(
            f"incomplete Gemini response (finish_reason={finish}, "
            f"{len(text or '')} chars); retrying")
    return text


PROVIDER_FN = {"anthropic": call_anthropic, "openai": call_openai, "gemini": call_gemini}

# --------------------------------------------------------------- prompts


def render_codebook(book):
    lines = []
    tiered = book["selection"] == "tiered"
    current_tier = None
    for c in book["codes"]:
        if tiered and c["tier"] != current_tier:
            current_tier = c["tier"]
            rule = book["tier_rules"][current_tier]
            if rule["max"] == 1:
                lines.append(f"\n== TIER {current_tier[-1]} (select exactly one) ==")
            else:
                lines.append(f"\n== TIER {current_tier[-1]} (select all that apply, zero or more) ==")
        lines.append(f"\n[{c['id']}] {c['name']}  (category: {c['category']})")
        lines.append(f"  Apply when: {c['description']}")
        if c.get("do_not_apply_when"):
            lines.append(f"  Do NOT apply when: {c['do_not_apply_when']}")
        if c.get("distinguish_from"):
            lines.append(f"  Distinguish from: {c['distinguish_from']}")
        if c.get("examples"):
            lines.append(f"  Example responses: {c['examples']}")
    return "\n".join(lines)


def build_system_prompt(book):
    excl = ", ".join(book["exclusive_codes"]) if book["exclusive_codes"] else "none"
    rules = "\n".join(f"- {r}" for r in book.get("extra_rules", []))
    return f"""You are an expert qualitative researcher applying a fixed codebook to short survey responses from undergraduate engineering students about their use of generative AI in a statics course.

SURVEY QUESTION shown to students:
"{book['question_text']}"

CODEBOOK:
{render_codebook(book)}

CODING RULES:
- Multiple codes may apply to one response; apply every code that fits, and only codes that fit. Base decisions on what the student wrote, not what they might have meant.
- Standalone (exclusive) codes that may never be combined with any other code: {excl}.
{rules}
- Use only the code IDs listed above. Do not invent codes.
- If you are genuinely uncertain about any assignment, still choose the best code(s) and set flag_for_review to true.

OUTPUT: respond with ONLY a JSON object, no markdown fences, in exactly this schema:
{{
  "codes": ["<code id>", ...],
  "rationale": "<one sentence, max 30 words, citing the response's wording>",
  "flag_for_review": true|false,
  "notes": "<required if an Other/Idiosyncratic residual code is applied, otherwise empty string>"
}}"""


def build_user_prompt(text):
    return f'Student response to code:\n"""\n{text}\n"""'

# ------------------------------------------------------------ validation


def extract_json(raw):
    raw = (raw or "").strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    try:
        return json.loads(raw)          # JSON-mode responses parse directly
    except json.JSONDecodeError:
        pass
    start = raw.find("{")
    if start == -1:
        raise ValueError("no JSON object found")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(raw)):
        ch = raw[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(raw[start:i + 1])
    raise ValueError("unbalanced JSON (response likely truncated)")


def validate(parsed, book):
    """Returns error string or None."""
    if not isinstance(parsed, dict) or "codes" not in parsed:
        return "output must be a JSON object with a 'codes' array"
    codes = parsed["codes"]
    if not isinstance(codes, list) or not all(isinstance(c, str) for c in codes):
        return "'codes' must be an array of code ID strings"
    codes = [c.strip() for c in codes]
    valid = {c["id"] for c in book["codes"]}
    bad = [c for c in codes if c not in valid]
    if bad:
        return f"unknown code IDs: {bad}; valid IDs are {sorted(valid)}"
    if len(set(codes)) != len(codes):
        return "duplicate code IDs"
    if len(codes) < book["min_codes"]:
        return f"at least {book['min_codes']} code required"
    for e in book["exclusive_codes"]:
        if e in codes and len(codes) > 1:
            return f"{e} is a standalone code and cannot be combined with other codes"
    if book["selection"] == "tiered":
        tier_of = {c["id"]: c["tier"] for c in book["codes"]}
        for tier, rule in book["tier_rules"].items():
            n = sum(1 for c in codes if tier_of[c] == tier)
            if n < rule["min"]:
                return f"tier {tier} requires at least {rule['min']} code(s), got {n}"
            if rule["max"] is not None and n > rule["max"]:
                return f"tier {tier} allows at most {rule['max']} code(s), got {n}"
    parsed["codes"] = codes
    return None

# --------------------------------------------------------------- runner


def code_one(provider_fn, model, book, text):
    system = build_system_prompt(book)
    user = build_user_prompt(text)
    last_err = None
    for attempt in range(REPAIR_RETRIES + 1):
        raw = call_with_retries(provider_fn, system, user, model)
        try:
            parsed = extract_json(raw)
            err = validate(parsed, book)
        except (ValueError, json.JSONDecodeError) as e:
            err = str(e)
            parsed = None
        if err is None:
            return parsed, raw, None
        last_err = err
        user = (build_user_prompt(text)
                + f"\n\nYour previous output was invalid: {err}. "
                  "Return ONLY a corrected JSON object in the required schema.")
    return parsed, raw, last_err


def call_with_retries(provider_fn, system, user, model):
    delay = 2.0
    for attempt in range(MAX_RETRIES):
        try:
            return provider_fn(system, user, model)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            print(f"    transient error ({e}); retrying in {delay:.0f}s")
            time.sleep(delay)
            delay *= 2


def load_data(path):
    # keep_default_na=False keeps typed non-answers ("NA", "N/A") as literal
    # strings so they reach the models and receive an Uncodable/Noise code;
    # only truly empty cells become "" and are skipped in main().
    # utf-8-sig strips a UTF-8 BOM when present and is a no-op otherwise.
    df = pd.read_csv(path, keep_default_na=False, encoding="utf-8-sig")
    df["QualID"] = df["QualID"].astype(str)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--codebooks", default="codebooks.json")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    ap.add_argument("--columns", nargs="+", default=None)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    books = json.loads(Path(args.codebooks).read_text())
    df = load_data(args.data)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    col_to_book = {col: bname for bname, b in books.items() for col in b["columns"]}
    columns = args.columns or [c for c in df.columns if c in col_to_book]

    for col in columns:
        book = books[col_to_book[col]]
        rows = df[["QualID", col]].copy()
        rows[col] = rows[col].astype(str).str.strip()
        rows = rows[rows[col] != ""]          # skip true blanks only
        if args.limit:
            rows = rows.head(args.limit)
        for mkey in args.models:
            cfg = MODELS[mkey]
            fn = PROVIDER_FN[cfg["provider"]]
            cache_path = outdir / f"{col}__{mkey}.jsonl"
            done = set()
            if cache_path.exists():
                with open(cache_path) as f:
                    for line in f:
                        if not line.strip():
                            continue
                        rec = json.loads(line)
                        # only successful codings count as done: failed calls
                        # are retried on the next run (the aggregator keeps the
                        # most recent successful record per item)
                        if not rec.get("error"):
                            done.add(rec["QualID"])
            todo = rows[~rows["QualID"].isin(done)]
            print(f"{col} / {mkey} ({cfg['model']}): {len(todo)} to code, {len(done)} cached")
            with open(cache_path, "a") as f:
                for _, r in todo.iterrows():
                    try:
                        parsed, raw, err = code_one(fn, cfg["model"], book, r[col])
                    except Exception as e:
                        parsed, raw, err = None, "", f"API failure: {e}"
                    rec = {
                        "QualID": r["QualID"],
                        "column": col,
                        "model_key": mkey,
                        "model": cfg["model"],
                        "text": r[col],
                        "codes": (parsed or {}).get("codes", []),
                        "rationale": (parsed or {}).get("rationale", ""),
                        "flag_for_review": (parsed or {}).get("flag_for_review", True),
                        "notes": (parsed or {}).get("notes", ""),
                        "error": err,
                        "raw": raw,
                        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    }
                    f.write(json.dumps(rec) + "\n")
                    f.flush()
                    time.sleep(SLEEP_BETWEEN_CALLS)
    print("Done. Next: python aggregate_results.py --outdir", args.outdir)


if __name__ == "__main__":
    main()
