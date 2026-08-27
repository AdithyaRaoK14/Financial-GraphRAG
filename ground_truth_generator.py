"""
ground_truth_generator.py
===========================
WHAT THIS FILE DOES:
Since there's no official "answer key" for these 5 companies (unlike big
benchmark datasets), this file MAKES one. This is what your sir meant by
"give the audio and the pdf stuff to an LLM as a groundtruth" — it's a
third pipeline, separate from GraphRAG and the baseline, whose only job is
to generate trustworthy question/answer pairs to test the other two against.

How it works, per company/year/quarter:
  1. Loads ALL processed chunks for that quarter (both PDF and audio) and
     builds a bounded-size context split fairly across PDF narrative text,
     tables, and audio commentary — this is the ONE place in the whole
     project where we deliberately skip retrieval and just give the LLM
     everything (up to the budget), because here we WANT maximum context
     to generate a reliable, well-grounded answer key. Giving every source
     type its own share of the character budget (instead of one blind
     `text[:N]` slice) avoids always favoring whatever appears first in
     the PDF.
  2. Asks the LLM to generate N question/answer pairs covering a mix of:
     - pure numeric/PDF questions (revenue, margin, etc.)
     - pure commentary/audio questions (guidance, strategy)
     - combined questions (numeric + commentary together)
  3. Cleans the raw output: drops hedged/unsupported answers ("not
     mentioned", "cannot be determined", ...) and near-duplicate questions.
  4. Re-verifies each surviving pair with a second LLM call ("is this
     answer explicitly supported by the text?") and drops anything that
     fails — see config.GROUND_TRUTH_VERIFY to toggle this off.
  5. Saves them to ground_truth.json in the same shape as benchmark.py
     expects.

IMPORTANT: this is NOT a pipeline you're benchmarking — it's the tool that
CREATES the benchmark. Run it once per quarter before running benchmark.py.

Run directly:
    python ground_truth_generator.py
"""

import difflib
import json
import re

import config
import llm_client

GEN_PROMPT = """You are creating a benchmark test set for a financial RAG
system about {company}, {quarter} {year}.

Below is text extracted from that quarter's PDF financial report and
earnings call transcript. Generate UP TO {n} question-answer pairs.
Quality matters far more than quantity — only generate a question if its
answer is explicitly supported by the text. Fewer well-grounded pairs is
better than {n} pairs padded with weak or unsupported ones.

STRICT GROUNDING RULES (most important):
- Every answer must be EXPLICITLY stated in the text below. Do not infer,
  summarize, speculate, or use outside financial knowledge.
- NEVER produce a question whose answer would be "not mentioned", "not
  available", "cannot be determined", "management did not say", or
  similar. If the text doesn't clearly and directly support an answer,
  don't ask that question at all — pick a different fact instead.
- Do not invent guidance, numbers, or attributions that aren't in the text.

Aim roughly for this split, but never sacrifice grounding to hit it:
  - up to {n_pdf} questions answerable from the PDF/numeric data alone
    (revenue, margins, profit, specific figures — quote the number as
    written)
  - up to {n_audio} questions answerable from the audio/commentary alone
    (management's tone, guidance, strategic comments — as actually said)
  - up to {n_both} "both" questions. A BOTH question must combine:
      • one numeric fact stated explicitly in the PDF, AND
      • one management statement stated explicitly in the transcript
    Both halves must be things the text actually says — do NOT compare,
    infer, or judge whether one matches the other (e.g. never "did X
    match guidance"). Just ask for both stated facts together, e.g.
    "What was the PAT increase, and what did management attribute it to?"

EXAMPLES OF GOOD QUESTIONS (follow this style):

Question: What was the revenue from operations in Q2 FY26?
Answer: ₹1,414.13 crore
Type: pdf
Evidence: Revenue from operations was ₹1,414.13 crore.

Question: What guidance did management provide regarding credit cost?
Answer: Management guided credit cost at 4-4.5%.
Type: audio
Evidence: "We expect credit cost to be in the 4 to 4.5% range going forward."

Question: What was the PAT increase, and what did management attribute it to?
Answer: PAT increased 18%, which management attributed to operating leverage.
Type: both
Evidence: PAT grew 18% YoY; management said "this was driven by operating leverage."

Before returning, verify: every answer appears in the text, every number
is copied accurately, no duplicate/near-duplicate questions, no "not
mentioned" answers.

Return STRICT JSON only, no markdown fences, in this exact shape:
{{
  "qa_pairs": [
    {{"question": "...", "answer": "...", "type": "pdf|audio|both", "evidence": "..."}}
  ]
}}

TEXT:
\"\"\"{text}\"\"\"

JSON:"""

# Answers matching any of these (case-insensitive substring) mean the model
# ignored the grounding rules and hedged instead of skipping the question —
# discard those pairs rather than keeping a QA pair with no real answer.
_BAD_ANSWER_MARKERS = (
    "not mentioned",
    "not directly answerable",
    "not explicitly",
    "cannot be determined",
    "cannot determine",
    "not available",
    "unknown",
    "did not provide",
    "did not say",
    "not stated",
    "n/a",
)


def _is_bad_answer(answer: str) -> bool:
    a = (answer or "").strip().lower()
    if not a:
        return True
    return any(marker in a for marker in _BAD_ANSWER_MARKERS)


def _clean_pairs(pairs: list[dict]) -> list[dict]:
    """Drop hedged/empty answers and de-duplicate near-identical questions
    (normalized: lowercase, whitespace-collapsed) before they ever reach
    ground_truth.json."""
    seen_questions = set()
    cleaned = []
    for p in pairs:
        question = (p.get("question") or "").strip()
        answer = (p.get("answer") or "").strip()
        if not question or _is_bad_answer(answer):
            continue
        key = re.sub(r"\s+", " ", question.lower())
        if key in seen_questions:
            continue
        seen_questions.add(key)
        cleaned.append(p)
    return cleaned


def _parse_qa_response(raw: str) -> list[dict] | None:
    """Accepts either {"qa_pairs": [...]} or a bare [...] array — Qwen
    sometimes returns the array directly despite the prompt's schema.
    Returns None on anything unparseable."""
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        pairs = parsed.get("qa_pairs")
        if isinstance(pairs, list):
            return pairs
        return None
    return None


VERIFY_PROMPT = """Context:
\"\"\"{text}\"\"\"

Question: {question}
Answer: {answer}

Is this answer explicitly and directly supported by the context above —
not inferred, not approximate, not from outside knowledge?

Return STRICT JSON only, no markdown fences, in this exact shape:
{{"supported": true}}
or
{{"supported": false}}

JSON:"""


def _verify_pair(question: str, answer: str, text: str) -> bool:
    """Re-asks the LLM whether a single QA pair is actually grounded in the
    source text, independent of whatever it claimed while generating it.
    Returns True (keep the pair) if verification is inconclusive/unparseable
    — a broken verification call shouldn't silently delete good pairs, it
    just means this safety net didn't catch anything for that pair."""
    prompt = VERIFY_PROMPT.format(text=text, question=question, answer=answer)
    raw = llm_client.generate(
        prompt, model=config.GROUND_TRUTH_MODEL, temperature=0, json_mode=True
    )
    cleaned = re.sub(r"^```(json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict) and "supported" in parsed:
            return bool(parsed["supported"])
    except json.JSONDecodeError:
        pass
    return True


def _verify_pairs(
    pairs: list[dict], text: str, company: str, year: str, quarter: str
) -> list[dict]:
    """Runs _verify_pair on every pair and drops the ones the model itself
    flags as unsupported. One extra LLM call per pair — see
    config.GROUND_TRUTH_VERIFY to disable."""
    if not config.GROUND_TRUTH_VERIFY or not pairs:
        return pairs

    kept = []
    rejected = 0
    for i, p in enumerate(pairs, start=1):
        try:
            supported = _verify_pair(p.get("question", ""), p.get("answer", ""), text)
        except Exception as e:
            # A verification-call failure (network/model hiccup) shouldn't
            # cost the pair — keep it and move on.
            print(
                f"  [warn] Verification call failed for pair {i}: {e} — keeping pair."
            )
            supported = True

        if supported:
            kept.append(p)
        else:
            rejected += 1

    if rejected:
        print(
            f"  [info] Verification dropped {rejected} unsupported pair(s) "
            f"for {company} {quarter}-{year}."
        )
    return kept


def _find_source_chunk(evidence: str, chunks: list[dict]) -> dict | None:
    """Finds which real chunk an `evidence` snippet actually came from, by
    matching text — NOT by trusting a chunk_id the LLM might output itself
    (it has no way to know the real SHA-256 chunk_ids from chunker.py, so
    asking it to produce one would just mean hallucinated-but-plausible
    IDs, which is worse than no ID). Exact containment wins immediately;
    otherwise falls back to fuzzy similarity for slightly paraphrased
    evidence, requiring a reasonably high match before accepting it."""
    if not evidence or len(evidence.strip()) < 8:
        return None

    norm_evidence = re.sub(r"\s+", " ", evidence.strip().lower())

    best_chunk = None
    best_score = 0.0
    for c in chunks:
        text = c.get("text") or c.get("embedding_text") or ""
        if not text:
            continue
        norm_text = re.sub(r"\s+", " ", text.lower())

        if norm_evidence in norm_text:
            return c  # exact substring match — can't do better than this

        # Coverage = longest contiguous run of evidence text found inside
        # the chunk, normalized by evidence length. Unlike ratio()/
        # quick_ratio() (which compare the two strings as wholes), this
        # stays meaningful even when the chunk is much longer than the
        # evidence snippet (typical here — chunks run up to ~1200 chars,
        # evidence is a short quote) since it's only measuring how much of
        # the evidence appears in one run inside the chunk, not overall
        # string similarity.
        matcher = difflib.SequenceMatcher(
            None, norm_evidence, norm_text, autojunk=False
        )
        match = matcher.find_longest_match(0, len(norm_evidence), 0, len(norm_text))
        score = match.size / len(norm_evidence) if norm_evidence else 0.0
        if score > best_score:
            best_score = score
            best_chunk = c

    return best_chunk if best_score >= 0.4 else None


def _attach_provenance(pairs: list[dict], chunks: list[dict]) -> list[dict]:
    """Attaches chunk_id + source to every pair based on its `evidence`
    field, so every QA pair in ground_truth.json can be traced back to the
    exact chunk it came from. Pairs whose evidence can't be matched to any
    chunk (LLM paraphrased too heavily, or omitted evidence) get
    chunk_id=None rather than a guessed value."""
    matched = 0
    for p in pairs:
        chunk = _find_source_chunk(p.get("evidence", ""), chunks)
        if chunk:
            p["chunk_id"] = chunk.get("chunk_id")
            p["source"] = chunk.get("source_type")  # "pdf" or "audio"
            matched += 1
        else:
            p["chunk_id"] = None
            p["source"] = None

    if pairs:
        print(f"  [info] Matched {matched}/{len(pairs)} pair(s) to a source chunk_id.")
    return pairs


def _load_quarter_chunks(company: str, year: str, quarter: str) -> list[dict]:
    q_dir = config.PROCESSED_DATA_DIR / company / year / quarter
    if not q_dir.exists():
        return []

    chunks = []
    for jf in q_dir.glob("*.json"):
        chunks.extend(json.loads(jf.read_text(encoding="utf-8")))
    return chunks


def _build_context_text(chunks: list[dict], budget: int = None) -> str:
    """
    Builds a bounded context that gives PDF narrative text, tables, and
    audio commentary each a fair share of the character budget, instead
    of concatenating everything and blindly slicing at N characters (which
    always favors whichever source happens to appear first).
    """
    budget = budget or config.MAX_CONTEXT

    pdf_text = [
        c
        for c in chunks
        if c.get("source_type") == "pdf" and c.get("chunk_type") == "text"
    ]
    tables = [c for c in chunks if c.get("chunk_type") == "table"]
    audio = [c for c in chunks if c.get("source_type") == "audio"]

    sections = [
        ("PDF NARRATIVE TEXT", pdf_text),
        ("TABLES", tables),
        ("AUDIO / EARNINGS CALL COMMENTARY", audio),
    ]

    non_empty = [(name, items) for name, items in sections if items]
    if not non_empty:
        return ""

    per_section_budget = budget // len(non_empty)

    parts = []
    for name, items in non_empty:
        section_text = "\n\n".join(
            c.get("embedding_text") or c.get("text", "") for c in items
        )
        parts.append(f"=== {name} ===\n{section_text[:per_section_budget]}")

    return "\n\n".join(parts)


def generate_for_quarter(
    company: str,
    year: str,
    quarter: str,
    n_pdf: int = 4,
    n_audio: int = 4,
    n_both: int = 2,
) -> list[dict]:
    chunks = _load_quarter_chunks(company, year, quarter)
    if not chunks:
        print(f"  No processed data found for {company} {quarter}-{year}, skipping.")
        return []

    text = _build_context_text(chunks)
    if not text:
        print(f"  No usable text found for {company} {quarter}-{year}, skipping.")
        return []

    print(f"  Context length: {len(text):,} chars (budget {config.MAX_CONTEXT:,})")

    n_total = n_pdf + n_audio + n_both
    prompt = GEN_PROMPT.format(
        company=company,
        quarter=quarter,
        year=year,
        n=n_total,
        n_pdf=n_pdf,
        n_audio=n_audio,
        n_both=n_both,
        text=text,
    )

    # temperature=0: ground truth should be deterministic — this is the
    # "answer key" everything else gets scored against, not creative output.
    max_retries = 2
    pairs = None
    for attempt in range(max_retries + 1):
        raw = llm_client.generate(
            prompt,
            model=config.GROUND_TRUTH_MODEL,
            temperature=0,
            json_mode=True,
            # Explicit floor, independent of config.OLLAMA_NUM_PREDICT —
            # up to n_total pairs (question+answer+type+evidence each) can
            # easily exceed a small shared default tuned for other callers
            # (e.g. short answer-pipeline responses), silently truncating
            # this response mid-JSON.
            num_predict=max(config.OLLAMA_NUM_PREDICT, 2048),
        )
        pairs = _parse_qa_response(raw)
        if pairs is not None:
            # Valid JSON — stop here regardless of count. A model that
            # returns fewer pairs than asked is usually doing the RIGHT
            # thing (it couldn't find {n_total} facts it could ground
            # cleanly per the strict rules above) — re-prompting the exact
            # same task rarely fixes that and just burns retries.
            break
        if attempt < max_retries:
            print(
                f"  [warn] Unparseable ground truth JSON for {company} "
                f"{quarter}-{year} (attempt {attempt + 1}/{max_retries + 1}), retrying..."
            )
            continue
        print(
            f"  [warn] Could not parse ground truth JSON for {company} "
            f"{quarter}-{year} after {max_retries + 1} attempts — skipping quarter."
        )
        return []

    pre_clean_count = len(pairs)
    pairs = _clean_pairs(pairs)
    dropped = pre_clean_count - len(pairs)
    if dropped:
        print(
            f"  [info] Dropped {dropped} hedged/duplicate pair(s) for "
            f"{company} {quarter}-{year}."
        )

    pairs = _attach_provenance(pairs, chunks)
    pairs = _verify_pairs(pairs, text, company, year, quarter)

    if len(pairs) < n_total:
        print(
            f"  [info] Got {len(pairs)}/{n_total} well-grounded QA pairs for "
            f"{company} {quarter}-{year} — kept as-is (fewer grounded pairs "
            f"beats padding with weak/unanswerable ones)."
        )

    for p in pairs:
        p["company"] = company
        p["year"] = year
        p["quarter"] = quarter
    return pairs


def generate_all():
    all_pairs = []
    for company in config.COMPANIES:
        for year in config.YEARS:
            for quarter in config.QUARTERS:
                print(f"Generating ground truth: {company} {quarter}-{year}")
                pairs = generate_for_quarter(company, year, quarter)
                all_pairs.extend(pairs)
                print(f"  -> {len(pairs)} QA pairs")

    out_path = config.RESULTS_DIR / "ground_truth.json"
    out_path.write_text(json.dumps(all_pairs, indent=2), encoding="utf-8")
    print(f"\nSaved {len(all_pairs)} total QA pairs to {out_path}")


if __name__ == "__main__":
    generate_all()
