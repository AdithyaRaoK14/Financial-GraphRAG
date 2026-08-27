"""
llm_client.py
=============
WHAT THIS FILE DOES:
One shared function, `generate()`, that every pipeline (GraphRAG, baseline,
ground-truth generator) calls to talk to an LLM. Right now it talks to your
local qwen2.5:7b through Ollama. When you're ready to switch to Groq for
speed, just set USE_GROQ=true in your .env — nothing else in the project
needs to change.

GROQ RATE LIMITS: free-tier Groq models enforce a tokens-per-minute (TPM)
cap much tighter than the daily budget suggests — e.g. 6,000 TPM for
llama-3.1-8b-instant. Two distinct failure modes come from this, and this
file now guards against both:
  1. A single request alone exceeds the cap (413 "Request too large") —
     GraphRAG's evidence-heavy prompts for multi-part/multi-company
     questions can genuinely run this large. Guarded by truncating the
     prompt to a safe size before sending.
  2. Several reasonably-sized requests fired back-to-back exceed the
     rolling 60s window (429) — benchmark.py has no pacing between calls.
     Guarded by tracking estimated usage and sleeping before a request
     that would tip over the limit.
Override the assumed cap with GROQ_TPM_LIMIT in .env if you switch to a
Groq model with a different limit.
"""

import collections
import re
import time

import ollama
import config

_groq_client = None
_ollama_client = None
_groq_usage_window: collections.deque = collections.deque()

# Answer prompts are financial Q&A — a short, direct answer, not an essay.
# Without capping this, Groq's TPM accounting for "Requested" tokens
# includes the reserved completion budget alongside the prompt, and an
# unset max_tokens likely defaults to something large enough that even a
# well-truncated prompt still reports as "over 6000" total.
MAX_RESPONSE_TOKENS = 700


def _get_groq_client():
    global _groq_client
    if _groq_client is None:
        from groq import Groq

        _groq_client = Groq(api_key=config.GROQ_API_KEY)
    return _groq_client


def _get_ollama_client():
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = ollama.Client(host=config.OLLAMA_HOST)
    return _ollama_client


def _groq_tpm_limit() -> int:
    return int(getattr(config, "GROQ_TPM_LIMIT", 6000))


def _estimate_tokens(text: str) -> int:
    """Rough ~4 chars/token estimate — not exact, only precise enough for
    client-side pacing so we don't blow through the TPM cap in the first
    place. Groq's own count in a 413/429 error is the real number; this
    just needs to be in the right ballpark ahead of time."""
    return max(1, len(text) // 4)


def _groq_wait_for_capacity(estimated_tokens: int):
    """Sleep if sending `estimated_tokens` now would exceed the rolling
    60-second TPM window. benchmark.py fires GraphRAG + baseline calls
    back-to-back with no pacing of its own, which reliably exceeds a
    6,000 TPM free-tier cap even when each individual request is a
    reasonable size on its own."""
    limit = _groq_tpm_limit()
    now = time.time()
    while _groq_usage_window and now - _groq_usage_window[0][0] > 60:
        _groq_usage_window.popleft()
    used = sum(t for _, t in _groq_usage_window)
    if used + estimated_tokens > limit:
        oldest_ts = _groq_usage_window[0][0] if _groq_usage_window else now
        sleep_for = max(0.0, 60 - (now - oldest_ts)) + 0.5
        time.sleep(sleep_for)
        now = time.time()
        while _groq_usage_window and now - _groq_usage_window[0][0] > 60:
            _groq_usage_window.popleft()
    _groq_usage_window.append((time.time(), estimated_tokens))


def _groq_max_prompt_chars(shrink_factor: float = 1.0) -> int:
    """Conservative char budget so a single request doesn't itself exceed
    the TPM cap. Uses ~3 chars/token (not the more common ~4) because
    GraphRAG's evidence is dense financial tables — long runs of numbers,
    pipes, and decimals tokenize less efficiently than ordinary prose, so
    a 4-chars/token estimate under-counts real usage on this content.
    shrink_factor < 1.0 shrinks further on a retry after an actual 413,
    which reports the real token count Groq measured — trust that over
    the estimate for how much more to cut."""
    limit = _groq_tpm_limit()
    # Reserve headroom for MAX_RESPONSE_TOKENS plus request overhead, not
    # just an arbitrary buffer.
    budget_tokens = max(500, limit - MAX_RESPONSE_TOKENS - 200)
    return max(1000, int(budget_tokens * 3 * shrink_factor))


def _truncate_for_groq(prompt: str, shrink_factor: float = 1.0) -> str:
    """Trim from the middle rather than the end — GraphRAG's task
    instructions live at the top of the prompt and the actual question at
    the bottom; the evidence in between is what's safe to shorten without
    breaking the model's ability to understand what's being asked."""
    max_chars = _groq_max_prompt_chars(shrink_factor)
    if len(prompt) <= max_chars:
        return prompt
    half = max_chars // 2
    return (
        prompt[:half]
        + "\n\n[...evidence truncated to fit the model's per-request token limit...]\n\n"
        + prompt[-half:]
    )


_RETRY_AFTER_RE = re.compile(
    r"try again in\s+(?:(\d+)h)?\s*(?:(\d+)m)?\s*([\d.]+)s", re.I
)


def _is_too_large_error(exc: Exception) -> bool:
    """True for Groq's 413 'Request too large' — distinct from a plain
    429 pacing error. A 413 means the prompt itself needs to shrink
    further; retrying with the identical prompt (as a plain rate-limit
    backoff would) fails identically every time."""
    msg = str(exc)
    return (
        "413" in msg or "Request too large" in msg or "reduce your message size" in msg
    )


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract a wait time (seconds) from a Groq rate-limit error message
    like "Please try again in 1.73s" or "...in 1h21m34.5s". Returns None
    for anything that isn't a rate/size error, or whose wait is too long
    to block on inline (benchmark.py's BENCHMARK_RESUME can pick a
    genuinely daily-quota-exhausted question back up later instead — no
    point stalling the whole run for over a minute on one question)."""
    msg = str(exc)
    if "rate_limit_exceeded" not in msg:
        return None
    m = _RETRY_AFTER_RE.search(msg)
    if not m:
        return 5.0
    hours, minutes, seconds = (float(g) if g else 0.0 for g in m.groups())
    total = hours * 3600 + minutes * 60 + seconds
    return total if total <= 60 else None


def _groq_generate(prompt: str, model: str, temperature: float, json_mode: bool) -> str:
    client = _get_groq_client()
    shrink_factor = 1.0
    prompt = _truncate_for_groq(prompt, shrink_factor)

    for attempt in range(3):
        _groq_wait_for_capacity(_estimate_tokens(prompt))
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=MAX_RESPONSE_TOKENS,
                response_format={"type": "json_object"} if json_mode else None,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            if attempt == 2:
                raise
            if _is_too_large_error(exc):
                # The prompt itself is still too big - shrink it further
                # and retry immediately, don't just wait and resend the
                # same oversized request.
                shrink_factor *= 0.7
                prompt = _truncate_for_groq(prompt, shrink_factor)
                continue
            wait = _parse_retry_after(exc)
            if wait is None:
                raise
            time.sleep(wait)
    raise RuntimeError("unreachable")  # pragma: no cover


def generate(
    prompt: str,
    model: str = None,
    temperature: float = 0.0,
    json_mode: bool = False,
    num_predict: int = None,
) -> str:
    """Send a prompt to the configured LLM and return its text response.

    json_mode=True constrains decoding so the response is syntactically
    valid JSON — use this for callers that parse the output (e.g.
    ground_truth_generator.py), NOT for the answer pipelines, which need
    plain prose.

    num_predict overrides config.OLLAMA_NUM_PREDICT for this call only —
    use this for callers whose JSON payload can run larger than the
    shared default comfortably covers, so a change to the shared default
    (tuned for other callers) can't silently truncate this one."""
    if config.USE_GROQ:
        return _groq_generate(
            prompt, model or config.GROQ_MODEL, temperature, json_mode
        )
    else:
        client = _get_ollama_client()
        resolved_model = model or config.ANSWER_MODEL
        response = client.generate(
            model=resolved_model,
            prompt=prompt,
            # qwen3-family models default to emitting a visible <think>...
            # </think> reasoning block before the actual answer, which
            # would otherwise land inside response["response"] and break
            # benchmark.py's number-extraction regex on the answer text.
            # NOTE: known Ollama issues (e.g. ollama-python#576) report
            # think=False doesn't always suppress this for qwen3:8b
            # specifically — verify actual output during a smoke test
            # rather than assuming this flag alone is sufficient.
            think=False if "qwen3" in resolved_model.lower() else None,
            options={
                "temperature": temperature,
                "num_ctx": config.OLLAMA_NUM_CTX,
                "num_predict": (
                    num_predict
                    if num_predict is not None
                    else config.OLLAMA_NUM_PREDICT
                ),
            },
            format="json" if json_mode else None,
        )
        return response["response"].strip()
