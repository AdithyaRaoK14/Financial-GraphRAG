"""
baseline_pipeline.py
=====================
WHAT THIS FILE DOES:
This is the pipeline you compare AGAINST GraphRAG to prove GraphRAG is
better. It's deliberately "dumb" by comparison — plain, standard RAG:

  1. Retrieve the top-k most relevant chunks (same retrieval.py as GraphRAG,
     so retrieval quality is a fair, identical comparison).
  2. Feed those raw chunk texts straight to the LLM. No knowledge graph, no
     structured facts, no entity/relationship extraction — just "here's some
     text, answer the question."

If GraphRAG scores meaningfully higher than this in benchmark.py — especially
on numeric accuracy — that's your evidence that the knowledge graph actually
adds value over plain retrieval.
"""

import time

import router
import retrieval
import llm_client
import config
from evidence_cleaning import (
    clean_evidence_text,
    format_table_markdown,
    statement_type_tag,
)
from graph_rag_pipeline import (
    FINAL_ANSWER_MARKER,
    _strip_unrequested_delta,
    _ensure_final_answer_line,
)

ANSWER_PROMPT = """You are a financial analyst assistant. Answer the question
using ONLY the text below. Be precise with numbers. If the text doesn't
contain the answer, say so — do not guess.

Your FINAL ANSWER line must contain ONLY the figures the question asks
for. Do not compute and append a change, difference or percentage unless
the question explicitly requests one — "why did revenue increase" and
"what was revenue in X and Y" do not ask for a computed delta, and
adding one counts against you even when the arithmetic is right. Copy
figures exactly as printed: never rescale, re-round, move a decimal
point or reorder digits.

WHICH FIGURE TO USE: the text often contains several similar-looking
figures. Prefer the one whose label matches the question's metric AND
whose period matches the period asked about. If exactly one figure
matches on both, that figure IS the answer — state it plainly. Other,
less precisely matched figures (a vaguer label, a different period, a
segment rather than the total) are not conflicting evidence and must not
stop you answering. Declining when a well-matched figure IS present is a
wrong answer, exactly as wrong as inventing a number. "Do not guess"
means do not invent figures that aren't there — it does not mean refuse
when the text is messy but does contain the answer.

Some questions are QUALITATIVE, asking what management discussed,
highlighted, or explained rather than asking for a number. For these,
"do not guess" means do not invent content that isn't there — it does
not mean the text must use the question's exact wording. If the text
contains commentary that is clearly the substance being asked about
(e.g. a funding/liability strategy update when asked what "new
initiatives" were discussed, even if the text never uses the word
"initiative"), summarize that commentary as your answer rather than
saying the information isn't present.

QUESTION: {question}

RETRIEVED TEXT:
{chunks}

QUESTION (repeated): {question}

Give a direct, concise answer. Decide once from the evidence — do not
narrate a multi-round self-correction process ("wait, actually",
"re-evaluating") in your response. If the question involves two or more
values (a comparison, a quarter-on-quarter or year-on-year change),
include every value asked for, not just one side of it — label each
value clearly (e.g. "Q2 FY25: X; Q2 FY26: Y") rather than running
numbers together with only a comma. You may explain briefly first if
needed, but end your response with one line in EXACTLY this format:
{final_answer_marker} <only the number(s)/fact(s) actually asked for —
no other numbers, dates, or page references on this line>"""


def answer(
    question: str,
    company: str = None,
    year: str = None,
    quarter: str = None,
    statement_type: str = None,
) -> dict:
    routing = router.route_question(question)
    # P1 fix (ablation support): matches graph_rag_pipeline.py's same
    # FORCE_SOURCE_FILTER override, applied identically to baseline for
    # the same fairness reason TOP_K_AUDIO is applied identically above
    # (equal retrieval-stage treatment between the two pipelines at every
    # ablation experiment, not just the final one).
    force_source_filter = getattr(config, "FORCE_SOURCE_FILTER", None)
    if force_source_filter:
        routing = dict(routing)
        routing["source_filter"] = force_source_filter
    # Timed separately from generation so RetLatency can be reported
    # alongside E2E latency - a system can be slow because retrieval is
    # slow or because the LLM is slow, and those need different fixes.
    _t0 = time.time()
    # P1 fix: matches graph_rag_pipeline.py's same audio-specific top_k
    # boost - see config.TOP_K_AUDIO's comment for the measured recall
    # gain. Applied identically to both pipelines deliberately (see
    # TOP_K_BASELINE's comment in config.py on equal budgets) - boosting
    # only GraphRAG's audio budget would reintroduce the "GraphRAG only
    # won because baseline was starved of context" confound that design
    # already rejected.
    baseline_top_k = (
        config.TOP_K_AUDIO
        if routing["source_filter"] == "audio"
        else config.TOP_K_BASELINE
    )
    chunks = retrieval.retrieve(
        question,
        company=company,
        year=year,
        quarter=quarter,
        statement_type=statement_type,
        source_filter=routing["source_filter"],
        top_k=baseline_top_k,
    )

    def _format_chunk(c: dict) -> str:
        if c.get("chunk_type") == "table":
            cleaned_text = format_table_markdown(c)
        else:
            # P0 fix: same statement-type tag as the table path and as
            # graph_rag_pipeline.py's _format_chunk(), so this stays a
            # fair, identical-evidence comparison between the two
            # pipelines (see evidence_cleaning.statement_type_tag()).
            cleaned_text = (
                statement_type_tag(c)
                + clean_evidence_text(c.get("text") or c.get("embedding_text") or "")[
                    :1200
                ]
            )
        return (
            f"[{c['source_type'].upper()} | {c['quarter']}-{c['year']} | "
            f"{c.get('document_name') or ''} | p.{c.get('page')}] "
            f"{cleaned_text}"
        )

    retrieval_sec = time.time() - _t0

    chunks_str = (
        "\n\n".join(_format_chunk(c) for c in chunks)
        if chunks
        else "(no chunks retrieved)"
    )

    prompt = ANSWER_PROMPT.format(
        question=question, chunks=chunks_str, final_answer_marker=FINAL_ANSWER_MARKER
    )
    response = llm_client.generate(
        prompt, model=config.ANSWER_MODEL, num_predict=config.OLLAMA_ANSWER_NUM_PREDICT
    )
    # Same deterministic backstops as graph_rag_pipeline.py, so neither a
    # missing FINAL ANSWER line nor a stray unrequested computed delta
    # gets scored against one pipeline and not the other — keeps this a
    # fair, identical-rules comparison.
    response = _ensure_final_answer_line(prompt, response)
    response = _strip_unrequested_delta(response, question)

    return {
        "question": question,
        "answer": response,
        "routing": routing,
        "num_chunks_used": len(chunks),
        # Dicts, matching graph_rag_pipeline.answer()'s "sources" shape.
        # benchmark.py's _retrieved_chunks() falls back to
        # `[s for s in sources if isinstance(s, dict)]` when there's no
        # separate "retrieved_chunks" key — plain chunk_id strings here
        # meant that filter always returned [], which silently zeroed out
        # baseline_avg_retrieval_score, baseline_hit_at_k, and
        # baseline_recall_at_k for every single row.
        "sources": [
            {
                "chunk_id": c["chunk_id"],
                "source_type": c["source_type"],
                "chunk_type": c.get("chunk_type"),
                "document_name": c.get("document_name"),
                "page": c.get("page"),
                "section": c.get("section"),
                "title": c.get("title"),
                "start": c.get("start"),
                "end": c.get("end"),
                "score": c.get("score"),
                "embedding_score": c.get("embedding_score"),
                "bm25_score": c.get("bm25_score"),
                "fastrp_score": c.get("fastrp_score"),
            }
            for c in chunks
        ],
        "retrieval_scores": [c.get("score") for c in chunks],
        "retrieval_sec": round(retrieval_sec, 4),
        "evidence_text": chunks_str,
    }


if __name__ == "__main__":
    result = answer(
        "What was the revenue growth for Q2 2026?",
        company="Tata Consumer Products",
        year="2026",
        quarter="Q2",
    )
    print(result["answer"])
