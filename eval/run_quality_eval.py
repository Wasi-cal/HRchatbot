#!/usr/bin/env python3
"""Formatted report generator for manual quality grading.

Reads eval/quality_eval_set.json, runs each case through ask_and_answer()
(no user attributes: nothing is filtered or asked), and prints a
clean per-case report: query, category, expected_behavior, the actual
result, and a blank "grade: PASS/FAIL" line for a human to fill in
while reading. This script does NOT auto-grade anything - answer
quality on paraphrase/exact_term cases is a judgment call for a human
who knows the corpus, not something to fake with keyword matching (that
kind of structural check belongs in eval/run_regression.py instead).

After grading, log the result in eval/GRADING_LOG.md so there's a
durable history across runs, not just whatever happened to be in the
terminal that day.

Each case has a "channel" ("text" | "voice", default "text") that selects the
system prompt style and retrieval depth, an optional "history" (prior
{"role", "content"} messages, oldest first) for multi-turn cases such as follow-ups, and an
optional "restrict_document_title" (substring): the runner marks the matching active
documents is_restricted for that case only, reports which documents the context came from
(flagging any restricted one - a structural check, not an answer-quality grade) and restores them. Pass --channel voice (or text) to run
every case on that channel instead.

Usage:
    python3 -m eval.run_quality_eval [--channel text|voice]
"""
import argparse
import json
from pathlib import Path

from access_control.ask_and_answer import ask_and_answer
from channels import get_channel
from vectorstore.db import get_pool

EVAL_SET_PATH = Path(__file__).parent / "quality_eval_set.json"


def _format_result(result: dict) -> str:
    if result["type"] == "smalltalk":
        return f"SMALLTALK (no retrieval, no LLM)\n    {result['answer']!r}"
    if result["type"] == "catalog":
        return f"CATALOG (0 LLM, 0 retrieval, 0 embedding)\n    {result['answer']!r}"
    if result["type"] in ("source_lookup", "section_lookup", "page_lookup"):
        return f"{result['type'].upper().replace('_', ' ')} (0 LLM calls, 1 retrieval; chunks {result.get('chunk_ids')})\n    {result['answer']!r}"
    return (
        f"ANSWER (shortcut={result['shortcut']}, contextualized={result.get('contextualized')}, "
        f"advisory={result.get('advisory')})\n"
        f"    {result['answer']!r}"
    )


def _restrict(title_substring: str) -> dict:
    """Marks active documents whose title contains the substring restricted; returns the
    previous {document_id: is_restricted} so it can be restored."""
    with get_pool().connection() as conn:
        rows = conn.execute("SELECT id, is_restricted FROM documents WHERE status = 'active' AND document_title ILIKE %s",
                            (f"%{title_substring}%",)).fetchall()
        conn.execute("UPDATE documents SET is_restricted = true WHERE id = ANY(%s)", ([r[0] for r in rows],))
        conn.commit()
    return {r[0]: r[1] for r in rows}


def _restore(previous: dict) -> None:
    with get_pool().connection() as conn:
        for document_id, flag in previous.items():
            conn.execute("UPDATE documents SET is_restricted = %s WHERE id = %s", (flag, document_id))
        conn.commit()


def _documents_for(chunk_ids: list) -> set:
    with get_pool().connection() as conn:
        return {r[0] for r in conn.execute("SELECT document_id FROM chunks WHERE chunk_id = ANY(%s)", (list(chunk_ids),)).fetchall()}


def run_case(case: dict, channel_override: str | None = None):
    channel = channel_override or case.get("channel", "text")
    print(f"--- {case['id']} [{case['category']}] ---")
    print(f"  channel:           {channel}")
    history = case.get("history") or []
    for message in history:
        print(f"  history:           {message['role']:9s} {message['content']!r}")
    print(f"  query:             {case['query']!r}")
    print(f"  expected_behavior: {case['expected_behavior']!r}")
    print(f"  notes:             {case['notes']}")

    # Mirror the API layer (api/chat.py, api/voice.py): small talk first, then a source lookup, then the catalog, then the pipeline.
    channel_module = get_channel(channel)
    category = channel_module.smalltalk_match(case["query"], history)
    source = None if category else channel_module.source_reply(case["query"], history)
    catalog_answer = None if (category or source) else channel_module.catalog_reply(case["query"], history)
    if category:
        result = {"type": "smalltalk", "answer": channel_module.smalltalk_reply(category)}
    elif source:
        result = {"type": source["type"], "answer": source["text"], "chunk_ids": source["chunk_ids"]}
    elif catalog_answer:
        result = {"type": "catalog", "answer": catalog_answer["text"]}
    else:
        restricted = _restrict(case["restrict_document_title"]) if case.get("restrict_document_title") else None
        try:
            result = ask_and_answer(case["query"], channel=channel, history=history)
        finally:
            if restricted is not None:
                _restore(restricted)
        if restricted is not None:
            used = _documents_for(result.get("chunk_ids", []))
            leaked = used & set(restricted)
            print(f"  restricted docs:   {len(restricted)} marked is_restricted for this case (restored afterwards)")
            print(f"  restricted check:  {'FAIL - a restricted document was used: ' + str(sorted(map(str, leaked))) if leaked else 'PASS - no restricted document in the context'}")
    print(f"  actual:            {_format_result(result)}")
    print("  grade:             PASS/FAIL   <- fill in by hand")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", choices=["text", "voice"], help="override every case's channel")
    args = ap.parse_args()
    cases = json.loads(EVAL_SET_PATH.read_text())
    print(f"=== Quality eval report: {len(cases)} cases from {EVAL_SET_PATH.name} ===\n")
    for case in cases:
        run_case(case, args.channel)
    print(
        "=== End of report. Log your grading results in eval/GRADING_LOG.md "
        "(date, pass/fail counts, notes on failures). ==="
    )


if __name__ == "__main__":
    main()
