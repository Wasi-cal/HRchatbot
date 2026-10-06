#!/usr/bin/env python3
"""Single regression gate for the HR RAG chatbot's retrieval, generation,
and access-control layers.

Consolidates and replaces (rather than wraps) three former print-and-
eyeball scripts: vectorstore/eval_retrieval.py, generation/
eval_generation.py, access_control/eval_access_control.py. Those relied
on a human reading terminal output; this asserts against fixed, known-
correct expectations and fails loudly (non-zero exit) when reality
diverges from them. See eval/quality_eval_set.json + run_quality_eval.py
for the manual-grading counterpart this doesn't replace - open-ended
answer quality still needs a human.

Sections, in order:
  (a) Retrieval backward-compatibility: retrieve() with no
      user_attributes must return the exact same ranked chunk_ids, in
      the exact same order, as eval/retrieval_baseline.json - a frozen
      fixture, not a live re-derivation. Run with --record-baseline to
      deliberately overwrite that fixture after a real, reviewed change
      (a new document, a re-embed, a fusion parameter change).
  (b) Generation shortcut/grounding: known out-of-corpus queries must
      take the low-confidence shortcut; the known BGV-disclaimer edge
      case (a real, on-topic chunk gets retrieved but doesn't actually
      contain the answer) must still produce a refusal, not a
      confident-sounding guess.
  (c) Access control: the 4 real-data scenarios + control from the
      retired eval_access_control.py, each as a real assertion instead
      of a printed comparison for a human to eyeball.
  (d) Shape/import sanity: asserts the exact field count/keys of shared
      internals (fetch_chunk_details, retrieve()'s result dicts,
      generate_answer()'s "done" event, ask_and_answer()'s result
      shapes) - this is the exact bug class that broke
      vectorstore/eval_retrieval.py silently after fetch_chunk_details()
      grew a field (see git history): a signature change here now fails
      THIS suite loudly instead of a sibling script quietly crashing
      the next time someone happens to run it.

Usage:
    python3 -m eval.run_regression                # run the gate
    python3 -m eval.run_regression --record-baseline   # rewrite (a)'s fixture
"""
import json
import sys
from pathlib import Path

from access_control.ask_and_answer import ask_and_answer
from generation.generate import generate_answer
from vectorstore.db import close_pool, get_pool
from vectorstore.retrieve import fetch_chunk_details, retrieve

BASELINE_PATH = Path(__file__).parent / "retrieval_baseline.json"

PARAPHRASE_QUERIES = [
    "can I do my job from home instead of coming into the office",
    "how much time off do I get when I have a baby",
    "if I quit, how long do I have to keep working before I can leave for good",
    "will I get paid extra for recommending a friend for a job here",
]
EXACT_TERM_QUERIES = [
    "POSH redressal mechanism",
    "PIP objective",
    "BGV disclaimer",
    "90 calendar days notice period buyout",
]
OUT_OF_CORPUS_QUERIES = [
    "what is the weather forecast for tomorrow",
    "can you recommend a good recipe for dinner",
]

# The known-good chunk_ids that touch the real-tagged referral policy
# (country=India, employment_type=full-time employee / intern - see
# access_control/verify_applicability.py).
TAGGED_QUERY = "will I get paid extra for recommending a friend for a job here"
TAGGED_CHUNK_IDS = {"Calfus_Crew_Employee_Referral_Policy_0004", "Calfus_Crew_Employee_Referral_Policy_1_2_0007"}
# "All IT Policies.pdf" carries no applicability tags at all.
UNTAGGED_QUERY = "what are the password requirements"

UNCERTAINTY_PHRASES = (
    "don't have", "do not have", "not sure", "unable to", "no information",
    "check with hr", "reach out to hr", "don't know", "do not know",
)

_results = []  # (name, passed, detail)


def check(name: str, condition: bool, detail: str = ""):
    passed = bool(condition)
    _results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if not passed and detail else ""))
    return passed


def indicates_uncertainty(answer: str) -> bool:
    lowered = answer.lower()
    return any(phrase in lowered for phrase in UNCERTAINTY_PHRASES)


def get_final_answer(query: str, **kwargs) -> dict:
    for event in generate_answer(query, **kwargs):
        if event["type"] == "done":
            return event
    raise AssertionError(f"generate_answer({query!r}) never yielded a 'done' event")


# --- (a) Retrieval backward-compatibility ---------------------------------

def run_retrieval_regression():
    print("\n=== (a) Retrieval backward-compatibility vs. frozen baseline ===")
    baseline = json.loads(BASELINE_PATH.read_text())
    for query, expected_chunk_ids in baseline["cases"].items():
        actual = [r["chunk_id"] for r in retrieve(query, top_k=baseline["top_k"])]
        check(
            f"retrieve({query!r}) matches baseline",
            actual == expected_chunk_ids,
            f"expected {expected_chunk_ids}, got {actual}",
        )


def record_baseline():
    cases = {}
    for query in PARAPHRASE_QUERIES + EXACT_TERM_QUERIES:
        cases[query] = [r["chunk_id"] for r in retrieve(query, top_k=3)]
    BASELINE_PATH.write_text(json.dumps({"top_k": 3, "cases": cases}, indent=2) + "\n")
    print(f"Recorded new baseline for {len(cases)} queries to {BASELINE_PATH}")


# --- (b) Generation shortcut/grounding behavior ----------------------------

def run_generation_regression():
    print("\n=== (b) Generation shortcut/grounding behavior ===")

    for query in OUT_OF_CORPUS_QUERIES:
        event = get_final_answer(query)
        check(f"shortcut fires for out-of-corpus query {query!r}", event["shortcut"] is True,
              f"got shortcut={event['shortcut']}, answer={event['answer']!r}")

    # "what is the capital of France" is known to score just above the
    # confidence threshold (see generation/generate.py's docstring for
    # the RRF-floor math) - it does NOT take the shortcut, but the model
    # must still refuse rather than answer from outside knowledge.
    event = get_final_answer("what is the capital of France")
    check("shortcut does NOT fire for 'capital of France' (known threshold edge case)",
          event["shortcut"] is False, f"got shortcut={event['shortcut']}")
    check("'capital of France' answer still indicates uncertainty/refusal",
          indicates_uncertainty(event["answer"]), f"answer={event['answer']!r}")

    # BGV disclaimer: a real, on-topic chunk (BGV_Policy_Ver1_0_0005) is
    # retrieved with a confident-looking fused score, but it doesn't
    # actually contain disclaimer content - correct behavior is refusal
    # despite retrieval "succeeding".
    event = get_final_answer("BGV disclaimer")
    check("shortcut does NOT fire for 'BGV disclaimer' (real chunk retrieved)",
          event["shortcut"] is False, f"got shortcut={event['shortcut']}")
    check("'BGV disclaimer' answer indicates uncertainty despite a confident retrieval",
          indicates_uncertainty(event["answer"]), f"answer={event['answer']!r}")


# --- (c) Access control scenarios ------------------------------------------

def run_access_control_regression():
    print("\n=== (c) Access control scenarios (real corpus data) ===")

    # (a) No user_attributes at all -> candidate pool touches tagged
    # content, nothing known about the user -> clarification, blocking
    # on "country" (first blocking tag_type alphabetically).
    result = ask_and_answer(TAGGED_QUERY, user_attributes=None)
    check("scenario (a) no attributes -> needs_clarification",
          result["type"] == "needs_clarification", f"got {result}")
    check("scenario (a) blocks on 'country'",
          result.get("blocking_tag_type") == "country", f"got {result}")

    # (b) country known, employment_type unknown -> clarification,
    # blocking on "employment_type".
    result = ask_and_answer(TAGGED_QUERY, user_attributes={"country": "India", "employment_type": None})
    check("scenario (b) missing employment_type -> needs_clarification",
          result["type"] == "needs_clarification", f"got {result}")
    check("scenario (b) blocks on 'employment_type'",
          result.get("blocking_tag_type") == "employment_type", f"got {result}")

    # (c) both known and matching -> real answer, tagged chunk included
    # in the filtered candidate pool.
    attrs_match = {"country": "India", "employment_type": "full-time employee"}
    result = ask_and_answer(TAGGED_QUERY, user_attributes=attrs_match)
    check("scenario (c) matching attributes -> answer (not clarification)",
          result["type"] == "answer", f"got {result}")
    filtered_ids = {r["chunk_id"] for r in retrieve(TAGGED_QUERY, top_k=5, user_attributes=attrs_match)}
    check("scenario (c) tagged chunk included when attributes match",
          bool(filtered_ids & TAGGED_CHUNK_IDS), f"filtered_ids={filtered_ids}")

    # (d) country known but wrong -> tagged chunks excluded from the
    # filtered candidate pool, and the fallback answer refuses rather
    # than answering from irrelevant leftover context.
    attrs_mismatch = {"country": "United States", "employment_type": "full-time employee"}
    unfiltered_ids = {r["chunk_id"] for r in retrieve(TAGGED_QUERY, top_k=5)}
    filtered_ids = {r["chunk_id"] for r in retrieve(TAGGED_QUERY, top_k=5, user_attributes=attrs_mismatch)}
    check("scenario (d) tagged chunks present unfiltered",
          bool(unfiltered_ids & TAGGED_CHUNK_IDS), f"unfiltered_ids={unfiltered_ids}")
    check("scenario (d) tagged chunks excluded once filtered on mismatched country",
          not (filtered_ids & TAGGED_CHUNK_IDS), f"filtered_ids={filtered_ids}")
    result = ask_and_answer(TAGGED_QUERY, user_attributes=attrs_mismatch)
    check("scenario (d) result type is 'answer' (attributes fully known, no clarification needed)",
          result["type"] == "answer", f"got {result}")
    check("scenario (d) answer indicates refusal (irrelevant fallback context)",
          indicates_uncertainty(result.get("answer", "")), f"answer={result.get('answer')!r}")

    # Control: untagged content, user_attributes must have zero effect.
    unfiltered_ids = {r["chunk_id"] for r in retrieve(UNTAGGED_QUERY, top_k=5)}
    filtered_ids = {r["chunk_id"] for r in retrieve(
        UNTAGGED_QUERY, top_k=5, user_attributes={"country": "United States", "employment_type": "contractor"}
    )}
    check("control: untagged query unaffected by user_attributes",
          unfiltered_ids == filtered_ids, f"unfiltered={unfiltered_ids} filtered={filtered_ids}")


# --- (d) Shape/import sanity ------------------------------------------------

def run_shape_sanity():
    print("\n=== (d) Shape/import sanity (the bug class we just found and fixed) ===")

    pool = get_pool()
    with pool.connection() as conn:
        results = retrieve("PIP objective", top_k=1)
        chunk_db_ids = [r for r in fetch_chunk_details(conn, [])]  # empty-input shape check
        check("fetch_chunk_details([]) returns empty dict", chunk_db_ids == [], f"got {chunk_db_ids}")

        # Re-derive a real chunk_db_id via a direct query so we can
        # exercise fetch_chunk_details() with a non-empty input too.
        row = conn.execute(
            "SELECT id FROM chunks WHERE chunk_id = %s", (results[0]["chunk_id"],)
        ).fetchone()
        details = fetch_chunk_details(conn, [row[0]])
        check("fetch_chunk_details() returns exactly one 5-field tuple per chunk_db_id",
              len(details) == 1 and len(next(iter(details.values()))) == 5,
              f"got {details}")

    expected_retrieve_keys = {
        "chunk_id", "section_path", "text", "document_title", "document_id",
        "fused_score", "dense_rank", "sparse_rank",
    }
    check("retrieve() result dicts have the expected keys",
          set(results[0].keys()) == expected_retrieve_keys, f"got {set(results[0].keys())}")

    event = get_final_answer("PIP objective")
    check("generate_answer()'s 'done' event has the expected keys",
          set(event.keys()) == {"type", "answer", "shortcut", "timing"}, f"got {set(event.keys())}")

    # UNTAGGED_QUERY, not "PIP objective" - PIP is now real-tagged
    # (country=India, employment_type=full-time employee), so it would
    # return needs_clarification here rather than an "answer" shape.
    answer_result = ask_and_answer(UNTAGGED_QUERY, user_attributes=None)
    check("ask_and_answer() 'answer' result has the expected keys",
          set(answer_result.keys()) == {"type", "answer", "shortcut", "timing"}, f"got {set(answer_result.keys())}")

    clarification_result = ask_and_answer(TAGGED_QUERY, user_attributes=None)
    check("ask_and_answer() 'needs_clarification' result has the expected keys",
          set(clarification_result.keys()) == {"type", "question", "blocking_tag_type"},
          f"got {set(clarification_result.keys())}")


def main():
    if "--record-baseline" in sys.argv:
        record_baseline()
        return

    run_retrieval_regression()
    run_generation_regression()
    run_access_control_regression()
    run_shape_sanity()
    close_pool()

    passed = sum(1 for _, ok, _ in _results if ok)
    failed = len(_results) - passed
    print(f"\n=== {passed} passed, {failed} failed ===")
    if failed:
        print("Failed assertions:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
