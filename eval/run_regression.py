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
  (c) No applicability filtering / clarification: retrieval is identical with or
      without user attributes, no turn ever becomes a clarifying question, restricted
      documents are never used (toggled and restored here).
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
    "couldn't find", "could not find", "can't find", "cannot find",
    "focus on hr",  # the non-policy-message rule's polite redirect ("...but I focus on HR policies")
)

_results = []  # (name, passed, detail)


def check(name: str, condition: bool, detail: str = ""):
    passed = bool(condition)
    _results.append((name, passed, detail))
    print(f"[{'PASS' if passed else 'FAIL'}] {name}" + (f" -- {detail}" if not passed and detail else ""))
    return passed


def indicates_uncertainty(answer: str) -> bool:
    # Normalise typographic apostrophes: the model often emits "don’t" (U+2019).
    lowered = answer.lower().replace("\u2019", "'")
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
    # The system prompt now tells the model to answer a non-policy message with one
    # friendly redirect sentence instead of "I don't have that information", so
    # phrase-matching uncertainty is no longer the right check for a non-policy
    # question: assert what actually matters - no outside knowledge leaked.
    check("'capital of France' answer does not leak outside knowledge (refuses or redirects)",
          "paris" not in event["answer"].lower(), f"answer={event['answer']!r}")

    # BGV disclaimer: a real, on-topic chunk (BGV_Policy_Ver1_0_0005) is
    # retrieved with a confident-looking fused score, but it doesn't
    # actually contain disclaimer content - correct behavior is refusal
    # despite retrieval "succeeding".
    event = get_final_answer("BGV disclaimer")
    check("shortcut does NOT fire for 'BGV disclaimer' (real chunk retrieved)",
          event["shortcut"] is False, f"got shortcut={event['shortcut']}")
    check("'BGV disclaimer' answer indicates uncertainty despite a confident retrieval",
          indicates_uncertainty(event["answer"]), f"answer={event['answer']!r}")


# --- (c) No applicability filtering / clarification -------------------------
# (Replaces the former access-control scenarios: the layer is unwired, see
# docs/applicability_layer_removed.md. Last version with them: commit fd317af.)

def run_unfiltered_regression():
    print("\n=== (c) No applicability filtering or clarification ===")
    attr_sets = [None, {}, {"country": "India", "employment_type": "full-time employee"},
                 {"country": "United States", "employment_type": "contractor"}]
    for query in (TAGGED_QUERY, UNTAGGED_QUERY, "how many leaves do interns get"):
        baseline = [(r["chunk_id"], r["fused_score"]) for r in retrieve(query, top_k=5)]
        same = all([(r["chunk_id"], r["fused_score"]) for r in retrieve(query, top_k=5, user_attributes=a)] == baseline
                   for a in attr_sets)
        check(f"retrieval identical with and without any attributes: {query!r}", same, f"baseline={baseline}")
        for a in attr_sets:
            result = ask_and_answer(query, user_attributes=a)
            check(f"ask_and_answer({query[:30]!r}..., attributes={a}) is an answer, never a clarification",
                  result["type"] == "answer", f"got {result.get('type')}")
    unfiltered = {r["chunk_id"] for r in retrieve(TAGGED_QUERY, top_k=5, user_attributes={"country": "United States"})}
    check("tagged (India-only) content is still retrieved for a non-India attribute (nothing is filtered)",
          bool(unfiltered & TAGGED_CHUNK_IDS), f"ids={unfiltered}")

    # Restricted documents are the ONLY thing hidden: toggle, confirm never used, restore.
    pool = get_pool()
    with pool.connection() as conn:
        ids = [r[0] for r in conn.execute(
            "SELECT d.id FROM documents d WHERE d.status = 'active' AND d.document_title ILIKE '%REFERRAL BONUS%'").fetchall()]
        was = {r[0]: r[1] for r in conn.execute("SELECT id, is_restricted FROM documents WHERE id = ANY(%s)", (ids,)).fetchall()}
    check("found the referral documents to restrict", len(ids) >= 1, str(ids))
    try:
        with pool.connection() as conn:
            conn.execute("UPDATE documents SET is_restricted = true WHERE id = ANY(%s)", (ids,))
            conn.commit()
        results = retrieve(TAGGED_QUERY, top_k=10)
        check("restricted documents never appear in retrieve()", not any(str(r["document_id"]) in map(str, ids) for r in results),
              str([(r["chunk_id"]) for r in results]))
        result = ask_and_answer(TAGGED_QUERY)
        check("restricted documents are never used for an answer", not (set(result.get("chunk_ids", [])) & TAGGED_CHUNK_IDS),
              str(result.get("chunk_ids")))
    finally:
        with pool.connection() as conn:
            for doc_id, flag in was.items():
                conn.execute("UPDATE documents SET is_restricted = %s WHERE id = %s", (flag, doc_id))
            conn.commit()
    back = {r["document_id"] for r in retrieve(TAGGED_QUERY, top_k=10)}
    check("restored: the referral documents are retrievable again", any(str(d) in map(str, ids) for d in back), str(back))


# --- (e) Per-turn LLM / retrieval / embedding calls, through the real endpoints -------

def run_call_budget():
    print("\n=== (e) Per-turn calls: normal turn = 1 retrieval, 1 embedding, 1 generation ===")
    import os

    from fastapi.testclient import TestClient

    import channels.text as text_ch
    import channels.voice as voice_ch
    import generation.generate as gen_mod
    import vectorstore.retrieve as retrieve_mod
    from api.main import app

    calls = {"llm": 0, "retrieve": 0, "embed": 0, "messages": []}
    client_llm = gen_mod.get_client()
    original_create = client_llm.chat.completions.create

    def counted_create(*a, **k):
        calls["llm"] += 1
        calls["messages"].append(k.get("messages"))
        return original_create(*a, **k)

    client_llm.chat.completions.create = counted_create
    embedder = retrieve_mod._get_embedder()
    original_embed = embedder.embed

    def counted_embed(*a, **k):
        calls["embed"] += 1
        return original_embed(*a, **k)

    embedder.embed = counted_embed
    for mod in (text_ch, voice_ch):
        orig_retrieve = mod.retrieve

        def counted_retrieve(*a, _o=orig_retrieve, **k):
            calls["retrieve"] += 1
            return _o(*a, **k)

        mod.retrieve = counted_retrieve

    def measured(fn):
        calls.update(llm=0, retrieve=0, embed=0, messages=[])
        retrieve_mod.embed_query.cache_clear()
        retrieve_mod.embed_queries.cache_clear()
        return fn()

    def voice_turn(client, text):
        headers = {"Authorization": f"Bearer {os.environ['THINK_ENDPOINT_SECRET']}"} if os.environ.get("THINK_ENDPOINT_SECRET") else {}
        msgs = [{"role": "user", "content": "-"}, {"role": "assistant", "content": "Hi, I'm the HR assistant."},
                {"role": "user", "content": text}]
        r = client.post("/api/voice/v1/chat/completions", json={"messages": msgs, "stream": True}, headers=headers)
        out = ""
        for line in r.text.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                for choice in json.loads(line[6:])["choices"]:
                    out += choice["delta"].get("content") or ""
        return out

    with get_pool().connection() as conn:
        first_id = conn.execute("SELECT coalesce(max(id), 0) FROM query_logs").fetchone()[0]

    with TestClient(app) as client:
        client.post("/api/chat", json={"query": "warm up: what is the leave policy"})
        r = measured(lambda: client.post("/api/chat", json={"query": "how many days of earned leave do I get"}))
        check("text normal turn: answer, exactly 1 generation call, no other LLM call",
              r.json()["type"] == "answer" and calls["llm"] == 1, f"{r.json()['type']} llm={calls['llm']}")
        check("text normal turn: retrieval runs once, one embedding call", calls["retrieve"] == 1 and calls["embed"] == 1,
              f"retrieve={calls['retrieve']} embed={calls['embed']}")
        text_v = measured(lambda: voice_turn(client, "how many days of earned leave do I get"))
        check("voice normal turn: spoken answer, exactly 1 generation call", bool(text_v) and calls["llm"] == 1, f"llm={calls['llm']}")
        check("voice normal turn: retrieval runs once, one embedding call", calls["retrieve"] == 1 and calls["embed"] == 1,
              f"retrieve={calls['retrieve']} embed={calls['embed']}")
        r = measured(lambda: client.post("/api/chat", json={"query": "what is the weather forecast for tomorrow"}))
        check("shortcut turn: 0 LLM calls, retrieval once", r.json()["type"] == "refusal_shortcut" and calls["llm"] == 0 and calls["retrieve"] == 1,
              f"{r.json()['type']} llm={calls['llm']} retrieve={calls['retrieve']}")

        # The old clarification-triggering query answers directly, on both channels, with and without legacy fields.
        r = measured(lambda: client.post("/api/chat", json={"query": TAGGED_QUERY}))
        check("referral query (formerly a clarification) is answered directly: 1 generation call, 1 retrieval",
              r.json()["type"] == "answer" and calls["llm"] == 1 and calls["retrieve"] == 1, f"{r.json()['type']} llm={calls['llm']} retr={calls['retrieve']}")
        label_seen = any("[Applies to: India; full-time employee, intern]" in m["content"] for m in calls["messages"][-1])
        check("retrieved chunks carry the plain '[Applies to: ...]' group label into the model input", label_seen)
        legacy = client.post("/api/chat", json={"query": TAGGED_QUERY, "user_attributes": {"country": "India"},
                                                 "pending_clarification": {"original_query": "x", "blocking_tag_type": "country"}})
        check("legacy request fields (user_attributes, pending_clarification) are ignored, still an answer",
              legacy.status_code == 200 and legacy.json()["type"] == "answer" and "user_attributes" not in legacy.json(), legacy.text[:200])
        spoken = measured(lambda: voice_turn(client, TAGGED_QUERY))
        check("voice endpoint never emits the clarification phrase or question",
              "quick check before i answer" not in spoken.lower() and "full-time employee or a contractor" not in spoken.lower(), spoken)
        check("voice referral turn: 1 generation call, 1 retrieval", calls["llm"] == 1 and calls["retrieve"] == 1, f"llm={calls['llm']} retr={calls['retrieve']}")

    with get_pool().connection() as conn:
        n = conn.execute("SELECT count(*) FROM query_logs WHERE id > %s AND response_type = 'clarification'", (first_id,)).fetchone()[0]
        cols = conn.execute("SELECT count(*) FROM query_logs WHERE id > %s AND (blocking_tag_type IS NOT NULL OR user_attributes IS NOT NULL)", (first_id,)).fetchone()[0]
    check("no query_logs row written by this run has response_type 'clarification'", n == 0, str(n))
    check("blocking_tag_type / user_attributes are no longer written", cols == 0, str(cols))


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
        check("fetch_chunk_details() returns exactly one 6-field tuple per chunk_db_id (incl. the applies_to label)",
              len(details) == 1 and len(next(iter(details.values()))) == 6,
              f"got {details}")

    expected_retrieve_keys = {
        "chunk_id", "section_path", "text", "document_title", "document_id", "applies_to",
        "fused_score", "dense_rank", "sparse_rank",
    }
    check("retrieve() result dicts have the expected keys",
          set(results[0].keys()) == expected_retrieve_keys, f"got {set(results[0].keys())}")

    event = get_final_answer("PIP objective")
    check("generate_answer()'s 'done' event has the expected keys",
          set(event.keys()) == {"type", "answer", "shortcut", "timing", "chunk_ids", "prompt_version", "contextualized", "advisory"}, f"got {set(event.keys())}")

    answer_result = ask_and_answer(UNTAGGED_QUERY, user_attributes=None)
    check("ask_and_answer() 'answer' result has the expected keys",
          set(answer_result.keys()) == {"type", "answer", "shortcut", "timing", "chunk_ids", "prompt_version", "contextualized", "advisory"}, f"got {set(answer_result.keys())}")


def main():
    if "--record-baseline" in sys.argv:
        record_baseline()
        return

    run_retrieval_regression()
    run_generation_regression()
    run_unfiltered_regression()
    run_shape_sanity()
    run_call_budget()
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
