#!/usr/bin/env python3
"""Tests for the deterministic small-talk path (channels/text.py, channels/voice.py), the topic
cache (config/topics.py) and the per-turn LLM-call budget.

Needs the same live services as eval/run_regression.py (Postgres, Redis,
OPENAI_API_KEY). Real LLM/retrieval calls are made but COUNTED by wrapping
the OpenAI client and retrieve(), so the per-turn budget is asserted:

    small-talk turn            0 LLM calls, 0 retrieval calls
    retrieval-shortcut turn    0 LLM calls
    normal answer turn         exactly 1 LLM call (the generation call)
    (no clarification turns any more: the referral query is a normal 1-call answer)

Usage:
    python3 -m eval.smalltalk_test
"""
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi.testclient import TestClient  # noqa: E402

import channels.text as text_ch  # noqa: E402
import channels.voice as voice_ch  # noqa: E402
import config.topics as topics_mod  # noqa: E402
import generation.generate as gen_mod  # noqa: E402
from api.main import app  # noqa: E402
from channels import common  # noqa: E402
from vectorstore.db import close_pool, get_pool  # noqa: E402

_results = []


def check(name: str, condition: bool, detail: str = ""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if not condition and detail else ""))
    return bool(condition)


# ---- call counting ---------------------------------------------------------

calls = {"llm": 0, "retrieval": 0}


def install_counters():
    client = gen_mod.get_client()
    original = client.chat.completions.create

    def counted_create(*a, **k):
        calls["llm"] += 1
        return original(*a, **k)

    client.chat.completions.create = counted_create
    for mod in (text_ch, voice_ch):
        orig_retrieve = mod.retrieve

        def counted_retrieve(*a, _o=orig_retrieve, **k):
            calls["retrieval"] += 1
            return _o(*a, **k)

        mod.retrieve = counted_retrieve


def measured(fn):
    calls["llm"] = calls["retrieval"] = 0
    start = time.perf_counter()
    out = fn()
    return out, calls["llm"], calls["retrieval"], (time.perf_counter() - start) * 1000


def last_log_row():
    with get_pool().connection() as conn:
        return conn.execute(
            "SELECT query_text, response_type, source FROM query_logs ORDER BY id DESC LIMIT 1"
        ).fetchone()


# ---- (a) matcher: pure, no services ----------------------------------------

POSITIVE = [
    "wassup", "hey", "thanks", "bye", "what can you do", "how are you",
    "Hi!", "hello there", "Hey, what's up?", "good morning", "thank you so much", "goodbye",
    "ok", "okay cool", "got it", "are you there", "who are you", "hi how are you", "WASSUP???",
]
FALSE_POSITIVE_GUARDS = [
    "hi, how many leaves do I get",
    "thanks, and what about notice period",
    "hello what is the referral bonus",
    "hey can you tell me about the leave policy",
    "ok so what is the notice period",
    "bye the way what is the probation period",
    "we need a standardized tech stack",
    "what is the leave policy",
]


def test_matcher():
    print("\n=== (a) Deterministic matcher (both channels) ===")
    for ch in (text_ch, voice_ch):
        name = ch.CHANNEL
        for text in POSITIVE:
            check(f"[{name}] matches small talk {text!r}", ch.smalltalk_match(text) is not None, f"got {ch.smalltalk_match(text)}")
        for text in FALSE_POSITIVE_GUARDS:
            check(f"[{name}] falls through (not small talk) {text!r}", ch.smalltalk_match(text) is None, f"got {ch.smalltalk_match(text)}")
        check(f"[{name}] empty / punctuation-only input does not match", ch.smalltalk_match("") is None and ch.smalltalk_match("?!") is None)
        check(f"[{name}] over max words never matches", ch.smalltalk_match("hi " * 7) is None)
    check("normalize: lowercase, no apostrophes, collapsed whitespace",
          common.normalize("  What\u2019s   UP?! ") == "whats up", common.normalize("  What\u2019s   UP?! "))

    n = 2000
    start = time.perf_counter()
    for _ in range(n):
        text_ch.smalltalk_match("hello what is the referral bonus")
        text_ch.smalltalk_match("wassup")
    per_call_ms = (time.perf_counter() - start) * 1000 / (2 * n)
    check("matcher handling time is trivially small (< 0.2 ms/call)", per_call_ms < 0.2, f"{per_call_ms:.4f} ms")
    print(f"       matcher: {per_call_ms * 1000:.1f} us per call")


def test_replies():
    print("\n=== (b) Replies (both channels) ===")
    for ch in (text_ch, voice_ch):
        name = ch.CHANNEL
        for category in ch._RESPONSES:
            seq = [ch.smalltalk_reply(category) for _ in range(30)]
            check(f"[{name}] no back-to-back repeat for {category!r}", all(a != b for a, b in zip(seq, seq[1:])))
            check(f"[{name}] {category!r} replies are short and spoken-friendly", all(len(r.split()) <= 20 for r in seq))
        cap = ch.smalltalk_reply(ch.CAPABILITIES)
        topics = topics_mod.get_topics()
        named = [t for t in topics if t in cap]
        check(f"[{name}] capabilities reply names 2-3 real topics", 2 <= len(named) <= 3, f"{cap!r} / named {named}")


# ---- (c) topics ---------------------------------------------------------------

def test_topics():
    print("\n=== (c) Topics cache ===")
    expected = {
        "Calfus Performance Improvement Plan (PIP) Policy": "performance improvement plan",
        "CALFUS CREW: REFERRAL BONUS PROGRAM - INDIA": "referral bonus program",
        "Calfus POSH Policy": "POSH",
        "IT Security Policies": "IT security",
        "Calfus Performance Appraisal & Promotion Policy": "performance appraisal and promotion",
        "Document Classification - Internal": "document classification",
    }
    for title, want in expected.items():
        got = topics_mod.clean_title(title)
        check(f"clean_title({title!r})", got == want, f"got {got!r}")

    topics_mod.invalidate_topics()
    real_load = topics_mod._load_topics
    loads = {"n": 0}

    def counting_load():
        loads["n"] += 1
        return real_load()

    topics_mod._load_topics = counting_load
    try:
        first = topics_mod.get_topics()
        second = topics_mod.get_topics()
        check("topics loaded from DB once, then served from Redis", loads["n"] == 1 and first == second and first,
              f"loads={loads['n']}")
        topics_mod.invalidate_topics()
        topics_mod.get_topics()
        check("invalidate_topics() forces a reload", loads["n"] == 2, f"loads={loads['n']}")
    finally:
        topics_mod._load_topics = real_load
    check("topics are de-duplicated", len(first) == len(set(first)))


# ---- (d) per-turn call budget through the real endpoints ---------------------------

def voice_messages(user_text, history=()):
    base = [{"role": "user", "content": "-"}, {"role": "assistant", "content": "Hi, I'm the HR assistant."}]
    return base + list(history) + [{"role": "user", "content": user_text}]


def voice_turn(client, messages):
    headers = {"Authorization": f"Bearer {os.environ['THINK_ENDPOINT_SECRET']}"} if os.environ.get("THINK_ENDPOINT_SECRET") else {}
    r = client.post("/api/voice/v1/chat/completions", json={"messages": messages, "stream": True}, headers=headers)
    text = ""
    for line in r.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for choice in json.loads(line[6:])["choices"]:
                text += choice["delta"].get("content") or ""
    return r.status_code, text


def test_call_budget():
    print("\n=== (d) Per-turn LLM / retrieval budget (real endpoints) ===")
    rows = []

    def record(label, llm, retr, ms):
        rows.append((label, llm, retr, ms))

    with TestClient(app) as client:
        install_counters()
        client.post("/api/chat", json={"query": "warm up: what is the leave policy"})  # model/pool warm-up

        # small talk, text
        (r, llm, retr, ms) = measured(lambda: client.post("/api/chat", json={"query": "wassup"}))
        record("text  small talk", llm, retr, ms)
        check("text small talk: type smalltalk", r.json().get("type") == "smalltalk", str(r.json()))
        check("text small talk: 0 LLM calls, 0 retrieval calls", (llm, retr) == (0, 0), f"llm={llm} retr={retr}")
        row = last_log_row()
        check("text small talk logged as response_type 'smalltalk', source 'text'",
              row == ("wassup", "smalltalk", "text"), str(row))

        for q in ("what can you do", "thanks", "bye", "hey", "how are you"):
            (r, llm, retr, ms) = measured(lambda q=q: client.post("/api/chat", json={"query": q}))
            check(f"text {q!r}: smalltalk with 0 LLM / 0 retrieval",
                  r.json().get("type") == "smalltalk" and (llm, retr) == (0, 0), f"{r.json()} llm={llm} retr={retr}")

        # small talk, voice
        (res, llm, retr, ms) = measured(lambda: voice_turn(client, voice_messages("wassup")))
        record("voice small talk", llm, retr, ms)
        check("voice small talk: streamed 200 with text", res[0] == 200 and res[1], str(res))
        check("voice small talk: 0 LLM calls, 0 retrieval calls", (llm, retr) == (0, 0), f"llm={llm} retr={retr}")
        row = last_log_row()
        check("voice small talk logged as 'smalltalk', source 'voice'", row == ("wassup", "smalltalk", "voice"), str(row))

        # false-positive guards through the API
        for q in ("hi, how many leaves do I get", "thanks, and what about notice period",
                  "hello what is the referral bonus"):
            (r, llm, retr, ms) = measured(lambda q=q: client.post("/api/chat", json={"query": q}))
            check(f"API falls through to normal pipeline: {q!r}",
                  r.json().get("type") in ("answer", "refusal_shortcut") and retr >= 1,
                  f"{r.json().get('type')} retr={retr}")

        # retrieval shortcut
        (r, llm, retr, ms) = measured(lambda: client.post("/api/chat", json={"query": "what is the weather forecast for tomorrow"}))
        record("text  retrieval shortcut", llm, retr, ms)
        body = r.json()
        check("retrieval shortcut: type refusal_shortcut", body.get("type") == "refusal_shortcut", str(body))
        check("retrieval shortcut: 0 LLM calls (as before)", llm == 0, f"llm={llm}")
        text = body.get("text", "")
        check("refusal copy says it couldn't find it and names topics, no blanket 'check with HR'",
              "couldn't find" in text and "I can help with things like" in text and "check with HR" not in text, text)

        (r, llm, retr, ms) = measured(lambda: client.post("/api/chat", json={"query": "we need a standardized tech stack"}))
        record("text  real transcript (tech stack)", llm, retr, ms)
        check("'we need a standardized tech stack' never gets the bare 'check with HR' reply",
              "check with HR directly" not in r.json().get("text", ""), r.json().get("text", ""))

        # normal answer
        (r, llm, retr, ms) = measured(lambda: client.post("/api/chat", json={"query": "whats my leave policy"}))
        record("text  normal answer", llm, retr, ms)
        check("normal answer: type answer", r.json().get("type") == "answer", str(r.json()))
        check("normal answer: exactly 1 LLM call (the generation call)", llm == 1, f"llm={llm}")

        # the former clarification-triggering query is simply answered now
        q = "will I get paid extra for recommending a friend for a job here"
        (r, llm, retr, ms) = measured(lambda: client.post("/api/chat", json={"query": q}))
        record("text  referral query (was clarif.)", llm, retr, ms)
        check("referral query: answered directly, exactly 1 LLM call, retrieval once",
              r.json().get("type") == "answer" and llm == 1 and retr == 1, f"{r.json().get('type')} llm={llm} retr={retr}")
        check("no turn type is a clarification any more", r.json().get("type") != "needs_clarification")

    print("\n    turn type                          LLM  retr      ms")
    for label, llm, retr, ms in rows:
        print(f"    {label:32s} {llm:5d} {retr:5d} {ms:7.0f}")


def test_quality_set():
    print("\n=== (e) Real-transcript cases in eval/quality_eval_set.json ===")
    cases = {c["id"]: c for c in json.loads((Path(__file__).parent / "quality_eval_set.json").read_text())}
    check("quality set has 'wassup' (smalltalk)", cases.get("s1", {}).get("query") == "wassup")
    check("quality set has 'we need a standardized tech stack'",
          cases.get("s2", {}).get("query") == "we need a standardized tech stack")


def main():
    test_matcher()
    test_topics()
    test_replies()
    test_call_budget()
    test_quality_set()
    failed = [r for r in _results if not r[1]]
    print(f"\n=== {len(_results) - len(failed)} passed, {len(failed)} failed ===")
    for name, _ok, detail in failed:
        print(f"  - {name}: {detail}")
    close_pool()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
