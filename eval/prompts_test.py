#!/usr/bin/env python3
"""Tests for DB-stored, per-channel generation prompts (config/prompts.py,
/api/admin/prompts), per-channel retrieval depth, and the no-added-LLM-calls
guarantee. Needs the same live services as eval/run_regression.py.

Everything it creates in prompt_templates / app_config is removed or restored
at the end (even on failure).

Usage:
    python3 -m eval.prompts_test
"""
import logging
import statistics
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi.testclient import TestClient  # noqa: E402

from eval.prompt_topics import has_topic  # noqa: E402
import config.prompts as prompts_mod  # noqa: E402
from config.runtime import get_app_config  # noqa: E402
import generation.generate as gen_mod  # noqa: E402
from api.main import app  # noqa: E402
from cache.redis_cache import MISS  # noqa: E402
from config.runtime import get_app_config, set_app_config  # noqa: E402
from generation.generate import generate_answer  # noqa: E402
from vectorstore.db import close_pool, get_pool  # noqa: E402
from vectorstore.retrieve import retrieve  # noqa: E402

TEST_NOTE = "prompts_test"
LENGTH_QUERIES = [
    "how many days of earned leave do I get and what are the rules for it",
    "what is the notice period and can it be bought out",
    "how does the employee referral bonus work",
]
VOICE_WORD_LIMIT = 70  # target is ~50 words; allow slack for model variance

_results = []


def check(name: str, condition: bool, detail: str = ""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if not condition and detail else ""))
    return bool(condition)


calls = {"llm": 0}


def install_counter():
    client = gen_mod.get_client()
    original = client.chat.completions.create

    def counted(*a, **k):
        calls["llm"] += 1
        return original(*a, **k)

    client.chat.completions.create = counted
    from eval.llm_count import wrap_with_options
    def _scoped(create):
        def scoped_counted(*a, **k):
            calls['llm'] += 1
            return create(*a, **k)
        return scoped_counted
    wrap_with_options(client, _scoped)


def final_event(query, channel):
    for event in generate_answer(query, channel=channel):
        if event["type"] == "done":
            return event
    raise AssertionError("no done event")


def active_ids():
    with get_pool().connection() as conn:
        return {(n, c): i for i, n, c in conn.execute(
            "SELECT id, name, channel FROM prompt_templates WHERE is_active AND created_by IS DISTINCT FROM %s",
            (TEST_NOTE,)).fetchall()}


# ---- (a) composition --------------------------------------------------------

def test_composition():
    print("\n=== (a) Prompt composition (each channel from its own rows) ===")
    with get_pool().connection() as conn:
        def active(name, channel):
            return conn.execute("SELECT content FROM prompt_templates WHERE name=%s AND channel=%s AND is_active",
                                (name, channel)).fetchone()[0]
        rows = {ch: (active("style", ch), active("grounding_core", ch)) for ch in ("text", "voice")}
    prompts = {ch: prompts_mod.get_system_prompt(ch) for ch in ("text", "voice")}
    for ch in ("text", "voice"):
        style, core = rows[ch]
        text, versions = prompts[ch]
        check(f"{ch} prompt = style/{ch} + grounding_core/{ch}", text == f"{style}\n\n{core}")
        check(f"{ch} version_info has grounding_core and style", set(versions) == {"grounding_core", "style"}, str(versions))
    check("legacy 'shared' grounding row is ignored by the loader",
          "shared" not in str(prompts) and prompts["text"][0] != prompts["voice"][0])
    check("voice style says spoken / ~50 words; text style does not cap at 50 words",
          "read aloud" in prompts["voice"][0] and "50 words" in prompts["voice"][0] and "50 words" not in prompts["text"][0])
    for topic in ("grounding: answer only from the excerpts", "grounding: no outside / general knowledge",
                  "no citations of titles, sections or pages", "grounding: do not guess"):
        check(f"grounding rule topic present in both channels: {topic!r}",
              has_topic(prompts["text"][0], topic) and has_topic(prompts["voice"][0], topic))


# ---- (b) admin API: versions, activation, validation, immediate effect -----------

def test_admin_versions(client):
    print("\n=== (b) Admin versioning + immediate effect ===")
    before = active_ids()
    _, v0 = prompts_mod.get_system_prompt("text")

    r = client.get("/api/admin/prompts")
    slots = {(g["name"], g["channel"]): g for g in r.json()}
    check("GET /prompts groups the four channel slots with an active version",
          set(slots) == {("grounding_core", "text"), ("grounding_core", "voice"), ("style", "text"), ("style", "voice")}
          and all(g["active_version"] for g in slots.values()), str(list(slots)))

    old_style_text = slots[("style", "text")]
    old_active = next(v for v in old_style_text["versions"] if v["is_active"])
    marker = "TESTMARKER-ZX81"
    r = client.post("/api/admin/prompts", json={
        "name": "style", "channel": "text", "content": old_active["content"] + f"\n- {marker}", "note": TEST_NOTE})
    check("POST /prompts creates a new version and activates it", r.status_code == 200 and r.json()["is_active"], r.text[:200])
    new = r.json()
    check("new version = previous max + 1", new["version"] == max(v["version"] for v in old_style_text["versions"]) + 1, str(new))

    prompt_now, ver_now = prompts_mod.get_system_prompt("text")
    check("next get_system_prompt('text') uses the new version immediately (write-through cache)",
          marker in prompt_now and ver_now["style"] == new["version"], str(ver_now))
    check("voice prompt untouched by a text style edit", marker not in prompts_mod.get_system_prompt("voice")[0])
    done = final_event("how many days of earned leave do I get", "text")
    check("generate_answer result reports the new version in prompt_version",
          done["prompt_version"] and done["prompt_version"]["style"] == new["version"], str(done.get("prompt_version")))

    with get_pool().connection() as conn:
        n_active = conn.execute("SELECT count(*) FROM prompt_templates WHERE name='style' AND channel='text' AND is_active").fetchone()[0]
        old_row = conn.execute("SELECT content, is_active FROM prompt_templates WHERE id=%s", (old_active["id"],)).fetchone()
    check("exactly one active row per slot; old row kept unmodified", n_active == 1 and old_row == (old_active["content"], False))

    r = client.post(f"/api/admin/prompts/{old_active['id']}/activate")
    check("POST /prompts/{id}/activate rolls back", r.status_code == 200 and r.json()["version"] == old_active["version"], r.text[:200])
    _, ver_back = prompts_mod.get_system_prompt("text")
    check("rollback takes effect on the next call", ver_back["style"] == old_active["version"] and marker not in prompts_mod.get_system_prompt("text")[0], str(ver_back))
    done = final_event("how many days of earned leave do I get", "text")
    check("generate_answer reports the rolled-back version", done["prompt_version"]["style"] == old_active["version"], str(done["prompt_version"]))

    check("empty content rejected (422)", client.post("/api/admin/prompts", json={"name": "style", "channel": "text", "content": "   "}).status_code == 422)
    check("over-long content rejected (422)", client.post("/api/admin/prompts", json={"name": "style", "channel": "text", "content": "x" * (int(get_app_config("prompts.max_chars", 8000)) + 1)}).status_code == 422)
    check("unknown slot rejected (422/404)", client.post("/api/admin/prompts", json={"name": "style", "channel": "shared", "content": "x"}).status_code in (404, 422))
    check("unknown name rejected (404)", client.post("/api/admin/prompts", json={"name": "nonsense", "channel": "text", "content": "x"}).status_code == 404)
    check("activate unknown id (404)", client.post("/api/admin/prompts/999999/activate").status_code == 404)
    check("active set restored after rollback", active_ids() == before)
    return v0


# ---- (c) failure degradation ------------------------------------------------------

def test_failure_modes():
    print("\n=== (c) Cache / DB failure degrade without exceptions ===")
    good_text, good_ver = prompts_mod.get_system_prompt("text")

    real_get, real_set = prompts_mod.get_cached, prompts_mod.set_cached
    try:
        prompts_mod.get_cached = lambda key: MISS
        prompts_mod.set_cached = lambda *a, **k: False
        text, ver = prompts_mod.get_system_prompt("text")
        check("Redis down (always miss): prompt still served from the DB", text == good_text and ver == good_ver)

        def boom(key):
            raise ConnectionError("redis exploded")
        prompts_mod.get_cached = boom
        text, ver = prompts_mod.get_system_prompt("text")
        check("cache read raising: no exception, last-good prompt served", text == good_text and ver == good_ver)

        prompts_mod.get_cached = lambda key: MISS
        real_load = prompts_mod._load_from_db

        def db_down(channel):
            raise ConnectionError("postgres is down")
        prompts_mod._load_from_db = db_down
        text, ver = prompts_mod.get_system_prompt("text")
        check("Redis miss + DB failure: last successfully loaded prompt from process memory", text == good_text and ver == good_ver)

        saved = dict(prompts_mod._last_good)
        prompts_mod._last_good.clear()
        records = []

        class Capture(logging.Handler):
            def emit(self, record):
                records.append(record)

        handler = Capture(level=logging.DEBUG)
        prompts_mod.logger.addHandler(handler)
        try:
            text, ver = prompts_mod.get_system_prompt("voice")
        finally:
            prompts_mod.logger.removeHandler(handler)
        check("nothing in memory: hardcoded last-resort prompt, no exception",
              text == prompts_mod.LAST_RESORT_PROMPT and ver == {"last_resort": 0}, f"{ver}")
        check("last-resort fallback logs loudly (CRITICAL)", any(r.levelno == logging.CRITICAL for r in records))
        check("last-resort prompt keeps the grounding essentials",
              "ONLY from the retrieved policy excerpts" in text and "Never mention document titles" in text)
        prompts_mod._last_good.update(saved)
        prompts_mod._load_from_db = real_load

        text, ver = prompts_mod.get_system_prompt("bogus-channel")
        check("unknown channel never raises", isinstance(text, str) and text)
    finally:
        prompts_mod.get_cached, prompts_mod.set_cached = real_get, real_set
        prompts_mod._load_from_db = real_load if "real_load" in dir() else prompts_mod._load_from_db


# ---- (d) per-channel top_k -----------------------------------------------------------

def test_top_k():
    print("\n=== (d) Per-channel retrieval depth ===")
    glob = get_app_config("retrieval.top_k")
    text_k, voice_k = get_app_config("retrieval.top_k.text"), get_app_config("retrieval.top_k.voice")
    print(f"       seeded values: retrieval.top_k={glob} text={text_k} voice={voice_k}")
    check("channel keys are seeded with the current effective value", text_k == glob and voice_k == glob, f"{text_k} {voice_k} {glob}")
    q = "how much notice do I need to serve"
    try:
        set_app_config("retrieval.top_k.voice", 1)
        check("channel-specific key beats global", len(retrieve(q, channel="voice")) == 1 and len(retrieve(q, channel="text")) == glob)
        check("explicit argument beats channel key", len(retrieve(q, top_k=2, channel="voice")) == 2)
        check("unknown channel falls back to the global key", len(retrieve(q, channel="nonexistent")) == glob)
        check("no channel -> global key (existing callers unchanged)", len(retrieve(q)) == glob)
    finally:
        set_app_config("retrieval.top_k.voice", voice_k)


# ---- (e) answers by channel + LLM call counts -------------------------------------------

def test_answers_and_calls(client):
    print("\n=== (e) Voice vs text answers; LLM calls per turn ===")
    words = {"text": [], "voice": []}
    for q in LENGTH_QUERIES:
        for channel in ("text", "voice"):
            ev = final_event(q, channel)
            n = len(ev["answer"].split())
            words[channel].append(n)
            print(f"       {channel:5s} {n:3d} words | {q[:50]!r}: {ev['answer'][:100]!r}")
    check(f"every voice answer within the spoken length target (<= {VOICE_WORD_LIMIT} words)",
          max(words["voice"]) <= VOICE_WORD_LIMIT, str(words["voice"]))
    check("text answers are not capped to the voice target (longest text answer > 60 words)",
          max(words["text"]) > 60, str(words["text"]))
    check("text answers are longer than voice answers on average",
          statistics.mean(words["text"]) > statistics.mean(words["voice"]), str(words))

    install_counter()
    for channel in ("text", "voice"):
        calls["llm"] = 0
        final_event("how many days of earned leave do I get", channel)
        check(f"normal {channel} answer: exactly 1 LLM call", calls["llm"] == 1, f"llm={calls['llm']}")
    calls["llm"] = 0
    r = client.post("/api/chat", json={"query": "what is the weather forecast for tomorrow"})
    check("retrieval-shortcut turn: 0 LLM calls", r.json()["type"] == "refusal_shortcut" and calls["llm"] == 0, f"llm={calls['llm']}")
    calls["llm"] = 0
    r = client.post("/api/chat", json={"query": "wassup"})
    check("small-talk turn: 0 LLM calls", r.json()["type"] == "smalltalk" and calls["llm"] == 0, f"llm={calls['llm']}")
    calls["llm"] = 0
    r = client.post("/api/chat", json={"query": "whats my leave policy"})
    check("normal /api/chat answer: exactly 1 LLM call", r.json()["type"] == "answer" and calls["llm"] == 1, f"llm={calls['llm']}")
    with get_pool().connection() as conn:
        pv = conn.execute("SELECT prompt_version, response_type FROM query_logs ORDER BY id DESC LIMIT 1").fetchone()
    check("query_logs.prompt_version recorded for the generation call", pv[1] == "answer" and pv[0] and set(pv[0]) == {"grounding_core", "style"}, str(pv))
    rec = client.get("/api/admin/usage/recent?limit=1").json()[0]
    check("/usage/recent includes prompt_version", "prompt_version" in rec and rec["prompt_version"], str(rec.keys()))


def main():
    try:
        with TestClient(app) as client:
            test_composition()
            test_admin_versions(client)
            test_failure_modes()
            test_top_k()
            test_answers_and_calls(client)
    finally:
        # remove rows this test created; the original active rows were restored by the test itself
        with get_pool().connection() as conn:
            conn.execute("DELETE FROM prompt_templates WHERE created_by = %s", (TEST_NOTE,))
            conn.commit()
        prompts_mod.refresh_prompt_cache()
    failed = [r for r in _results if not r[1]]
    print(f"\n=== {len(_results) - len(failed)} passed, {len(failed)} failed ===")
    for name, _ok, detail in failed:
        print(f"  - {name}: {detail}")
    close_pool()
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
