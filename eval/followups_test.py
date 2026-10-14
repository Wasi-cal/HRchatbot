#!/usr/bin/env python3
"""Tests for: the group/leave-counting prompt rules (v4), config-driven small talk incl. the new acknowledgements
and the affirmation-after-an-offer rule, and deterministic source lookups ("where is this from") on both channels.

    python3 -m eval.followups_test

Needs the same live services as eval/run_regression.py. Real LLM / retrieval / embedding calls are made but COUNTED
by wrapping the OpenAI client, the channel retrieve() functions and the BGE embedder. Settings it changes are
restored at the end (even on failure).
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi.testclient import TestClient  # noqa: E402

import channels.text as text_ch  # noqa: E402
import channels.voice as voice_ch  # noqa: E402
from eval.prompt_topics import has_topic  # noqa: E402
import config.prompts as prompts_mod  # noqa: E402
import generation.generate as gen_mod  # noqa: E402
import vectorstore.retrieve as retrieve_mod  # noqa: E402
from api.main import app  # noqa: E402
from channels import common  # noqa: E402
from config.runtime import get_app_config, set_app_config  # noqa: E402
from vectorstore.db import close_pool, get_pool  # noqa: E402

REFERRAL_Q = "will I get paid extra for recommending a friend for a job here"
TRIP_Q = "i need to go to my hometown for a week and i have only 3 leaves left"
INTERN_Q = "tell me everything you know about the interns"
OFFER = "Yes, a referral bonus is paid after probation. Would you like the eligibility details?"
STATEMENT = "You get 21 days of earned leave a year."
CHANNELS = (("text", text_ch), ("voice", voice_ch))

_results = []
calls = {"llm": 0, "retrieve": 0, "embed": 0, "messages": []}


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if not condition and detail else ""))
    return bool(condition)


def install_counters():
    client = gen_mod.get_client()
    original = client.chat.completions.create

    def counted(*a, **k):
        calls["llm"] += 1
        calls["messages"].append(k.get("messages"))
        return original(*a, **k)

    client.chat.completions.create = counted
    from eval.llm_count import wrap_with_options
    def _scoped(create):
        def scoped_counted(*a, **k):
            calls['llm'] += 1
            calls['messages'].append(k.get('messages'))
            return create(*a, **k)
        return scoped_counted
    wrap_with_options(client, _scoped)
    embedder = retrieve_mod._get_embedder()
    original_embed = embedder.embed

    def counted_embed(*a, **k):
        calls["embed"] += 1
        return original_embed(*a, **k)

    embedder.embed = counted_embed
    for mod in (text_ch, voice_ch):
        orig = mod.retrieve

        def counted_retrieve(*a, _o=orig, **k):
            calls["retrieve"] += 1
            return _o(*a, **k)

        mod.retrieve = counted_retrieve


def reset_counts():
    calls.update(llm=0, retrieve=0, embed=0, messages=[])
    retrieve_mod.embed_query.cache_clear()
    retrieve_mod.embed_queries.cache_clear()


def last_log():
    with get_pool().connection() as conn:
        return conn.execute("SELECT response_type, source, chunk_ids_used, retrieval_time_ms, contextualized, advisory "
                            "FROM query_logs ORDER BY id DESC LIMIT 1").fetchone()


def msgs(*pairs):
    return [{"role": r, "content": c} for r, c in pairs]


# ---- (a) prompt v4 ----------------------------------------------------------------------------------------

def test_prompt():
    print("\n=== (a) Prompt rules (grounding_core, both channels) ===")
    # Rule TOPICS (eval/prompt_topics.py), not exact phrases: a rewrite of the wording must not break these, a dropped rule must.
    needles = {
        "never infers the user's group from earlier topics": "never infer the user's group from earlier topics",
        "group is the user's own only if stated in the first person": "group is the user's own only when stated in the first person",
        "no 'not entitled / typically doesn't get / gets less'": "no 'not entitled' / 'gets less' claims",
        "no general knowledge about how companies treat groups": "grounding: no outside / general knowledge",
        "couldn't-find phrasing for a stated group": "couldn't-find phrasing for a stated group",
        "leave-day counting states the assumption": "leave counting states the assumption",
        "working days only, a week is five": "working days only, a week is five",
        "existing: group labels": "group labels ([Applies to: ...])",
        "existing: no clarifying question": "no clarifying question about the group",
        "existing: never say excerpts/context": "never say 'excerpts' / 'context' wording",
        "existing: grounding": "grounding: answer only from the excerpts",
        "existing: no citations": "no citations of titles, sections or pages",
    }
    for channel in ("text", "voice"):
        prompt, versions = prompts_mod.get_system_prompt(channel)
        check(f"[{channel}] grounding_core is version >= 4 (earlier versions kept for rollback)", versions["grounding_core"] >= 4, str(versions))
        for label, needle in needles.items():
            check(f"[{channel}] {label}", has_topic(prompt, needle), needle)
    with get_pool().connection() as conn:
        n = conn.execute("SELECT count(*) FROM prompt_templates WHERE name='grounding_core' AND channel IN ('text','voice')").fetchone()[0]
    check("earlier grounding_core versions still exist (v1..v3 per channel + v4)", n >= 8, str(n))


# ---- (b) small talk from config + offers ----------------------------------------------------------------------

POSITIVE_ACKS = ["yes thank you", "yessir", "yes sir", "thank you so much", "thanks a lot", "cheers", "great thanks",
                 "perfect thanks", "ok thanks", "got it thanks", "Yes Thank You!", "YESSIR", "yes", "yeah", "yep", "sure", "ok", "okay"]
STILL_PIPELINE = ["yes thanks, and what about notice period", "yes what is the referral bonus", "thanks but how many sick leaves do i get",
                  "yes please explain more about eligibility and approval rules"]


def test_smalltalk(client):
    print("\n=== (b) Small talk: config-driven phrases, acknowledgements, affirmations after an offer ===")
    for name, ch in CHANNELS:
        for u in POSITIVE_ACKS:
            check(f"[{name}] after a statement, {u!r} is small talk", ch.smalltalk_match(u, msgs(("user", "x"), ("assistant", STATEMENT))) is not None)
            check(f"[{name}] with no history, {u!r} is small talk", ch.smalltalk_match(u) is not None)
        for u in STILL_PIPELINE:
            check(f"[{name}] real question starting with an acknowledgement is NOT small talk: {u!r}", ch.smalltalk_match(u) is None)
        for u in ("yes", "yeah", "yep", "sure", "ok", "okay", "yessir", "yes sir", "yes thank you"):
            for label, reply in (("ends with '?'", "Do you want the notice period too?"), ("'would you like'", "Would you like more detail. It is paid after probation."),
                                 ("'want me to'", "I can check that. Want me to go on."), ("'i can also'", "That is 21 days. I can also list the other leave types.")):
                check(f"[{name}] {u!r} after an assistant turn that {label} is NOT small talk",
                      ch.smalltalk_match(u, msgs(("user", "x"), ("assistant", reply))) is None)
        check(f"[{name}] a thanks-only phrase after an offer is still small talk",
              ch.smalltalk_match("thank you so much", msgs(("user", "x"), ("assistant", OFFER))) == "thanks")
        check(f"[{name}] after the USER's own last message (no assistant yet) a bare yes is small talk",
              ch.smalltalk_match("yes", msgs(("user", "x"))) == "ack")
    long_answer = "A" * 700 + " Would you like me to go through the other leave types?"
    check("offer is detected on the RAW (untruncated) last assistant message", common.previous_turn_invites_reply(msgs(("user", "x"), ("assistant", long_answer))))
    cleaned = common.clean_history(msgs(("user", "x"), ("assistant", long_answer)), "text")
    check("history truncation keeps the TAIL of a long assistant message (the offer) within the 400-char cap",
          len(cleaned[-1]["content"]) <= 400 and cleaned[-1]["content"].endswith("other leave types?"), cleaned[-1]["content"][-60:])

    orig = get_app_config("smalltalk.phrases.text")
    try:
        check("editing the phrase list via the admin API takes effect at once (new phrase 'zorp')",
              client.patch("/api/admin/config/smalltalk.phrases.text", json={"value": {**orig, "ack": orig["ack"] + ["zorp"]}}).status_code == 200
              and text_ch.smalltalk_match("zorp") == "ack" and voice_ch.smalltalk_match("zorp") is None)
        check("phrase sets are per channel (text edit did not touch voice)", voice_ch.smalltalk_match("zorp") is None)
        check("invalid category rejected (422)", client.patch("/api/admin/config/smalltalk.phrases.text", json={"value": {"nonsense": ["x"]}}).status_code == 422)
        check("empty phrase list rejected (422)", client.patch("/api/admin/config/smalltalk.phrases.text", json={"value": {"ack": []}}).status_code == 422)
    finally:
        client.patch("/api/admin/config/smalltalk.phrases.text", json={"value": orig})
    check("restored: 'zorp' is no longer small talk", text_ch.smalltalk_match("zorp") is None)

    with get_pool().connection() as conn:
        row = conn.execute("SELECT count(*) FROM app_config WHERE key IN ('smalltalk.phrases.text','smalltalk.phrases.voice','smalltalk.affirmations','smalltalk.offer_markers')").fetchone()[0]
    check("small-talk patterns are stored in app_config (not hardcoded in the channel modules)", row == 4 and not hasattr(text_ch, "_PATTERNS") and not hasattr(voice_ch, "_PATTERNS"))


# ---- (c) source lookup --------------------------------------------------------------------------------------------

def chunk_ids_of(mod, query, history=None):
    return [e for e in mod.answer_stream(query, history=history) if e["type"] == "done"][0]["chunk_ids"]


def test_source(client):
    print("\n=== (c) Source lookup: detection, recomputed retrieval, templates ===")
    for name, ch in CHANNELS:
        a = ch.answer(REFERRAL_Q)
        h = msgs(("user", REFERRAL_Q), ("assistant", a["answer"]))
        for u in ("whats the source of this/", "What's the source?", "where is this from", "where did you get that", "which policy is this",
                  "which document", "reference", "where is that mentioned", "source"):
            check(f"[{name}] {u!r} with a previous answer is a source lookup", ch.source_reply(u, h) is not None)
            check(f"[{name}] {u!r} with no history falls through (None)", ch.source_reply(u, []) is None)
        for u in ("okay whats the source of that", "ok where did u get that", "so where is this from", "whats the source of that please", "and which document", "Okay, what's the source of that?"):
            check(f"[{name}] with filler words / informal forms {u!r} is still a source lookup", ch.source_reply(u, h) is not None)
        for u in ("so which document do i submit for joining", "okay where is the office", "and what is the source of the referral bonus formula", "well reference check policy please"):
            check(f"[{name}] a real question with filler words {u!r} is NOT a source lookup", ch.source_reply(u, h) is None)
        check(f"[{name}] fillers alone are not a source question", ch.source_reply("okay so please", h) is None)
        for u in ("what is the source of the referral bonus formula and who approves it", "where is the office", "reference check policy"):
            check(f"[{name}] real question {u!r} is NOT a source lookup", ch.source_reply(u, h) is None)
        s = ch.source_reply("whats the source of this", h)
        check(f"[{name}] recomputed retrieval equals the answer's retrieval (same chunks, same order)", s["chunk_ids"] == a["chunk_ids"], f"{s['chunk_ids']} vs {a['chunk_ids']}")
        h2 = h + msgs(("user", "whats the source of this"), ("assistant", s["text"]))
        check(f"[{name}] asking for the source twice still refers to the real answer", ch.source_reply("source", h2)["chunk_ids"] == a["chunk_ids"])

        # a contextualised follow-up: its retrieval used the preceding user turns; the lookup must reproduce it
        follow = "and what about interns?"
        hf = h
        a2 = ch.answer(follow, history=hf)
        hh = hf + msgs(("user", follow), ("assistant", a2["answer"]))
        if a2["shortcut"] or a2["answer"].lower().startswith(common.REFUSAL_PREFIX):
            # the model itself declined this time (LLM variance): the lookup rightly says it wasn't from a document
            check(f"[{name}] contextualised follow-up (model declined this run): source lookup says it didn't come from a document",
                  ch.source_reply("where is this from", hh)["chunk_ids"] == [])
        else:
            check(f"[{name}] contextualised follow-up: source lookup reproduces ITS retrieval",
                  ch.source_reply("where is this from", hh)["chunk_ids"] == a2["chunk_ids"], f"{ch.source_reply('where is this from', hh)['chunk_ids']} vs {a2['chunk_ids']}")
        # an advisory turn (wider top_k, diversity cap, expansions)
        advisory_q = "is there anything else that i can do regarding that as i dont want LOP"
        hist_adv = msgs(("user", "whats my leave policy"), ("assistant", "21 days."), ("user", TRIP_Q), ("assistant", "2 unpaid days."))
        adv_chunks = chunk_ids_of(ch, advisory_q, hist_adv)
        hist_after = hist_adv + msgs(("user", advisory_q), ("assistant", "Try comp-off or work from home."))
        check(f"[{name}] advisory turn (top_k.advisory, per-doc cap, expansions): lookup reproduces its retrieval",
              ch.source_reply("which document", hist_after)["chunk_ids"] == adv_chunks, f"{ch.source_reply('which document', hist_after)['chunk_ids']} vs {adv_chunks}")

        for label, prev_user, prev_assistant in (("small talk", "wassup", "Hello! Ask me anything about our HR policies."),
                                                  ("a retrieval refusal", "what is the weather forecast for tomorrow", ch.no_match_response()),
                                                  ("a model 'couldn't find' reply", "how many leaves do interns get", "I couldn't find leave for interns in the HR policies.")):
            r = ch.source_reply("where is this from", msgs(("user", prev_user), ("assistant", prev_assistant)))
            check(f"[{name}] previous answer was {label}: templated 'didn't come from a policy document', no retrieval chunks",
                  r["text"] == "That answer didn't come from a policy document." and r["chunk_ids"] == [], str(r))

    # formatting
    docs = lambda *spec: [{"chunk_id": f"c{i}", "document_id": d, "document_title": t, "section_path": sp} for i, (d, t, sp) in enumerate(spec)]
    tt = get_app_config("source.templates.text")
    tv = get_app_config("source.templates.voice")
    one = docs(("d1", "Employee Handbook", "Employee Handbook > 5 Time > 5.3 Leave Policy"), ("d1", "Employee Handbook", "Employee Handbook > 5 Time > 5.3 Leave Policy"))
    check("text, one document, all chunks in one section: names the section", common.format_source_reply(one, tt, spoken=False) == "That came from the Employee Handbook, in the section 5.3 Leave Policy.", common.format_source_reply(one, tt, spoken=False))
    mixed = docs(("d1", "Employee Handbook", "Employee Handbook > 5.3 Leave Policy"), ("d1", "Employee Handbook", "Employee Handbook > 5.5 Comp Off Policy"))
    check("text, one document, chunks in different sections: document only", common.format_source_reply(mixed, tt, spoken=False) == "That came from the Employee Handbook.")
    check("voice, one document: document name only, no section path", common.format_source_reply(one, tv, spoken=True) == "That's from the employee handbook policy.", common.format_source_reply(one, tv, spoken=True))
    many = docs(("d1", "CALFUS CREW: REFERRAL BONUS PROGRAM - INDIA", "x"), ("d2", "Employee Handbook", "y"), ("d1", "CALFUS CREW: REFERRAL BONUS PROGRAM - INDIA", "x"),
                ("d3", "Domestic Travel Policy", "z"), ("d4", "India Relocation Policy", "w"))
    check("text, several documents: 'drew on', distinct, rank order, at most 3, ALL-CAPS title made readable",
          common.format_source_reply(many, tt, spoken=False) == "That drew on the Calfus Crew: Referral Bonus Program - India, Employee Handbook and Domestic Travel Policy.", common.format_source_reply(many, tt, spoken=False))
    check("voice, several documents: 'drew on' ... policies, names only", common.format_source_reply(many, tv, spoken=True).startswith("That drew on the referral bonus program, employee handbook and domestic travel") , common.format_source_reply(many, tv, spoken=True))
    check("no retrieved documents: not_found line", common.format_source_reply([], tt, spoken=False) == tt["not_found"])
    check("a broken template falls back to the seeded default instead of failing",
          common.format_source_reply(one, {**tt, "one_with_section": "{nope}"}, spoken=False) == "That came from the Employee Handbook, in the section 5.3 Leave Policy.")

    orig = get_app_config("source.templates.text")
    try:
        check("template edit via admin API accepted and used at once",
              client.patch("/api/admin/config/source.templates.text", json={"value": {**orig, "one": "Source: {title}"}}).status_code == 200
              and common.format_source_reply(mixed, get_app_config("source.templates.text"), spoken=False) == "Source: Employee Handbook")
        check("template with an unknown placeholder rejected (422)", client.patch("/api/admin/config/source.templates.text", json={"value": {**orig, "one": "{bogus}"}}).status_code == 422)
        check("template missing required keys rejected (422)", client.patch("/api/admin/config/source.templates.text", json={"value": {"one": "x"}}).status_code == 422)
        check("markers: blank entry rejected (422)", client.patch("/api/admin/config/source.markers", json={"value": ["ok", " "]}).status_code == 422)
    finally:
        client.patch("/api/admin/config/source.templates.text", json={"value": orig})


# ---- (d) per-turn counts through the real endpoints ----------------------------------------------------------------

def voice_turn(client, messages):
    headers = {"Authorization": f"Bearer {os.environ['THINK_ENDPOINT_SECRET']}"} if os.environ.get("THINK_ENDPOINT_SECRET") else {}
    r = client.post("/api/voice/v1/chat/completions", json={"messages": messages, "stream": True}, headers=headers)
    out = ""
    for line in r.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for choice in json.loads(line[6:])["choices"]:
                out += choice["delta"].get("content") or ""
    return out


GREET = [{"role": "user", "content": "-"}, {"role": "assistant", "content": "Hi, I'm the HR assistant."}]


def test_counts(client):
    print("\n=== (d) LLM / retrieval / embedding calls per turn ===")
    rows = []

    def run(label, fn):
        reset_counts()
        t = time.perf_counter()
        out = fn()
        rows.append((label, calls["llm"], calls["retrieve"], calls["embed"], (time.perf_counter() - t) * 1000))
        return out

    # a real referral answer to refer back to
    first = client.post("/api/chat", json={"query": REFERRAL_Q}).json()
    hist = msgs(("user", REFERRAL_Q), ("assistant", first["text"]))

    r = run("text source lookup", lambda: client.post("/api/chat", json={"query": "whats the source of this/", "history": hist}))
    j = r.json()
    check("text source lookup: type source_lookup, 0 LLM calls, exactly 1 retrieval and 1 embedding", j["type"] == "source_lookup" and calls["llm"] == 0 and calls["retrieve"] == 1 and calls["embed"] == 1,
          f"{j['type']} llm={calls['llm']} retr={calls['retrieve']} emb={calls['embed']}")
    row = last_log()
    check("logged as response_type 'source_lookup', source 'text', with the chunk_ids it drew on", row[0] == "source_lookup" and row[1] == "text" and bool(row[2]) and row[3] is not None, str(row))
    print(f"       text source reply: {j['text']}")

    v_msgs = GREET + [{"role": "user", "content": REFERRAL_Q}, {"role": "assistant", "content": first["text"]}, {"role": "user", "content": "where is this from"}]
    out = run("voice source lookup", lambda: voice_turn(client, v_msgs))
    check("voice source lookup: spoken templated reply, 0 LLM calls, 1 retrieval, 1 embedding", bool(out) and calls["llm"] == 0 and calls["retrieve"] == 1 and calls["embed"] == 1, f"llm={calls['llm']} retr={calls['retrieve']} emb={calls['embed']} {out!r}")
    row = last_log()
    check("voice source lookup logged as 'source_lookup' with source 'voice'", row[0] == "source_lookup" and row[1] == "voice" and bool(row[2]), str(row))
    check("voice source reply names documents only (no section path, no '>')", ">" not in out and "section" not in out.lower(), out)
    print(f"       voice source reply: {out}")

    r = run("source question, no history", lambda: client.post("/api/chat", json={"query": "whats the source of this"}))
    check("source question with no previous answer falls through to the normal pipeline (1 LLM, 1 retrieval)", r.json()["type"] in ("answer", "refusal_shortcut") and calls["retrieve"] == 1, f"{r.json()['type']} llm={calls['llm']}")

    s = client.get("/api/admin/usage/summary?days=1").json()
    check("usage summary breakdown includes source_lookup", s["by_response_type"].get("source_lookup", 0) >= 2, str(s["by_response_type"]))

    stm = msgs(("user", "x"), ("assistant", STATEMENT))
    for u in ("yes thank you", "yessir", "yes sir", "thank you so much", "thanks a lot", "cheers", "great thanks", "perfect thanks", "ok thanks", "got it thanks"):
        r = run(f"text {u!r}", lambda u=u: client.post("/api/chat", json={"query": u, "history": stm}))
        check(f"text {u!r} after a statement: small talk, 0 LLM and 0 retrieval calls", r.json()["type"] == "smalltalk" and calls["llm"] == 0 and calls["retrieve"] == 0, f"{r.json()['type']} {calls['llm']} {calls['retrieve']}")
    out = run("voice 'yes thank you'", lambda: voice_turn(client, GREET + [{"role": "user", "content": "x"}, {"role": "assistant", "content": STATEMENT}, {"role": "user", "content": "yes thank you"}]))
    check("voice 'yes thank you' after a statement: 0 LLM, 0 retrieval", calls["llm"] == 0 and calls["retrieve"] == 0 and bool(out), f"{calls['llm']} {calls['retrieve']}")
    out = run("voice 'yessir'", lambda: voice_turn(client, GREET + [{"role": "user", "content": "x"}, {"role": "assistant", "content": STATEMENT}, {"role": "user", "content": "yessir"}]))
    check("voice 'yessir' after a statement: 0 LLM, 0 retrieval", calls["llm"] == 0 and calls["retrieve"] == 0, f"{calls['llm']} {calls['retrieve']}")

    offer_hist = msgs(("user", REFERRAL_Q), ("assistant", OFFER))
    r = run("text 'yes' after an offer", lambda: client.post("/api/chat", json={"query": "yes", "history": offer_hist}))
    check("bare 'yes' after the bot's offer goes through retrieval + exactly 1 generation call", r.json()["type"] in ("answer", "refusal_shortcut") and calls["llm"] == 1 and calls["retrieve"] == 1, f"{r.json()['type']} llm={calls['llm']} retr={calls['retrieve']}")
    check("...and the model saw the offer in its history", any(OFFER[:40] in m["content"] for m in calls["messages"][-1][1:-1]))
    check("...logged as a normal contextualised turn, not small talk", last_log()[0] in ("answer", "refusal_shortcut") and last_log()[4] is True, str(last_log()))
    print(f"       'yes' after an offer -> {r.json().get('text')}")
    out = run("voice 'yes' after an offer", lambda: voice_turn(client, GREET + [{"role": "user", "content": REFERRAL_Q}, {"role": "assistant", "content": OFFER}, {"role": "user", "content": "yes"}]))
    check("voice: bare 'yes' after the agent's offer goes through retrieval + exactly 1 generation call", calls["llm"] == 1 and calls["retrieve"] == 1, f"llm={calls['llm']} retr={calls['retrieve']}")
    r = run("text 'yes' after a statement", lambda: client.post("/api/chat", json={"query": "yes", "history": stm}))
    check("bare 'yes' after a STATEMENT may still be an acknowledgement (0 LLM, 0 retrieval)", r.json()["type"] == "smalltalk" and calls["llm"] == 0 and calls["retrieve"] == 0)

    # everything else is unchanged versus the pre-change baseline
    r = run("normal answer", lambda: client.post("/api/chat", json={"query": "how many days of earned leave do I get"}))
    check("normal answer: 1 LLM, 1 retrieval, 1 embedding (unchanged)", r.json()["type"] == "answer" and (calls["llm"], calls["retrieve"], calls["embed"]) == (1, 1, 1), str((calls["llm"], calls["retrieve"], calls["embed"])))
    r = run("retrieval shortcut", lambda: client.post("/api/chat", json={"query": "what is the weather forecast for tomorrow"}))
    check("retrieval shortcut: 0 LLM, 1 retrieval (unchanged)", r.json()["type"] == "refusal_shortcut" and (calls["llm"], calls["retrieve"]) == (0, 1))
    r = run("small talk", lambda: client.post("/api/chat", json={"query": "wassup"}))
    check("small talk: 0 LLM, 0 retrieval (unchanged)", r.json()["type"] == "smalltalk" and (calls["llm"], calls["retrieve"]) == (0, 0))
    adv_hist = msgs(("user", "whats my leave policy"), ("assistant", "21 days."), ("user", TRIP_Q), ("assistant", "2 unpaid days."))
    r = run("advisory follow-up", lambda: client.post("/api/chat", json={"query": "is there anything else that i can do regarding that as i dont want LOP", "history": adv_hist}))
    check("advisory turn: 1 LLM, 1 retrieval, 1 batched embedding (unchanged)", r.json()["type"] == "answer" and (calls["llm"], calls["retrieve"], calls["embed"]) == (1, 1, 1), str((calls["llm"], calls["retrieve"], calls["embed"])))
    out = run("voice normal", lambda: voice_turn(client, GREET + [{"role": "user", "content": "how many days of earned leave do I get"}]))
    check("voice normal turn: 1 LLM, 1 retrieval, 1 embedding (unchanged)", (calls["llm"], calls["retrieve"], calls["embed"]) == (1, 1, 1), str((calls["llm"], calls["retrieve"], calls["embed"])))

    print("\n    turn                              LLM retr embed      ms")
    for label, llm, retr, emb, ms in rows:
        print(f"    {label:32s} {llm:4d} {retr:4d} {emb:5d} {ms:7.0f}")


# ---- (e) behaviour of the real model with the v4 rules -------------------------------------------------------------

# When the user has SAID they are an intern, "as an intern" / "you are an intern" is correct wording; only the
# unsupported claims count there.
BAD_CLAIM_STATED = re.compile(r"not entitled|are not eligible|typically|differs|different (?:leave )?polic", re.I)
BAD_CLAIM = re.compile(r"as an intern|you are an intern|not entitled|are not eligible|typically|differs|different (?:leave )?polic", re.I)


def test_model_behaviour():
    print("\n=== (e) Model behaviour with the v4 rules (3 runs each; manual grading covers the rest) ===")
    for name, ch in CHANNELS:
        a1 = ch.answer(INTERN_Q)["answer"]
        h = msgs(("user", INTERN_Q), ("assistant", a1))
        answers = [ch.answer("how many leaves do i get", history=h)["answer"] for _ in range(3)]
        check(f"[{name}] 'how many leaves do i get' after intern talk: standard-employee answer, no 'As an intern' framing or unsupported claim",
              all(not BAD_CLAIM.search(a) and re.search(r"21|twenty-one", a) for a in answers), " || ".join(a[:90] for a in answers))
        h2 = h + msgs(("user", "how many leaves do i get"), ("assistant", answers[0]))
        flow = [ch.answer("i am an intern", history=h2)["answer"] for _ in range(3)]
        check(f"[{name}] after 'i am an intern': 'couldn't find' wording, no 'not entitled' claim",
              all("couldn't find" in a.lower().replace("\u2019", "'") and not BAD_CLAIM_STATED.search(a) for a in flow), " || ".join(a[:90] for a in flow))
        trips = [ch.answer(TRIP_Q)["answer"] for _ in range(3)]
        check(f"[{name}] trip / 3 leaves left: states the working-week assumption and says 2 unpaid days, never 4",
              all(re.search(r"five-day|5-day|working week|working days", a, re.I) and re.search(r"\b(2|two)\b", a) and not re.search(r"\b(4|four)\b", a) for a in trips), " || ".join(a[:110] for a in trips))


def test_quality_set():
    print("\n=== (f) Cases in eval/quality_eval_set.json ===")
    cases = {c["id"]: c for c in json.loads((Path(__file__).parent / "quality_eval_set.json").read_text())}
    for cid in ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"):
        check(f"quality set has {cid}", cid in cases)
    check("f1 carries the intern conversation as history", len(cases["f1"]["history"]) == 2 and "interns" in cases["f1"]["history"][0]["content"])


def main():
    try:
        with TestClient(app) as client:
            install_counters()
            client.post("/api/chat", json={"query": "warm up: what is the leave policy"})
            test_prompt()
            test_smalltalk(client)
            test_source(client)
            test_counts(client)
            test_model_behaviour()
            test_quality_set()
    finally:
        close_pool()
    failed = [r for r in _results if not r[1]]
    print(f"\n=== {len(_results) - len(failed)} passed, {len(failed)} failed ===")
    for name, _ok, detail in failed:
        print(f"  - {name}: {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
