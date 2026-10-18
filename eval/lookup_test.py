#!/usr/bin/env python3
"""Tests for the deterministic source / section / page lookups ("what's the source", "what section is that in",
"what page number") on both channels.

    python3 -m eval.lookup_test

Needs the same live services as eval/run_regression.py. Real LLM / retrieval / embedding / DB / Redis calls are made but COUNTED
(eval/turn_calls.py wraps them). Settings it changes are restored at the end (even on failure).

  (a) detection: 25 positive phrasings x text and voice - each 0 LLM, exactly 1 retrieval, 1 embedding, and logged with the
      right response_type; 11 false-positive guards that must reach the normal pipeline
  (b) chain: answer -> source -> page -> section -> source phrasing; every lookup recomputes against the ORIGINAL answered
      question (chunk_ids equal to the original answer's)
  (c) walk-back past small talk and catalog replies; replies when nothing came from a document
  (d) reply wording for 10 typical questions (printed), section-name cleaning, no reply ends in a question or an offer
  (e) admin validation of the new config, usage breakdown, voice streaming format
  (f) per-turn LLM / retrieval / embedding / DB / Redis counts of every other turn type equal eval/turn_calls_baseline.json
"""
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi.testclient import TestClient

import channels.text as text_ch
import channels.voice as voice_ch
from api.main import app
from channels import common
from config.runtime import get_app_config
from eval import turn_calls as tc
from vectorstore.db import close_pool, get_pool

REFERRAL_Q = "will I get paid extra for recommending a friend for a job here"
LEAVE_Q = "how many days of earned leave do I get"

_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if not condition and detail else ""))
    return bool(condition)


def last_log():
    with get_pool().connection() as conn:
        return conn.execute("SELECT response_type, source, chunk_ids_used FROM query_logs ORDER BY id DESC LIMIT 1").fetchone()


def text_turn(client, query, history):
    tc.reset()
    r = client.post("/api/chat", json={"query": query, "history": history}).json()
    return r["type"], r["text"], dict(tc.calls)


def voice_say(client, query, history):
    tc.reset()
    out = tc.voice_turn(client, tc.GREET + history + [{"role": "user", "content": query}])
    return out, dict(tc.calls)


def original_chunk_ids(question, channel):
    module = text_ch if channel == "text" else voice_ch
    plan = common.plan_turn(question, [], channel)
    return [r["chunk_id"] for r in module._retrieve(plan)]


POSITIVES = [  # (phrasing, expected intent)
    ("What's the source?", "source"), ("What page number?", "page"),
    ("What section of the HR policies did you get it from?", "section"), ("And where are you sourcing this from?", "source"),
    ("where did you get that", "source"), ("where are you getting this information", "source"),
    ("which policy is this from", "source"), ("what section is that in", "section"), ("which part of the handbook", "section"),
    ("how do you know that", "source"), ("how do you know", "source"), ("what is that based on", "source"),
    ("can you cite that", "source"), ("what's your source", "source"), ("which document", "source"),
    ("where is this written", "source"), ("where is that mentioned", "source"), ("what page is that on", "page"),
    ("which page", "page"), ("exact page", "page"), ("which section", "section"), ("what heading", "section"),
    ("okay where did u get that", "source"), ("where are you pulling this from", "source"),
    ("which section did you find that in", "section"), ("so what page number please", "page"),
]
GUARDS = ["where can I find the leave form", "what is the source of truth for payroll", "where do I submit my leave request",
          "how does the company source candidates", "what section of the policy covers sick leave", "which policy covers notice period",
          "what do you know about sourcing", "what part of the policy covers it", "what page does the leave policy start on",
          "where did you find the leave form", "how do you know my leave balance"]
LOOKUP_TYPES = {"source_lookup", "section_lookup", "page_lookup"}
INTENT_TYPE = {"source": "source_lookup", "section": "section_lookup", "page": "page_lookup"}
TYPICAL = ["how many days of earned leave do I get", "what is the notice period", "will I get paid extra for recommending a friend for a job here",
           "can I work from home", "what is comp-off", "what are the travel limits for domestic travel", "what is a PIP", "what is the POSH policy",
           "remote connectivity allowance", "certification reimbursement"]


def ends_badly(text):
    t = text.strip()
    return t.endswith("?") or any(m in t.lower() for m in ("would you like", "want me to", "i can also", "let me know", "feel free"))


def test_detection(client):
    print("\n=== (a) detection, 0 LLM / 1 retrieval / 1 embedding ===")
    answered = [{"role": "user", "content": REFERRAL_Q}, {"role": "assistant", "content": "Yes. A referral bonus is paid after the person you referred completes probation."}]
    bad = []
    for phrase, intent in POSITIVES:
        got = common.lookup_intent(phrase)
        if got != intent:
            bad.append((phrase, got, intent))
    check(f"{len(POSITIVES)} positive phrasings detect the expected intent (source / section / page)", not bad, str(bad))
    for channel in ("text", "voice"):
        failures = []
        for phrase, intent in POSITIVES:
            if channel == "text":
                typ, text, c = text_turn(client, phrase, answered)
            else:
                text, c = voice_say(client, phrase, answered)
                typ = last_log()[0]
            ok = typ == INTENT_TYPE[intent] and (c["llm"], c["retrieve"], c["embed"]) == (0, 1, 1) and bool(text) and not ends_badly(text)
            if not ok:
                failures.append((phrase, typ, c, text))
        check(f"[{channel}] every positive: right response type, 0 LLM, 1 retrieval, 1 embedding, reply not a question or offer", not failures, str(failures[:3]))
    for phrase in GUARDS:
        check(f"guard {phrase!r}: not a lookup (normal pipeline)", common.lookup_intent(phrase) is None)
    for phrase in GUARDS[:7]:
        typ, text, c = text_turn(client, phrase, answered)
        check(f"guard through the endpoint {phrase!r}: reaches the normal pipeline", typ not in LOOKUP_TYPES | {"smalltalk", "catalog"}, f"{typ}: {text}")
    typ, _text, c = text_turn(client, "What section is that in?", [])
    check("a section question with NO earlier answer falls through to the normal pipeline", typ not in LOOKUP_TYPES, typ)
    long_q = "what section is that in " + "really " * 12
    check("an utterance over source.max_words (14) is never a lookup", common.lookup_intent(long_q) is None)


def test_chain(client):
    print("\n=== (b) chain: answer -> source -> page -> section -> source phrasing ===")
    for channel in ("text", "voice"):
        original = original_chunk_ids(LEAVE_Q, channel)
        if channel == "text":
            first = client.post("/api/chat", json={"query": LEAVE_Q}).json()["text"]
        else:
            first, _ = voice_say(client, LEAVE_Q, [])
        hist = [{"role": "user", "content": LEAVE_Q}, {"role": "assistant", "content": first}]
        answer_row = last_log()
        check(f"[{channel}] the original answer's logged chunks equal a fresh retrieval of the question", list(answer_row[2]) == original, f"{answer_row[2]} vs {original}")
        for q in ("What's the source?", "What page number?", "What section of the HR policies did you get it from?", "And where are you sourcing this from?"):
            if channel == "text":
                typ, text, c = text_turn(client, q, hist)
            else:
                text, c = voice_say(client, q, hist)
                typ = last_log()[0]
            row = last_log()
            check(f"[{channel}] {q!r} -> {typ}, 0 LLM, 1 retrieval, 1 embedding, chunks == the ORIGINAL answer's",
                  typ in LOOKUP_TYPES and (c["llm"], c["retrieve"], c["embed"]) == (0, 1, 1) and list(row[2]) == original, f"{typ} {c} {row}")
            print(f"       {channel}: {q!r} -> {text}")
            hist += [{"role": "user", "content": q}, {"role": "assistant", "content": text}]


def test_walkback(client):
    print("\n=== (c) walk-back and no-document replies ===")
    answered = [{"role": "user", "content": REFERRAL_Q}, {"role": "assistant", "content": "Yes. A referral bonus is paid after the person you referred completes probation."}]
    original = original_chunk_ids(REFERRAL_Q, "text")
    catalog_turn = [{"role": "user", "content": "what documents do you have"}, {"role": "assistant", "content": "I have 14 documents."}]
    small_turn = [{"role": "user", "content": "thanks a lot"}, {"role": "assistant", "content": "You're welcome!"}]
    for label, extra in (("a catalog reply", catalog_turn), ("small talk", small_turn), ("small talk and a catalog reply", small_turn + catalog_turn)):
        typ, text, c = text_turn(client, "what section is that in", answered + extra)
        check(f"lookup after {label} walks back to the answered turn (chunks == its retrieval)", typ == "section_lookup" and list(last_log()[2]) == original, f"{typ} {text}")
    for q in ("where is this from", "what section is that in", "what page number"):
        typ, text, c = text_turn(client, q, [{"role": "user", "content": "wassup"}, {"role": "assistant", "content": "Hello! Ask me anything about our HR policies."}])
        check(f"{q!r} when the only earlier reply was small talk: 'didn't come from a policy document', 0 retrieval", typ in LOOKUP_TYPES and "didn't come from a policy document" in text and c["retrieve"] == 0, f"{typ} {text} {c}")
    typ, text, c = text_turn(client, "what section is that in", catalog_turn)
    check("only a catalog reply before: 'didn't come from a policy document'", "didn't come from a policy document" in text and c["retrieve"] == 0, text)
    refusal_hist = [{"role": "user", "content": "what is the weather forecast for tomorrow"}, {"role": "assistant", "content": text_ch.no_match_response()}]
    for q in ("what section is that in", "what page number"):
        typ, text, c = text_turn(client, q, refusal_hist)
        check(f"{q!r} after a refusal: 'didn't come from a policy document', 0 retrieval", "didn't come from a policy document" in text and c["retrieve"] == 0, text)
    typ, text, c = text_turn(client, "what section is that in", [{"role": "user", "content": "how many leaves do interns get"}, {"role": "assistant", "content": "I couldn't find leave for interns in the HR policies."}])
    check("after a model 'I couldn't find' reply: 'didn't come from a policy document'", "didn't come from a policy document" in text and c["retrieve"] == 0, text)


def test_wording(client):
    print("\n=== (d) reply wording: 10 typical questions ===")
    tt = get_app_config("lookup.templates.text")
    unnumbered = lambda s: not re.search(r"\b\d+\.\d+\b", s) and ">" not in s
    ok_all = True
    for q in TYPICAL:
        for channel, module, spoken in (("text", text_ch, False), ("voice", voice_ch, True)):
            plan = common.plan_turn(q, [], channel)
            results = module._retrieve(plan)
            sec = common.format_location_reply(results, get_app_config(f"lookup.templates.{channel}"), spoken, "section")
            pag = common.format_location_reply(results, get_app_config(f"lookup.templates.{channel}"), spoken, "page")
            print(f"  [{channel}] {q}\n        section: {sec}\n        page:    {pag}")
            ok_all &= unnumbered(sec) and unnumbered(pag) and not ends_badly(sec) and not ends_badly(pag)
    check("no reply has numbering or a path separator, none ends in a question or an offer", ok_all)

    clean = common.clean_section_name
    cases = [("3.0 Policy Statement", "IT Security Policies", False, None), ("5.2 Holidays", "Employee Handbook", False, "Holidays"),
             ("8.1 Voluntary Resignation:", "Employee Handbook", False, "Voluntary Resignation"), ("1.2 ELIGIBILITY", "X Policy", False, "Eligibility"),
             ("1.2 ELIGIBILITY", "X Policy", True, "eligibility"), ("4.2. Risks", "T", False, "Risks"), ("1.6 WHO CAN COMPLAIN?", "P", True, "who can complain"),
             ("1.5 APPLICATION & APPROVAL PROCESS", "P", False, "Application and Approval Process"), ("Calfus POSH Policy", "Calfus POSH Policy", False, None),
             ("Table of Contents", "Employee Handbook", False, None), ("for India", "Employee Handbook", False, None),
             ("Item | Amount | Limit", "Travel Policy", False, None), ("Sr. No 4", "Travel Policy", False, None), ("", "T", False, None), (None, "T", False, None)]
    bad = [(raw, got) for raw, title, spoken, want in cases if (got := clean(raw, title, spoken)) != want]
    check("section-name cleaning: numbering, punctuation, caps, '&', table rows, title-equal and generic headings", not bad, str(bad))
    path = "IT Security Policies > 16. Teleworking Policy > 4.0 Policy Statement"
    check("a generic leaf gives way to its parent heading", common._best_section(path, "IT Security Policies", False) == "Teleworking Policy")
    docs = lambda *spec: [{"chunk_id": f"c{i}", "document_id": d, "document_title": t, "section_path": sp} for i, (d, t, sp) in enumerate(spec)]
    hb = "Employee Handbook"
    three = docs(("d1", hb, f"{hb} > for India > 5 Time > 5.3 Leave Policy"), ("d1", hb, f"{hb} > for India > 5 Time > 5.5 Comp Off Policy"),
                 ("d1", hb, f"{hb} > 5 Time > 5.1 Attendance Policy"), ("d1", hb, f"{hb} > 6 Employee Referral Policy"), ("d1", hb, f"{hb} > 5 Time > 5.3 Leave Policy"))
    tv = get_app_config("lookup.templates.voice")
    check("text: up to 3 distinct sections joined by 'and' in rank order", common.format_location_reply(three, tt, False, "section") == "That's in the Leave Policy, Comp Off Policy and Attendance Policy section of the Employee Handbook.", common.format_location_reply(three, tt, False, "section"))
    check("voice: at most 2 sections, '... policy' suffix", common.format_location_reply(three, tv, True, "section") == "That's in the Leave Policy and Comp Off Policy section of the employee handbook policy.", common.format_location_reply(three, tv, True, "section"))
    title_only = docs(("d1", "Calfus POSH Policy", "Calfus POSH Policy"), ("d1", "Calfus POSH Policy", "Calfus POSH Policy"))
    check("section equal to the title: document only (section)", common.format_location_reply(title_only, tt, False, "section") == "That's from the Calfus POSH Policy, but I can't point to a specific section.", common.format_location_reply(title_only, tt, False, "section"))
    check("page, no clean section: title-only form", common.format_location_reply(title_only, tt, False, "page") == "I don't have page numbers for that document, but it came from the Calfus POSH Policy.", common.format_location_reply(title_only, tt, False, "page"))
    check("page, with a section: says there are no page numbers and names the section", common.format_location_reply(three, tt, False, "page").startswith("I don't have page numbers for that document, but it's in the Leave Policy, Comp Off Policy and Attendance Policy section of the Employee Handbook."), common.format_location_reply(three, tt, False, "page"))
    two_docs = docs(("d1", hb, f"{hb} > 5.3 Leave Policy"), ("d2", "Domestic Travel Policy", "Domestic Travel Policy > 1.2 ELIGIBILITY"))
    check("several documents: one 'also' sentence per further document", common.format_location_reply(two_docs, tt, False, "section") == "That's in the Leave Policy section of the Employee Handbook. It's also in the Eligibility section of the Domestic Travel Policy.", common.format_location_reply(two_docs, tt, False, "section"))
    check("no retrieved documents: not_found line", common.format_location_reply([], tt, False, "section") == tt["not_found"])
    check("a broken template falls back to the seeded default instead of failing", common.format_location_reply(title_only, {**tt, "section_title_only": "{nope}"}, False, "section") == "That's from the Calfus POSH Policy, but I can't point to a specific section.")


def test_admin_and_stream(client):
    print("\n=== (e) admin validation, usage breakdown, voice streaming ===")
    keys = ["lookup.templates.text", "lookup.skip_section_patterns", "section.markers", "lookup.max_sections.text"]
    saved = {k: get_app_config(k) for k in keys}
    try:
        good = {**saved["lookup.templates.text"], "page": "No page numbers here, but it is in the {section} section of the {title}."}
        check("template edit accepted via admin PATCH and used at once", client.patch("/api/admin/config/lookup.templates.text", json={"value": good}).status_code == 200
              and common.format_location_reply(common_docs(), get_app_config("lookup.templates.text"), False, "page").startswith("No page numbers here"))
        check("template with an unknown placeholder rejected (422)", client.patch("/api/admin/config/lookup.templates.text", json={"value": {**good, "page": "{bogus}"}}).status_code == 422)
        check("template ending in a question rejected (422)", client.patch("/api/admin/config/lookup.templates.text", json={"value": {**good, "section": "That's in {section} of {title}, ok?"}}).status_code == 422)
        check("template containing an offer rejected (422)", client.patch("/api/admin/config/lookup.templates.text", json={"value": {**good, "none": "Not from a document. Would you like to ask something else."}}).status_code == 422)
        check("template missing a key rejected (422)", client.patch("/api/admin/config/lookup.templates.text", json={"value": {"section": "x"}}).status_code == 422)
        check("invalid regular expression rejected (422)", client.patch("/api/admin/config/lookup.skip_section_patterns", json={"value": ["(unclosed"]}).status_code == 422)
        check("blank marker rejected (422)", client.patch("/api/admin/config/section.markers", json={"value": ["ok", " "]}).status_code == 422)
        check("max_sections must be a positive integer (422)", client.patch("/api/admin/config/lookup.max_sections.text", json={"value": 0}).status_code == 422)
        check("a new section marker is used at once", client.patch("/api/admin/config/section.markers", json={"value": saved["section.markers"] + ["which bit is that"]}).status_code == 200
              and common.lookup_intent("which bit is that") == "section")
    finally:
        for k, v in saved.items():
            client.patch(f"/api/admin/config/{k}", json={"value": v})
        common.clear_config_memo()
    check("the markers edit was restored", common.lookup_intent("which bit is that") is None)

    s = client.get("/api/admin/usage/summary?days=1").json()["by_response_type"]
    check("usage breakdown includes source_lookup, section_lookup and page_lookup", all(s.get(k, 0) >= 1 for k in LOOKUP_TYPES), str(s))

    msgs = tc.GREET + [{"role": "user", "content": REFERRAL_Q}, {"role": "assistant", "content": "A referral bonus is paid after probation."}, {"role": "user", "content": "what section is that in"}]
    headers = {"Authorization": f"Bearer {__import__('os').environ['THINK_ENDPOINT_SECRET']}"} if __import__("os").environ.get("THINK_ENDPOINT_SECRET") else {}
    r = client.post("/api/voice/v1/chat/completions", json={"messages": msgs, "stream": True}, headers=headers)
    lines = [l for l in r.text.splitlines() if l.startswith("data: ")]
    chunks = [json.loads(l[6:]) for l in lines if l != "data: [DONE]"]
    check("voice lookup streams OpenAI-compatible chunks (role, content, finish_reason stop, [DONE])",
          r.headers["content-type"].startswith("text/event-stream") and lines[-1] == "data: [DONE]" and all(c["object"] == "chat.completion.chunk" for c in chunks)
          and any(c["choices"][0]["delta"].get("content") for c in chunks) and chunks[-1]["choices"][0]["finish_reason"] == "stop", r.text[:300])


def common_docs():
    return [{"chunk_id": "c0", "document_id": "d1", "document_title": "Employee Handbook", "section_path": "Employee Handbook > 5.3 Leave Policy"}]


def test_other_turns_unchanged(client):
    print("\n=== (f) every other turn type: LLM / retrieval / embedding / DB / Redis counts equal the baseline ===")
    baseline = json.loads((Path(__file__).parent / "turn_calls_baseline.json").read_text())
    table = tc.run_all(client)
    tc.print_table(table)
    lookups = {"text source lookup", "voice source lookup"}  # the lookup turn types themselves: may only get cheaper
    diffs = {label: (baseline[label], table[label]) for label in baseline if label in table and label not in lookups and baseline[label] != table[label]}
    check(f"all {len(baseline) - len(lookups)} other baseline turn types have identical counts", not diffs and set(baseline) <= set(table), str(diffs))
    worse = {label: (baseline[label], table[label]) for label in lookups
             if any(table[label][k] != baseline[label][k] for k in ("llm", "retrieve", "embed")) or any(table[label][k] > baseline[label][k] for k in ("db", "redis"))}
    check("source lookups: 0 LLM, 1 retrieval, 1 embedding, and no more DB / Redis calls than the baseline", not worse, str(worse))
    for label in lookups:
        if table[label] != baseline[label]:
            print(f"       note: {label}: baseline {baseline[label]} -> now {table[label]}")


def test_quality_set():
    data = json.loads((Path(__file__).parent / "quality_eval_set.json").read_text())
    ids = {c["id"] for c in data}
    check("transcript turns are in eval/quality_eval_set.json", {"lk_page", "lk_section", "lk_source2", "lk_source"} <= ids)


def main():
    with TestClient(app) as client:
        tc.install()
        client.post("/api/chat", json={"query": "warm up: what is the leave policy"})
        try:
            test_detection(client)
            test_chain(client)
            test_walkback(client)
            test_wording(client)
            test_admin_and_stream(client)
            test_other_turns_unchanged(client)
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
