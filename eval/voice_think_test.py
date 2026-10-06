#!/usr/bin/env python3
"""Simulates Deepgram calling the think endpoint (no audio, no tunnel).

    ./run_api.sh   # in another terminal
    python3 -m eval.voice_think_test [--base-url http://localhost:8000]

Each call POSTs an OpenAI-style {"messages": [...]} body with the FULL
history, ending in the newest user utterance - the contract assumed in
api/voice.py - and reads the SSE stream the way Deepgram would. The
history is built from the endpoint's own previous replies, exactly as a
real session would accumulate it. Scenarios:
  1. fresh query
  2. clarification-triggering query + resolving reply
  3. unusable reply -> same question re-asked, reworded
  4. later query in the same conversation does NOT re-ask a resolved attribute
Needs a query that triggers a clarification on the loaded corpus
(--clarify-query); with default country=India that means an
employment_type-gated topic.
"""
import argparse
import json
import sys

import httpx

from api.voice import CLARIFY_LEAD_IN, parse_clarification

failures = 0


def check(label, ok, detail=""):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    failures += 0 if ok else 1


class Conversation:
    def __init__(self, client, secret):
        self.c, self.headers = client, ({"authorization": f"Bearer {secret}"} if secret else {})
        self.messages = [{"role": "system", "content": "ignored: Deepgram's own prompt"},
                         {"role": "assistant", "content": "Hi, I'm the HR assistant."}]

    def say(self, utterance: str) -> str:
        self.messages.append({"role": "user", "content": utterance})
        text, finished, done = "", False, False
        with self.c.stream("POST", "/api/voice/v1/chat/completions", headers=self.headers,
                           json={"model": "gpt-4o-mini", "messages": self.messages, "stream": True}) as r:
            assert r.status_code == 200, r.read()
            for line in r.iter_lines():
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    done = True
                    continue
                choice = json.loads(line[6:])["choices"][0]
                text += choice["delta"].get("content") or ""
                finished = finished or choice["finish_reason"] == "stop"
        assert finished and done, "stream did not end with finish_reason=stop + [DONE]"
        self.messages.append({"role": "assistant", "content": text})
        print(f"    user: {utterance}\n    agent: {text}")
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--secret", default=None, help="THINK_ENDPOINT_SECRET if the server enforces one")
    ap.add_argument("--fresh-query", default="How many days of leave do I get?")
    ap.add_argument("--clarify-query", default="What is the domestic travel policy?")
    ap.add_argument("--replies", default='{"employment_type": "I\'m a full-time employee", "country": "I work in India"}')
    args = ap.parse_args()
    client = httpx.Client(base_url=args.base_url, timeout=120)
    replies = json.loads(args.replies)

    print("0. Marker round-trip (offline)")
    for tag in ("country", "employment_type"):
        for i in (0, 1):
            from api.voice import clarifying_question
            check(f"{tag} variant {i} parses back", parse_clarification(clarifying_question(tag, i)) == (tag, i))
    check("normal answer is not a clarification", parse_clarification("You get twenty days of leave.") is None)

    print("1. Fresh query")
    conv = Conversation(client, args.secret)
    t = conv.say(args.fresh_query)
    check("got a spoken reply", bool(t.strip()))

    print("2. Clarification-triggering query + resolving reply")
    conv = Conversation(client, args.secret)
    q = conv.say(args.clarify_query)
    parsed = parse_clarification(q)
    if not parsed:
        print("  [SKIP] query triggered no clarification on this corpus; pass --clarify-query"); sys.exit(1)
    tag, _ = parsed
    check("starts with the fixed lead-in", q.startswith(CLARIFY_LEAD_IN), f"tag={tag}")

    print("3. Unusable reply -> re-ask")
    again = conv.say("banana purple forty two")
    p2 = parse_clarification(again)
    check("same attribute asked again", p2 is not None and p2[0] == tag)
    check("reworded, not identical", again != q)

    ans = conv.say(replies.get(tag, "I'm not sure how to say"))
    check("resolving reply gets a real answer (no new lead-in for same tag)",
          not (parse_clarification(ans) and parse_clarification(ans)[0] == tag), ans[:80])

    print("4. Later clarification-worthy query, same conversation")
    later = conv.say(args.clarify_query)
    pl = parse_clarification(later)
    check("does not re-ask the resolved attribute", not (pl and pl[0] == tag), later[:80])

    print("5. Non-streaming request + auth")
    r = client.post("/api/voice/v1/chat/completions", json={"messages": conv.messages[:2] + [
        {"role": "user", "content": args.fresh_query}], "stream": False},
        headers=conv.headers)
    check("stream:false returns chat.completion JSON", r.status_code == 200 and r.json()["object"] == "chat.completion")
    r = client.post("/api/voice/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "hi"}]},
                    headers=conv.headers)
    check("no user message -> 400", r.status_code == 400)

    rows = client.get("/api/admin/usage/recent", params={"limit": 20}).json()
    check("turns logged with source=voice", any(x["source"] == "voice" for x in rows))

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
