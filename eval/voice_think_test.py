#!/usr/bin/env python3
"""Simulates Deepgram calling the think endpoint (no audio, no tunnel).

    uvicorn api.main:app --port 8000   # in another terminal
    python3 -m eval.voice_think_test [--base-url http://localhost:8000]

Each call POSTs an OpenAI-style {"messages": [...]} body with the FULL
history, ending in the newest user utterance - the contract assumed in
api/voice.py - and reads the SSE stream the way Deepgram would. The
history is built from the endpoint's own previous replies, exactly as a
real session would accumulate it. Scenarios:
  1. fresh query
  2. a query that used to trigger a clarifying question is answered directly
     (the clarification layer is unwired: docs/applicability_layer_removed.md)
  3. a follow-up in the same conversation is answered too, and the endpoint
     never says the old clarification phrase
  4. non-streaming request, auth shape, turn logging
"""
import argparse
import json
import sys

import httpx

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
    ap.add_argument("--former-clarify-query", default="What is the domestic travel policy?")
    args = ap.parse_args()
    client = httpx.Client(base_url=args.base_url, timeout=120)

    print("1. Fresh query")
    conv = Conversation(client, args.secret)
    t = conv.say(args.fresh_query)
    check("got a spoken reply", bool(t.strip()))

    print("2. A query that used to trigger a clarifying question")
    conv = Conversation(client, args.secret)
    q = conv.say(args.former_clarify_query)
    check("answered directly, no clarification phrase",
          bool(q.strip()) and "quick check before i answer" not in q.lower() and "full-time employee or a contractor" not in q.lower(), q[:80])

    print("3. Follow-up in the same conversation")
    later = conv.say("and what about the referral bonus?")
    check("answered, never the old clarification phrase",
          bool(later.strip()) and "quick check before i answer" not in later.lower(), later[:80])

    print("4. Non-streaming request + auth")
    r = client.post("/api/voice/v1/chat/completions", json={"messages": conv.messages[:2] + [
        {"role": "user", "content": args.fresh_query}], "stream": False},
        headers=conv.headers)
    check("stream:false returns chat.completion JSON", r.status_code == 200 and r.json()["object"] == "chat.completion")
    r = client.post("/api/voice/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "hi"}]},
                    headers=conv.headers)
    check("no user message -> 400", r.status_code == 400)

    rows = client.get("/api/admin/usage/recent", params={"limit": 20}).json()
    check("turns logged with source=voice", any(x["source"] == "voice" for x in rows))
    check("no new turn is a clarification", all(x["response_type"] != "clarification" for x in rows[:6]))

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
