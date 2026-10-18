#!/usr/bin/env python3
"""Counts LLM, retrieval, embedding, DB and Redis calls per turn type through the real endpoints.

    python3 -m eval.turn_calls [--json out.json]

Each scenario is run once to warm the caches / in-process config copies, then measured 3 times and the
element-wise minimum kept, so the numbers show the steady state of a turn (a config key's once-per-30s refresh
landing inside one run is not counted).
Used by eval/catalog_test.py to prove that every turn type except a catalog turn makes exactly the calls it
made before the catalog intent existed (see CALLS_BASELINE in that file).
"""
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import psycopg
import redis
from fastapi.testclient import TestClient

import channels.text as text_ch
import channels.voice as voice_ch
import generation.generate as gen_mod
import vectorstore.retrieve as retrieve_mod
from api.main import app

calls = {"llm": 0, "retrieve": 0, "embed": 0, "db": 0, "redis": 0}
_installed = False


def install():
    global _installed
    if _installed:
        return
    _installed = True
    client = gen_mod.get_client()
    original = client.chat.completions.create

    def counted(*a, **k):
        calls["llm"] += 1
        return original(*a, **k)

    client.chat.completions.create = counted

    from eval.llm_count import wrap_with_options

    def _scoped(create):
        def scoped_counted(*a, **k):
            calls["llm"] += 1
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
    original_execute = psycopg.Cursor.execute

    def counted_db(self, *a, **k):
        calls["db"] += 1
        return original_execute(self, *a, **k)

    psycopg.Cursor.execute = counted_db
    original_cmd = redis.Redis.execute_command

    def counted_redis(self, *a, **k):
        calls["redis"] += 1
        return original_cmd(self, *a, **k)

    redis.Redis.execute_command = counted_redis


def reset():
    calls.update(llm=0, retrieve=0, embed=0, db=0, redis=0)
    retrieve_mod.embed_query.cache_clear()
    retrieve_mod.embed_queries.cache_clear()


GREET = [{"role": "user", "content": "-"}, {"role": "assistant", "content": "Hi, I'm the HR assistant."}]
REFERRAL_Q = "will I get paid extra for recommending a friend for a job here"
OFFER = "Yes, a referral bonus is paid after probation. Would you like the eligibility details?"
STATEMENT = "You get 21 days of earned leave a year."
ADV_HIST = [{"role": "user", "content": "whats my leave policy"}, {"role": "assistant", "content": "21 days."},
            {"role": "user", "content": "i need to go to my hometown for a week but i only have 3 leaves left"},
            {"role": "assistant", "content": "2 unpaid days."}]


def voice_turn(client, messages):
    headers = {"Authorization": f"Bearer {os.environ['THINK_ENDPOINT_SECRET']}"} if os.environ.get("THINK_ENDPOINT_SECRET") else {}
    r = client.post("/api/voice/v1/chat/completions", json={"messages": messages, "stream": True}, headers=headers)
    out = ""
    for line in r.text.splitlines():
        if line.startswith("data: ") and line != "data: [DONE]":
            for choice in json.loads(line[6:])["choices"]:
                out += choice["delta"].get("content") or ""
    return out


def scenarios(client):
    chat = lambda q, history=None: (lambda: client.post("/api/chat", json={"query": q, "history": history or []}).json())
    say = lambda q, hist=(): (lambda: voice_turn(client, GREET + list(hist) + [{"role": "user", "content": q}]))
    h = lambda *pairs: [{"role": r, "content": c} for r, c in pairs]
    ref_hist = h(("user", REFERRAL_Q), ("assistant", "A referral bonus is paid after probation."))
    return {
        "text normal answer": chat("how many days of earned leave do I get"),
        "text question containing 'your' (what is your leave policy)": chat("what is your leave policy"),
        "text advisory follow-up": chat("is there anything else that i can do regarding that as i dont want LOP", ADV_HIST),
        "text retrieval shortcut": chat("what is the weather forecast for tomorrow"),
        "text small talk (wassup)": chat("wassup"),
        "text small talk (what can you do)": chat("what can you do"),
        "text thanks after statement": chat("yes thank you", h(("user", "x"), ("assistant", STATEMENT))),
        "text source lookup": chat("whats the source of this/", ref_hist),
        "text 'yes' after an offer": chat("yes", h(("user", REFERRAL_Q), ("assistant", OFFER))),
        "text fresh short question after unrelated turns": chat("Tell me how change of work location", h(("user", "My dog died."), ("assistant", "I'm sorry."), ("user", "Yeah. I want some time off."), ("assistant", "OK."))),
        "text follow-up with marker": chat("can I carry it forward", h(("user", "how many days of earned leave do I get"), ("assistant", "21 days."))),
        "voice short follow-up": say("for interns?", h(("user", "how many days of earned leave do I get"), ("assistant", "Twenty-one days."))),
        "voice normal turn": say("how many days of earned leave do I get"),
        "voice small talk (wassup)": say("wassup"),
        "voice source lookup": say("where is this from", h(("user", REFERRAL_Q), ("assistant", "A referral bonus is paid after probation."))),
    }


def run_all(client) -> dict:
    out = {}
    for label, fn in scenarios(client).items():
        fn()  # warm-up
        runs = []
        for _ in range(3):
            reset()
            fn()
            runs.append(dict(calls))
        # element-wise minimum: a 30-second cache-aside refresh landing inside one run adds a DB read and some
        # Redis commands to THAT run only; the steady state of a turn is the minimum
        out[label] = {k: min(r[k] for r in runs) for k in calls}
    return out


def print_table(table: dict) -> None:
    print(f"{'turn type':64s} {'LLM':>4} {'retr':>5} {'emb':>4} {'DB':>4} {'Redis':>6}")
    for label, c in table.items():
        print(f"{label:64s} {c['llm']:4d} {c['retrieve']:5d} {c['embed']:4d} {c['db']:4d} {c['redis']:6d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json")
    args = ap.parse_args()
    with TestClient(app) as client:
        install()
        client.post("/api/chat", json={"query": "warm up: what is the leave policy"})
        table = run_all(client)
    print_table(table)
    if args.json:
        Path(args.json).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
