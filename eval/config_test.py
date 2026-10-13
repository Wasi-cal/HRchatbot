#!/usr/bin/env python3
"""Tests for runtime config, Redis cache-aside, Redis-outage degradation and
pooled concurrency. Needs the API running (./run_api.sh), Postgres and Redis up.

    python3 -m eval.config_test [--base-url http://localhost:8000]

Takes ~40s (waits out the 30s cache TTL once). Restores every value it changes.
"""
import argparse
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from cache import redis_cache
from cache.redis_cache import MISS
from config.runtime import CACHE_TTL_SECONDS, DEFAULT_APP_CONFIG, get_app_config
from vectorstore.db import get_pool
from vectorstore.retrieve import DEFAULT_CANDIDATE_N, DEFAULT_RRF_K, DEFAULT_TOP_K, retrieve

failures = 0
FULL_USER = {"country": "India", "employment_type": "full-time employee"}


def check(label, ok, detail=""):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    failures += 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base_url, timeout=120)
    query = "how much time off do I get when I have a baby"

    print("1. Seeded values match the code constants / SQL")
    code = {"retrieval.top_k": DEFAULT_TOP_K, "retrieval.candidate_n": DEFAULT_CANDIDATE_N, "retrieval.rrf_k": DEFAULT_RRF_K}
    for key in code:
        seed = DEFAULT_APP_CONFIG[key][0]
        check(f"{key}: seed {seed} == retrieve.py constant {code[key]}", seed == code[key])
    rows = {r["key"]: r["value"] for r in c.get("/api/admin/config").json()["app_config (table: app_config)"]}
    check("DB rows currently equal seeds", all(rows.get(k) == DEFAULT_APP_CONFIG[k][0] for k in code), str(rows))
    check("per-channel top_k rows are seeded with the global's effective value",
          rows.get("retrieval.top_k.text") == rows["retrieval.top_k"] and rows.get("retrieval.top_k.voice") == rows["retrieval.top_k"], str(rows))

    print("2. PATCH is reflected in GET, retrieve() and the chat pipeline")
    r = c.patch("/api/admin/config/retrieval.top_k", json={"value": 1})
    check("PATCH top_k=1 -> 200", r.status_code == 200, r.text[:100])
    rows = {r["key"]: r["value"] for r in c.get("/api/admin/config").json()["app_config (table: app_config)"]}
    check("GET shows 1", rows["retrieval.top_k"] == 1)
    check("retrieve() (this process, via cache) returns exactly 1 chunk", len(retrieve(query)) == 1)
    check("explicit top_k argument still wins", len(retrieve(query, top_k=2)) == 2)
    # /api/chat is the "text" channel, whose own key (seeded = global) overrides the global one.
    c.patch("/api/admin/config/retrieval.top_k.text", json={"value": 1})
    r = c.post("/api/chat", json={"query": query, "user_attributes": FULL_USER})
    time.sleep(0.5)
    recent = c.get("/api/admin/usage/recent", params={"limit": 1}).json()[0]
    check("chat pipeline used exactly 1 chunk", r.json()["type"] == "answer" and len(recent["chunk_ids_used"] or []) == 1,
          f"chunk_ids_used={recent['chunk_ids_used']}")
    c.patch("/api/admin/config/retrieval.top_k.text", json={"value": DEFAULT_TOP_K})
    c.patch("/api/admin/config/retrieval.top_k", json={"value": DEFAULT_TOP_K})
    check("restored: retrieve() returns 3 again", len(retrieve(query)) == DEFAULT_TOP_K)

    print("3. Provider-config key routing + validation")
    r = c.patch("/api/admin/config/generation.confidence_threshold", json={"value": 0.5})
    check("generation.confidence_threshold=0.5 routed to provider_configs",
          r.status_code == 200 and r.json()["source"] == "provider_configs" and r.json()["config"]["confidence_threshold"] == 0.5)
    from generation.generate import get_confidence_threshold
    check("get_confidence_threshold() sees it immediately", get_confidence_threshold() == 0.5)
    check("the global provider threshold does NOT leak into the text channel (own row)",
          c.post("/api/chat", json={"query": query, "user_attributes": FULL_USER}).json()["type"] == "answer")
    # threshold 0.5 > any fused score -> everything takes the shortcut (text channel's own row)
    c.patch("/api/admin/config/generation.confidence_threshold.text", json={"value": 0.5})
    r = c.post("/api/chat", json={"query": query, "user_attributes": FULL_USER})
    check("text channel threshold change alters behavior (shortcut now fires)", r.json()["type"] == "refusal_shortcut", r.json()["type"])
    c.patch("/api/admin/config/generation.confidence_threshold.text", json={"value": 0.02})
    c.patch("/api/admin/config/generation.confidence_threshold", json={"value": 0.02})
    for key, bad in [("retrieval.top_k", 0), ("retrieval.candidate_n", "abc"), ("retrieval.rrf_k", 2.5),
                     ("retrieval.top_k", True), ("generation.confidence_threshold", 1.5),
                     ("generation.confidence_threshold", "high"), ("tts.voice", "")]:
        r = c.patch(f"/api/admin/config/{key}", json={"value": bad})
        check(f"rejects {key}={bad!r}", r.status_code == 422, str(r.status_code))
    check("unknown key -> 404", c.patch("/api/admin/config/nope.nope", json={"value": 1}).status_code == 404)

    print(f"4. Cache-aside: direct DB edit shows up within the {CACHE_TTL_SECONDS}s TTL")
    c.patch("/api/admin/config/retrieval.candidate_n", json={"value": 20})  # prime cache
    check("primed cache", get_app_config("retrieval.candidate_n") == 20)
    with get_pool().connection() as conn:
        conn.execute("UPDATE app_config SET value = '25'::jsonb WHERE key = 'retrieval.candidate_n'")
        conn.commit()
    check("immediately after the direct edit the cache is still stale (20)", get_app_config("retrieval.candidate_n") == 20)
    deadline = time.time() + CACHE_TTL_SECONDS + 5
    while time.time() < deadline and get_app_config("retrieval.candidate_n") != 25:
        time.sleep(1)
    check("then converges to 25", get_app_config("retrieval.candidate_n") == 25)
    c.patch("/api/admin/config/retrieval.candidate_n", json={"value": DEFAULT_CANDIDATE_N})

    print("5. Redis outage degrades to DB reads, never raises")
    redis_cache.configure("redis://localhost:1/0")  # nothing listens here
    check("get_cached -> MISS", redis_cache.get_cached("x") is MISS)
    check("set_cached -> False, no exception", redis_cache.set_cached("x", 1) is False)
    check("get_app_config still answers from the DB", get_app_config("retrieval.top_k") == DEFAULT_TOP_K)
    redis_cache.configure()
    check("cached None is distinguishable from a miss",
          redis_cache.set_cached("t:none", None, ttl=5) and redis_cache.get_cached("t:none") is None)

    print("6. Burst of 10 concurrent chat requests through the shared pool")
    def one(i):
        r = httpx.post(f"{args.base_url}/api/chat", timeout=180,
                       json={"query": f"{query}{' ' * i}", "user_attributes": FULL_USER})
        return r.status_code, r.json().get("type")
    with ThreadPoolExecutor(10) as ex:
        results = list(ex.map(one, range(10)))
    check("all 10 returned 200 with a valid type", all(s == 200 and t in ("answer", "refusal_shortcut") for s, t in results), str(results))

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
