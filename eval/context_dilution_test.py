#!/usr/bin/env python3
"""Dual-query fusion for contextualised turns (eval/context_dilution_set.json). Run: python3 -m eval.context_dilution_test"""
import sys

from fastapi.testclient import TestClient

from api.main import app
from channels import common
from config.runtime import get_app_config
from eval import context_dilution_eval as E
from vectorstore.db import close_pool

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if not ok and detail else ""))


def main():
    rows = E.evaluate(quiet=True)
    fresh = [r for r in rows if r["group"] == "fresh"]
    follow = [r for r in rows if r["group"] == "followup"]
    check("fresh short questions after unrelated turns keep their standalone top result in the top 3 (10/10)", all(r["rank"] and r["rank"] <= 3 for r in fresh), str([(r["query"], r["rank"]) for r in fresh if not r["rank"] or r["rank"] > 3]))
    check("...and the relocation question now reaches the context (was rank 4)", next(r for r in fresh if r["query"].startswith("Tell me how change"))["rank"] <= 3)
    check("true follow-ups keep the preceding topic in the top 3 (12 of 13, as before the change)", sum(1 for r in follow if r["rank"] and r["rank"] <= 3) >= 12, str([(r["query"], r["rank"]) for r in follow]))
    check("no follow-up is lost (every expected chunk within the top 10)", all(r["rank"] and r["rank"] <= 10 for r in follow))
    ident = [r for r in rows if r["group"] == "fresh_identity"]
    check("fresh questions with identity phrases ('I am relocating ...', 'as an employee ...') keep their standalone top result at rank 1 (11 of 12; was 6)", sum(1 for r in ident if r["rank"] == 1) >= 11, str([(r["query"][:30], r["rank"]) for r in ident if r["rank"] != 1]))
    held = E.evaluate(quiet=True, which="heldout")
    hf = [r for r in held if r["group"] == "fresh"]; hu = [r for r in held if r["group"] == "followup"]
    check("held-out fresh questions: expected at rank 1 for all 10", all(r["rank"] == 1 for r in hf), str([(r["query"], r["rank"]) for r in hf if r["rank"] != 1]))
    check("held-out follow-ups: expected in the top 3 for at least 9 of 10, none beyond rank 10", sum(1 for r in hu if r["rank"] and r["rank"] <= 3) >= 9 and all(r["rank"] and r["rank"] <= 10 for r in hu), str([(r["query"], r["rank"]) for r in hu]))
    hist = [{"role": "user", "content": "how do I refer a friend"}, {"role": "assistant", "content": "y"}]
    long_id = common.plan_turn("I am relocating to another city, what does the policy say", hist, "text")
    check("a long fresh question containing 'i am' is NOT contextualised (single standalone retrieval)", not long_id.contextualized and long_id.standalone_query is None)
    short_id = common.plan_turn("i am an intern", hist, "text")
    check("a short identity statement ('i am an intern') is contextualised with the identity weights (standalone 0.25, ctx 1.0)", short_id.contextualized and (short_id.standalone_weight, short_id.ctx_weight) == (0.25, 1.0), str((short_id.contextualized, short_id.standalone_weight, short_id.ctx_weight)))
    check("identity markers are not in the follow-up marker list", not set(get_app_config("retrieval.context.identity_markers")) & set(get_app_config("retrieval.context.followup_markers")))
    plan = common.plan_turn("what is the notice period", [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], "text")
    check("a qualifying turn plans a standalone query next to the contextualised one", plan.contextualized and plan.standalone_query == "what is the notice period")
    check("no history: single-query plan (retrieval baseline untouched)", common.plan_turn("what is the notice period", [], "text").standalone_query is None)
    long_q = "can you tell me what the policy is on carrying forward earned leave into next year please"
    check("a long non-marker question after history is not contextualised and has no standalone query", common.plan_turn(long_q, [{"role": "user", "content": "x"}, {"role": "assistant", "content": "y"}], "text").standalone_query is None)
    w_marker = common._context_weights("what about interns", "text")
    w_plain = common._context_weights("what is the notice period", "text")
    check("a follow-up marker shifts the weights toward the contextualised query (standalone weight lower)", w_marker[0] < w_marker[1] and w_marker[0] < w_plain[0], f"{w_marker} {w_plain}")
    check("a very short utterance counts as marked", common._context_weights("for interns?", "text")[0] < 0.5)
    with TestClient(app) as cl:
        p = lambda k, v: cl.patch(f"/api/admin/config/{k}", json={"value": v}).status_code
        check("weights validation: bad shapes rejected (422)", all(p("retrieval.context.weights_with_marker", v) == 422 for v in ({}, {"x": 1}, {"standalone": -1}, {"standalone": 0, "ctx": 0}, [1, 2], "a")))
        check("identity keys validated: weights_identity shape, identity_max_words int >= 1, identity_markers a list of strings (422)",
              p("retrieval.context.weights_identity", {"x": 1}) == 422 and p("retrieval.context.identity_max_words", 0) == 422 and p("retrieval.context.identity_max_words", "6") == 422 and p("retrieval.context.identity_markers", "i am") == 422)
        check("weight_standalone / weight_ctx must be numbers >= 0; ctx_heavy_below_words an int >= 0; dual_query.enabled a bool",
              p("retrieval.context.weight_ctx", -1) == 422 and p("retrieval.context.weight_standalone", "1") == 422 and p("retrieval.context.ctx_heavy_below_words", 1.5) == 422 and p("retrieval.context.dual_query.enabled", "yes") == 422)
    seeds = (get_app_config("retrieval.context.weights_with_marker"), get_app_config("retrieval.context.weights_without_marker"))
    check("seeded weights: marker -> standalone 0.05 / ctx 1.0; no marker -> 1.0 / 1.0", seeds == ({"standalone": 0.05, "ctx": 1.0}, {"standalone": 1.0, "ctx": 1.0}), str(seeds))
    close_pool()
    failed = results.count(False)
    print(f"\n=== {len(results) - failed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
