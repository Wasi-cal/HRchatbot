#!/usr/bin/env python3
"""Short-query dilution set: does a fresh short question after unrelated turns still retrieve what it retrieves alone,
and do true follow-ups still keep their topic?

    python3 -m eval.context_dilution_eval --build     # freeze expected chunks into eval/context_dilution_set.json
    python3 -m eval.context_dilution_eval             # evaluate the current retrieval (text channel depth) and print the tables

fresh:     expected = the top chunk of the query retrieved ALONE (no history). History = unrelated user turns.
follow-up: expected = the top-3 chunks of the PRECEDING question retrieved alone (the topic the follow-up refers to).
Rank = best (lowest) rank of an expected chunk in the fused list (top 30 requested, same fusion as a real turn; > 30 = not found).
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import channels.text as text_ch  # noqa: E402
from channels import common  # noqa: E402
from vectorstore.db import close_pool  # noqa: E402

SET_PATH = Path(__file__).parent / "context_dilution_set.json"

UNRELATED = [
    ["My dog died.", "Yeah. I want some time off."],
    ["how many days of earned leave do i get", "can i work from home if i am sick"],
    ["what is the probation period", "who do i report a security incident to"],
]
FRESH = ["Tell me how change of work location", "what is the notice period", "referral bonus amount", "how do I report a whistleblower concern",
         "performance appraisal cycle", "remote connectivity rules", "certification reimbursement process", "how to file a POSH complaint",
         "travel reimbursement limits", "what is the confirmation process"]
FOLLOWUPS = [
    ("how many earned leaves do i get", "what about interns"),
    ("how do I refer a friend for a job", "and for contractors"),
    ("how many days of earned leave do i get", "can I carry it forward"),
    ("what is the notice period", "what happens if I exceed it"),
    ("how many earned leaves do i get", "is it the same for managers"),
    ("how does work from home work", "what else"),
    ("what are the work from home rules", "instead?"),
    ("can i encash my earned leaves", "how many days is that"),
    ("what is the referral bonus", "what about interns"),
    ("how does the performance improvement plan work", "how long does it last"),
    # no follow-up marker word: only their shortness says they depend on the previous turn
    ("how many earned leaves do i get", "for interns?"),
    ("how do I refer a friend for a job", "contractors too?"),
    ("what is the referral bonus", "how long to wait?"),
]


def ranked(plan, n=30):
    return [r["chunk_id"] for r in text_ch._retrieve(plan, top_k=n)]


def build():
    cases = []
    for i, q in enumerate(FRESH):
        alone = ranked(common.plan_turn(q, [], "text"))
        hist = []
        for u in UNRELATED[i % len(UNRELATED)]:
            hist += [{"role": "user", "content": u}, {"role": "assistant", "content": "Here is what the policy says."}]
        cases.append({"group": "fresh", "history": hist, "query": q, "expected": alone[:1]})
    for prev, q in FOLLOWUPS:
        topic = ranked(common.plan_turn(prev, [], "text"))[:3]
        hist = [{"role": "user", "content": prev}, {"role": "assistant", "content": "Here is what the policy says."}]
        cases.append({"group": "followup", "history": hist, "query": q, "expected": topic, "previous_query": prev})
    SET_PATH.write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {SET_PATH}")


def short(c):
    return c[:3] + "." + c.split("_")[-1]


def evaluate(quiet=False):
    cases = json.loads(SET_PATH.read_text())
    rows = []
    for c in cases:
        plan = common.plan_turn(c["query"], c["history"], "text")
        ids = ranked(plan)
        ranks = [ids.index(e) + 1 for e in c["expected"] if e in ids]
        rows.append({**c, "top3": ids[:3], "rank": min(ranks) if ranks else None, "contextualized": plan.contextualized})
    if not quiet:
        for g in ("fresh", "followup"):
            print(f"\n=== {g} ===")
            print(f"{'query':44s} {'ctx':3s} {'expected':14s} {'top-3':32s} rank")
            for r in [x for x in rows if x["group"] == g]:
                print(f"{r['query'][:43]:44s} {'Y' if r['contextualized'] else 'n':3s} {','.join(map(short, r['expected']))[:13]:14s} {','.join(map(short, r['top3'])):32s} {r['rank'] or '>30'}")
            rs = [r for r in rows if r["group"] == g]
            top1 = sum(1 for r in rs if r["rank"] == 1)
            in3 = sum(1 for r in rs if r["rank"] and r["rank"] <= 3)
            print(f"-> expected at rank 1: {top1}/{len(rs)}; in top 3: {in3}/{len(rs)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    args = ap.parse_args()
    build() if args.build else evaluate()
    close_pool()


if __name__ == "__main__":
    main()
