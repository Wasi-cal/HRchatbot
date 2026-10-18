#!/usr/bin/env python3
"""Short-query dilution set: does a fresh short question after unrelated turns still retrieve what it retrieves alone,
and do true follow-ups still keep their topic?

    python3 -m eval.context_dilution_eval --build [--set tuned|heldout]   # freeze expected chunks (eval/context_dilution_set.json | _heldout.json)
    python3 -m eval.context_dilution_eval [--set tuned|heldout]           # evaluate the current retrieval (text channel depth), print the tables

Two sets: "tuned" (used to choose the weights / markers) and "heldout" (frozen BEFORE any tuning and never used to choose anything).
Groups: fresh (short fresh question), fresh_identity (8+ word fresh question that contains an identity phrase such as "I am ..."),
followup.

fresh:     expected = the top chunk of the query retrieved ALONE (no history). History = unrelated user turns.
follow-up: expected = the top-3 chunks of the PRECEDING question retrieved alone (the topic the follow-up refers to).
Rank = best (lowest) rank of an expected chunk in the fused list (top 30 requested, same fusion as a real turn; > 30 = not found).
"""
import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import channels.text as text_ch
from channels import common
from vectorstore.db import close_pool

SET_PATHS = {"tuned": Path(__file__).parent / "context_dilution_set.json", "heldout": Path(__file__).parent / "context_dilution_heldout.json",
             "identity_heldout": Path(__file__).parent / "context_dilution_identity_heldout.json"}
SET_PATH = SET_PATHS["tuned"]

UNRELATED = [
    ["My dog died.", "Yeah. I want some time off."],
    ["how many days of earned leave do i get", "can i work from home if i am sick"],
    ["what is the probation period", "who do i report a security incident to"],
]
FRESH = ["Tell me how change of work location", "what is the notice period", "referral bonus amount", "how do I report a whistleblower concern",
         "performance appraisal cycle", "remote connectivity rules", "certification reimbursement process", "how to file a POSH complaint",
         "travel reimbursement limits", "what is the confirmation process"]
FRESH_IDENTITY = [
    "I am relocating to another city, what does the policy say",
    "as an employee on probation what is the notice period",
    "I am thinking of getting a certification, what does the company reimburse",
    "as a new joiner how do I get my referral bonus",
    "im planning to travel for work, what are the reimbursement rules",
    "I am facing harassment from a colleague, how do I file a complaint",
    "as an employee who works remotely what connectivity rules apply to me",
    "what about the performance appraisal cycle, when does it happen each year",
    "and for people leaving the company, what is the notice period",
    "I am a manager and want to put an employee on a PIP, what is the process",
    "as a team lead how do I nominate someone for rewards and recognition",
    "I am worried about a data breach on my laptop, who should I tell",
]
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


IDENTITY_HELDOUT = [  # 4 long (8+ words) then 4 short (<= 6 words); frozen before any evaluation, never used for tuning
    "I am travelling abroad for work, how are my expenses reimbursed",
    "as an employee under a PIP what happens if I do not improve",
    "im worried about being asked to work on a public holiday, what are the rules",
    "as a remote employee what do I need to do to keep my laptop secure",
    "as a manager, referral bonus rules?",
    "i am on probation, confirmation criteria",
    "im reporting harassment, what next",
    "as an employee, travel limits?",
]
HELDOUT_FRESH = ["how do I apply for a referral bonus", "what is the process to claim travel expenses", "who is on the POSH internal committee",
                 "can I get my certification fee reimbursed", "how many days notice do I need to serve", "what are the rules for working remotely",
                 "tell me about the performance improvement plan", "how is my probation confirmed", "where do I report a data breach",
                 "what is the disciplinary process"]
HELDOUT_FOLLOWUPS = [
    ("how many sick leaves do i get", "what if I run out"),
    ("what is the notice period", "can it be shortened"),
    ("how do referrals work", "what about the payout timing"),
    ("how does work from home work", "how many days a week"),
    ("what is the probation period", "can it be extended"),
    ("tell me about the performance improvement plan", "how long is it"),
    ("how many earned leaves do i get", "does it expire"),
    ("what is the certification reimbursement policy", "what is the limit"),
    ("what are the travel rules", "and for international trips"),
    ("how does comp off work", "how do I apply for it"),
]


def ranked(plan, n=30):
    return [r["chunk_id"] for r in text_ch._retrieve(plan, top_k=n)]


def build(which="tuned"):
    cases = []
    if which == "identity_heldout":
        fresh, identity, followups = [], IDENTITY_HELDOUT, []
    else:
        fresh, identity, followups = (FRESH, FRESH_IDENTITY, FOLLOWUPS) if which == "tuned" else (HELDOUT_FRESH, [], HELDOUT_FOLLOWUPS)
    for i, q in enumerate(identity):
        alone = ranked(common.plan_turn(q, [], "text"))
        hist = []
        for u in UNRELATED[i % len(UNRELATED)]:
            hist += [{"role": "user", "content": u}, {"role": "assistant", "content": "Here is what the policy says."}]
        cases.append({"group": "fresh_identity", "history": hist, "query": q, "expected": alone[:1]})
    for i, q in enumerate(fresh):
        alone = ranked(common.plan_turn(q, [], "text"))
        hist = []
        for u in UNRELATED[i % len(UNRELATED)]:
            hist += [{"role": "user", "content": u}, {"role": "assistant", "content": "Here is what the policy says."}]
        cases.append({"group": "fresh", "history": hist, "query": q, "expected": alone[:1]})
    for prev, q in followups:
        topic = ranked(common.plan_turn(prev, [], "text"))[:3]
        hist = [{"role": "user", "content": prev}, {"role": "assistant", "content": "Here is what the policy says."}]
        cases.append({"group": "followup", "history": hist, "query": q, "expected": topic, "previous_query": prev})
    SET_PATHS[which].write_text(json.dumps(cases, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {len(cases)} cases to {SET_PATHS[which]}")


def short(c):
    return c[:3] + "." + c.split("_")[-1]


def evaluate(quiet=False, which="tuned"):
    cases = json.loads(SET_PATHS[which].read_text())
    rows = []
    for c in cases:
        plan = common.plan_turn(c["query"], c["history"], "text")
        ids = ranked(plan)
        ranks = [ids.index(e) + 1 for e in c["expected"] if e in ids]
        rows.append({**c, "top3": ids[:3], "rank": min(ranks) if ranks else None, "contextualized": plan.contextualized})
    if not quiet:
        for g in ("fresh", "fresh_identity", "followup"):
            print(f"\n=== {g} ===")
            print(f"{'query':44s} {'ctx':3s} {'expected':14s} {'top-3':32s} rank")
            for r in [x for x in rows if x["group"] == g]:
                print(f"{r['query'][:43]:44s} {'Y' if r['contextualized'] else 'n':3s} {','.join(map(short, r['expected']))[:13]:14s} {','.join(map(short, r['top3'])):32s} {r['rank'] or '>30'}")
            rs = [r for r in rows if r["group"] == g]
            if not rs:
                continue
            top1 = sum(1 for r in rs if r["rank"] == 1)
            in3 = sum(1 for r in rs if r["rank"] and r["rank"] <= 3)
            print(f"-> expected at rank 1: {top1}/{len(rs)}; in top 3: {in3}/{len(rs)}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--set", choices=["tuned", "heldout", "identity_heldout"], default="tuned")
    args = ap.parse_args()
    build(args.set) if args.build else evaluate(which=args.set)
    close_pool()


if __name__ == "__main__":
    main()
