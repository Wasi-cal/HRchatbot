#!/usr/bin/env python3
"""Measures ungrounded reassurance / situation-fit on the grief scenario.

    python3 -m eval.grounding_fit_measure --label v6 [--runs 8]

Runs two turns N times per channel against the ACTIVE prompt versions (real model, real retrieval):
  Q1 "can I apply for bereavement leave for my dog"  after the grief opening
  Q2 "do I need to send proof"                       after Q1
and writes the raw answers to eval/results/grounding_fit_<label>.json for manual grading. It also counts, with simple
text heuristics (a screen, not a grade - read the raw answers):
  (a) Q1 answers that say the leave applies to the pet without saying the policy doesn't mention it
  (b) Q2 answers that state what is / isn't required to be submitted when no retrieved chunk says so
The earlier assistant turns in the history are fixed neutral text, so every run (and every prompt version) sees the same input.
"""
import argparse
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from channels import common, get_channel  # noqa: E402
from config.prompts import get_system_prompt  # noqa: E402
from vectorstore.db import close_pool, get_pool  # noqa: E402

H0 = [
    {"role": "user", "content": "My dog died."},
    {"role": "assistant", "content": "I'm so sorry for your loss. How can I help?"},
    {"role": "user", "content": "Yeah. I want some time off."},
    {"role": "assistant", "content": "Earned leave is one option. Want me to go through the leave types?"},
]
Q1 = "can I apply for bereavement leave for my dog"
A1 = "The handbook lists bereavement leave as two days."
Q2 = "do I need to send proof"

NOT_MENTIONED = re.compile(r"(doesn'?t|does not|don'?t|do not|isn'?t|is not|not) (mention|say|state|specify|cover|list|include|clear)|couldn'?t find|could not find|no mention|nothing (in|about|on)|not (stated|specified|covered|mentioned|clear)", re.I)
APPLIES = re.compile(r"\b(yes|you can (apply|take|request|use)|you('re| are) (eligible|entitled)|can apply|two days|2 days|entitled to)\b", re.I)
REQUIREMENT = re.compile(r"(don'?t|do not|doesn'?t|does not|no|won'?t|not) (need|require|necessary|needed|required|have to|must)|\b(need|needs|required|must|have to|should|requires?) (to )?(provide|submit|send|show|attach|upload|give|produce|bring|have)|\b(no|any) (proof|documents?|documentation|certificate|evidence|photos?)\b|(proof|documentation|certificate|evidence|photos?) (is|are) (needed|required|necessary|not needed|not required)|approval (isn'?t|is not|is) (required|needed)", re.I)


def run_turn(channel, query, history):
    module = get_channel(channel)
    plan = common.plan_turn(query, history, channel)
    results = module._retrieve(plan)
    answer = module.answer(query, history=history)
    return answer["answer"], [r["chunk_id"] for r in results], answer.get("prompt_version")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=8)
    args = ap.parse_args()
    out = {"label": args.label, "prompts": {c: get_system_prompt(c)[1] for c in ("text", "voice")}, "runs": []}
    flag_a = {"text": 0, "voice": 0}
    flag_b = {"text": 0, "voice": 0}
    for channel in ("text", "voice"):
        for i in range(args.runs):
            a1, c1, _ = run_turn(channel, Q1, H0)
            h2 = H0 + [{"role": "user", "content": Q1}, {"role": "assistant", "content": A1}]
            a2, c2, _ = run_turn(channel, Q2, h2)
            fa = bool(APPLIES.search(a1)) and not NOT_MENTIONED.search(a1)
            fb = bool(REQUIREMENT.search(a2)) and not NOT_MENTIONED.search(a2)
            flag_a[channel] += fa
            flag_b[channel] += fb
            out["runs"].append({"channel": channel, "run": i + 1, "q1": Q1, "a1": a1, "chunks1": c1, "flag_a": fa,
                                "q2": Q2, "a2": a2, "chunks2": c2, "flag_b": fb})
    out["flagged"] = {"a_applies_without_saying_not_mentioned": flag_a, "b_requirement_stated_without_support": flag_b}
    path = Path(__file__).parent / "results" / f"grounding_fit_{args.label}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[{args.label}] prompts {out['prompts']}")
    print(f"[{args.label}] (a) {flag_a}  (b) {flag_b}  of {args.runs} per channel; raw answers: {path}")
    close_pool()


if __name__ == "__main__":
    main()
