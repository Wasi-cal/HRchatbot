#!/usr/bin/env python3
"""Ungrounded coverage / requirement claims and needless hedging, per prompt version.

    python3 -m eval.coverage_claims_measure --label v7 [--runs 8]

Six single-turn questions (no history), N runs per channel against the ACTIVE prompts, real model + retrieval. Raw answers go to
eval/results/coverage_claims_<label>.json. The regexes are a SCREEN, not a grade - read the raw answers:
  (a) coverage claim: names who a leave covers (family, relative, immediate, dependent, spouse, parent, ...) with no "doesn't say" hedge
  (b) requirement/proof claim: says what must or needn't be provided (proof, certificate, document, approval, ...) with no "couldn't find" hedge
  (c) hedging on a stated fact (the two count questions only): typically / may / might / approximately / check with ... on "2 days" or "7 days"
"""
import argparse
import json
import re
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from channels import get_channel  # noqa: E402
from config.prompts import get_system_prompt  # noqa: E402
from vectorstore.db import close_pool  # noqa: E402

TURNS = [
    ("dog", "can I take bereavement leave for my dog", "a"),
    ("proof_bereavement", "do I need to send proof for bereavement leave", "b"),
    ("who_bereavement", "who does bereavement leave cover", "a"),
    ("proof_sick", "do I need proof for sick leave", "b"),
    ("days_bereavement", "how many days of bereavement leave", "c"),
    ("days_sick", "how many sick leaves do I get", "c"),
]
HEDGE_UNSTATED = re.compile(r"(doesn'?t|does not|don'?t|do not|isn'?t|is not|not) (mention|say|state|specify|cover|list|include|clear|provided)|couldn'?t find|could not find|no mention|nothing (in|about|on)|not (stated|specified|covered|mentioned|clear)|doesn'?t (really )?(go into|detail)", re.I)
COVERAGE = re.compile(r"\b(family|families|relatives?|immediate|dependents?|spouse|parents?|children|siblings?|in-?laws?|loved ones|close (family|relatives))\b", re.I)
REQUIREMENT = re.compile(r"(don'?t|do not|doesn'?t|does not|no|won'?t|not) (need|require|necessary|needed|required|have to|must)|\b(need|needs|required|must|have to|should|requires?) (to )?(provide|submit|send|show|attach|upload|give|produce|bring|have|get)\b|\b(no|any|some) (proof|documents?|documentation|certificates?|evidence|photos?)\b|\b(proof|documentation|certificates?|evidence|photos?|documents?) (is|are) (needed|required|necessary|not needed|not required)|\b(medical )?certificates?\b|approval (isn'?t|is not|is) (required|needed)|\bautomatically\b", re.I)
HEDGE_FACT = re.compile(r"\b(typically|usually|generally|may|might|approximately|around|roughly|likely|probably|depends|up to|check with|confirm with|not sure|couldn'?t find|I believe|should be)\b", re.I)


def screen(text, kinds):
    """(a, b, c) flags for one answer; curly apostrophes are normalised first."""
    text = text.replace("\u2019", "'")
    fa = "a" in kinds and any(COVERAGE.search(x) and not HEDGE_UNSTATED.search(x) for x in re.split(r"(?<=[.!?])\s+", text))
    fb = "b" in kinds and bool(REQUIREMENT.search(text)) and not HEDGE_UNSTATED.search(text)
    fc = "c" in kinds and bool(HEDGE_FACT.search(text))
    return fa, fb, fc


def rescore(label):
    """Re-applies the current screen to saved raw answers (no model calls)."""
    data = json.loads((Path(__file__).parent / "results" / f"coverage_claims_{label}.json").read_text())
    kinds = {k: v for k, _q, v in TURNS}
    tally = {}
    for r in data["runs"]:
        for name, f in zip("abc", screen(r["answer"], kinds[r["turn"]])):
            tally.setdefault((r["channel"], r["turn"]), {"a": 0, "b": 0, "c": 0})[name] += f
    for (c, t), v in tally.items():
        print(f"[{label} rescored] {c:5s} {t:18s} a={v['a']} b={v['b']} c={v['c']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--rescore", action="store_true", help="re-screen saved answers, no model calls")
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.label)
    out = {"label": args.label, "prompts": {c: get_system_prompt(c)[1] for c in ("text", "voice")}, "runs": []}
    flags = {c: {"a": 0, "b": 0, "c": 0} for c in ("text", "voice")}
    per_turn = {}
    for channel in ("text", "voice"):
        module = get_channel(channel)
        for key, question, kinds_for_turn in TURNS:
            for i in range(args.runs):
                text = module.answer(question)["answer"]
                fa, fb, fc = screen(text, kinds_for_turn)
                for k, f in (("a", fa), ("b", fb), ("c", fc)):
                    flags[channel][k] += f
                    per_turn.setdefault((channel, key), {"a": 0, "b": 0, "c": 0})[k] += f
                out["runs"].append({"channel": channel, "turn": key, "question": question, "run": i + 1, "answer": text, "a": fa, "b": fb, "c": fc})
    out["flags_by_channel"] = flags
    out["flags_by_turn"] = {f"{c}/{t}": v for (c, t), v in per_turn.items()}
    path = Path(__file__).parent / "results" / f"coverage_claims_{args.label}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[{args.label}] prompts {out['prompts']}")
    for (c, t), v in per_turn.items():
        print(f"[{args.label}] {c:5s} {t:18s} a={v['a']} b={v['b']} c={v['c']}  (of {args.runs})")
    print(f"[{args.label}] raw answers: {path}")
    close_pool()


if __name__ == "__main__":
    main()
