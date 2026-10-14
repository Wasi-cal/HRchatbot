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
import logging
import re
from concurrent.futures import ThreadPoolExecutor
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
    ("mother_in_law", "can I take bereavement leave when my mother-in-law passes away", "ab"),
    ("adoptive_paternity", "does paternity leave cover adoptive parents", "ab"),
    ("contractor_earned", "can a contractor take earned leave", "a"),
    ("intern_sick", "do interns get sick leave", "a"),
    ("sister_wedding", "can I use marriage leave for my sister's wedding", "ab"),
    ("approval_earned", "do I need my manager's approval to take earned leave", "b"),
    ("days_bereavement", "how many days of bereavement leave", "c"),
    ("days_sick", "how many sick leaves do I get", "c"),
    ("days_earned", "how many earned leaves do I get", "c"),
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
    ap.add_argument("--grounding-version", type=int, help="activate this grounding_core version on both channels for the run (restored afterwards)")
    ap.add_argument("--reminder-file", help="set prompts.turn_reminder from this file for the run ('' / omitted = off); restored afterwards")
    ap.add_argument("--temperature", type=float, help="set generation.temperature for the run (restored afterwards)")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    if args.rescore:
        return rescore(args.label)
    with _settings(args):
        run(args)


class _LogTap(logging.Handler):
    """Collects the 'turn type=answer ... ttft_ms= prompt_tokens= cached_tokens=' log lines."""
    def __init__(self):
        super().__init__(logging.INFO)
        self.rows = []

    def emit(self, record):
        m = record.getMessage()
        if m.startswith("turn type=answer"):
            self.rows.append({k: float(v) for k, v in re.findall(r"\b(ttft_ms|prompt_tokens|cached_tokens|completion_tokens)=([0-9.]+)", m)})


class _settings:
    """Applies the A/B settings through the admin API and restores the previous ones."""
    def __init__(self, args):
        self.args = args

    def __enter__(self):
        from fastapi.testclient import TestClient

        from api.main import app
        from config.runtime import get_active_provider_config, get_app_config
        self.client = TestClient(app)
        self.client.__enter__()
        self.saved = {"reminder": get_app_config("prompts.turn_reminder", ""), "temperature": get_active_provider_config("generation").get("temperature", 1.0)}
        a = self.args
        self.rows = {}
        if a.grounding_version:
            groups = self.client.get("/api/admin/prompts").json()
            for ch in ("text", "voice"):
                versions = next(g for g in groups if g["name"] == "grounding_core" and g["channel"] == ch)["versions"]
                active = next(r for r in versions if r["is_active"])
                target = next(r for r in versions if r["version"] == a.grounding_version)
                self.rows[ch] = active["id"]
                assert self.client.post(f"/api/admin/prompts/{target['id']}/activate").status_code == 200
        text = Path(a.reminder_file).read_text().strip() if a.reminder_file else ""
        assert self.client.patch("/api/admin/config/prompts.turn_reminder", json={"value": text}).status_code == 200
        if a.temperature is not None:
            assert self.client.patch("/api/admin/config/generation.temperature", json={"value": a.temperature}).status_code == 200
        return self

    def __exit__(self, *exc):
        for ch, prompt_id in self.rows.items():
            self.client.post(f"/api/admin/prompts/{prompt_id}/activate")
        self.client.patch("/api/admin/config/prompts.turn_reminder", json={"value": self.saved["reminder"]})
        self.client.patch("/api/admin/config/generation.temperature", json={"value": self.saved["temperature"]})
        self.client.__exit__(*exc)


def run(args):
    tap = _LogTap()
    for name in ("hrchatbot.text", "hrchatbot.voice"):
        logging.getLogger(name).addHandler(tap)
        logging.getLogger(name).setLevel(logging.INFO)
    out = {"label": args.label, "prompts": {c: get_system_prompt(c)[1] for c in ("text", "voice")}, "runs": [],
           "settings": {"grounding_version": args.grounding_version, "reminder": bool(args.reminder_file), "temperature": args.temperature}}
    flags = {c: {"a": 0, "b": 0, "c": 0} for c in ("text", "voice")}
    per_turn = {}
    jobs = [(channel, key, question, kinds, i) for channel in ("text", "voice") for key, question, kinds in TURNS for i in range(args.runs)]

    def work(job):
        channel, key, question, kinds_for_turn, i = job
        return job, get_channel(channel).answer(question)["answer"]

    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(work, jobs))
    for (channel, key, question, kinds_for_turn, i), text in results:
        fa, fb, fc = screen(text, kinds_for_turn)
        for k, f in (("a", fa), ("b", fb), ("c", fc)):
            flags[channel][k] += f
            per_turn.setdefault((channel, key), {"a": 0, "b": 0, "c": 0})[k] += f
        out["runs"].append({"channel": channel, "turn": key, "question": question, "run": i + 1, "answer": text, "a": fa, "b": fb, "c": fc})
    rows = tap.rows
    if rows:
        mean = lambda k: round(sum(r.get(k, 0) for r in rows) / len(rows), 1)
        out["usage"] = {"turns": len(rows), "ttft_ms_mean": mean("ttft_ms"), "prompt_tokens_mean": mean("prompt_tokens"),
                        "cached_tokens_mean": mean("cached_tokens"), "completion_tokens_mean": mean("completion_tokens")}
    out["flags_by_channel"] = flags
    out["flags_by_turn"] = {f"{c}/{t}": v for (c, t), v in per_turn.items()}
    path = Path(__file__).parent / "results" / f"coverage_claims_{args.label}.json"
    path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"[{args.label}] prompts {out['prompts']}")
    for (c, t), v in per_turn.items():
        print(f"[{args.label}] {c:5s} {t:18s} a={v['a']} b={v['b']} c={v['c']}  (of {args.runs})")
    print(f"[{args.label}] usage {out.get('usage')}")
    tot = {k: sum(v[k] for v in per_turn.values()) for k in "abc"}
    print(f"[{args.label}] TOTAL a={tot['a']} b={tot['b']} c={tot['c']} of {args.runs * 2} per turn x {len(TURNS)} turns")
    print(f"[{args.label}] raw answers: {path}")
    close_pool()


if __name__ == "__main__":
    main()
