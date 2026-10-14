#!/usr/bin/env python3
"""Interleaved A/B of prompt configurations (prompt version + turn reminder + temperature) on grounding-claim turns.

    python3 -m eval.prompt_ab --label ab1 [--runs 8]

For run 1..N, EVERY configuration answers every turn once before run N+1 starts (settings are switched through the admin API
between configurations), so network / provider drift hits all configurations equally. Original settings are restored at the end.
Writes eval/results/ab_<label>.json (raw answers + screens + usage), a BLINDED side-by-side for manual grading
(ab_<label>_blinded.md, labels A/B/C shuffled per turn) and its key (ab_<label>_key.json).

Screens are regexes, a screen only - grade the blinded file by hand:
  a   unhedged coverage claim (who a benefit covers) not in the text
  b1  requirement/proof/approval stated as required without the text's scope (for sick-leave proof and approvals only a claim
      scoped to the notice period is accepted)
  b2  negated-requirement reassurance ("don't need", "no need", "not required", "doesn't require") near approval, proof, document,
      photo or certificate, without a "doesn't specify" in the same sentence
  c   hedging on a fact the text states plainly (day counts)
  d   intern case: says "couldn't find" and makes no "not entitled"-style claim
  t   trip case: states the working-week assumption and 2 unpaid days, never 4
"""
import argparse
import json
import logging
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from channels import get_channel  # noqa: E402
from eval.coverage_claims_measure import COVERAGE, HEDGE_FACT, HEDGE_UNSTATED, _LogTap  # noqa: E402
from vectorstore.db import close_pool  # noqa: E402

RESULTS = Path(__file__).parent / "results"
CONFIGS = {
    "v8_t1.0": {"version": 8, "reminder": None, "temperature": 1.0},
    "v9b_t0.2": {"version": 10, "reminder": RESULTS / "turn_reminder_v9b.txt", "temperature": 0.2},
    "v9b_t1.0": {"version": 10, "reminder": RESULTS / "turn_reminder_v9b.txt", "temperature": 1.0},
}
INTERN_HISTORY = [
    {"role": "user", "content": "tell me everything you know about the interns"},
    {"role": "assistant", "content": "I couldn't find information specifically about interns in the HR policies."},
    {"role": "user", "content": "how many leaves do i get"},
    {"role": "assistant", "content": "You get 21 earned leaves every calendar year."},
]
# key, question, history, kinds
TURNS = [
    ("dog", "can I take bereavement leave for my dog", None, "a"),
    ("who_bereavement", "who does bereavement leave cover", None, "a"),
    ("proof_bereavement", "do I need to send proof for bereavement leave", None, "b"),
    ("proof_sick", "do I need proof for sick leave", None, "b"),
    ("approval_earned", "do I need my manager's approval to take earned leave", None, "b"),
    ("mother_in_law", "can I take bereavement leave when my mother-in-law passes away", None, "ab"),
    ("adoptive_paternity", "does paternity leave cover adoptive parents", None, "ab"),
    ("contractor_earned", "can a contractor take earned leave", None, "a"),
    ("intern_sick", "do interns get sick leave", None, "a"),
    ("sister_wedding", "can I use marriage leave for my sister's wedding", None, "ab"),
    ("days_bereavement", "how many days of bereavement leave", None, "c"),
    ("days_sick", "how many sick leaves do I get", None, "c"),
    ("days_earned", "how many earned leaves do I get", None, "c"),
    ("intern", "i am an intern", INTERN_HISTORY, "d"),
    ("trip", "i need to go to my hometown for a week and i have only 3 leaves left", None, "t"),
]
KEY_TURNS = ["dog", "who_bereavement", "proof_bereavement", "proof_sick", "approval_earned", "intern", "trip"]

REQ_POS = re.compile(r"\b(need|needs|required|requires?|must|have to|should)\b[^.]{0,40}\b(proof|approval|approved|documents?|documentation|certificates?|photos?|provide|submit|send|notice)|\b(proof|approval|documents?|documentation|certificates?|photos?)\b[^.]{0,25}\b(is|are) (needed|required|necessary)|\bautomatically\b", re.I)
NEGATED = re.compile(r"(don'?t|do not|doesn'?t|does not|no|not) (need|require|required|necessary|needed)|no need|not required|isn'?t required|isn'?t needed", re.I)
NOUNS = re.compile(r"approval|approve|proof|documents?|documentation|photos?|certificates?|evidence", re.I)
BAD_GROUP_CLAIM = re.compile(r"not entitled|are not eligible|typically|differs|different (?:leave )?polic", re.I)


def _sentences(text):
    return re.split(r"(?<=[.!?])\s+", text.replace("’", "'"))


def screen(turn, kinds, text):
    t = text.replace("’", "'")
    flags = {"a": False, "b1": False, "b2": False, "c": False, "d": None, "t": None}
    sents = _sentences(t)
    if "a" in kinds:
        flags["a"] = any(COVERAGE.search(x) and not HEDGE_UNSTATED.search(x) for x in sents)
    if "b" in kinds:
        flags["b1"] = any(REQ_POS.search(x) and not NEGATED.search(x) and not HEDGE_UNSTATED.search(x) and "notice" not in x.lower() for x in sents)
        flags["b2"] = any(NEGATED.search(x) and NOUNS.search(x) and not HEDGE_UNSTATED.search(x) for x in sents)
    if "c" in kinds:
        flags["c"] = bool(HEDGE_FACT.search(t))
    if "d" in kinds:
        flags["d"] = "couldn't find" in t.lower() and not BAD_GROUP_CLAIM.search(t)
    if "t" in kinds:
        flags["t"] = bool(re.search(r"five-day|5-day|working week|working days", t, re.I) and re.search(r"\b(2|two)\b", t) and not re.search(r"\b(4|four)\b", t))
    return flags


class Switcher:
    def __init__(self):
        from fastapi.testclient import TestClient

        from api.main import app
        from config.runtime import get_active_provider_config, get_app_config
        self.client = TestClient(app)
        self.client.__enter__()
        self.saved = {"reminder": get_app_config("prompts.turn_reminder", ""), "temperature": get_active_provider_config("generation").get("temperature", 1.0)}
        groups = self.client.get("/api/admin/prompts").json()
        self.versions = {}  # (channel, version) -> id ; plus the originally active ids
        self.original = {}
        for g in groups:
            if g["name"] != "grounding_core":
                continue
            for r in g["versions"]:
                self.versions[(g["channel"], r["version"])] = r["id"]
                if r["is_active"]:
                    self.original[g["channel"]] = r["id"]
        self.current = None

    def apply(self, name):
        if self.current == name:
            return
        cfg = CONFIGS[name]
        for ch in ("text", "voice"):
            assert self.client.post(f"/api/admin/prompts/{self.versions[(ch, cfg['version'])]}/activate").status_code == 200
        text = Path(cfg["reminder"]).read_text().strip() if cfg["reminder"] else ""
        assert self.client.patch("/api/admin/config/prompts.turn_reminder", json={"value": text}).status_code == 200
        assert self.client.patch("/api/admin/config/generation.temperature", json={"value": cfg["temperature"]}).status_code == 200
        self.current = name

    def restore(self):
        for ch, prompt_id in self.original.items():
            self.client.post(f"/api/admin/prompts/{prompt_id}/activate")
        self.client.patch("/api/admin/config/prompts.turn_reminder", json={"value": self.saved["reminder"]})
        self.client.patch("/api/admin/config/generation.temperature", json={"value": self.saved["temperature"]})
        self.client.__exit__(None, None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    tap = _LogTap()
    for name in ("hrchatbot.text", "hrchatbot.voice"):
        logging.getLogger(name).addHandler(tap)
        logging.getLogger(name).setLevel(logging.INFO)
    sw = Switcher()
    records, usage_rows = [], {c: [] for c in CONFIGS}
    try:
        for run in range(1, args.runs + 1):
            for cname in CONFIGS:
                sw.apply(cname)
                tap.rows.clear()
                jobs = [(ch, t) for ch in ("text", "voice") for t in TURNS]

                def work(job):
                    ch, (key, q, hist, kinds) = job
                    return job, get_channel(ch).answer(q, history=hist)["answer"]

                with ThreadPoolExecutor(args.workers) as pool:
                    for (ch, (key, q, hist, kinds)), text in pool.map(work, jobs):
                        records.append({"config": cname, "run": run, "channel": ch, "turn": key, "question": q, "answer": text, "flags": screen(key, kinds, text)})
                usage_rows[cname] += list(tap.rows)
                print(f"run {run} {cname} done", file=sys.stderr, flush=True)
    finally:
        sw.restore()
        close_pool()
    # ---- tables
    cols = ["a", "b1", "b2", "c", "d", "t"]
    summary, per_turn = {}, {}
    for r in records:
        s = summary.setdefault(r["config"], {k: 0 for k in cols})
        pt = per_turn.setdefault((r["config"], r["channel"], r["turn"]), {k: 0 for k in cols})
        for k in cols:
            v = r["flags"][k]
            if v is True and k in ("a", "b1", "b2", "c"):
                s[k] += 1; pt[k] += 1
            if k in ("d", "t") and v is True:  # d / t count PASSES
                s[k] += 1; pt[k] += 1
    usage = {}
    for c, rows in usage_rows.items():
        if rows:
            m = lambda k: round(sum(x.get(k, 0) for x in rows) / len(rows), 1)
            usage[c] = {"turns": len(rows), "ttft_ms": m("ttft_ms"), "prompt_tokens": m("prompt_tokens"), "cached_tokens": m("cached_tokens"), "completion_tokens": m("completion_tokens")}
    out = {"label": args.label, "runs": args.runs, "configs": {k: {**v, "reminder": str(v["reminder"]) if v["reminder"] else None} for k, v in CONFIGS.items()},
           "summary_flags": summary, "usage": usage, "per_turn": {"|".join(k): v for k, v in per_turn.items()}, "records": records}
    (RESULTS / f"ab_{args.label}.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    # ---- blinded side by side
    rng = random.Random(20261014)
    key, lines = {}, [f"# Blinded side-by-side ({args.label})", "", "Each block shows the answers of THREE configurations as sets A/B/C (order shuffled per turn and channel). "
                      "Grade them blind; the mapping is in the key file.", ""]
    by = {}
    for r in records:
        by.setdefault((r["turn"], r["channel"], r["config"]), []).append(r)
    for t in KEY_TURNS:
        q = next(x[1] for x in TURNS if x[0] == t)
        for ch in ("text", "voice"):
            order = list(CONFIGS)
            rng.shuffle(order)
            labels = dict(zip("ABC", order))
            key[f"{t}/{ch}"] = labels
            lines += [f"## {t} ({ch}): {q}", ""]
            for lab, cfg in labels.items():
                lines.append(f"**Set {lab}**")
                for r in sorted(by[(t, ch, cfg)], key=lambda x: x["run"]):
                    lines.append(f"- {r['answer'].strip()!r}")
                lines.append("")
    (RESULTS / f"ab_{args.label}_blinded.md").write_text("\n".join(lines))
    (RESULTS / f"ab_{args.label}_key.json").write_text(json.dumps(key, indent=2))
    n = args.runs * 2
    print(f"AB-RESULT runs={args.runs} (per turn per config {n} answers)")
    for c, s in summary.items():
        print("AB-RESULT", c, s, usage.get(c))
    print("AB-FILES", RESULTS / f"ab_{args.label}.json")


if __name__ == "__main__":
    main()
