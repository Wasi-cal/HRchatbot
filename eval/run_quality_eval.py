#!/usr/bin/env python3
"""Formatted report generator for manual quality grading.

Reads eval/quality_eval_set.json, runs each case through ask_and_answer()
(user_attributes from the case if present, else None), and prints a
clean per-case report: query, category, expected_behavior, the actual
result, and a blank "grade: PASS/FAIL" line for a human to fill in
while reading. This script does NOT auto-grade anything - answer
quality on paraphrase/exact_term cases is a judgment call for a human
who knows the corpus, not something to fake with keyword matching (that
kind of structural check belongs in eval/run_regression.py instead).

After grading, log the result in eval/GRADING_LOG.md so there's a
durable history across runs, not just whatever happened to be in the
terminal that day.

Usage:
    python3 -m eval.run_quality_eval
"""
import json
from pathlib import Path

from access_control.ask_and_answer import ask_and_answer

EVAL_SET_PATH = Path(__file__).parent / "quality_eval_set.json"


def _format_result(result: dict) -> str:
    if result["type"] == "needs_clarification":
        return (
            f"CLARIFICATION\n"
            f"    question:         {result['question']!r}\n"
            f"    blocking_tag_type: {result['blocking_tag_type']!r}"
        )
    return (
        f"ANSWER (shortcut={result['shortcut']})\n"
        f"    {result['answer']!r}"
    )


def run_case(case: dict):
    print(f"--- {case['id']} [{case['category']}] ---")
    print(f"  query:             {case['query']!r}")
    print(f"  expected_behavior: {case['expected_behavior']!r}")
    print(f"  notes:             {case['notes']}")
    user_attributes = case.get("user_attributes")
    if user_attributes is not None:
        print(f"  user_attributes:   {user_attributes!r}")

    result = ask_and_answer(case["query"], user_attributes=user_attributes)
    print(f"  actual:            {_format_result(result)}")
    print("  grade:             PASS/FAIL   <- fill in by hand")
    print()


def main():
    cases = json.loads(EVAL_SET_PATH.read_text())
    print(f"=== Quality eval report: {len(cases)} cases from {EVAL_SET_PATH.name} ===\n")
    for case in cases:
        run_case(case)
    print(
        "=== End of report. Log your grading results in eval/GRADING_LOG.md "
        "(date, pass/fail counts, notes on failures). ==="
    )


if __name__ == "__main__":
    main()
