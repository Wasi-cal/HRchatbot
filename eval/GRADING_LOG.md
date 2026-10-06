# Quality Eval Grading Log

Manual grading history for `eval/quality_eval_set.json`, run via
`python3 -m eval.run_quality_eval`. This is a durable record across
runs - the terminal output from `run_quality_eval.py` is not kept
anywhere, so a result only exists once it's logged here.

Log one entry per grading pass, newest first. Copy the template below.

## How to use this

1. Run `python3 -m eval.run_quality_eval` and read through the report.
2. For each case, decide PASS or FAIL against its `expected_behavior`
   and your own judgment of answer quality (this script does not
   auto-grade anything).
3. Add a new entry below with the date, pass/fail counts, and specific
   notes on any failures (which case id, what was wrong, whether it's a
   retrieval issue, a generation issue, or an access-control issue).

---

## Template

```
## YYYY-MM-DD

- Cases graded: N
- Passed: N
- Failed: N
- Failures:
  - <case id>: <what was wrong>
- Notes: <anything else worth remembering - e.g. corpus changed,
  extraction re-run, model swapped>
```

---

## History

(No grading runs logged yet.)
