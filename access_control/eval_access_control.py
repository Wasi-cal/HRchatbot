#!/usr/bin/env python3
"""Exercises ask_and_answer() and retrieve()'s hard filtering under the
four scenarios from this task's spec: no user_attributes, an incomplete
user_attributes that should trigger a clarifying question, a fully
matching user_attributes, and a fully provided but NON-matching
user_attributes (to confirm exclusion actually happens).

Now backed by real extracted data, not demo fixtures
------------------------------------------------------
An earlier version of this script had to seed manual demo tags, because
the old regex-gated extraction pipeline found nothing real to filter
on. access_control/extract_applicability.py is now a single-stage LLM
extraction over every document's title + opening text (see that
module's docstring for why), and re-running it against this corpus
produced real tags on most documents - including
"Calfus Crew - Employee Referral Policy.pdf" (country=India,
employment_type=full-time employee / intern), which this script uses
below. No fixtures are seeded here anymore; run
access_control.extract_applicability first if the tag tables are
empty.

Its near-duplicate "Calfus Crew - Employee Referral Policy_1.2.pdf" got
the SAME real tags (both files really do contain the same India /
full-time-or-intern scoping language), so scenario (d) below correctly
excludes both duplicates when the mismatch is on country - unlike the
old demo-fixture version, where only one copy was tagged. The control
query intentionally targets "All IT Policies.pdf" instead of a PIP or
POSH query, since those are now genuinely tagged too (real signal, not
a demo artifact) and would no longer make a valid "untagged" control.

    python3 -m access_control.eval_access_control
"""
from access_control.ask_and_answer import ask_and_answer
from vectorstore.retrieve import retrieve

# Touches "Calfus Crew - Employee Referral Policy.pdf" /
# "..._1.2.pdf", both real-tagged: country=India,
# employment_type=full-time employee, employment_type=intern.
TAGGED_QUERY = "will I get paid extra for recommending a friend for a job here"
# "All IT Policies.pdf" has no applicability tags at all (see
# access_control.verify_applicability) - a genuine untagged control.
UNTAGGED_QUERY = "what are the password requirements"


def _print_unfiltered_vs_filtered(query: str, user_attributes: dict | None):
    unfiltered = retrieve(query, top_k=5)
    filtered = retrieve(query, top_k=5, user_attributes=user_attributes)
    unfiltered_ids = {r["chunk_id"] for r in unfiltered}
    filtered_ids = {r["chunk_id"] for r in filtered}
    excluded = unfiltered_ids - filtered_ids
    print(f"  Unfiltered top-5 chunk_ids: {sorted(unfiltered_ids)}")
    print(f"  Filtered top-5 chunk_ids:   {sorted(filtered_ids)}")
    print(f"  Excluded by hard filtering: {sorted(excluded) if excluded else '(none)'}")


def run_scenario(label: str, query: str, user_attributes: dict | None):
    print(f"--- {label} ---")
    print(f"  query: {query!r}")
    print(f"  user_attributes: {user_attributes!r}")
    _print_unfiltered_vs_filtered(query, user_attributes)

    result = ask_and_answer(query, user_attributes=user_attributes)
    if result["type"] == "needs_clarification":
        print(f"  ask_and_answer -> NEEDS CLARIFICATION: {result['question']!r} "
              f"(blocking_tag_type={result['blocking_tag_type']!r})")
    else:
        print(f"  ask_and_answer -> ANSWER: {result['answer']!r}")
    print()


def main():
    print("############ Scenario (a): no user_attributes at all ############")
    run_scenario("Tagged-content query, no attributes (backward-compatible: no filtering)", TAGGED_QUERY, None)

    print("############ Scenario (b): user_attributes missing the relevant field ############")
    run_scenario(
        "Tagged-content query, country known but employment_type unknown -> should ask",
        TAGGED_QUERY,
        {"country": "India", "employment_type": None},
    )

    print("############ Scenario (c): user_attributes fully provided, matching a real tag ############")
    run_scenario(
        "Tagged-content query, both attributes known and matching -> normal answer, doc included",
        TAGGED_QUERY,
        {"country": "India", "employment_type": "full-time employee"},
    )

    print("############ Scenario (d): user_attributes fully provided, NOT matching ############")
    run_scenario(
        "Tagged-content query, country known but WRONG -> both tagged duplicates excluded from context",
        TAGGED_QUERY,
        {"country": "United States", "employment_type": "full-time employee"},
    )

    print("############ Control: untagged-content query, attributes should have zero effect ############")
    run_scenario("Untagged query with attributes provided (no tags exist to filter on)", UNTAGGED_QUERY, {
        "country": "United States",
        "employment_type": "contractor",
    })
    run_scenario("Same untagged query, no attributes (should match the above exactly)", UNTAGGED_QUERY, None)


if __name__ == "__main__":
    main()
