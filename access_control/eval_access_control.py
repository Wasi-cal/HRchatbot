#!/usr/bin/env python3
"""Exercises ask_and_answer() and retrieve()'s hard filtering under the
four scenarios from this task's spec: no user_attributes, an incomplete
user_attributes that should trigger a clarifying question, a fully
matching user_attributes, and a fully provided but NON-matching
user_attributes (to confirm exclusion actually happens).

Demo fixtures, not real extracted data
---------------------------------------
access_control.extract_applicability (task 1) found zero usable tags in
this real corpus - the only two documents with a raw applicability
fragment ("All IT Policies" and the India employee handbook) don't name
a specific country or employment type in that fragment (confirmed via
access_control.verify_applicability). Extraction only ever looks at
that narrow ingestion-level raw field, by design, so it never gets a
chance to look at the rest of a document's text.

To exercise hard filtering against something more meaningful than an
empty tag table, this script manually seeds two tags directly grounded
in real corpus content that extraction's narrow scope doesn't reach:
  - "Calfus Crew - Employee Referral Policy.pdf" is titled "REFERRAL
    BONUS PROGRAM - INDIA" and its body text scopes the policy to
    "permanent employees" - both explicit statements in the source
    document, just not in its ingestion-level applicability field.
  - Its near-duplicate "Calfus Crew - Employee Referral Policy_1.2.pdf"
    is deliberately left untagged, to also demonstrate that untagged
    documents keep applying to everyone even when a similar document
    IS tagged.

These are clearly-labeled fixtures for THIS SCRIPT only - not a claim
that extract_applicability.py produced them.

    python3 -m access_control.eval_access_control
"""
from access_control.ask_and_answer import ask_and_answer
from vectorstore.db import close_pool, get_pool
from vectorstore.retrieve import retrieve

TAGGED_SOURCE_FILE = "Calfus Crew - Employee Referral Policy.pdf"
DEMO_TAGS = [
    (TAGGED_SOURCE_FILE, "country", "India"),
    (TAGGED_SOURCE_FILE, "employment_type", "permanent employee"),
]

# Touches the tagged referral policy document (see DEMO_TAGS above).
TAGGED_QUERY = "will I get paid extra for recommending a friend for a job here"
# Untagged content, used as a control to confirm user_attributes has no
# effect at all when nothing in the candidate pool carries tags.
UNTAGGED_QUERY = "PIP objective"


def _seed_demo_tags(conn):
    for source_file, tag_type, tag_value in DEMO_TAGS:
        doc_row = conn.execute(
            "SELECT id FROM documents WHERE source_file = %s AND status = 'active'",
            (source_file,),
        ).fetchone()
        if not doc_row:
            print(f"  WARNING: {source_file!r} not found - skipping seed tag {tag_type}={tag_value}")
            continue
        document_id = doc_row[0]

        tag_row = conn.execute(
            "SELECT id FROM applicability_tags WHERE tag_type = %s AND tag_value = %s",
            (tag_type, tag_value),
        ).fetchone()
        tag_id = tag_row[0] if tag_row else conn.execute(
            "INSERT INTO applicability_tags (tag_type, tag_value) VALUES (%s, %s) RETURNING id",
            (tag_type, tag_value),
        ).fetchone()[0]

        conn.execute(
            "INSERT INTO document_applicability (document_id, tag_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (document_id, tag_id),
        )
    conn.commit()
    print(f"Seeded demo tags on {TAGGED_SOURCE_FILE!r}: {[(t, v) for _, t, v in DEMO_TAGS]}\n")


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
    pool = get_pool()
    with pool.connection() as conn:
        _seed_demo_tags(conn)

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
        {"country": "India", "employment_type": "permanent employee"},
    )

    print("############ Scenario (d): user_attributes fully provided, NOT matching ############")
    run_scenario(
        "Tagged-content query, country known but WRONG -> tagged doc excluded from context",
        TAGGED_QUERY,
        {"country": "United States", "employment_type": "permanent employee"},
    )

    print("############ Control: untagged-content query, attributes should have zero effect ############")
    run_scenario("Untagged query with attributes provided (no tags exist to filter on)", UNTAGGED_QUERY, {
        "country": "United States",
        "employment_type": "contractor",
    })
    run_scenario("Same untagged query, no attributes (should match the above exactly)", UNTAGGED_QUERY, None)

    close_pool()


if __name__ == "__main__":
    main()
