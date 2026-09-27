#!/usr/bin/env python3
"""Single-stage LLM extraction of structured applicability tags
(country, employment_type) for every active document.

History / why this is single-stage now
---------------------------------------
This used to be a two-stage pipeline: ingestion's detect_applicability()
(ingestion/structure.py) regex-scanned for a raw "applicable to ..." /
"applicability: ..." fragment, and this script only ran an LLM call on
documents where that regex fired. Diagnosed against the real corpus,
that regex turned out to be both too narrow and too easily truncated:
  - It only matches the literal phrasing "applicable to"/"applicability:",
    never the far more common "This policy applies to ..." construction
    used throughout this corpus - so it silently produced nothing for
    documents that do state a clear scope, e.g. the India referral
    policy's "This policy applies to all full-time employee/ Interns
    and candidates ..." sentence (title: "REFERRAL BONUS PROGRAM -
    INDIA").
  - On the rare block where "applicable to" IS present verbatim, it
    only captured whatever text followed within that SAME block/table
    cell, routinely truncating mid-sentence into noise (e.g. "1. Date
    of purchase").
See ingestion/structure.py's detect_applicability() docstring for the
full diagnosis. It's kept in that file for reference but is no longer
called from the ingestion pipeline (ingestion/pipeline.py always sets
applicability=None now).

This script instead reads directly from the loaded database - title
plus an opening window of chunk text (in document order) for every
active document, no regex pre-filter - and asks the LLM to extract
explicit country/employment_type mentions from that window, the same
"do not guess" constraint as before.

Window size: TOKEN_WINDOW_BUDGET tokens of chunk text (concatenated in
chunk-id / document order, stopping once the running total reaches the
budget). Confirmed sufficient against the known failing case: the India
referral document's applicability-bearing chunk ends at a cumulative
601 tokens from the start of the document, comfortably inside an
800-token budget - and every corpus document that states its scope at
all does so in its introductory objective/scope paragraph, well before
that point.

Usage:
    python3 -m access_control.extract_applicability

Then spot-check the result:
    python3 -m access_control.verify_applicability
"""
import json

from generation.generate import get_client, get_generation_model
from vectorstore.db import close_pool, get_pool

TOKEN_WINDOW_BUDGET = 800

EXTRACTION_SYSTEM_PROMPT = """You extract structured applicability metadata from the opening of an HR policy document (its title plus the start of its body text).

Determine two things:

- country: a list of specific countries this document explicitly restricts applicability to (e.g. ["India"]). Empty list if no country is stated as a restriction - a country appearing only as an example, an office location mentioned in passing, or a currency reference does not count. A country's name appearing ONLY as part of a company/legal-entity name (e.g. "Calfus Technologies India Pvt Ltd") is weaker evidence than an explicit applicability sentence naming the country (e.g. "applies to ... in India", "for Calfus India employees"). You may still report it in that weaker case, but you MUST then set "ambiguous": true.

- employment_type: a list of the SPECIFIC employment categories this document narrows its scope to (e.g. ["full-time employee"], ["full-time employee", "intern"]). Empty list if no such narrowing is stated.
  CRITICAL - read this carefully: an "inclusive" scope statement that enumerates many or all kinds of workers together, to make clear that coverage is BROAD and nobody in that list is excluded (e.g. "applies to all employees, contractors, consultants, and third-party personnel", or a long list like "regular, temporary, ad-hoc, daily wage, contractor, trainees, apprentices, interns, consultants, contract workers, probationers, volunteers" used to ensure a protection policy covers everyone), is NOT a restriction - it is functionally the same as "applies to everyone". Report an EMPTY list for these, even though specific words like "contractor" or "intern" appear in the sentence.
  Only report non-empty values when the text narrows scope DOWN to a smaller group than "everyone associated with the company" - e.g. "This policy applies only to full-time employees", "Eligibility: full-time employees" stated as the sole named eligible group (with other groups implicitly or explicitly excluded, such as a separate "exceptions" list).
  If you are genuinely torn between "inclusive, applies to everyone" and "a real narrowing restriction" for a given passage, give your best-judgment answer and set "ambiguous": true.

Do not guess, infer, or use outside knowledge about the company or its policies - only report what this specific text explicitly states. When you do report a value, use a short, normalized label (e.g. "India" for country; "full-time employee", "contractor", "intern", "consultant", "part-time employee", or "temporary employee" for employment_type when the text's own wording maps cleanly onto one of those - otherwise use a short version of the text's own wording).

Respond with a JSON object with exactly three keys: "country" (array of strings, possibly empty), "employment_type" (array of strings, possibly empty), and "ambiguous" (boolean)."""


def extract_applicability_tags(client, model_name: str, title: str, window_text: str) -> dict:
    user_content = f"Document title: {title}\n\nOpening text:\n{window_text}"
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)

    def _clean_list(value):
        if not isinstance(value, list):
            return []
        return [str(v).strip() for v in value if v and str(v).strip()]

    return {
        "country": _clean_list(data.get("country")),
        "employment_type": _clean_list(data.get("employment_type")),
        "ambiguous": bool(data.get("ambiguous")),
    }


def _get_active_documents(conn):
    return conn.execute(
        "SELECT id, source_file, document_title FROM documents WHERE status = 'active' ORDER BY source_file"
    ).fetchall()


def _get_opening_window_text(conn, document_id, token_budget: int = TOKEN_WINDOW_BUDGET) -> str:
    rows = conn.execute(
        """
        SELECT text, token_count FROM chunks
        WHERE document_id = %s AND NOT is_superseded
        ORDER BY id
        """,
        (document_id,),
    ).fetchall()
    pieces = []
    running = 0
    for text, token_count in rows:
        pieces.append(text)
        running += token_count or 0
        if running >= token_budget:
            break
    return "\n\n".join(pieces)


def _get_or_create_tag(conn, tag_type: str, tag_value: str) -> int:
    row = conn.execute(
        "SELECT id FROM applicability_tags WHERE tag_type = %s AND tag_value = %s",
        (tag_type, tag_value),
    ).fetchone()
    if row:
        return row[0]
    return conn.execute(
        "INSERT INTO applicability_tags (tag_type, tag_value) VALUES (%s, %s) RETURNING id",
        (tag_type, tag_value),
    ).fetchone()[0]


def _link_tag(conn, document_id, tag_id: int):
    conn.execute(
        "INSERT INTO document_applicability (document_id, tag_id) VALUES (%s, %s) "
        "ON CONFLICT DO NOTHING",
        (document_id, tag_id),
    )


def extract_all():
    client = get_client()
    model_name = get_generation_model()
    pool = get_pool()

    results = []
    with pool.connection() as conn:
        for document_id, source_file, title in _get_active_documents(conn):
            window_text = _get_opening_window_text(conn, document_id)
            extracted = extract_applicability_tags(client, model_name, title, window_text)

            inserted = []
            for tag_type in ("country", "employment_type"):
                for value in extracted.get(tag_type, []):
                    tag_id = _get_or_create_tag(conn, tag_type, value)
                    _link_tag(conn, document_id, tag_id)
                    inserted.append((tag_type, value))
            conn.commit()

            results.append(
                {
                    "source_file": source_file,
                    "title": title,
                    "inserted": inserted,
                    "ambiguous": extracted["ambiguous"],
                }
            )
            flag = "  [AMBIGUOUS - spot-check this one]" if extracted["ambiguous"] else ""
            tag_str = inserted if inserted else "no tags extracted"
            print(f"  {source_file} ({title}){flag}")
            print(f"      -> {tag_str}")

    tags_inserted = sum(len(r["inserted"]) for r in results)
    ambiguous_docs = [r["source_file"] for r in results if r["ambiguous"]]
    print("\n=== Extraction Summary ===")
    print(f"Documents processed: {len(results)}")
    print(f"Tag rows inserted: {tags_inserted}")
    print(f"Flagged ambiguous: {len(ambiguous_docs)}{(' - ' + ', '.join(ambiguous_docs)) if ambiguous_docs else ''}")
    close_pool()
    return results


def main():
    extract_all()


if __name__ == "__main__":
    main()
