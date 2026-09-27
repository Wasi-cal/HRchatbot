#!/usr/bin/env python3
"""LLM extraction of structured applicability tags (country,
employment_type) from the raw, unstructured applicability text captured
during ingestion.

Ingestion's detect_applicability() (ingestion/structure.py) only ever
produces a document-level {"raw": "..."} string when it finds a regex
match on an "applicability"/"applicable to" style sentence - it never
writes structured tag_type/tag_value pairs itself (see the note in
vectorstore/load.py's module docstring on why loader deliberately skips
this raw shape). This script closes that gap: for every ingestion
output document that has a raw applicability string, it asks the active
generation model to pull out an explicit country and/or employment_type
mention - nothing invented or inferred - and inserts what it finds into
applicability_tags / document_applicability.

Usage:
    python3 -m access_control.extract_applicability [ingestion_output_dir]

Then spot-check the result:
    python3 -m access_control.verify_applicability
"""
import json
import os
import sys
from pathlib import Path

from vectorstore.db import close_pool, get_pool
from vectorstore.load import _load_document_jsons
from generation.generate import get_client, get_generation_model

EXTRACTION_SYSTEM_PROMPT = """You extract structured applicability metadata from a short fragment of HR policy text.

Given the fragment, determine two things, each reported only if the fragment explicitly says so:
- country: the specific country this text restricts applicability to (e.g. "India", "United States"). Null if no country is named.
- employment_type: the specific employment type this text restricts applicability to (e.g. "contractor", "full-time employee", "intern"). Null if no specific employment type is named as a restriction - in particular, phrasing like "applies to all employees" is NOT an employment_type restriction and should be null.

Do not guess, infer, or use outside knowledge about the company or its policies. Only report what this exact fragment explicitly states. Respond with a JSON object with exactly two keys, "country" and "employment_type", each either a short string or null."""


def extract_applicability_tags(client, model_name: str, raw_text: str) -> dict:
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": raw_text},
        ],
        response_format={"type": "json_object"},
        temperature=0,
    )
    data = json.loads(response.choices[0].message.content)
    return {
        "country": data.get("country") or None,
        "employment_type": data.get("employment_type") or None,
    }


def _get_active_document_id(conn, source_file: str):
    row = conn.execute(
        "SELECT id FROM documents WHERE source_file = %s AND status = 'active'",
        (source_file,),
    ).fetchone()
    return row[0] if row else None


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


def extract_all(output_dir: Path):
    client = get_client()
    model_name = get_generation_model()
    pool = get_pool()

    processed, no_document, no_tags_found, tags_inserted = 0, 0, 0, 0

    with pool.connection() as conn:
        for path, doc_json in _load_document_jsons(output_dir):
            applicability = doc_json.get("applicability")
            if not isinstance(applicability, dict) or "raw" not in applicability:
                continue
            processed += 1
            raw_text = applicability["raw"]

            document_id = _get_active_document_id(conn, doc_json["source_file"])
            if document_id is None:
                no_document += 1
                print(f"  {path.name}: no active document row found in DB - skipping")
                continue

            extracted = extract_applicability_tags(client, model_name, raw_text)
            inserted = []
            for tag_type in ("country", "employment_type"):
                value = extracted.get(tag_type)
                if value:
                    tag_id = _get_or_create_tag(conn, tag_type, value)
                    _link_tag(conn, document_id, tag_id)
                    inserted.append((tag_type, value))
            conn.commit()

            if inserted:
                tags_inserted += len(inserted)
            else:
                no_tags_found += 1
            print(f"  {path.name}: raw={raw_text!r}")
            print(f"      -> {inserted if inserted else 'no country/employment_type stated - no tags inserted'}")

    print("\n=== Extraction Summary ===")
    print(f"Documents with a raw applicability string: {processed}")
    print(f"  no matching active document row: {no_document}")
    print(f"  no tags extracted (nothing explicit stated): {no_tags_found}")
    print(f"Tag rows inserted: {tags_inserted}")
    close_pool()


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        os.environ.get("INGESTION_OUTPUT_DIR", "out_chunks")
    )
    if not output_dir.is_dir():
        print(f"Ingestion output folder not found: {output_dir}", file=sys.stderr)
        sys.exit(1)
    extract_all(output_dir)


if __name__ == "__main__":
    main()
