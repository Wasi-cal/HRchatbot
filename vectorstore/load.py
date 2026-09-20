#!/usr/bin/env python3
"""Load ingestion output (chunk JSON files) into the vector store.

For each document JSON:
  - Upsert into `documents`, matched on source_file. If an active document
    already exists for that source_file and its chunk content is
    unchanged, the document is skipped entirely (idempotent re-runs). If
    the content differs, the existing document's chunks are marked
    is_superseded = true (and the document status flipped to
    'superseded') rather than deleted, and the new document + chunks are
    inserted as fresh rows.
  - Applicability tags are only inserted when the ingestion JSON's
    `applicability` field is structured (keys other than the current
    ingestion output's raw-text shape {"raw": "..."}) - seeing only a raw
    string with no declared tag_type is not something this loader guesses
    at (see README note in this file's module docstring for the gap this
    leaves, until ingestion emits structured tags).
  - Chunks and their dense+sparse embeddings (via vectorstore.embedder)
    are inserted for the new document.

Usage:
    python3 -m vectorstore.load [ingestion_output_dir]
"""
import json
import re
import sys
from pathlib import Path

from psycopg.types.json import Jsonb

from vectorstore.db import close_pool, get_pool
from vectorstore.embedder import BGEM3Embedder, ensure_embedding_model_registered

EMBED_BATCH_SIZE = 12

# A conservative date-shaped pattern: DD-Mon-YYYY, DD/MM/YYYY, YYYY-MM-DD,
# etc. Anything not matching this is treated as a version label instead
# (see _split_effective_date_version).
_DATE_LIKE_RE = re.compile(
    r"\d{1,2}[-/][A-Za-z]{3,9}[-/]\d{2,4}|\d{4}-\d{2}-\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
)


def _split_effective_date_version(raw: str | None):
    """The ingestion pipeline emits a single conflated 'effective_date'
    field (it can hold either an actual date or a version number, e.g.
    "1.0"), but this schema has separate effective_date/version columns.
    Split on whether the value looks date-shaped."""
    if not raw:
        return None, None
    if _DATE_LIKE_RE.search(raw):
        return raw, None
    return None, raw


def _load_document_jsons(output_dir: Path):
    for path in sorted(output_dir.glob("*.json")):
        if path.name == "ingestion_summary.json":
            continue
        yield path, json.loads(path.read_text(encoding="utf-8"))


def _existing_active_document(conn, source_file: str):
    return conn.execute(
        "SELECT id FROM documents WHERE source_file = %s AND status = 'active'",
        (source_file,),
    ).fetchone()


def _existing_chunk_fingerprint(conn, document_id) -> set:
    rows = conn.execute(
        "SELECT chunk_id, content_hash FROM chunks WHERE document_id = %s AND NOT is_superseded",
        (document_id,),
    ).fetchall()
    return {(r[0], r[1]) for r in rows}


def _supersede_document(conn, document_id):
    conn.execute("UPDATE documents SET status = 'superseded', updated_at = now() WHERE id = %s", (document_id,))
    conn.execute("UPDATE chunks SET is_superseded = true WHERE document_id = %s", (document_id,))


def _insert_document(conn, doc_json: dict) -> str:
    effective_date, version = _split_effective_date_version(doc_json.get("effective_date"))
    row = conn.execute(
        """
        INSERT INTO documents (source_file, document_title, effective_date, version, status)
        VALUES (%s, %s, %s, %s, 'active')
        RETURNING id
        """,
        (doc_json["source_file"], doc_json.get("document_title"), effective_date, version),
    ).fetchone()
    return row[0]


def _insert_applicability(conn, document_id, applicability: dict | None):
    if not applicability:
        return
    # Today's ingestion output shape is {"raw": "..."} - an unstructured
    # string, not tag_type/tag_value pairs. Inserting it under a
    # fabricated tag_type like "raw" would be guessing at a category the
    # source document never explicitly declared, which the spec forbids.
    # Only structured shapes (any key other than "raw") are loaded.
    tag_pairs = []
    for tag_type, value in applicability.items():
        if tag_type == "raw":
            continue
        values = value if isinstance(value, list) else [value]
        for v in values:
            if v:
                tag_pairs.append((tag_type, str(v)))

    for tag_type, tag_value in tag_pairs:
        row = conn.execute(
            "SELECT id FROM applicability_tags WHERE tag_type = %s AND tag_value = %s",
            (tag_type, tag_value),
        ).fetchone()
        if row:
            tag_id = row[0]
        else:
            tag_id = conn.execute(
                "INSERT INTO applicability_tags (tag_type, tag_value) VALUES (%s, %s) RETURNING id",
                (tag_type, tag_value),
            ).fetchone()[0]
        conn.execute(
            "INSERT INTO document_applicability (document_id, tag_id) VALUES (%s, %s) "
            "ON CONFLICT DO NOTHING",
            (document_id, tag_id),
        )


def _insert_chunks_and_embeddings(conn, document_id, chunks: list[dict], embedder: BGEM3Embedder, model_id: int):
    chunk_db_ids = []
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i:i + EMBED_BATCH_SIZE]
        texts = [c["text"] for c in batch]
        dense, sparse, token_counts = embedder.embed(texts)

        for c, dense_vec, sparse_dict, token_count in zip(batch, dense, sparse, token_counts):
            row = conn.execute(
                """
                INSERT INTO chunks (
                    document_id, chunk_id, section_path, chunk_type, source_type,
                    ocr_confidence, page_number, content_hash, text, token_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    document_id, c["chunk_id"], c.get("section_path"), c["chunk_type"],
                    c["source_type"], c.get("ocr_confidence"), c.get("page_number"),
                    c["content_hash"], c["text"], token_count,
                ),
            ).fetchone()
            chunk_db_id = row[0]
            chunk_db_ids.append(chunk_db_id)

            conn.execute(
                """
                INSERT INTO chunk_embeddings (chunk_id, embedding_model_id, dense_vector, sparse_vector)
                VALUES (%s, %s, %s, %s)
                """,
                (chunk_db_id, model_id, dense_vec, Jsonb(sparse_dict)),
            )
    return chunk_db_ids


def load_all(output_dir: Path):
    pool = get_pool()
    embedder = BGEM3Embedder()

    with pool.connection() as conn:
        model_id = ensure_embedding_model_registered(conn, embedder)
        conn.commit()

    stats = {"inserted": 0, "superseded": 0, "unchanged": 0, "errors": []}

    for path, doc_json in _load_document_jsons(output_dir):
        try:
            with pool.connection() as conn:
                new_fingerprint = {(c["chunk_id"], c["content_hash"]) for c in doc_json["chunks"]}
                existing = _existing_active_document(conn, doc_json["source_file"])

                if existing:
                    existing_id = existing[0]
                    if _existing_chunk_fingerprint(conn, existing_id) == new_fingerprint:
                        print(f"  {path.name}: unchanged, skipping")
                        stats["unchanged"] += 1
                        conn.commit()
                        continue
                    _supersede_document(conn, existing_id)
                    stats["superseded"] += 1

                document_id = _insert_document(conn, doc_json)
                _insert_applicability(conn, document_id, doc_json.get("applicability"))
                chunk_ids = _insert_chunks_and_embeddings(conn, document_id, doc_json["chunks"], embedder, model_id)
                conn.commit()
                print(f"  {path.name}: loaded {len(chunk_ids)} chunks")
                stats["inserted"] += 1
        except Exception as exc:
            stats["errors"].append((path.name, str(exc)))
            print(f"  {path.name}: FAILED - {exc}")

    return stats


def main():
    output_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        __import__("os").environ.get("INGESTION_OUTPUT_DIR", "out_chunks")
    )
    if not output_dir.is_dir():
        print(f"Ingestion output folder not found: {output_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading from {output_dir} ...")
    stats = load_all(output_dir)

    print("\n=== Load Summary ===")
    print(f"Documents inserted (new or superseding): {stats['inserted']}")
    print(f"Documents superseded: {stats['superseded']}")
    print(f"Documents unchanged (skipped): {stats['unchanged']}")
    if stats["errors"]:
        print(f"Errors: {len(stats['errors'])}")
        for name, err in stats["errors"]:
            print(f"  - {name}: {err}")
    close_pool()


if __name__ == "__main__":
    main()
