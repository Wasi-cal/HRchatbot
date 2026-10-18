#!/usr/bin/env python3
"""Post-load verification for the vector store.

    python3 -m vectorstore.verify "what is the parental leave policy?"

Checks:
  a. Every non-superseded chunk has exactly one chunk_embeddings row under
     the currently active embedding model.
  b. Embeds the given (or default) query string and runs a pgvector
     cosine-similarity search against dense_vector, printing the top 5
     matches for manual sanity-checking.
  c. Prints row counts for every table.
"""
import sys

from vectorstore.db import close_pool, get_pool
from vectorstore.embedder import BGEM3Embedder

DEFAULT_QUERY = "what is the parental leave policy?"

TABLES = [
    "documents", "applicability_tags", "document_applicability",
    "chunks", "embedding_models", "chunk_embeddings", "provider_configs",
]


def check_embedding_coverage(conn):
    active_model = conn.execute(
        "SELECT id, name, version FROM embedding_models em "
        "WHERE EXISTS (SELECT 1 FROM provider_configs pc "
        "              WHERE pc.category = 'embedding' AND pc.is_active "
        "              AND (pc.config->>'model_id')::int = em.id)"
    ).fetchone()
    if not active_model:
        print("No active embedding model found in provider_configs.")
        return

    model_id, name, version = active_model
    print(f"Active embedding model: {name} ({version}), id={model_id}")

    missing = conn.execute(
        """
        SELECT count(*) FROM chunks c
        WHERE NOT c.is_superseded
          AND NOT EXISTS (
            SELECT 1 FROM chunk_embeddings ce
            WHERE ce.chunk_id = c.id AND ce.embedding_model_id = %s
          )
        """,
        (model_id,),
    ).fetchone()[0]

    total_active_chunks = conn.execute(
        "SELECT count(*) FROM chunks WHERE NOT is_superseded"
    ).fetchone()[0]

    duplicates = conn.execute(
        """
        SELECT count(*) FROM (
            SELECT chunk_id FROM chunk_embeddings
            WHERE embedding_model_id = %s
            GROUP BY chunk_id HAVING count(*) > 1
        ) dup
        """,
        (model_id,),
    ).fetchone()[0]

    print(f"Active (non-superseded) chunks: {total_active_chunks}")
    print(f"Chunks missing an embedding under the active model: {missing}")
    print(f"chunk_id values with >1 embedding row under the active model: {duplicates}")
    if missing == 0 and duplicates == 0:
        print("OK: every active chunk has exactly one embedding under the active model.")
    return model_id


def run_test_query(conn, query: str, embedder: BGEM3Embedder, model_id: int, top_k: int = 5):
    dense, _sparse, _tokens = embedder.embed([query])
    query_vec = dense[0]

    rows = conn.execute(
        """
        SELECT c.chunk_id, c.section_path, c.text, 1 - (ce.dense_vector <=> %s) AS similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        WHERE ce.embedding_model_id = %s AND NOT c.is_superseded
        ORDER BY ce.dense_vector <=> %s
        LIMIT %s
        """,
        (query_vec, model_id, query_vec, top_k),
    ).fetchall()

    print(f"\nTop {top_k} matches for query: {query!r}")
    for chunk_id, section_path, text, similarity in rows:
        snippet = text[:150].replace("\n", " ")
        print(f"  [{similarity:.4f}] {chunk_id} | {section_path}")
        print(f"          {snippet}...")


def print_row_counts(conn):
    print("\n=== Row counts ===")
    for table in TABLES:
        count = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count}")


def main():
    query = " ".join(sys.argv[1:]) or DEFAULT_QUERY
    pool = get_pool()
    embedder = BGEM3Embedder()

    with pool.connection() as conn:
        print("=== Embedding coverage ===")
        model_id = check_embedding_coverage(conn)

        if model_id is not None:
            run_test_query(conn, query, embedder, model_id)

        print_row_counts(conn)
    close_pool()


if __name__ == "__main__":
    main()
