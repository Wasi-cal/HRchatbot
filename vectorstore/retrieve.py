"""Hybrid (dense + sparse) retrieval over the pgvector store, fused with
Reciprocal Rank Fusion (RRF).

No generation, no serving layer here - this module's contract ends at
retrieve() returning ranked chunk results.

Query-side prefixing: per embedder.py's docstring, BGE-M3 does not
require (or want) an instruction prefix for either queries or passages,
unlike earlier BGE v1/v1.5 models. Queries are embedded unmodified here,
consistent with how passages are embedded in embedder.py.
"""
from vectorstore.db import get_pool
from vectorstore.embedder import BGEM3Embedder

DEFAULT_CANDIDATE_N = 20
DEFAULT_TOP_K = 3
DEFAULT_RRF_K = 60

_embedder = None


def _get_embedder() -> BGEM3Embedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEM3Embedder()
    return _embedder


def embed_query(query: str):
    """Returns (dense_vector, sparse_dict) for a single query string."""
    embedder = _get_embedder()
    dense, sparse, _token_counts = embedder.embed([query])
    return dense[0], sparse[0]


def get_active_embedding_model_id(conn) -> int:
    row = conn.execute(
        "SELECT id FROM embedding_models em "
        "WHERE EXISTS (SELECT 1 FROM provider_configs pc "
        "              WHERE pc.category = 'embedding' AND pc.is_active "
        "              AND (pc.config->>'model_id')::int = em.id)"
    ).fetchone()
    if not row:
        raise RuntimeError("No active embedding model found in provider_configs.")
    return row[0]


def dense_search(conn, query_dense, model_id: int, candidate_n: int = DEFAULT_CANDIDATE_N):
    """Cosine-similarity search via the HNSW index. Returns a list of
    (chunk_db_id, similarity) ordered by similarity descending - list
    position gives the dense rank (1-based)."""
    rows = conn.execute(
        """
        SELECT ce.chunk_id, 1 - (ce.dense_vector <=> %s) AS similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        WHERE ce.embedding_model_id = %s AND NOT c.is_superseded
        ORDER BY ce.dense_vector <=> %s
        LIMIT %s
        """,
        (query_dense, model_id, query_dense, candidate_n),
    ).fetchall()
    return [(chunk_db_id, float(similarity)) for chunk_db_id, similarity in rows]


def _sparse_dot(query_sparse: dict, chunk_sparse: dict) -> float:
    """Weighted dot product over shared lexical (token-id) weights."""
    if len(query_sparse) > len(chunk_sparse):
        query_sparse, chunk_sparse = chunk_sparse, query_sparse
    return sum(weight * chunk_sparse[tok] for tok, weight in query_sparse.items() if tok in chunk_sparse)


def sparse_search(conn, query_sparse: dict, model_id: int, candidate_n: int = DEFAULT_CANDIDATE_N):
    """Sparse (lexical) search computed in application code: pulls every
    active chunk's sparse_vector under the active model once, scores each
    against the query in memory, and returns the top candidate_n as
    (chunk_db_id, score) ordered by score descending."""
    rows = conn.execute(
        """
        SELECT ce.chunk_id, ce.sparse_vector
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        WHERE ce.embedding_model_id = %s AND NOT c.is_superseded
        """,
        (model_id,),
    ).fetchall()

    scored = [
        (chunk_db_id, _sparse_dot(query_sparse, chunk_sparse or {}))
        for chunk_db_id, chunk_sparse in rows
    ]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:candidate_n]


def rrf_fuse(dense_results, sparse_results, k: int = DEFAULT_RRF_K):
    """Reciprocal Rank Fusion. A chunk absent from one list contributes 0
    for that term (no penalty, just no contribution). Returns a list of
    (chunk_db_id, fused_score, dense_rank_or_None, sparse_rank_or_None)
    sorted by fused_score descending."""
    dense_rank = {chunk_db_id: rank for rank, (chunk_db_id, _score) in enumerate(dense_results, start=1)}
    sparse_rank = {chunk_db_id: rank for rank, (chunk_db_id, _score) in enumerate(sparse_results, start=1)}

    fused = []
    for chunk_db_id in dense_rank.keys() | sparse_rank.keys():
        dr = dense_rank.get(chunk_db_id)
        sr = sparse_rank.get(chunk_db_id)
        score = (1.0 / (k + dr) if dr is not None else 0.0) + (1.0 / (k + sr) if sr is not None else 0.0)
        fused.append((chunk_db_id, score, dr, sr))

    fused.sort(key=lambda item: item[1], reverse=True)
    return fused


def fetch_chunk_details(conn, chunk_db_ids: list) -> dict:
    """Returns {chunk_db_id: (chunk_id, section_path, text, document_title)}."""
    if not chunk_db_ids:
        return {}
    rows = conn.execute(
        """
        SELECT c.id, c.chunk_id, c.section_path, c.text, d.document_title
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY(%s)
        """,
        (list(chunk_db_ids),),
    ).fetchall()
    return {row[0]: row[1:] for row in rows}


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    candidate_n: int = DEFAULT_CANDIDATE_N,
    k: int = DEFAULT_RRF_K,
) -> list[dict]:
    """Hybrid dense+sparse retrieval fused with RRF. Returns up to top_k
    results, each: {chunk_id, section_path, text, document_title,
    fused_score, dense_rank, sparse_rank}."""
    query_dense, query_sparse = embed_query(query)

    pool = get_pool()
    with pool.connection() as conn:
        model_id = get_active_embedding_model_id(conn)
        dense_results = dense_search(conn, query_dense, model_id, candidate_n)
        sparse_results = sparse_search(conn, query_sparse, model_id, candidate_n)
        fused = rrf_fuse(dense_results, sparse_results, k)[:top_k]
        details = fetch_chunk_details(conn, [chunk_db_id for chunk_db_id, *_ in fused])

    results = []
    for chunk_db_id, score, dr, sr in fused:
        chunk_id, section_path, text, document_title = details[chunk_db_id]
        results.append(
            {
                "chunk_id": chunk_id,
                "section_path": section_path,
                "text": text,
                "document_title": document_title,
                "fused_score": score,
                "dense_rank": dr,
                "sparse_rank": sr,
            }
        )
    return results


def main():
    import sys

    query = " ".join(sys.argv[1:]) or "what is the parental leave policy?"
    for result in retrieve(query):
        snippet = result["text"][:150].replace("\n", " ")
        print(f"[{result['fused_score']:.4f}] (dense={result['dense_rank']}, sparse={result['sparse_rank']}) "
              f"{result['chunk_id']} | {result['section_path']}")
        print(f"    {snippet}...")


if __name__ == "__main__":
    main()
