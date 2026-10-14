"""Hybrid (dense + sparse) retrieval over the pgvector store, fused with
Reciprocal Rank Fusion (RRF).

No generation, no serving layer here - this module's contract ends at
retrieve() returning ranked chunk results.

Query-side prefixing: per embedder.py's docstring, BGE-M3 does not
require (or want) an instruction prefix for either queries or passages,
unlike earlier BGE v1/v1.5 models. Queries are embedded unmodified here,
consistent with how passages are embedded in embedder.py.

Visibility: every search excludes superseded chunks (is_superseded) and documents
flagged documents.is_restricted (a manual admin toggle), BEFORE scoring so excluded
chunks never consume a candidate_n slot. Nothing else filters results: everyone is
answered from everything that is not restricted. Applicability tags are NOT used for
filtering any more (see docs/applicability_layer_removed.md); they are fetched with each
chunk's details only so generation can say which group a rule applies to.
"""
from functools import lru_cache

from config.channel_settings import channel_setting
from config.labels import compute_label
from config.runtime import get_app_config
from vectorstore.db import get_pool
from vectorstore.embedder import BGEM3Embedder

# Code-level fallbacks only. The live defaults are the app_config rows
# retrieval.top_k / candidate_n / rrf_k (seeded with these same values by
# config/runtime.py); an explicit argument to retrieve() overrides both.
DEFAULT_CANDIDATE_N = 20
DEFAULT_TOP_K = 3
DEFAULT_RRF_K = 60

_embedder = None


def _get_embedder() -> BGEM3Embedder:
    global _embedder
    if _embedder is None:
        _embedder = BGEM3Embedder()
    return _embedder


@lru_cache(maxsize=256)
def embed_query(query: str):
    """Returns (dense_vector, sparse_dict) for a single query string. Memoized per
    process: a turn retrieves twice with the SAME text (the unfiltered applicability
    probe, then the filtered retrieval), and this keeps that to ONE embedding call.
    The returned objects must be treated as read-only."""
    embedder = _get_embedder()
    dense, sparse, _token_counts = embedder.embed([query])
    return dense[0], sparse[0]


@lru_cache(maxsize=64)
def embed_queries(queries: tuple[str, ...]):
    """Returns (dense_vectors, sparse_dicts) for several query strings in ONE batched
    encode call (the advisory main query plus its expansion queries). Memoized on the
    tuple so the probe and the filtered retrieval of one turn share a single encode."""
    dense, sparse, _token_counts = _get_embedder().embed(list(queries))
    return list(dense), sparse


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


def _visibility_filter_sql() -> str:
    """WHERE-clause fragment for a query that already joins chunks AS c and
    documents AS d: restricted documents are never searchable."""
    return "NOT d.is_restricted"


def dense_search(
    conn,
    query_dense,
    model_id: int,
    candidate_n: int = DEFAULT_CANDIDATE_N,
):
    """Cosine-similarity search via the HNSW index. Returns a list of
    (chunk_db_id, similarity) ordered by similarity descending - list
    position gives the dense rank (1-based)."""
    filter_sql = _visibility_filter_sql()
    rows = conn.execute(
        f"""
        SELECT ce.chunk_id, 1 - (ce.dense_vector <=> %s) AS similarity
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE ce.embedding_model_id = %s AND NOT c.is_superseded AND {filter_sql}
        ORDER BY ce.dense_vector <=> %s
        LIMIT %s
        """,
        [query_dense, model_id, query_dense, candidate_n],
    ).fetchall()
    return [(chunk_db_id, float(similarity)) for chunk_db_id, similarity in rows]


def dense_search_many(conn, query_denses: list, model_id: int, limits: list[int]):
    """dense_search() for several query vectors in ONE statement (UNION ALL of per-query HNSW searches), so a turn that
    searches with two queries still makes one dense round trip. limits[i] is candidate_n for query i."""
    filter_sql = _visibility_filter_sql()
    parts, params = [], []
    for i, (vec, limit) in enumerate(zip(query_denses, limits)):
        parts.append(f"""(SELECT {i} AS q, ce.chunk_id, 1 - (ce.dense_vector <=> %s) AS similarity
            FROM chunk_embeddings ce
            JOIN chunks c ON c.id = ce.chunk_id
            JOIN documents d ON d.id = c.document_id
            WHERE ce.embedding_model_id = %s AND NOT c.is_superseded AND {filter_sql}
            ORDER BY ce.dense_vector <=> %s
            LIMIT %s)""")
        params += [vec, model_id, vec, limit]
    rows = conn.execute(" UNION ALL ".join(parts) + " ORDER BY q, similarity DESC", params).fetchall()
    out = [[] for _ in query_denses]
    for q, chunk_db_id, similarity in rows:
        out[q].append((chunk_db_id, float(similarity)))
    return out


def _sparse_dot(query_sparse: dict, chunk_sparse: dict) -> float:
    """Weighted dot product over shared lexical (token-id) weights."""
    if len(query_sparse) > len(chunk_sparse):
        query_sparse, chunk_sparse = chunk_sparse, query_sparse
    return sum(weight * chunk_sparse[tok] for tok, weight in query_sparse.items() if tok in chunk_sparse)


def sparse_search(
    conn,
    query_sparse: dict,
    model_id: int,
    candidate_n: int = DEFAULT_CANDIDATE_N,
):
    """Sparse (lexical) search computed in application code: pulls every
    active chunk's sparse_vector under the active model once, scores each
    against the query in memory, and returns the top candidate_n as
    (chunk_db_id, score) ordered by score descending."""
    filter_sql = _visibility_filter_sql()
    rows = conn.execute(
        f"""
        SELECT ce.chunk_id, ce.sparse_vector
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE ce.embedding_model_id = %s AND NOT c.is_superseded AND {filter_sql}
        """,
        [model_id],
    ).fetchall()

    scored = [
        (chunk_db_id, _sparse_dot(query_sparse, chunk_sparse or {}))
        for chunk_db_id, chunk_sparse in rows
    ]
    scored = [item for item in scored if item[1] > 0]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:candidate_n]


def sparse_search_many(conn, query_sparses: list[dict], model_id: int, limits: list[int]):
    """sparse_search() for several queries over ONE fetch of the chunk sparse vectors
    (the fetch dominates its cost). limits[i] is candidate_n for query i."""
    filter_sql = _visibility_filter_sql()
    rows = conn.execute(
        f"""
        SELECT ce.chunk_id, ce.sparse_vector
        FROM chunk_embeddings ce
        JOIN chunks c ON c.id = ce.chunk_id
        JOIN documents d ON d.id = c.document_id
        WHERE ce.embedding_model_id = %s AND NOT c.is_superseded AND {filter_sql}
        """,
        [model_id],
    ).fetchall()
    results = []
    for query_sparse, limit in zip(query_sparses, limits):
        scored = [(chunk_db_id, _sparse_dot(query_sparse, chunk_sparse or {})) for chunk_db_id, chunk_sparse in rows]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        results.append(scored[:limit])
    return results


def add_expansion_scores(fused, expansion_lists, weight: float, k: int = DEFAULT_RRF_K):
    """Adds weight/(k+rank) for every ranked expansion list (dense and sparse lists of
    each expansion query) to the fused scores, including chunks only an expansion found
    (their dense/sparse rank stays None). Returns the list re-sorted by score."""
    scores = {chunk_db_id: [score, dr, sr] for chunk_db_id, score, dr, sr in fused}
    for ranked in expansion_lists:
        for rank, (chunk_db_id, _s) in enumerate(ranked, start=1):
            entry = scores.setdefault(chunk_db_id, [0.0, None, None])
            entry[0] += weight / (k + rank)
    out = [(chunk_db_id, score, dr, sr) for chunk_db_id, (score, dr, sr) in scores.items()]
    out.sort(key=lambda item: item[1], reverse=True)
    return out


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


# The document's applicability tags as {tag_type: [values]} (JSON), fetched in the SAME query as the chunk
# details (no extra round trip). retrieve() turns them into a label with config.labels.compute_label(), which
# shows a tag type only where the document deviates from the norm (norms come from the cache). NULL when the
# document has no tags.
_TAGS_SQL = """
    (SELECT json_object_agg(g.tag_type, g.vals)
       FROM (SELECT at.tag_type, array_agg(at.tag_value ORDER BY at.tag_value) AS vals
               FROM document_applicability da
               JOIN applicability_tags at ON at.id = da.tag_id
              WHERE da.document_id = d.id
              GROUP BY at.tag_type) g)
"""


def fetch_chunk_details(conn, chunk_db_ids: list) -> dict:
    """Returns {chunk_db_id: (chunk_id, section_path, text, document_title,
    document_id, tags)}. tags is the document's {tag_type: [values]} (None when it has none)."""
    if not chunk_db_ids:
        return {}
    rows = conn.execute(
        f"""
        SELECT c.id, c.chunk_id, c.section_path, c.text, d.document_title, d.id, {_TAGS_SQL} AS tags
        FROM chunks c
        JOIN documents d ON d.id = c.document_id
        WHERE c.id = ANY(%s)
        """,
        (list(chunk_db_ids),),
    ).fetchall()
    return {row[0]: row[1:] for row in rows}


def _diversify(fused, details, top_k: int, max_per_doc: int):
    """Walks the RRF-ordered list keeping at most max_per_doc chunks per document
    until top_k are chosen. details maps chunk_db_id -> (chunk_id, section_path,
    text, document_title, document_id)."""
    per_doc: dict = {}
    chosen = []
    for item in fused:
        document_id = details[item[0]][4]
        if per_doc.get(document_id, 0) >= max_per_doc:
            continue
        per_doc[document_id] = per_doc.get(document_id, 0) + 1
        chosen.append(item)
        if len(chosen) == top_k:
            break
    return chosen


def retrieve(
    query: str,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    user_attributes: dict | None = None,
    channel: str | None = None,
    advisory: bool = False,
    expansions: list[str] | None = None,
    expansion_weight: float | None = None,
    expansion_candidate_n: int | None = None,
    standalone_query: str | None = None,
    standalone_weight: float = 1.0,
    ctx_weight: float = 1.0,
) -> list[dict]:
    """Hybrid dense+sparse retrieval fused with RRF. Returns up to top_k
    results, each: {chunk_id, section_path, text, document_title,
    document_id, fused_score, dense_rank, sparse_rank}.

    user_attributes is accepted and IGNORED (kept so older callers don't break):
    results are not filtered by who is asking. Restricted and superseded content
    is always excluded - see this module's docstring.

    top_k / candidate_n / k default to the app_config values (None = use
    the runtime-configured value; an explicit int always wins). Each
    resolves: explicit argument > <key>.<channel> (when a channel is
    given: "text" or "voice", see config/channel_settings.py) > <key> >
    the module default.

    advisory=True (the user is asking for options, see channels/common.py
    detect_advisory) widens the context: top_k comes from
    retrieval.top_k.advisory.<channel> and, AFTER RRF fusion over the same
    candidate pool (no extra queries), at most retrieval.advisory.max_chunks_per_doc
    chunks per document are kept, so the context can span several policies.

    expansions (advisory turns only; see channels/common.py select_expansions): extra
    queries embedded in ONE batched encode with the main query, searched, and fused in
    with weight retrieval.expansion.weight (RRF), so related policies the main query's
    wording doesn't reach (e.g. work from home, comp-off) can enter the pool. expansion_weight overrides
    retrieval.expansion.weight and expansion_candidate_n retrieval.expansion.candidate_n for this call (the
    affirmation-after-an-offer query fuses the offer sentence with the last user message this way).

    standalone_query (a follow-up-contextualised turn, see channels/common.py plan_turn): `query` is the contextualised
    text and standalone_query the user's bare utterance. Both are encoded in the SAME batched call and searched (one
    dense statement, one sparse fetch); the contextualised lists are fused as usual and scaled by ctx_weight, the
    standalone lists are added with standalone_weight (RRF), so a fresh short question is not drowned by earlier
    unrelated turns while a true follow-up keeps its topic. Without standalone_query the path is exactly the old one."""
    if top_k is None and advisory:
        top_k = channel_setting("retrieval.top_k.advisory", channel)
    if top_k is None:
        top_k = channel_setting("retrieval.top_k", channel, DEFAULT_TOP_K)
    if candidate_n is None:
        candidate_n = channel_setting("retrieval.candidate_n", channel, DEFAULT_CANDIDATE_N)
    if k is None:
        k = channel_setting("retrieval.rrf_k", channel, DEFAULT_RRF_K)
    expansions = list(expansions) if expansions else []
    texts = [query] + ([standalone_query] if standalone_query else []) + expansions
    n_main = 2 if standalone_query else 1  # query vectors before the expansions
    if len(texts) > 1:
        dense_vecs, sparse_vecs = embed_queries(tuple(texts))
        query_dense, query_sparse = dense_vecs[0], sparse_vecs[0]
    else:
        query_dense, query_sparse = embed_query(query)

    pool = get_pool()
    with pool.connection() as conn:
        model_id = get_active_embedding_model_id(conn)
        if standalone_query:
            dense_pair = dense_search_many(conn, dense_vecs[:2], model_id, [candidate_n, candidate_n])
            dense_results = dense_pair[0]
        else:
            dense_results = dense_search(conn, query_dense, model_id, candidate_n)
        if expansions or standalone_query:
            expansion_n = 0
            if expansions:
                expansion_n = expansion_candidate_n if expansion_candidate_n is not None else get_app_config("retrieval.expansion.candidate_n", 5)
            limits = [candidate_n] + ([candidate_n] if standalone_query else []) + [expansion_n] * len(expansions)
            sparse_all = sparse_search_many(conn, sparse_vecs, model_id, limits)
            sparse_results = sparse_all[0]
            expansion_lists = []
            for i, dense_vec in enumerate(dense_vecs[n_main:], start=n_main):
                expansion_lists.append(dense_search(conn, dense_vec, model_id, expansion_n))
                expansion_lists.append(sparse_all[i])
        else:
            sparse_results = sparse_search(conn, query_sparse, model_id, candidate_n)
        fused = rrf_fuse(dense_results, sparse_results, k)
        if standalone_query:
            fused = [(cid, score * ctx_weight, dr, sr) for cid, score, dr, sr in fused]
            fused = add_expansion_scores(fused, [dense_pair[1], sparse_all[1]], standalone_weight, k)
        if expansions:
            fused = add_expansion_scores(fused, expansion_lists,
                                         expansion_weight if expansion_weight is not None else get_app_config("retrieval.expansion.weight", 1.0), k)
        if advisory:
            details = fetch_chunk_details(conn, [chunk_db_id for chunk_db_id, *_ in fused])
            fused = _diversify(fused, details, top_k, channel_setting("retrieval.advisory.max_chunks_per_doc", channel, 2))
        else:
            fused = fused[:top_k]
            details = fetch_chunk_details(conn, [chunk_db_id for chunk_db_id, *_ in fused])

    results = []
    for chunk_db_id, score, dr, sr in fused:
        chunk_id, section_path, text, document_title, document_id, tags = details[chunk_db_id]
        results.append(
            {
                "chunk_id": chunk_id,
                "section_path": section_path,
                "text": text,
                "document_title": document_title,
                "document_id": document_id,
                "applies_to": compute_label(tags),
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
