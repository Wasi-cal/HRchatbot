#!/usr/bin/env python3
"""Side-by-side comparison of dense-only, sparse-only, and RRF-fused
retrieval against a small hardcoded set of test queries.

Not an auto-judge: this just prints the three result sets so a human who
knows the corpus can eyeball whether fusion pulls in anything either
method alone would have missed.

    python3 -m vectorstore.eval_retrieval
"""
from vectorstore.db import close_pool, get_pool
from vectorstore.retrieve import (
    dense_search,
    embed_query,
    fetch_chunk_details,
    get_active_embedding_model_id,
    rrf_fuse,
    sparse_search,
)

# (a) Paraphrase-style: no keyword overlap with the source policy wording,
# grounded in real topics in this corpus (work-from-home, maternity/
# paternity leave, notice period, referral bonuses - see
# Calfus_India_Employee_Handbook_v4 and the referral policy docs).
PARAPHRASE_QUERIES = [
    "can I do my job from home instead of coming into the office",
    "how much time off do I get when I have a baby",
    "if I quit, how long do I have to keep working before I can leave for good",
    "will I get paid extra for recommending a friend for a job here",
]

# (b) Exact-term/acronym-style: specific terms verified present verbatim
# in the corpus (section headers / policy names as chunked).
EXACT_TERM_QUERIES = [
    "POSH redressal mechanism",
    "PIP objective",
    "BGV disclaimer",
    "90 calendar days notice period buyout",
]

TOP_N_DISPLAY = 3
CANDIDATE_N = 20
RRF_K = 60


def _snippet(text: str, length: int = 120) -> str:
    return text[:length].replace("\n", " ")


def _print_ranked(label: str, ranked_chunk_db_ids, details: dict):
    print(f"  -- {label} --")
    if not ranked_chunk_db_ids:
        print("     (no results)")
        return
    for rank, chunk_db_id in enumerate(ranked_chunk_db_ids[:TOP_N_DISPLAY], start=1):
        chunk_id, section_path, text, _document_title, _document_id = details[chunk_db_id]
        print(f"     {rank}. {chunk_id} | {section_path}")
        print(f"        {_snippet(text)}...")


def compare(conn, model_id: int, query: str):
    print(f"\n=== Query: {query!r} ===")

    query_dense, query_sparse = embed_query(query)
    dense_results = dense_search(conn, query_dense, model_id, CANDIDATE_N)
    sparse_results = sparse_search(conn, query_sparse, model_id, CANDIDATE_N)
    fused = rrf_fuse(dense_results, sparse_results, RRF_K)

    dense_ids = [chunk_db_id for chunk_db_id, _score in dense_results]
    sparse_ids = [chunk_db_id for chunk_db_id, _score in sparse_results]
    fused_ids = [chunk_db_id for chunk_db_id, *_rest in fused]

    all_ids = set(dense_ids[:TOP_N_DISPLAY]) | set(sparse_ids[:TOP_N_DISPLAY]) | set(fused_ids[:TOP_N_DISPLAY])
    details = fetch_chunk_details(conn, list(all_ids))

    _print_ranked("Dense-only top-3", dense_ids, details)
    _print_ranked("Sparse-only top-3", sparse_ids, details)
    _print_ranked("Fused (RRF) top-3", fused_ids, details)


def main():
    pool = get_pool()
    with pool.connection() as conn:
        model_id = get_active_embedding_model_id(conn)

        print("############ Paraphrase-style queries (no keyword overlap) ############")
        for query in PARAPHRASE_QUERIES:
            compare(conn, model_id, query)

        print("\n############ Exact-term / acronym-style queries ############")
        for query in EXACT_TERM_QUERIES:
            compare(conn, model_id, query)
    close_pool()


if __name__ == "__main__":
    main()
