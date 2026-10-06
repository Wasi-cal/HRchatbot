"""Missing-attribute detection and the ask_and_answer() wrapper.

ask_and_answer() sits on top of vectorstore.retrieve.retrieve() and
generation.generate.generate_answer() without changing either's core
algorithm - it only adds an unfiltered probe call before the real,
filtered call, to decide whether a clarifying question is needed first.
"""
from generation.generate import generate_answer
from vectorstore.db import get_pool
from vectorstore.retrieve import DEFAULT_CANDIDATE_N, DEFAULT_RRF_K, DEFAULT_TOP_K, retrieve

CLARIFYING_QUESTIONS = {
    "country": "Which country are you based in?",
    "employment_type": "Are you a full-time employee or a contractor?",
}


def detect_blocking_tag_types(conn, candidate_results: list[dict], user_attributes: dict | None) -> list[str]:
    """Given chunk results from an UNFILTERED retrieve() call and the
    user's known attributes, returns which tag_type(s) (if any) are
    "blocking": a tag_type for which at least one candidate's document
    carries applicability tags, but user_attributes has no known
    (non-None) value for that tag_type - meaning hard filtering can't
    yet tell whether those candidates should be included or excluded."""
    user_attributes = user_attributes or {}
    document_ids = {r["document_id"] for r in candidate_results}
    if not document_ids:
        return []

    rows = conn.execute(
        """
        SELECT DISTINCT at.tag_type
        FROM document_applicability da
        JOIN applicability_tags at ON at.id = da.tag_id
        WHERE da.document_id = ANY(%s)
        """,
        (list(document_ids),),
    ).fetchall()
    tag_types_present = {row[0] for row in rows}
    return sorted(tag_type for tag_type in tag_types_present if user_attributes.get(tag_type) is None)


def ask_and_answer(
    query: str,
    user_attributes: dict | None = None,
    top_k: int = DEFAULT_TOP_K,
    candidate_n: int = DEFAULT_CANDIDATE_N,
    k: int = DEFAULT_RRF_K,
    confidence_threshold: float | None = None,
) -> dict:
    """Wraps retrieve() + generate_answer() with access-control-aware
    clarification. Returns one of two discriminated shapes:

        {"type": "needs_clarification", "question": str, "blocking_tag_type": str}
            The candidate pool touches applicability-tagged content for
            a tag_type the caller's user_attributes doesn't have a
            known value for. No generation call was made. Re-invoke
            ask_and_answer() with that attribute filled in to proceed.

        {"type": "answer", "answer": str, "shortcut": bool, "timing": dict,
         "chunk_ids": list[str]}
            Same shape as generate_answer()'s final "done" event, with
            hard filtering (via user_attributes) applied throughout.
    """
    # (a) Unfiltered probe: what would the candidate pool look like with
    # no access-control filtering at all? Used only to decide whether a
    # clarifying question is needed - never shown to the user as-is.
    unfiltered_candidates = retrieve(query, top_k=top_k, candidate_n=candidate_n, k=k)

    pool = get_pool()
    with pool.connection() as conn:
        blocking_tag_types = detect_blocking_tag_types(conn, unfiltered_candidates, user_attributes)

    if blocking_tag_types:
        blocking_tag_type = blocking_tag_types[0]
        return {
            "type": "needs_clarification",
            "question": CLARIFYING_QUESTIONS.get(
                blocking_tag_type, f"Could you tell me your {blocking_tag_type.replace('_', ' ')}?"
            ),
            "blocking_tag_type": blocking_tag_type,
        }

    # (b) No blocking attribute: proceed with the real, hard-filtered
    # retrieve() + generate_answer() call.
    final_event = None
    for event in generate_answer(
        query,
        top_k=top_k,
        candidate_n=candidate_n,
        k=k,
        confidence_threshold=confidence_threshold,
        user_attributes=user_attributes,
    ):
        if event["type"] == "done":
            final_event = event

    return {
        "type": "answer",
        "answer": final_event["answer"],
        "shortcut": final_event["shortcut"],
        "timing": final_event["timing"],
        "chunk_ids": final_event.get("chunk_ids", []),
    }
