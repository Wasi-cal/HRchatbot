"""Group labels on retrieved chunks, shown ONLY WHEN A DOCUMENT DEVIATES from the norm.

Per tag_type the NORM is the most common tag-value SET among active, unrestricted documents that carry that
tag_type (e.g. country -> {India}, employment_type -> {full-time employee}). A document's label includes a
tag_type only if its value set differs from that type's norm, and then shows that type's full set, values only:

    India-only handbook                      -> no label
    referral bonus program (full-time + intern) -> "full-time employee, intern"   (rendered "[Applies to: ...]")

app_config: label.mode = deviation | always | off (always = every type present, as before; off = never) and
label.norm_min_share (a set counts as the norm only if at least this share of the documents carrying the type
have it; otherwise the type is always shown).

The norms are cached like the catalog title list (in-process copy over Redis over the DB) and invalidated by
config.doc_caches.invalidate_document_caches(); the mode / share are read through the in-process config copy.
Nothing here touches the DB or network on a chat turn once the copies are warm.
"""
from collections import Counter, defaultdict

from cache.redis_cache import delete_cached
from config.runtime import memo_config
from config.topics import _memo, _memo_lock, memoized, redis_cached
from vectorstore.db import get_pool

LABEL_NORMS_CACHE_KEY = "labels:norms"
TYPE_ORDER = {"country": 0, "employment_type": 1}


def _load_norms() -> dict:
    """{tag_type: {"norm": [values], "share": float, "docs": int}} over active, unrestricted documents."""
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT d.id, at.tag_type, at.tag_value FROM documents d "
            "JOIN document_applicability da ON da.document_id = d.id "
            "JOIN applicability_tags at ON at.id = da.tag_id "
            "WHERE d.status = 'active' AND NOT d.is_restricted"
        ).fetchall()
    per_doc: dict = defaultdict(lambda: defaultdict(set))
    for doc_id, tag_type, value in rows:
        per_doc[tag_type][doc_id].add(value)
    norms = {}
    for tag_type, docs in per_doc.items():
        counts = Counter(tuple(sorted(values)) for values in docs.values())
        # most common set; ties broken by the set itself so the choice is stable
        norm, count = min(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        norms[tag_type] = {"norm": list(norm), "share": count / len(docs), "docs": len(docs)}
    return norms


def get_label_norms() -> dict:
    return memoized(LABEL_NORMS_CACHE_KEY, lambda: redis_cached(LABEL_NORMS_CACHE_KEY, _load_norms))


def invalidate_label_norms() -> None:
    delete_cached(LABEL_NORMS_CACHE_KEY)
    with _memo_lock:
        _memo.pop(LABEL_NORMS_CACHE_KEY, None)


def compute_label(tags: dict | None) -> str | None:
    """Label text (values only, no brackets) for a document's tags {tag_type: [values]}, or None."""
    if not tags:
        return None
    mode = memo_config("label.mode", "deviation")
    if mode == "off":
        return None
    shown = []
    norms = get_label_norms() if mode != "always" else {}
    min_share = memo_config("label.norm_min_share", 0.5)
    for tag_type in sorted(tags, key=lambda t: (TYPE_ORDER.get(t, 2), t)):
        values = sorted(tags[tag_type])
        info = norms.get(tag_type)
        is_norm = bool(info) and info["share"] >= min_share and values == info["norm"]
        if mode == "always" or not is_norm:
            shown.append(", ".join(values))
    return "; ".join(shown) or None
