"""Document fingerprints and duplicate detection (upload-time and on-demand only; never on a chat turn).

Everything here is READ-ONLY with respect to document status, titles and tags: it reports, an admin
decides (see api/admin.py). Similarity is computed over normalized document text (lowercase, collapsed
whitespace) as 5-word shingles: Jaccard, plus containment in both directions (a shorter document fully
contained in a longer one has containment 1.0 but a low Jaccard).
"""
import hashlib
import re

from cache.redis_cache import MISS, delete_cached, get_cached, set_cached

SHINGLE_WORDS = 5
CHUNK_MATCH_JACCARD = 0.9  # a chunk "is in" the other document when its best shingle Jaccard is >= this
_WS = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    return _WS.sub(" ", (text or "").lower()).strip()


def shingles(normalized: str, n: int = SHINGLE_WORDS) -> set[str]:
    words = normalized.split()
    if len(words) <= n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def document_shingles(chunk_texts: list[str]) -> set[str]:
    return shingles(normalize_text(" ".join(chunk_texts)))


def fingerprint(chunk_texts: list[str]) -> str:
    """sha256 over the ORDERED normalized chunk texts."""
    digest = hashlib.sha256()
    for text in chunk_texts:
        digest.update(normalize_text(text).encode("utf-8"))
        digest.update(b"\x1f")
    return digest.hexdigest()


def similarity(a: set, b: set) -> dict:
    inter = len(a & b)
    union = len(a | b)
    return {
        "jaccard": inter / union if union else 0.0,
        "containment_a_in_b": inter / len(a) if a else 0.0,
        "containment_b_in_a": inter / len(b) if b else 0.0,
    }


def normalize_title(title: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def diff_summary(chunks_a: list[dict], chunks_b: list[dict]) -> dict:
    """Chunk-level diff of two documents. Each chunk dict needs `text` and `section_path`. A chunk of A is
    "also in B" when its normalized text equals one of B's chunks or its best 5-word-shingle Jaccard against
    a chunk of B is >= 0.9 (and vice versa)."""
    def prep(chunks):
        return [(normalize_text(c["text"]), shingles(normalize_text(c["text"])), c.get("section_path")) for c in chunks]

    pa, pb = prep(chunks_a), prep(chunks_b)

    def only_in(src, other):
        texts = {t for t, _s, _p in other}
        missing = []
        for text, sh, path in src:
            if text in texts:
                continue
            best = max((similarity(sh, osh)["jaccard"] for _t, osh, _p in other), default=0.0)
            if best < CHUNK_MATCH_JACCARD:
                missing.append(path)
        return missing

    only_a, only_b = only_in(pa, pb), only_in(pb, pa)
    paths_a = {p for _t, _s, p in pa}
    paths_b = {p for _t, _s, p in pb}
    return {
        "chunks_a": len(chunks_a), "chunks_b": len(chunks_b),
        "chunks_only_in_a": len(only_a), "chunks_only_in_b": len(only_b),
        "section_paths_only_in_a": sorted(p for p in paths_a - paths_b if p),
        "section_paths_only_in_b": sorted(p for p in paths_b - paths_a if p),
        "sections_of_chunks_only_in_a": sorted({p for p in only_a if p}),
        "sections_of_chunks_only_in_b": sorted({p for p in only_b if p}),
    }


def load_active_documents(conn, with_chunks: bool = True) -> list[dict]:
    """Active documents with their ordered chunks (insertion order)."""
    docs = [
        {"id": r[0], "title": r[1], "source_file": r[2], "chunks": []}
        for r in conn.execute(
            "SELECT id, document_title, source_file FROM documents WHERE status = 'active' "
            "ORDER BY document_title, source_file"
        ).fetchall()
    ]
    if with_chunks and docs:
        by_id = {d["id"]: d for d in docs}
        for doc_id, text, section_path in conn.execute(
            "SELECT document_id, text, section_path FROM chunks WHERE document_id = ANY(%s) ORDER BY id",
            (list(by_id),),
        ).fetchall():
            by_id[doc_id]["chunks"].append({"text": text, "section_path": section_path})
    return docs


SCAN_CACHE_KEY = "dups:scan"
SCAN_CACHE_TTL_SECONDS = 3600


def invalidate_scan_cache() -> None:
    delete_cached(SCAN_CACHE_KEY)


def classify(sim: dict, titles_equal: bool, same_fingerprint: bool, thresholds: dict) -> list[str]:
    """Reasons two documents are possible duplicates: 'exact' (same ordered chunk text), 'near' (Jaccard >=
    jaccard_threshold OR containment in either direction >= containment_threshold), 'title' (normalized
    titles equal, when title_match is on). Two VERSIONS of one policy typically score a middling Jaccard but a
    high containment, which is why both are checked."""
    reasons = []
    if same_fingerprint:
        reasons.append("exact")
    elif (sim["jaccard"] >= thresholds["jaccard"]
          or max(sim["containment_a_in_b"], sim["containment_b_in_a"]) >= thresholds["containment"]):
        reasons.append("near")
    if thresholds.get("title_match", True) and titles_equal:
        reasons.append("title")
    return reasons


def pairwise_scan(docs: list[dict], thresholds: dict, with_diff: bool = True) -> list[dict]:
    """All pairs of documents that are exact, near or share a normalized title (see classify()), with the
    similarity numbers and a diff summary, highest Jaccard first."""
    prepared = []
    for d in docs:
        texts = [c["text"] for c in d["chunks"]]
        prepared.append((d, document_shingles(texts), fingerprint(texts)))
    pairs = []
    for i in range(len(prepared)):
        for j in range(i + 1, len(prepared)):
            (a, sa, fa), (b, sb, fb) = prepared[i], prepared[j]
            sim = similarity(sa, sb)
            titles_equal = bool(normalize_title(a["title"])) and normalize_title(a["title"]) == normalize_title(b["title"])
            reasons = classify(sim, titles_equal, fa == fb, thresholds)
            if not reasons:
                continue
            pair = {
                "a": {"document_id": str(a["id"]), "title": a["title"], "source_file": a["source_file"]},
                "b": {"document_id": str(b["id"]), "title": b["title"], "source_file": b["source_file"]},
                "reasons": reasons,
                **{k: round(v, 4) for k, v in sim.items()},
            }
            if with_diff:
                pair["diff"] = diff_summary(a["chunks"], b["chunks"])
            pairs.append(pair)
    pairs.sort(key=lambda p: -p["jaccard"])
    return pairs


def candidate_matches(conn, chunk_texts: list[str], title: str | None, source_file: str, thresholds: dict) -> list[dict]:
    """Active documents (other than ones with the same source_file, which an upload would supersede) that a
    document about to be ingested may duplicate: [{document_id, title, source_file, jaccard, containment,
    reason}] where reason is 'exact' | 'near' | 'title' (exact outranks near outranks title)."""
    sh = document_shingles(chunk_texts)
    fp = fingerprint(chunk_texts)
    found = []
    for d in load_active_documents(conn):
        if d["source_file"] == source_file:
            continue
        texts = [c["text"] for c in d["chunks"]]
        sim = similarity(sh, document_shingles(texts))
        titles_equal = bool(normalize_title(title)) and normalize_title(title) == normalize_title(d["title"])
        reasons = classify(sim, titles_equal, fp == fingerprint(texts), thresholds)
        if reasons:
            found.append({
                "document_id": str(d["id"]), "title": d["title"], "source_file": d["source_file"],
                "jaccard": round(sim["jaccard"], 4),
                "containment": round(max(sim["containment_a_in_b"], sim["containment_b_in_a"]), 4),
                "reason": reasons[0] if len(reasons) == 1 else "+".join(reasons),
            })
    found.sort(key=lambda m: -m["jaccard"])
    return found


def backfill_fingerprints(conn) -> int:
    """Idempotent: computes content_fingerprint for documents that have none. Returns how many were set."""
    rows = conn.execute("SELECT id FROM documents WHERE content_fingerprint IS NULL").fetchall()
    for (doc_id,) in rows:
        texts = [r[0] for r in conn.execute("SELECT text FROM chunks WHERE document_id = %s ORDER BY id", (doc_id,)).fetchall()]
        conn.execute("UPDATE documents SET content_fingerprint = %s WHERE id = %s", (fingerprint(texts), doc_id))
    conn.commit()
    return len(rows)


def exact_duplicate_of(conn, fingerprint_value: str, source_file: str) -> dict | None:
    """An ACTIVE document with this exact content under a DIFFERENT source_file, if any."""
    row = conn.execute(
        "SELECT id, document_title, source_file FROM documents WHERE content_fingerprint = %s AND status = 'active' "
        "AND source_file <> %s ORDER BY created_at LIMIT 1", (fingerprint_value, source_file)).fetchone()
    return {"document_id": str(row[0]), "title": row[1], "source_file": row[2]} if row else None


def thresholds_from_config(get) -> dict:
    return {"jaccard": get("dup.jaccard_threshold", 0.6), "containment": get("dup.containment_threshold", 0.75),
            "title_match": get("dup.title_match", True)}


def cached_scan(conn_factory, get) -> list[dict]:
    """Cache-aside scan result for the admin documents list (badges); invalidated by the shared helper."""
    cached = get_cached(SCAN_CACHE_KEY)
    if cached is not MISS:
        return cached
    with conn_factory() as conn:
        docs = load_active_documents(conn)
    pairs = pairwise_scan(docs, thresholds_from_config(get), with_diff=False)
    set_cached(SCAN_CACHE_KEY, pairs, ttl=SCAN_CACHE_TTL_SECONDS)
    return pairs
