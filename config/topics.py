"""Spoken-friendly topic names derived from active, unrestricted document
titles, used in small-talk replies and the "couldn't find that" reply.

The list is cached in Redis with a long TTL and invalidated when the
document set changes (upload, restrict toggle - see api/admin.py), so a
turn that needs topics costs one Redis read, not a DB query. If Redis is
down, cache.redis_cache degrades to "always miss" and this falls back to
a DB read.
"""
import random
import re
import threading
import time

from cache.redis_cache import MISS, delete_cached, get_cached, set_cached
from vectorstore.db import get_pool

TOPICS_CACHE_KEY = "topics:active_documents"
TOPICS_TTL_SECONDS = 24 * 3600  # long; explicit invalidation is the primary mechanism
KEEP_UPPER_MAX_LEN = 4  # acronyms like POSH, PIP, IT stay upper-case

_PREFIX_RE = re.compile(r"^\s*calfus(?:\s+crew)?\s*:?\s*", re.IGNORECASE)
_SUFFIX_RE = re.compile(r"\s+(?:policy|policies)\s*$", re.IGNORECASE)
_PARENS_RE = re.compile(r"\s*\([^)]*\)")


def clean_title(title: str) -> str:
    """'Calfus Performance Improvement Plan (PIP) Policy' -> 'performance improvement plan',
    'CALFUS CREW: REFERRAL BONUS PROGRAM - INDIA' -> 'referral bonus program'."""
    t = title.split(" - ")[0]
    t = _PARENS_RE.sub("", t)
    t = _PREFIX_RE.sub("", t)
    t = _SUFFIX_RE.sub("", t).replace("&", "and")
    words = [w if (w.isupper() and len(w) <= KEEP_UPPER_MAX_LEN) else w.lower() for w in t.split()]
    return " ".join(words).strip()


def _load_topics() -> list[str]:
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT document_title FROM documents "
            "WHERE status = 'active' AND NOT is_restricted AND document_title IS NOT NULL "
            "ORDER BY document_title"
        ).fetchall()
    topics = []
    for (title,) in rows:
        topic = clean_title(title)
        if topic and topic not in topics:
            topics.append(topic)
    return topics


# ---- catalog titles: same cache, same invalidation as the topic list ---------------------------------------------
CATALOG_CACHE_KEY = "topics:catalog_titles"
MEMO_TTL_SECONDS = 30.0  # in-process copy so a chat turn makes no Redis or DB call
_memo: dict = {}
_memo_lock = threading.Lock()


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _load_catalog_titles() -> list[str]:
    """Titles of active (not superseded), unrestricted documents, de-duplicated case-insensitively on the
    normalized title, in a stable order (normalized title, then source file)."""
    with get_pool().connection() as conn:
        rows = conn.execute(
            "SELECT document_title FROM documents WHERE status = 'active' AND NOT is_restricted AND document_title IS NOT NULL "
            "ORDER BY lower(document_title), source_file"
        ).fetchall()
    seen, titles = set(), []
    for (title,) in rows:
        key = _normalize_title(title)
        if key and key not in seen:
            seen.add(key)
            titles.append(title.strip())
    return sorted(titles, key=_normalize_title)


def _memoized(key: str, loader):
    now = time.monotonic()
    with _memo_lock:
        hit = _memo.get(key)
        if hit is not None and hit[0] > now:
            return hit[1]
    value = loader()
    with _memo_lock:
        _memo[key] = (now + MEMO_TTL_SECONDS, value)
    return value


def _redis_cached(cache_key: str, loader) -> list[str]:
    cached = get_cached(cache_key)
    if cached is not MISS:
        return cached
    value = loader()
    set_cached(cache_key, value, ttl=TOPICS_TTL_SECONDS)
    return value


def get_catalog_titles() -> list[str]:
    """Cached title list for the catalog reply: in-process copy (30s) over Redis (24h) over the DB."""
    return _memoized(CATALOG_CACHE_KEY, lambda: _redis_cached(CATALOG_CACHE_KEY, _load_catalog_titles))


def get_topics_cached() -> list[str]:
    """get_topics() behind the same in-process copy (no Redis call on a hit)."""
    return _memoized(TOPICS_CACHE_KEY, get_topics)


def get_topics() -> list[str]:
    cached = get_cached(TOPICS_CACHE_KEY)
    if cached is not MISS:
        return cached
    topics = _load_topics()
    set_cached(TOPICS_CACHE_KEY, topics, ttl=TOPICS_TTL_SECONDS)
    return topics


def invalidate_topics() -> None:
    """Drops every cached document-derived list (topics and catalog titles), here and in this process's
    in-process copies. Call on upload, status change, title change and restrict toggle; other workers'
    in-process copies expire within MEMO_TTL_SECONDS."""
    delete_cached(TOPICS_CACHE_KEY)
    delete_cached(CATALOG_CACHE_KEY)
    with _memo_lock:
        _memo.clear()


def topic_phrase(n: int = 3) -> str | None:
    """'a, b, or c' for up to n randomly chosen topics; None if there are none."""
    topics = get_topics()
    if not topics:
        return None
    picked = random.sample(topics, min(n, len(topics)))
    if len(picked) == 1:
        return picked[0]
    return ", ".join(picked[:-1]) + (", or " if len(picked) > 2 else " or ") + picked[-1]
