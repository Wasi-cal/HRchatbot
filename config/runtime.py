"""Runtime configuration read from Postgres with Redis cache-aside.

Two stores:
  - app_config: pipeline tunables (retrieval.top_k, ...), one row per key.
  - provider_configs: per-provider settings for the ACTIVE row of a
    category ('generation', 'stt', 'tts'), as one jsonb blob.

Reads check Redis first (CACHE_TTL_SECONDS), then Postgres, then
repopulate. Writes (admin API) go to Postgres and then straight into the
cache (write-through), so the writer's next read is immediately fresh;
other readers converge within the TTL even if the DB is edited directly.

Secrets (OPENAI_API_KEY, DEEPGRAM_API_KEY, THINK_ENDPOINT_SECRET, DB
credentials) are env-only and must never be stored in either table.
"""
import logging
import os

from psycopg.types.json import Jsonb

from cache.redis_cache import MISS, get_cached, set_cached
from vectorstore.db import get_pool

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30

# Defaults: used to seed rows and as a last-resort fallback if a read fails.
# The retrieval values must equal vectorstore/retrieve.py's DEFAULT_* (checked
# by eval/config_test.py).
DEFAULT_APP_CONFIG = {
    "retrieval.top_k": (3, "Chunks returned by retrieve() after RRF fusion."),
    "retrieval.candidate_n": (20, "Candidates pulled from each of dense and sparse search before fusion."),
    "retrieval.rrf_k": (60, "Reciprocal Rank Fusion constant k."),
}
DEFAULT_GENERATION_MODEL = "gpt-4o-mini"
DEFAULT_CONFIDENCE_THRESHOLD = 0.02
DEFAULT_STT_MODEL = "nova-3"
DEFAULT_TTS_VOICE = "aura-2-thalia-en"
DEFAULT_GREETING = "Hi, I'm the HR assistant. What would you like to know about company policy?"

PROVIDER_CATEGORIES = ("generation", "stt", "tts")


def _app_key(key: str) -> str:
    return f"app_config:{key}"


def _provider_key(category: str) -> str:
    return f"provider_config:{category}"


# ---- reads ----------------------------------------------------------------

def get_app_config(key: str, default=None):
    cached = get_cached(_app_key(key))
    if cached is not MISS:
        return cached
    try:
        with get_pool().connection() as conn:
            row = conn.execute("SELECT value FROM app_config WHERE key = %s", (key,)).fetchone()
    except Exception:
        logger.exception("app_config read failed for %s; using default", key)
        return default
    if row is None:
        return default
    set_cached(_app_key(key), row[0], ttl=CACHE_TTL_SECONDS)
    return row[0]


def get_active_provider_config(category: str) -> dict:
    cached = get_cached(_provider_key(category))
    if cached is not MISS:
        return cached
    with get_pool().connection() as conn:
        row = conn.execute(
            "SELECT config FROM provider_configs WHERE category = %s AND is_active LIMIT 1", (category,)
        ).fetchone()
    config = row[0] if row and row[0] else {}
    if row:
        set_cached(_provider_key(category), config, ttl=CACHE_TTL_SECONDS)
    return config


# ---- writes (write-through) ------------------------------------------------

def set_app_config(key: str, value) -> bool:
    """Updates an EXISTING app_config row; returns False if the key doesn't exist."""
    with get_pool().connection() as conn:
        updated = conn.execute(
            "UPDATE app_config SET value = %s, updated_at = now() WHERE key = %s",
            (Jsonb(value), key),
        ).rowcount
        conn.commit()
    if updated:
        set_cached(_app_key(key), value, ttl=CACHE_TTL_SECONDS)
    return bool(updated)


def set_provider_setting(category: str, field: str, value) -> dict | None:
    """Merges {field: value} into the active provider_configs row for the
    category; returns the full updated config, or None if no active row."""
    with get_pool().connection() as conn:
        row = conn.execute(
            "UPDATE provider_configs SET config = COALESCE(config, '{}'::jsonb) || %s, updated_at = now() "
            "WHERE category = %s AND is_active RETURNING config",
            (Jsonb({field: value}), category),
        ).fetchone()
        conn.commit()
    if row is None:
        return None
    set_cached(_provider_key(category), row[0], ttl=CACHE_TTL_SECONDS)
    return row[0]


# ---- seeding ---------------------------------------------------------------

def _seed_provider(conn, category: str, provider_name: str, defaults: dict) -> None:
    """Ensures an active provider_configs row exists for the category and
    that it has every key in `defaults` (never overwrites existing values)."""
    row = conn.execute(
        "SELECT id, config FROM provider_configs WHERE category = %s AND is_active", (category,)
    ).fetchone()
    if row is None:
        conn.execute(
            "INSERT INTO provider_configs (category, provider_name, config, is_active) VALUES (%s, %s, %s, true)",
            (category, provider_name, Jsonb(defaults)),
        )
        return
    missing = {k: v for k, v in defaults.items() if k not in (row[1] or {})}
    if missing:
        conn.execute(
            "UPDATE provider_configs SET config = COALESCE(config, '{}'::jsonb) || %s, updated_at = now() WHERE id = %s",
            (Jsonb(missing), row[0]),
        )


def seed_runtime_config() -> None:
    """Idempotent. One-time env -> DB hand-off: values below come from the
    pre-existing env vars (or code defaults) so seeding changes no
    behavior; afterwards the DB is the source of truth and those env vars
    are ignored for these settings."""
    with get_pool().connection() as conn:
        for key, (value, description) in DEFAULT_APP_CONFIG.items():
            conn.execute(
                "INSERT INTO app_config (key, value, description) VALUES (%s, %s, %s) ON CONFLICT (key) DO NOTHING",
                (key, Jsonb(value), description),
            )
        _seed_provider(conn, "generation", "openai", {
            "model": os.environ.get("GENERATION_MODEL_NAME", DEFAULT_GENERATION_MODEL),
            "confidence_threshold": float(os.environ.get("GENERATION_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD)),
        })
        _seed_provider(conn, "stt", "deepgram", {
            "model": os.environ.get("DEEPGRAM_LISTEN_MODEL", DEFAULT_STT_MODEL),
        })
        _seed_provider(conn, "tts", "deepgram", {
            "voice": os.environ.get("DEEPGRAM_TTS_MODEL", DEFAULT_TTS_VOICE),
            "greeting": os.environ.get("VOICE_GREETING", DEFAULT_GREETING),
        })
        conn.commit()
