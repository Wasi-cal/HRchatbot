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
from config.prompts import seed_prompts
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
    "conversation.history_messages": (6, "Max prior user/assistant messages accepted per turn (server-side cap on client/Deepgram history)."),
    "conversation.history_max_chars": (400, "Each history message is truncated to this many characters."),
    "retrieval.context.max_words_followup": (8, "An utterance shorter than this many words is treated as a follow-up and gets prior user turns prepended to its retrieval query."),
    "retrieval.context.prior_user_turns": (2, "How many previous user messages are prepended to a follow-up's retrieval query."),
    "retrieval.context.followup_markers": (
        ["that", "it", "this", "those", "these", "them", "there", "same", "anything else", "what else",
         "instead", "other option", "other options"],
        "Words/phrases that mark an utterance as a follow-up (whole-word match on the normalized text).",
    ),
    "retrieval.advisory.markers": (
        ["anything else i can do", "anything else that i can do", "what are my options", "how can i avoid",
         "how do i avoid", "is there a way", "alternatives", "alternative", "i dont want", "without losing",
         "can i instead", "workaround", "any other option", "other options", "avoid losing"],
        "Phrases (normalized: lowercase, no punctuation) that mark a turn as advisory: broader, multi-policy retrieval.",
    ),
    "retrieval.expansions": (
        [{
            "topic": "leave_shortage",
            "triggers": ["lop", "loss of pay", "leave without pay", "lwp", "leaves left", "no leaves", "leave balance",
                         "not enough leave", "run out of leave", "out of leave", "short of leave", "lose pay",
                         "losing pay", "lose my pay"],
            # Wordings chosen by probing the corpus (eval notes in the PR): each puts its policy's chunks
            # at the top of BOTH the dense and sparse rankings for this handbook's wording.
            "queries": [
                "work from home WFH request approval",
                "compensatory leave comp off earn and utilize",
                "leave without pay LWP when leave balance is exhausted",
            ],
        }],
        "Topic -> expansion queries for advisory turns. A topic fires when any trigger phrase appears in the "
        "current utterance or prior user messages; its queries are embedded in one batched encode and fused in.",
    ),
    "retrieval.expansion.enabled": (True, "Whether advisory turns run the expansion queries (per channel copies override)."),
    "retrieval.expansion.weight": (2.0, "RRF weight of each expansion query's ranked lists relative to the main query's."),
    "retrieval.expansion.candidate_n": (2, "Candidates pulled per expansion query (dense and sparse each): only its top hits should steer the context."),
    # 4, not 2: in this corpus leave, work-from-home and comp-off all live inside ONE document (the Employee
    # Handbook), so a cap of 2 cannot hold the leave rule AND both alternatives (measured; see the report).
    "retrieval.advisory.max_chunks_per_doc": (4, "Per-document cap on chunks in an advisory turn's context, applied after RRF fusion."),
    "retrieval.top_k.advisory.text": (6, "Chunks retrieved for an advisory text turn."),
    "retrieval.top_k.advisory.voice": (4, "Chunks retrieved for an advisory voice turn."),
    "user.default_attributes": (
        {"country": "India"},
        "Applicability attributes assumed when the caller doesn't supply them (no auth yet; all documents are India-only).",
    ),
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


def _seed_channel_settings(conn) -> None:
    """Gives text and voice their own copy of every tunable they use, seeded at
    the CURRENT effective value (so nothing changes on day one) and never
    overwriting an existing row. Retrieval and user-default settings copy the
    global app_config rows; generation settings copy the active 'generation' provider config."""
    from config.channel_settings import CHANNEL_SETTINGS, CHANNELS  # lazy: channel_settings imports this module

    generation = conn.execute(
        "SELECT config FROM provider_configs WHERE category = 'generation' AND is_active"
    ).fetchone()
    generation_config = (generation[0] if generation else None) or {}
    for channel in CHANNELS:
        for key, description in CHANNEL_SETTINGS.items():
            text = f"{description} - {channel} channel; overrides {key} for it."
            if key.startswith(("retrieval.", "user.", "conversation.")):
                conn.execute(
                    "INSERT INTO app_config (key, value, description) SELECT %s, value, %s FROM app_config "
                    "WHERE key = %s ON CONFLICT (key) DO NOTHING",
                    (f"{key}.{channel}", text, key),
                )
            elif generation_config.get(key.split(".", 1)[1]) is not None:
                conn.execute(
                    "INSERT INTO app_config (key, value, description) VALUES (%s, %s, %s) ON CONFLICT (key) DO NOTHING",
                    (f"{key}.{channel}", Jsonb(generation_config[key.split(".", 1)[1]]), text),
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
        seed_prompts(conn)
        _seed_provider(conn, "generation", "openai", {
            "model": os.environ.get("GENERATION_MODEL_NAME", DEFAULT_GENERATION_MODEL),
            "confidence_threshold": float(os.environ.get("GENERATION_CONFIDENCE_THRESHOLD", DEFAULT_CONFIDENCE_THRESHOLD)),
        })
        _seed_channel_settings(conn)
        _seed_provider(conn, "stt", "deepgram", {
            "model": os.environ.get("DEEPGRAM_LISTEN_MODEL", DEFAULT_STT_MODEL),
        })
        _seed_provider(conn, "tts", "deepgram", {
            "voice": os.environ.get("DEEPGRAM_TTS_MODEL", DEFAULT_TTS_VOICE),
            "greeting": os.environ.get("VOICE_GREETING", DEFAULT_GREETING),
        })
        conn.commit()
