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
import time

from psycopg.types.json import Jsonb

from cache.redis_cache import MISS, get_cached, set_cached
from config.prompts import seed_prompts
from vectorstore.db import get_pool

logger = logging.getLogger(__name__)

CACHE_TTL_SECONDS = 30

# Defaults: used to seed rows and as a last-resort fallback if a read fails.
# The retrieval values must equal vectorstore/retrieve.py's DEFAULT_* (checked
# by eval/config_test.py).
# ---- small talk and source questions (editable JSON, see channels/common.py) -------------------------------
# Phrases are NORMALIZED (lowercase, no punctuation or apostrophes). A turn is small talk only when the WHOLE
# utterance is one of a category's phrases (optionally with a trailing address word like "there"/"bot", or
# a leading greeting for how_are_you / capabilities / greeting).
_SMALLTALK_COMMON = {
    "greeting": ["hi", "hii", "hello", "helloo", "hey", "heyy", "heya", "hiya", "yo", "sup", "wassup", "wazzup",
                 "whats up", "whatsup", "howdy", "greetings", "good morning", "good afternoon", "good evening", "good day"],
    "how_are_you": ["how are you", "how are you doing", "how are you today", "how r u", "how do you do", "hows it going",
                    "how is it going", "how are things", "hows everything", "how you doing", "you good", "are you ok",
                    "are you okay"],
    "thanks": ["thanks", "thank you", "thx", "ty", "cheers", "much appreciated", "appreciate it", "many thanks",
               "thanks a lot", "thanks so much", "thanks very much", "thanks a ton", "thank you so much",
               "thank you very much", "thank you a lot", "ok thanks", "okay thanks", "cool thanks", "great thanks",
               "alright thanks", "perfect thanks", "got it thanks", "ok thank you", "okay thank you", "cool thank you",
               "great thank you", "alright thank you", "perfect thank you", "got it thank you", "yes thank you",
               "yes thanks", "yes thank you so much"],
    "goodbye": ["bye", "bye bye", "goodbye", "see you", "see you later", "see you soon", "see ya", "later", "good night",
                "take care", "talk later", "ttyl", "cya", "catch you later"],
    "capabilities": ["who are you", "what are you", "what is your name", "whats your name", "what can you do",
                     "what can you do for me", "what do you do", "what can you help with", "what can you help me with",
                     "how can you help", "how can you help me", "what can i ask", "what can i ask you",
                     "what do you know", "help", "help me"],
    "ack": ["ok", "okay", "k", "kk", "cool", "got it", "alright", "all right", "understood", "i see", "noted", "great",
            "nice", "awesome", "perfect", "makes sense", "sounds good", "ok cool", "okay cool", "ok great", "ok got it",
            "cool great", "cool got it", "great got it", "yes", "yeah", "yep", "yup", "yea", "sure", "yessir", "yes sir"],
    "are_you_there": ["are you there", "you there", "anyone there", "is anyone there"],
}
DEFAULT_SMALLTALK_PHRASES = {
    "text": _SMALLTALK_COMMON,
    # Voice-only phrasings: people check the line is live before they ask.
    "voice": {**_SMALLTALK_COMMON, "are_you_there": _SMALLTALK_COMMON["are_you_there"] + [
        "are you listening", "can you hear me", "hello hello", "testing", "test", "mic check", "is this thing on"]},
}
DEFAULT_SOURCE_MARKERS = [
    "source", "the source", "whats the source", "what is the source", "whats the source of this",
    "whats the source of that", "what is the source of this", "what is the source of that", "source of this",
    "source of that", "where is this from", "where is that from", "where did you get that", "where did you get this",
    "where did that come from", "where did this come from", "which policy is this", "which policy is that",
    "which document", "which document is this", "which document is that", "reference", "where is that mentioned",
    "where is this mentioned",
]
DEFAULT_SOURCE_TEMPLATES = {
    "text": {
        "one": "That came from the {title}.",
        "one_with_section": "That came from the {title}, in the section {section}.",
        "many": "That drew on the {titles}.",
        "none": "That answer didn't come from a policy document.",
        "not_found": "I couldn't work out which document that came from.",
    },
    "voice": {
        "one": "That's from the {title} policy.",
        "many": "That drew on the {titles} policies.",
        "none": "That answer didn't come from a policy document.",
        "not_found": "I couldn't work out which document that came from.",
    },
}

# ---- catalog intent ("what documents do you have"): editable, see channels/catalog.py ---------------------------
DEFAULT_CATALOG_VOCAB = [
    "document", "documents", "policy", "policies", "topic", "topics", "file", "files", "source", "sources",
    "information", "knowledge", "hr", "have", "has", "know", "cover", "covers", "available", "access", "list",
    "show", "tell", "give", "which", "what", "all", "everything", "anything", "answer", "ask", "able", "can",
]
DEFAULT_CATALOG_NOUNS = [
    "document", "documents", "policy", "policies", "topic", "topics", "file", "files", "source", "sources",
    "knowledge", "information", "what you know", "what do you know",
]
DEFAULT_CATALOG_NORMALIZE_MAP = {
    "u": "you", "ur": "your", "docs": "documents", "doc": "document", "rn": "", "pls": "", "plz": "",
    "right now": "", "currently": "", "whats": "what", "wats": "what",
}
DEFAULT_CATALOG_TEMPLATES = {
    "text": {
        "main": "I can answer questions from {n} HR documents: {list}.",
        "more": "and {k} more",
        "topics": "Example topics: {topics}.",
        "closing": "Just ask me about any of them.",
        "empty": "There are no HR documents loaded yet.",
    },
    "voice": {
        "main": "I have {n} HR policy documents, including {names}.",
        "closing": "Just ask me about any of them.",
        "empty": "I don't have any HR documents loaded yet.",
    },
}


DEFAULT_TITLE_GENERIC_PHRASES = ["document classification", "confidential", "internal", "table of contents"]
DEFAULT_CATALOG_STOPWORDS = [
    "the", "a", "an", "of", "to", "do", "does", "did", "is", "are", "am", "be", "been", "i", "me", "my", "we", "our",
    "us", "it", "its", "this", "that", "these", "those", "in", "on", "at", "for", "with", "about", "any", "some",
    "there", "here", "please", "and", "or", "so", "if", "just", "also", "right", "now",
]

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
    "smalltalk.phrases.text": (DEFAULT_SMALLTALK_PHRASES["text"], "Text channel small-talk phrases by category (normalized; whole-utterance match). Editable."),
    "smalltalk.phrases.voice": (DEFAULT_SMALLTALK_PHRASES["voice"], "Voice channel small-talk phrases by category (normalized; whole-utterance match). Editable."),
    "smalltalk.affirmations": (
        ["yes", "yeah", "yep", "yup", "yea", "sure", "ok", "okay", "yessir", "yes sir", "yes thank you", "yes thanks", "yes please"],
        "Affirmation phrases: NOT small talk when the previous assistant turn ended with a question or made an offer - they go through the normal pipeline so the offer is followed up.",
    ),
    "smalltalk.offer_markers": (
        ["would you like", "want me to", "i can also"],
        "Phrases that mark an assistant message as an offer (a message ending in '?' also counts).",
    ),
    "catalog.enabled": (True, "Whether 'what documents do you have' questions get the deterministic catalog reply."),
    "catalog.vocab": (DEFAULT_CATALOG_VOCAB, "A catalog question may contain ONLY these content words (after stopword removal and the normalize map)."),
    "catalog.nouns": (DEFAULT_CATALOG_NOUNS, "A catalog question must contain at least one of these (multi-word entries match the whole utterance)."),
    "catalog.normalize_map": (DEFAULT_CATALOG_NORMALIZE_MAP, "Informal forms expanded before matching; an empty replacement drops the word (u->you, docs->documents, rn->drop)."),
    "catalog.stopwords": (DEFAULT_CATALOG_STOPWORDS, "Function words ignored by the catalog vocabulary test (moved here from code; same list)."),
    "catalog.max_words": (12, "A catalog question is at most this many words."),
    "catalog.text_max_titles": (12, "Titles listed in a text catalog reply before 'and N more'."),
    "catalog.voice_max_titles": (4, "Short spoken names listed in a voice catalog reply."),
    "catalog.voice_max_title_words": (5, "Titles longer than this many words are not read aloud."),
    "catalog.templates.text": (DEFAULT_CATALOG_TEMPLATES["text"], "Text catalog reply parts: main ({n},{list}), more ({k}), topics ({topics}), closing, empty. Must not end with a question or offer."),
    "catalog.templates.voice": (DEFAULT_CATALOG_TEMPLATES["voice"], "Voice catalog reply parts: main ({n},{names}), closing, empty. Must not end with a question or offer."),
    "label.mode": ("deviation", "Group label on retrieved chunks: deviation (only types where the document differs from the norm) | always | off."),
    "label.norm_min_share": (0.5, "A tag-value set counts as the norm for its type only if at least this share of documents with that type carry it; otherwise the type is always shown."),
    "dup.enabled": (True, "Upload-time duplicate detection (exact fingerprint 409 and possible_duplicates)."),
    "dup.jaccard_threshold": (0.6, "Flag a possible duplicate when the 5-word-shingle Jaccard of two documents is at least this."),
    "dup.containment_threshold": (0.75, "...or when either document's shingles are contained in the other's at least this much."),
    "dup.title_match": (True, "...or when the normalized titles are equal."),
    "title.generic_phrases": (DEFAULT_TITLE_GENERIC_PHRASES, "A detected title containing one of these phrases triggers a title_warning at upload (never changed automatically)."),
    "source.markers": (DEFAULT_SOURCE_MARKERS, "Whole-utterance phrases (normalized, <= 8 words) that ask where the previous answer came from."),
    "source.templates.text": (DEFAULT_SOURCE_TEMPLATES["text"], "Text replies for a source question. Keys: one, one_with_section, many, none, not_found."),
    "source.templates.voice": (DEFAULT_SOURCE_TEMPLATES["voice"], "Voice replies for a source question (document name only). Keys: one, many, none, not_found."),
    "retrieval.advisory.max_chunks_per_doc": (4, "Per-document cap on chunks in an advisory turn's context, applied after RRF fusion."),
    "retrieval.top_k.advisory.text": (6, "Chunks retrieved for an advisory text turn."),
    "retrieval.top_k.advisory.voice": (4, "Chunks retrieved for an advisory voice turn."),
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
            if key.startswith(("retrieval.", "conversation.")):
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


# ---- in-process copy of app_config values read on every chat turn ----------------------------------------------------
_CONFIG_MEMO: dict = {}
CONFIG_MEMO_TTL_SECONDS = 30.0


def memo_config(key: str, default=None):
    """get_app_config() with a 30s in-process copy, so keys read on EVERY turn cost no Redis round trip except
    once per TTL. Admin writes clear it (api/admin.py) so a change applies at once in the process that handled it;
    other workers converge within the TTL."""
    now = time.monotonic()
    hit = _CONFIG_MEMO.get(key)
    if hit is not None and hit[0] > now:
        return hit[1]
    value = get_app_config(key, default)
    _CONFIG_MEMO[key] = (now + CONFIG_MEMO_TTL_SECONDS, value)
    return value


def clear_config_memo() -> None:
    _CONFIG_MEMO.clear()
