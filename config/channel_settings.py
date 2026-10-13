"""Per-channel setting resolution shared by retrieval and generation.

Text and voice each own their tunables as separate app_config rows named
"<setting>.<channel>" (e.g. retrieval.top_k.voice). Resolution is:

    channel-specific row  >  global row (the "<setting>" key)  >  default

Rows are seeded once, at the current effective value, by
config.runtime.seed_runtime_config(), so editing one channel's row never
changes the other, and the global row is only a fallback for tooling that
passes no channel.
"""
from config.runtime import get_app_config

CHANNELS = ("text", "voice")

# Settings each channel owns (global key -> description used when seeding).
CHANNEL_SETTINGS = {
    "retrieval.top_k": "Chunks retrieved after RRF fusion",
    "retrieval.candidate_n": "Candidates pulled from each of dense and sparse search before fusion",
    "retrieval.rrf_k": "Reciprocal Rank Fusion constant k",
    "generation.model": "Chat model used to generate answers",
    "generation.confidence_threshold": "Minimum top fused_score required to attempt generation",
    "conversation.history_messages": "Max prior user/assistant messages accepted per turn",
    "conversation.history_max_chars": "Each history message is truncated to this many characters",
    "retrieval.context.max_words_followup": "Utterances shorter than this many words count as follow-ups",
    "retrieval.context.prior_user_turns": "Prior user messages prepended to a follow-up's retrieval query",
    "retrieval.advisory.max_chunks_per_doc": "Per-document chunk cap in an advisory turn's context",
    "retrieval.expansion.enabled": "Whether advisory turns run the expansion queries",
}


def channel_setting(key: str, channel: str | None = None, default=None):
    if channel:
        value = get_app_config(f"{key}.{channel}")
        if value is not None:
            return value
    value = get_app_config(key)
    return default if value is None else value
