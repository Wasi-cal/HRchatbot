"""Shared generation plumbing for the HR RAG chatbot: the OpenAI client, the
per-channel model / confidence-threshold getters, and generate_answer(), a
thin compatibility router used by eval tooling.

The pipelines themselves are NOT here. Text and voice are fully separate,
one module each (channels/text.py, channels/voice.py): each owns its
retrieval -> grounding -> generation flow, its copy and its settings.
Production code (api/chat.py, api/voice.py) calls those modules directly;
generate_answer(channel=...) only dispatches to them.

Low-confidence threshold: RRF (see vectorstore/retrieve.py) bounds a
top-1 result's fused_score to a narrow, rank-derived range. With the
default RRF k=60, a chunk that lands at rank 1 in only ONE of the two
candidate lists scores exactly 1/(60+1) ~= 0.0164 - that's the floor
for a top-1 result, since retrieve() always returns whatever chunk had
the best fused score even when nothing in the corpus is a real match.
A chunk ranked 1 in BOTH lists (the strongest possible signal) scores
2/(60+1) ~= 0.0328. Empirically, over this corpus's known-good eval
queries (see eval/run_regression.py), genuine matches cluster at 0.027-0.033
(agreement in both lists), while a query with no real match in the
corpus can still drift up to that floor by pure coincidence. The default
threshold (0.02, now the generation provider_configs' confidence_threshold, editable at runtime) sits just above the theoretical floor and below the
observed good-match cluster, so it only short-circuits the clearest
retrieval failures (single-list, uncorroborated top-1 hits) rather than
trying to be a general relevance classifier - that job is left to the
system prompt's instruction to admit when the excerpts don't answer the
question. Treat this as a tunable starting point, not a tuned constant.
"""
import os

from openai import OpenAI

from config.channel_settings import channel_setting
from config.runtime import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_GENERATION_MODEL,
    get_active_provider_config,
)

_client = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENAI_API_KEY - copy .env.example to .env and set a real key."
            )
        _client = OpenAI(api_key=api_key)
    return _client


def get_generation_model(channel: str | None = None) -> str:
    """Model name. With a channel: generation.model.<channel> (app_config), falling
    back to the active category='generation' provider_configs row (cached for
    config.runtime.CACHE_TTL_SECONDS; seeded by config.runtime.seed_runtime_config).
    Without one (clarification extraction, applicability extraction): the provider row."""
    if channel:
        value = channel_setting("generation.model", channel)
        if value:
            return value
    return get_active_provider_config("generation").get("model") or DEFAULT_GENERATION_MODEL


def get_confidence_threshold(channel: str | None = None) -> float:
    if channel:
        value = channel_setting("generation.confidence_threshold", channel)
        if value is not None:
            return float(value)
    value = get_active_provider_config("generation").get("confidence_threshold")
    return DEFAULT_CONFIDENCE_THRESHOLD if value is None else float(value)


def generate_answer(
    query: str,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    user_attributes: dict | None = None,
    channel: str = "text",
    history: list | None = None,
):
    """Compatibility router: yields the same events as the channel module's
    answer_stream() ("token" events, then one "done" event; see channels/text.py).
    user_attributes is accepted and ignored (nothing is filtered by who is asking).
    Exists so eval tooling has one entry point; the API calls the channel
    modules directly."""
    from channels import get_channel  # lazy: channels import this module

    yield from get_channel(channel).answer_stream(
        query, top_k=top_k, candidate_n=candidate_n, k=k,
        confidence_threshold=confidence_threshold, history=history,
    )


def main():
    import sys

    query = " ".join(sys.argv[1:]) or "what is the parental leave policy?"
    for event in generate_answer(query):
        if event["type"] == "token":
            print(event["text"], end="", flush=True)
        else:
            print()
            print(event["timing"])


if __name__ == "__main__":
    main()
