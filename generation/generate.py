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


def build_client() -> OpenAI:
    """The process-wide OpenAI client. Built once (FastAPI lifespan, api/main.py; lazily on first use in scripts and tests).

    SDK-internal retries are OFF (max_retries=0): channels/common.py stream_completion owns retry / fallback, so the two never
    stack. The per-attempt timeouts (connect vs first token vs stream gap) are applied per request (channels/common.py
    _with_options); the client-level timeout below is only a generous backstop. keepalive_expiry comes from app_config
    generation.keepalive_expiry_s (read at creation: needs an API restart). Measured (eval/openai_stall_probe.py): connect + TLS
    take about 30 ms and first-byte stalls happen on fresh and reused connections alike, so HTTP/2, IPv4-only and keep-alive tuning
    were NOT adopted; the settings below are the httpx/SDK defaults, made explicit and configurable."""
    import httpx

    from config.runtime import get_app_config

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing OPENAI_API_KEY - copy .env.example to .env and set a real key."
        )
    try:
        keepalive_expiry = float(get_app_config("generation.keepalive_expiry_s", 5.0))
    except Exception:  # config store unavailable: fall back to httpx's default  # noqa: BLE001
        keepalive_expiry = 5.0
    http_client = httpx.Client(
        limits=httpx.Limits(max_connections=1000, max_keepalive_connections=100, keepalive_expiry=keepalive_expiry),
        timeout=httpx.Timeout(30.0, connect=5.0),
    )
    return OpenAI(api_key=api_key, max_retries=0, http_client=http_client)


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = build_client()
    return _client


def init_client(warm_up: bool = True) -> OpenAI:
    """Creates the process-wide client (idempotent) and, when asked, opens the first connection with ONE lightweight non-LLM
    request (GET /models/<model>: DNS, TCP, TLS and SDK lazy initialisation happen now, not on the first caller's turn).
    Warm-up failures are logged and ignored."""
    client = get_client()
    if warm_up:
        import logging

        try:
            client.models.retrieve(get_generation_model())
        except Exception as exc:  # noqa: BLE001
            logging.getLogger("hrchatbot.generation").warning("think event=warmup_failed reason=%s: %s", type(exc).__name__, str(exc)[:120])
    return client


def close_client() -> None:
    """Closes the process-wide client and its connection pool (FastAPI shutdown). A later get_client() builds a new one."""
    global _client
    client, _client = _client, None
    if client is not None:
        try:
            client.close()
        except Exception:  # noqa: BLE001, S110
            pass


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
