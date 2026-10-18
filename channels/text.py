"""Text channel (/api/chat): retrieval -> grounding -> generation, plus this
channel's small talk, copy and settings, all in one file.

Fully separate from the voice channel: this module never imports channels.voice, and every
setting it reads resolves to a "<key>.text" row first (see config/channel_settings.py), its
prompt is the text style + text grounding_core (config/prompts.py), and its copy below is
written for reading. Shared, behaviour-free helpers live in channels/common.py."""
import logging
import time

from channels import catalog, common
from config.prompts import get_system_prompt
from config.topics import get_catalog_titles, get_topics_cached, topic_phrase
from generation.generate import (
    get_client,
    get_confidence_threshold,
    get_generation_model,
)
from vectorstore.retrieve import retrieve

CHANNEL = "text"
logger = logging.getLogger("hrchatbot.text")

# --------------------------------------------------------------------------
# Copy
# --------------------------------------------------------------------------

NO_MATCH_BASE = "I couldn't find anything on that in the HR policies I have."


def no_match_response() -> str:
    """Retrieval-shortcut reply. Names a few real topics (cached list, see
    config/topics.py); falls back to the base sentence if the lookup fails or finds none."""
    try:
        topics = topic_phrase()
    except Exception:  # noqa: BLE001
        topics = None
    return f"{NO_MATCH_BASE} I can help with things like {topics}." if topics else NO_MATCH_BASE


# --------------------------------------------------------------------------
# Small talk (deterministic; no retrieval, no DB, no LLM - see channels/common.py)
# --------------------------------------------------------------------------
_RESPONSES = {
    "greeting": [
        "Hi there! What would you like to know about company policy?",
        "Hello! Ask me anything about our HR policies.",
        "Hey! What can I help you with today?",
    ],
    "how_are_you": [
        "I'm doing well, thanks! What can I help you with?",
        "All good here! Is there a policy question I can help with?",
        "Doing great, thanks for asking. What would you like to know?",
    ],
    "thanks": [
        "You're welcome! Let me know if you have another question.",
        "Happy to help! Ask away if there's anything else.",
        "Anytime! I'm here if you need more.",
    ],
    "goodbye": [
        "Bye! Come back any time you have a policy question.",
        "Take care! I'm here whenever you need me.",
        "Goodbye! Have a great day.",
    ],
    "ack": [
        "Great! Let me know if you have another question.",
        "Sounds good. Anything else I can help with?",
        "Okay! Ask me anything about HR policy.",
    ],
    "are_you_there": [
        "Yes, I'm here! What would you like to know?",
        "I'm here. Go ahead and ask.",
        "Here and ready - what can I help you with?",
    ],
}
_CAPABILITIES_WITH_TOPICS = [
    "I'm the HR assistant. I can help with things like {topics}. What would you like to know?",
    "I answer questions about company HR policies, for example {topics}. What can I help with?",
    "I can help with HR policy questions, like {topics}. Just ask.",
]
_CAPABILITIES_FALLBACK = [
    "I'm the HR assistant. I can answer questions about company HR policies. What would you like to know?",
]

_matcher = common.PhraseMatcher(CHANNEL)
_picker = common.ReplyPicker()
GREETING, HOW_ARE_YOU, THANKS, GOODBYE, CAPABILITIES, ACK, ARE_YOU_THERE = (
    "greeting", "how_are_you", "thanks", "goodbye", "capabilities", "ack", "are_you_there",
)


def smalltalk_match(text: str, history: list | None = None) -> str | None:
    """Small-talk category for a pure small-talk utterance, else None. Phrase sets are editable in
    app_config (smalltalk.phrases.text). history is the RAW prior messages: a bare affirmation after an
    assistant question or offer is not small talk (it goes through the normal pipeline)."""
    return common.smalltalk_category(_matcher, text, history)


def smalltalk_reply(category: str) -> str:
    """Reply for a matched category. Only the capabilities reply touches topics
    (Redis-cached; see config/topics.py)."""
    if category == CAPABILITIES:
        topics = topic_phrase()
        if topics is None:
            reply = _picker.pick(category, _CAPABILITIES_FALLBACK)
        else:
            reply = _picker.pick(category, _CAPABILITIES_WITH_TOPICS).format(topics=topics)
    else:
        reply = _picker.pick(category, _RESPONSES[category])
    logger.info("turn type=smalltalk category=%s", category)
    return reply


# --------------------------------------------------------------------------
# Source lookup ("where is this from"): deterministic, 0 LLM calls, 1 retrieval
# --------------------------------------------------------------------------

def source_reply(query: str, history: list | None) -> dict | None:
    """A source, section or page question ("where is this from", "what section", "what page number") when the history has
    an earlier answer: 0 LLM calls, one re-retrieval of that answer's turn. See common.lookup_reply. Returns
    {"text", "chunk_ids", "timing", "type"} or None (the caller falls through to the normal pipeline)."""
    return common.lookup_reply(query, history, CHANNEL, smalltalk_match, _retrieve, NO_MATCH_BASE)


# --------------------------------------------------------------------------
# Catalog ("what documents do you have"): deterministic, 0 LLM / retrieval / embedding calls
# --------------------------------------------------------------------------

def catalog_reply(query: str, history: list | None) -> dict | None:
    """If query is a catalog question (see channels/catalog.py) return {"text"} built from the cached title
    list (in-process copy over Redis over the DB - no call on the turn itself); else None (the caller falls
    through to the normal pipeline)."""
    if not catalog.is_catalog_question(query, history):
        return None
    from config.runtime import DEFAULT_CATALOG_TEMPLATES

    text = catalog.build_reply(
        get_catalog_titles(),
        [] if False else get_topics_cached(),
        common.memo_config("catalog.templates.text", {}) or {},
        spoken=False,
        max_titles=common.memo_config("catalog.text_max_titles", 12),
        max_title_words=common.memo_config("catalog.voice_max_title_words", 5),
        defaults=DEFAULT_CATALOG_TEMPLATES["text"],
    )
    logger.info("turn type=catalog")
    return {"text": text}


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def _retrieve(plan: common.TurnPlan, top_k: int | None = None, candidate_n: int | None = None, k: int | None = None):
    """THE retrieval of a turn: the planned (follow-up-contextualised) query with this channel's depth, advisory
    breadth and expansions. A normal turn and a source lookup both call exactly this, so the retrieval a
    source lookup recomputes cannot drift from the one the answer was built on."""
    return retrieve(plan.retrieval_query, top_k=top_k, candidate_n=candidate_n, k=k,
                    channel=CHANNEL, advisory=plan.advisory, expansions=plan.expansions,
                    expansion_weight=plan.expansion_weight, expansion_candidate_n=plan.expansion_candidate_n,
                    standalone_query=plan.standalone_query, standalone_weight=plan.standalone_weight, ctx_weight=plan.ctx_weight)


def answer_stream(
    query: str,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    history: list | None = None,
    handle: common.TurnHandle | None = None,
):
    """One turn: retrieve ONCE, generate ONCE, stream. A generator yielding dict events:

        {"type": "token", "text": str}
            One streamed piece of the answer, in order.
        {"type": "done", "answer": str, "shortcut": bool, "timing": dict,
         "chunk_ids": list[str], "prompt_version": dict | None,
         "contextualized": bool, "advisory": bool}
            Always the final event. "shortcut" is True when retrieval was too weak
            to attempt generation (no LLM call; canned no_match_response()). "timing"
            has "retrieval_time_ms" always, plus "time_to_first_token_ms" and
            "total_generation_time_ms" when a generation call was made.
            "prompt_version" is the prompt_templates versions used (None on a shortcut).

    Every setting (top_k, candidate_n, rrf k, confidence threshold, model, prompt)
    resolves for THIS channel; an explicit argument wins. history (prior
    user/assistant messages, untrusted) is sanitised, may widen the retrieval query for
    follow-ups, and is sent to the model as prior chat turns (see channels/common.py). handle (optional) lets the
    caller cancel the upstream stream and read what happened (request id, attempts, usage)."""
    handle = handle or common.TurnHandle()
    plan = common.plan_turn(query, history, CHANNEL)
    threshold = get_confidence_threshold(CHANNEL) if confidence_threshold is None else confidence_threshold

    retrieval_start = time.monotonic()
    results = _retrieve(plan, top_k, candidate_n, k)
    retrieval_time_ms = (time.monotonic() - retrieval_start) * 1000

    top_score = results[0]["fused_score"] if results else 0.0
    if not results or top_score < threshold:
        no_match = no_match_response()
        logger.info("turn type=refusal_shortcut top_score=%.4f threshold=%.4f retrieval_ms=%.0f",
                    top_score, threshold, retrieval_time_ms)
        yield {"type": "token", "text": no_match}
        yield {
            "type": "done",
            "answer": no_match,
            "shortcut": True,
            "timing": {"retrieval_time_ms": retrieval_time_ms},
            "chunk_ids": [],
            "prompt_version": None,
            "contextualized": plan.contextualized,
            "advisory": plan.advisory,
        }
        return

    model = get_generation_model(CHANNEL)
    system_prompt, prompt_version = get_system_prompt(CHANNEL)
    messages = common.build_messages(query, results, system_prompt, plan.history)
    stats = yield from common.stream_completion(get_client(), model, messages, channel=CHANNEL, handle=handle)

    timing = {
        "retrieval_time_ms": retrieval_time_ms,
        "time_to_first_token_ms": stats["time_to_first_token_ms"],
        "total_generation_time_ms": stats["total_generation_time_ms"],
    }
    g = handle.info
    logger.info(
        "turn type=%s model=%s prompt=%s chunks=%d contextualized=%s advisory=%s retrieval_ms=%.0f headers_ms=%s ttft_ms=%s total_ms=%.0f "
        "request_id=%s attempts=%s prompt_tokens=%s cached_tokens=%s completion_tokens=%s",
        "error" if stats["fallback"] else "answer", model, prompt_version, len(results), plan.contextualized, plan.advisory, retrieval_time_ms,
        None if g.get("headers_ms") is None else round(g["headers_ms"]),
        None if stats["time_to_first_token_ms"] is None else round(stats["time_to_first_token_ms"]),
        stats["total_generation_time_ms"], g.get("request_id"), g.get("attempts"), g.get("prompt_tokens"), g.get("cached_tokens"), g.get("completion_tokens"))
    yield {
        "type": "done",
        "answer": stats["answer"],
        "shortcut": False,
        "timing": timing,
        "chunk_ids": [r["chunk_id"] for r in results],
        "prompt_version": prompt_version,
        "contextualized": plan.contextualized,
        "advisory": plan.advisory,
    }


def answer(
    query: str,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    history: list | None = None,
) -> dict:
    """Non-streaming answer_stream(): {"type": "answer", "answer", "shortcut", "timing",
    "chunk_ids", "prompt_version", "contextualized", "advisory"}."""
    final_event = None
    for event in answer_stream(query, top_k=top_k, candidate_n=candidate_n, k=k,
                               confidence_threshold=confidence_threshold, history=history):
        if event["type"] == "done":
            final_event = event
    return {
        "type": "answer",
        "answer": final_event["answer"],
        "shortcut": final_event["shortcut"],
        "timing": final_event["timing"],
        "chunk_ids": final_event.get("chunk_ids", []),
        "prompt_version": final_event.get("prompt_version"),
        "contextualized": final_event.get("contextualized", False),
        "advisory": final_event.get("advisory", False),
    }
