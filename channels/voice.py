"""Voice channel (the Deepgram think endpoint, api/voice.py): retrieval -> grounding ->
generation, plus this channel's small talk, copy and settings, all in one file.

Fully separate from the text channel: this module never imports channels.text, and every
setting it reads resolves to a "<key>.voice" row first (see config/channel_settings.py), its
prompt is the voice style + voice grounding_core (config/prompts.py), and its copy below is
written to be spoken aloud (short, no symbols). Shared, behaviour-free helpers live in
channels/common.py. The HTTP shape, history parsing and spoken clarification wording stay in
api/voice.py."""
import logging
import time

from channels import common
from config.channel_settings import channel_setting
from config.prompts import get_system_prompt
from config.topics import topic_phrase
from generation.generate import get_client, get_confidence_threshold, get_generation_model
from vectorstore.db import get_pool
from vectorstore.retrieve import retrieve

CHANNEL = "voice"
logger = logging.getLogger("hrchatbot.voice")

# --------------------------------------------------------------------------
# Copy
# --------------------------------------------------------------------------

# Closed set of attribute names the clarification flow can ask about; the spoken wording of the
# question itself (lead-in plus rephrasings) lives in api/voice.py with the history parsing.
CLARIFYING_QUESTIONS = {
    "country": "Which country are you based in?",
    "employment_type": "Are you a full-time employee or a contractor?",
}

NO_MATCH_BASE = "I couldn't find that in the HR policies."


def no_match_response() -> str:
    """Retrieval-shortcut reply, short enough to sound natural spoken. Names a few real
    topics (cached list, see config/topics.py); falls back to the base sentence."""
    try:
        topics = topic_phrase(2)
    except Exception:
        topics = None
    return f"{NO_MATCH_BASE} I can help with things like {topics}." if topics else NO_MATCH_BASE


# --------------------------------------------------------------------------
# Small talk (deterministic; no retrieval, no DB, no LLM - see channels/common.py)
# --------------------------------------------------------------------------
_PATTERNS = {
    "greeting": rf"{common.GREETING_WORD}(?: {common.GREETING_WORD})?{common.ADDRESS}",  # "hey whats up", "hi there"
    "how_are_you": (
        common.LEAD + r"(?:how are you(?: doing| today)?|how r u|how do you do|hows it going|how is it going|"
        r"how are things|hows everything|how you doing|you good|are you ok(?:ay)?)" + common.ADDRESS
    ),
    "thanks": (
        r"(?:(?:ok|okay|cool|great|alright) )?(?:thanks|thank you|thx|ty|cheers|much appreciated|"
        r"appreciate it|many thanks)(?: (?:a lot|so much|very much|a ton))?" + common.ADDRESS
    ),
    "goodbye": (
        r"(?:bye(?: bye)?|goodbye|see you(?: later| soon)?|see ya|later|good night|take care|"
        r"talk later|ttyl|cya|catch you later)" + common.ADDRESS
    ),
    "capabilities": (
        common.LEAD + r"(?:who are you|what are you|what is your name|whats your name|what can you do(?: for me)?|"
        r"what do you do|what can you help(?: me)? with|how can you help(?: me)?|what can i ask(?: you)?|"
        r"what do you know|help|help me)"
    ),
    "ack": (
        r"(?:ok(?:ay)?|k|kk|cool|got it|alright|all right|understood|i see|noted|great|nice|awesome|"
        r"perfect|makes sense|sounds good)(?: (?:cool|great|got it))?"
    ),
    # Voice-only phrasings: people check the line is live before they ask.
    "are_you_there": (
        r"(?:are you (?:there|listening)|you there|anyone there|is anyone there|can you hear me|"
        r"hello hello|testing|test|mic check|is this thing on)" + common.ADDRESS
    ),
}

_RESPONSES = {
    "greeting": [
        "Hi! What would you like to know about company policy?",
        "Hello! Ask me about our HR policies.",
        "Hey there! How can I help?",
    ],
    "how_are_you": [
        "Doing well, thanks! What can I help you with?",
        "All good! Got a policy question?",
        "Great, thanks for asking. What would you like to know?",
    ],
    "thanks": [
        "You're welcome! Anything else?",
        "Happy to help! Ask away if there's more.",
        "Anytime!",
    ],
    "goodbye": [
        "Bye! Call back any time.",
        "Take care! I'm here whenever you need me.",
        "Goodbye! Have a great day.",
    ],
    "ack": [
        "Great! Anything else?",
        "Okay. Ask me anything else about policy.",
        "Sounds good. What else can I help with?",
    ],
    "are_you_there": [
        "Yes, I'm here. What would you like to know?",
        "I'm here, loud and clear. Go ahead.",
        "Yep, I can hear you. What can I help with?",
    ],
}
_CAPABILITIES_WITH_TOPICS = [
    "I'm the HR assistant. I can help with things like {topics}. What would you like to know?",
    "I can answer questions about HR policy, like {topics}. What can I help with?",
    "Ask me about HR policies, for example {topics}.",
]
_CAPABILITIES_FALLBACK = [
    "I'm the HR assistant. I can answer questions about company HR policies. What would you like to know?",
]

_matcher = common.SmalltalkMatcher(_PATTERNS)
_picker = common.ReplyPicker()
GREETING, HOW_ARE_YOU, THANKS, GOODBYE, CAPABILITIES, ACK, ARE_YOU_THERE = (
    "greeting", "how_are_you", "thanks", "goodbye", "capabilities", "ack", "are_you_there",
)


def smalltalk_match(text: str, *, after_clarification: bool = False) -> str | None:
    """Small-talk category for a pure small-talk utterance, else None. Callers pass
    after_clarification=True when the previous assistant turn was a clarifying question."""
    return _matcher.match(text, after_clarification=after_clarification)


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
# Settings
# --------------------------------------------------------------------------

def default_attributes() -> dict:
    """Attributes assumed for this channel when the caller supplies none
    (app_config user.default_attributes.voice)."""
    return channel_setting("user.default_attributes", CHANNEL, {}) or {}


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def generate_stream(
    query: str,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    user_attributes: dict | None = None,
    history: list | None = None,
    _plan: common.TurnPlan | None = None,
):
    """Retrieves grounding chunks for query, then streams a generated answer. A
    generator yielding dict events:

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
    follow-ups, and is sent to the model as prior chat turns (see channels/common.py)."""
    plan = _plan or common.plan_turn(query, history, CHANNEL)
    threshold = get_confidence_threshold(CHANNEL) if confidence_threshold is None else confidence_threshold

    retrieval_start = time.monotonic()
    results = retrieve(plan.retrieval_query, top_k=top_k, candidate_n=candidate_n, k=k,
                       user_attributes=user_attributes, channel=CHANNEL, advisory=plan.advisory,
                       expansions=plan.expansions)
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
    stats = yield from common.stream_completion(get_client(), model, messages)

    timing = {
        "retrieval_time_ms": retrieval_time_ms,
        "time_to_first_token_ms": stats["time_to_first_token_ms"],
        "total_generation_time_ms": stats["total_generation_time_ms"],
    }
    logger.info("turn type=answer model=%s prompt=%s chunks=%d contextualized=%s advisory=%s retrieval_ms=%.0f ttft_ms=%s total_ms=%.0f",
                model, prompt_version, len(results), plan.contextualized, plan.advisory, retrieval_time_ms,
                None if stats["time_to_first_token_ms"] is None else round(stats["time_to_first_token_ms"]),
                stats["total_generation_time_ms"])
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


def answer_stream(
    query: str,
    user_attributes: dict | None = None,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    history: list | None = None,
):
    """generate_stream() preceded by the applicability check: yields a single
    {"type": "needs_clarification", "question", "blocking_tag_type"} event when the
    user's answer would change which documents apply, else generate_stream()'s events."""
    attributes = common.with_default_attributes(user_attributes, CHANNEL)
    plan = common.plan_turn(query, history, CHANNEL)

    # Unfiltered probe: the SAME context selection generation will use (same query, depth and
    # diversity cap), just without attribute filtering - so only chunks that could actually
    # reach generation are checked, and a broader advisory context is checked no wider than
    # the context it produces. Same query text => the embedding is computed once per turn.
    candidates = retrieve(plan.retrieval_query, top_k=top_k, candidate_n=candidate_n, k=k,
                          channel=CHANNEL, advisory=plan.advisory, expansions=plan.expansions)
    with get_pool().connection() as conn:
        blocking = common.detect_blocking_tag_types(conn, candidates, attributes)
    if blocking:
        tag_type = blocking[0]
        logger.info("turn type=clarification blocking_tag_type=%s", tag_type)
        yield {
            "type": "needs_clarification",
            "question": CLARIFYING_QUESTIONS.get(tag_type, f"Could you tell me your {tag_type.replace('_', ' ')}?"),
            "blocking_tag_type": tag_type,
            "contextualized": plan.contextualized,
            "advisory": plan.advisory,
        }
        return

    yield from generate_stream(
        query, top_k=top_k, candidate_n=candidate_n, k=k,
        confidence_threshold=confidence_threshold, user_attributes=attributes, history=history, _plan=plan,
    )


def answer(
    query: str,
    user_attributes: dict | None = None,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    history: list | None = None,
) -> dict:
    """Non-streaming answer_stream(). Returns either
    {"type": "needs_clarification", "question", "blocking_tag_type"} or
    {"type": "answer", "answer", "shortcut", "timing", "chunk_ids", "prompt_version", "contextualized", "advisory"}."""
    final_event = None
    for event in answer_stream(query, user_attributes, top_k=top_k, candidate_n=candidate_n, k=k,
                               confidence_threshold=confidence_threshold, history=history):
        if event["type"] == "needs_clarification":
            return event
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
