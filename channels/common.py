"""Primitives shared by channels/text.py and channels/voice.py.

Only things with NO channel behaviour live here: message building, the LLM
streaming loop, the applicability check, and the small-talk matching /
reply-picking machinery. Which patterns count as small talk, what the
replies say, how settings are resolved and what the pipeline does with them
are all decided in each channel module.
"""
import random
import re
import threading
import time
from dataclasses import dataclass, field

from config.channel_settings import channel_setting
from config.runtime import get_app_config
from vectorstore.db import get_pool

# ---- attributes -----------------------------------------------------------------


def with_default_attributes(user_attributes: dict | None, channel: str) -> dict:
    """Fills attributes the caller doesn't know with the channel's
    'user.default_attributes.<channel>' defaults (e.g. {"country": "India"} while
    every document is India-only). Known caller values always win."""
    defaults = channel_setting("user.default_attributes", channel, {}) or {}
    known = {k: v for k, v in (user_attributes or {}).items() if v is not None}
    return {**defaults, **known}


def detect_blocking_tag_types(conn, candidate_results: list[dict], user_attributes: dict | None) -> list[str]:
    """Given chunk results from an UNFILTERED retrieve() call and the user's known
    attributes, returns which tag_type(s) (if any) are "blocking": a tag_type for
    which the candidates' documents carry MORE THAN ONE distinct tag value (so the
    user's answer would change which documents apply) but user_attributes has no
    known (non-None) value for it. A tag_type with a single value across the
    candidates (e.g. every document is India-only) is never asked about."""
    user_attributes = user_attributes or {}
    document_ids = {r["document_id"] for r in candidate_results}
    if not document_ids:
        return []
    rows = conn.execute(
        """
        SELECT at.tag_type
        FROM document_applicability da
        JOIN applicability_tags at ON at.id = da.tag_id
        WHERE da.document_id = ANY(%s)
        GROUP BY at.tag_type
        HAVING COUNT(DISTINCT at.tag_value) > 1
        """,
        (list(document_ids),),
    ).fetchall()
    return sorted(tag_type for (tag_type,) in rows if user_attributes.get(tag_type) is None)


# ---- generation ---------------------------------------------------------------------


def build_messages(query: str, chunks: list[dict], system_prompt: str, history: list[dict] | None = None) -> list[dict]:
    """[system prompt (stable first, so provider prefix caching applies)] + prior chat
    turns + ONE final user message holding the retrieved excerpts and the current
    question. Excerpts attach to the current turn only, never to history turns."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        blocks.append(
            f"[Excerpt {i} - internal source label, never speak this: "
            f"{chunk['document_title']} > {chunk['section_path']}]\n{chunk['text']}"
        )
    context = "\n\n".join(blocks)
    user_content = (
        "Retrieved policy excerpts (internal grounding only - never mention the "
        "excerpt labels, source titles, or section names in your answer):\n\n"
        f"{context}\n\nEmployee question: {query}"
    )
    return [
        {"role": "system", "content": system_prompt},
        *(history or []),
        {"role": "user", "content": user_content},
    ]


def stream_completion(client, model: str, messages: list[dict]):
    """Generator: yields {"type": "token", "text": ...} per streamed piece and
    RETURNS {"answer", "time_to_first_token_ms", "total_generation_time_ms"}
    (use `stats = yield from stream_completion(...)`). Exactly one LLM call."""
    start = time.monotonic()
    first_token_time = None
    pieces = []
    for event in client.chat.completions.create(model=model, messages=messages, stream=True):
        delta = event.choices[0].delta.content
        if not delta:
            continue
        if first_token_time is None:
            first_token_time = time.monotonic()
        pieces.append(delta)
        yield {"type": "token", "text": delta}
    end = time.monotonic()
    return {
        "answer": "".join(pieces),
        "time_to_first_token_ms": (first_token_time - start) * 1000 if first_token_time is not None else None,
        "total_generation_time_ms": (end - start) * 1000,
    }


# ---- conversation context, follow-ups and advisory turns (deterministic, no LLM) ------------------

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_history(history: list | None, channel: str) -> list[dict]:
    """Server-side sanitisation of client/Deepgram-supplied history, which is
    UNTRUSTED: only user/assistant roles with string content survive, control
    characters are dropped, the list is capped to the last
    conversation.history_messages messages and each is truncated to
    conversation.history_max_chars characters."""
    if not history:
        return []
    max_messages = channel_setting("conversation.history_messages", channel, 6)
    max_chars = channel_setting("conversation.history_max_chars", channel, 400)
    kept = []
    for message in history:
        if not isinstance(message, dict) or message.get("role") not in ("user", "assistant"):
            continue
        content = message.get("content")
        if not isinstance(content, str):
            continue
        content = _CONTROL_CHARS.sub("", content).strip()
        if content:
            kept.append({"role": message["role"], "content": content[:max_chars]})
    return kept[-max_messages:]


def _contains_phrase(normalized_text: str, phrases) -> bool:
    padded = f" {normalized_text} "
    return any(p and f" {normalize(p)} " in padded for p in phrases or [])


def build_retrieval_query(current: str, history: list[dict], channel: str) -> tuple[str, bool]:
    """(query, contextualized). A SHORT utterance (fewer than
    retrieval.context.max_words_followup words) or one containing a follow-up marker
    gets the previous retrieval.context.prior_user_turns USER messages prepended, so
    "is there anything else I can do about that" keeps its referent. Otherwise, or
    with no prior user message, the query is returned unchanged. One string in, one
    embedding call out."""
    prior = [m["content"] for m in history if m["role"] == "user"]
    if not prior:
        return current, False
    max_words = channel_setting("retrieval.context.max_words_followup", channel, 8)
    norm = normalize(current)
    follow_up = len(norm.split()) < max_words or _contains_phrase(
        norm, get_app_config("retrieval.context.followup_markers", []))
    if not follow_up:
        return current, False
    n = channel_setting("retrieval.context.prior_user_turns", channel, 2)
    return " ".join([*prior[-n:], current]), True


def detect_advisory(current: str, history: list[dict]) -> bool:
    """True when the user is asking for options / ways around a problem (matched
    against retrieval.advisory.markers on the normalized text). A bare follow-up
    ("what else?", "any other option") counts as advisory when an earlier user
    message in the history was advisory."""
    markers = get_app_config("retrieval.advisory.markers", [])
    norm = normalize(current)
    if _contains_phrase(norm, markers):
        return True
    if _contains_phrase(norm, ("what else", "anything else", "any other")):
        return any(_contains_phrase(normalize(m["content"]), markers) for m in history if m["role"] == "user")
    return False


def select_expansions(current: str, history: list[dict], channel: str) -> list[str]:
    """Expansion queries for an advisory turn: for every topic in retrieval.expansions
    whose trigger phrases appear in the current utterance or the prior USER messages,
    that topic's queries (de-duplicated, in order). [] when this channel has expansion
    disabled (retrieval.expansion.enabled.<channel>)."""
    if not channel_setting("retrieval.expansion.enabled", channel, True):
        return []
    text = normalize(" ".join([m["content"] for m in history if m["role"] == "user"] + [current]))
    queries: list[str] = []
    for topic in get_app_config("retrieval.expansions", []) or []:
        if _contains_phrase(text, topic.get("triggers", [])):
            queries += [q for q in topic.get("queries", []) if q not in queries]
    return queries


@dataclass
class TurnPlan:
    history: list = field(default_factory=list)
    retrieval_query: str = ""
    contextualized: bool = False
    advisory: bool = False
    expansions: list = field(default_factory=list)


def plan_turn(current: str, history: list | None, channel: str) -> TurnPlan:
    """Everything the pipeline decides about a turn BEFORE retrieval, deterministically."""
    cleaned = clean_history(history, channel)
    query, contextualized = build_retrieval_query(current, cleaned, channel)
    advisory = detect_advisory(current, cleaned)
    return TurnPlan(cleaned, query, contextualized, advisory,
                    select_expansions(current, cleaned, channel) if advisory else [])


# ---- small-talk machinery -----------------------------------------------------------

# Regex building blocks a channel's pattern sets can compose.
ADDRESS = r"(?: (?:there|bot|assistant|team|everyone|all|buddy))?"
GREETING_WORD = (
    r"(?:hi+|hello+|hey+|heya|hiya|yo|sup|wassup|wazzup|whats up|whatsup|howdy|greetings|"
    r"good (?:morning|afternoon|evening|day))"
)
LEAD = rf"(?:{GREETING_WORD}{ADDRESS} )?"  # optional greeting before how-are-you / capabilities

_APOSTROPHES = re.compile(r"['’`]")
_NON_WORD = re.compile(r"[^a-z0-9]+")


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes (what's -> whats), turn other punctuation into
    spaces, collapse whitespace."""
    return _NON_WORD.sub(" ", _APOSTROPHES.sub("", text.lower())).strip()


class SmalltalkMatcher:
    """Whole-utterance matcher over a channel's pattern sets: category -> regex
    source. An utterance matches only if all of it (<= max_words words) is a pure
    small-talk phrase, so a real question is never swallowed."""

    def __init__(self, patterns: dict[str, str], max_words: int = 6):
        self.max_words = max_words
        self._compiled = {cat: re.compile(rf"^{pat}$") for cat, pat in patterns.items()}

    def match(self, text: str, *, after_clarification: bool = False) -> str | None:
        """Category for a pure small-talk utterance, else None. With
        after_clarification=True (the previous assistant turn was a clarifying
        question) never matches, so a short reply like "ok" is not misread."""
        if after_clarification:
            return None
        norm = normalize(text)
        if not norm or len(norm.split()) > self.max_words:
            return None
        for category, pattern in self._compiled.items():
            if pattern.match(norm):
                return category
        return None


class ReplyPicker:
    """Random choice that never repeats the previous pick for a category."""

    def __init__(self):
        self._last: dict[str, int] = {}
        self._lock = threading.Lock()

    def pick(self, category: str, options: list[str]) -> str:
        with self._lock:
            choices = [i for i in range(len(options)) if i != self._last.get(category)] or [0]
            idx = random.choice(choices)
            self._last[category] = idx
        return options[idx]
