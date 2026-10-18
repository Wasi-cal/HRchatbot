"""Primitives shared by channels/text.py and channels/voice.py.

Only things with NO channel behaviour live here: message building, the LLM
streaming loop, the applicability check, and the small-talk matching /
reply-picking machinery. Which patterns count as small talk, what the
replies say, how settings are resolved and what the pipeline does with them
are all decided in each channel module.
"""
import json
import logging
import random
import re
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache

from config.channel_settings import channel_setting
from config.runtime import get_app_config, memo_provider_config

# ---- generation ---------------------------------------------------------------------


def build_messages(query: str, chunks: list[dict], system_prompt: str, history: list[dict] | None = None) -> list[dict]:
    """[system prompt (stable first, so provider prefix caching applies)] + prior chat
    turns + ONE final user message holding the retrieved excerpts and the current
    question. Excerpts attach to the current turn only, never to history turns."""
    blocks = []
    for i, chunk in enumerate(chunks, start=1):
        label = f"[Applies to: {chunk['applies_to']}]\n" if chunk.get("applies_to") else ""
        blocks.append(
            f"[Excerpt {i} - internal source label, never speak this: "
            f"{chunk['document_title']} > {chunk['section_path']}]\n{label}{chunk['text']}"
        )
    context = "\n\n".join(blocks)
    reminder = (memo_config("prompts.turn_reminder", "") or "").strip()  # prompts.turn_reminder: after the excerpts, before the question
    user_content = (
        "Retrieved policy excerpts (internal grounding only - never mention the "
        "excerpt labels, source titles, or section names in your answer):\n\n"
        f"{context}\n\n" + (f"{reminder}\n\n" if reminder else "") + f"Employee question: {query}"
    )
    return [
        {"role": "system", "content": system_prompt},
        *(history or []),
        {"role": "user", "content": user_content},
    ]


class TurnCancelled(Exception):
    """The turn was abandoned (the client disconnected, e.g. Deepgram barge-in). Raised inside the generation so the
    pipeline stops; callers log the turn as aborted."""


class TurnHandle:
    """Shared by the voice endpoint and the generation call of ONE turn: lets the endpoint cancel the upstream OpenAI
    stream from another thread (closing it, so the provider stops generating) and collects what happened
    (`info`: request_id, attempts, headers_ms, ttft_ms, token counts, fallback, ...)."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stream = None
        self.cancelled = False
        self.expired = False
        self.info: dict = {}

    def attach(self, stream) -> None:
        with self._lock:
            self._stream = stream
            if self.cancelled or self.expired:
                self._close_locked()

    def _close_locked(self) -> None:
        stream = self._stream
        if stream is not None:
            try:
                stream.close()
            except Exception:  # noqa: BLE001, S110
                pass

    def cancel(self) -> None:
        with self._lock:
            self.cancelled = True
            self.info["closed_upstream"] = self._stream is not None
            self._close_locked()

    def expire(self) -> None:
        with self._lock:
            self.expired = True
            self._close_locked()


def generation_settings(channel: str) -> dict:
    """Timeout / retry / fallback settings for a channel (app_config, read through the in-process copy)."""
    provider = memo_provider_config("generation")
    return {
        "temperature": provider.get("temperature"),
        "top_p": provider.get("top_p"),
        "first_token_timeout_s": float(memo_config("generation.first_token_timeout_s", 2.0)),
        "connect_timeout_s": float(memo_config("generation.connect_timeout_s", 3.0)),
        "stream_gap_timeout_s": float(memo_config("generation.stream_gap_timeout_s", 6.0)),
        "request_timeout_s": float(memo_config("generation.request_timeout_s", 20)),
        "max_retries": int(memo_config(f"generation.max_retries.{channel}", 1)),
        "fallback_message": memo_config("generation.fallback_message", "Sorry, I'm having trouble right now. Please try again."),
    }


_gen_logger = logging.getLogger("hrchatbot.generation")


def _with_options(client, cfg: dict):
    """The same client (shared connection pool) with per-attempt options: no automatic SDK retries (retrying is done
    here, with a log line per retry), a connect timeout (generation.connect_timeout_s) and a read timeout of
    first_token_timeout_s, which bounds the wait for the response headers and the first token. Once the response has started,
    _apply_stream_gap() relaxes the read timeout to generation.stream_gap_timeout_s."""
    if not hasattr(client, "with_options"):
        return client
    import httpx

    return client.with_options(
        timeout=httpx.Timeout(connect=cfg.get("connect_timeout_s", 3.0), read=cfg["first_token_timeout_s"], write=10.0, pool=5.0),
        max_retries=0,
    )


def _apply_stream_gap(stream, cfg: dict) -> None:
    """The first-token timeout is the read timeout of the request, which httpx would also apply between streamed chunks; a
    mid-answer gap cannot be retried (speech has started), so after the response headers have arrived the read timeout of THIS
    request is raised to generation.stream_gap_timeout_s. httpcore re-reads request.extensions["timeout"] for every read of the
    body, so changing it takes effect for the rest of the stream. Best effort: any failure leaves the first-token timeout in place."""
    try:
        timeouts = stream.response.request.extensions.get("timeout")
        if isinstance(timeouts, dict):
            timeouts["read"] = cfg.get("stream_gap_timeout_s", 6.0)
    except Exception:  # noqa: BLE001, S110
        pass


def _usage_fields(usage) -> dict:
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "cached_tokens": getattr(details, "cached_tokens", None) if details is not None else None,
    }


def stream_completion(client, model: str, messages: list[dict], *, channel: str = "text", handle: "TurnHandle | None" = None,
                      settings: dict | None = None):
    """Generator: yields {"type": "token", "text": ...} per streamed piece and RETURNS
    {"answer", "time_to_first_token_ms", "total_generation_time_ms", "fallback"} (use `stats = yield from
    stream_completion(...)`). A normal turn makes exactly ONE OpenAI call.

    The retry / timeout / cancel logic only runs on failure or cancel: an attempt that fails or gets no first token
    within first_token_timeout_s is closed and, while retries remain, repeated; after the last retry the configured
    fallback message is streamed instead of hanging. Once speech has started nothing is retried. If `handle` is
    cancelled (client disconnect) the upstream stream is closed and TurnCancelled is raised. `handle.info` records
    request id, attempts, time to response headers, time to first token and usage (the usage chunk is requested in the
    same call via stream_options)."""
    cfg = settings or generation_settings(channel)
    handle = handle or TurnHandle()
    info = handle.info
    start = time.monotonic()
    first_token_time = None
    pieces: list[str] = []
    info.update(attempts=0, headers_ms=None, ttft_ms=None, request_id=None, chunks=0, fallback=False, error=None)
    max_attempts = 1 + max(0, cfg["max_retries"])
    for attempt in range(1, max_attempts + 1):
        info["attempts"] = attempt
        stream = None
        timer = None
        completed = False
        try:
            attempt_start = time.monotonic()
            sampling = {k: cfg[k] for k in ("temperature", "top_p") if cfg.get(k) is not None}
            stream = _with_options(client, cfg).chat.completions.create(
                model=model, messages=messages, stream=True, stream_options={"include_usage": True}, **sampling)
            info["headers_ms"] = (time.monotonic() - attempt_start) * 1000
            headers = getattr(getattr(stream, "response", None), "headers", None) or {}
            info["request_id"] = headers.get("x-request-id") if hasattr(headers, "get") else None
            _apply_stream_gap(stream, cfg)
            handle.attach(stream)
            timer = threading.Timer(max(cfg["request_timeout_s"] - (time.monotonic() - start), 0.1), handle.expire)
            timer.daemon = True
            timer.start()
            for event in stream:
                if handle.cancelled:
                    raise TurnCancelled()
                usage = getattr(event, "usage", None)
                if usage is not None:
                    info.update(_usage_fields(usage))
                choices = getattr(event, "choices", None)
                if not choices:
                    continue
                delta = choices[0].delta.content
                if not delta:
                    continue
                if first_token_time is None:
                    first_token_time = time.monotonic()
                    info["ttft_ms"] = (first_token_time - start) * 1000
                pieces.append(delta)
                info["chunks"] = len(pieces)
                yield {"type": "token", "text": delta}
            if handle.cancelled:
                raise TurnCancelled()
            completed = True
            break
        except (TurnCancelled, GeneratorExit):
            raise
        except Exception as exc:
            if handle.cancelled:
                raise TurnCancelled() from exc
            reason = f"{type(exc).__name__}: {str(exc)[:120]}"
            elapsed_ms = (time.monotonic() - start) * 1000
            if pieces:  # speech already started: cannot retry, stop here
                info["error"] = reason
                _gen_logger.error("think event=error stage=mid_stream attempt=%d elapsed_ms=%.0f reason=%s", attempt, elapsed_ms, reason)
                break
            if attempt < max_attempts:
                _gen_logger.warning("think event=retry attempt=%d of %d elapsed_ms=%.0f first_token_timeout_s=%.1f reason=%s",
                                    attempt, max_attempts, elapsed_ms, cfg["first_token_timeout_s"], reason)
                continue
            info["error"] = reason
            info["fallback"] = True
            _gen_logger.error("think event=fallback attempts=%d elapsed_ms=%.0f reason=%s", attempt, elapsed_ms, reason)
            pieces.append(cfg["fallback_message"])
            yield {"type": "token", "text": cfg["fallback_message"]}
            break
        finally:
            if timer is not None:
                timer.cancel()
            if stream is not None and not completed:  # a fully read stream returns its connection to the pool for reuse
                try:
                    stream.close()
                except Exception:  # noqa: BLE001, S110
                    pass
    end = time.monotonic()
    return {
        "answer": "".join(pieces),
        "time_to_first_token_ms": info["ttft_ms"],
        "total_generation_time_ms": (end - start) * 1000,
        "fallback": info["fallback"],
    }


# ---- conversation context, follow-ups and advisory turns (deterministic, no LLM) ------------------

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _truncate(content: str, max_chars: int, keep_tail: bool) -> str:
    if len(content) <= max_chars:
        return content
    if not keep_tail:
        return content[:max_chars]
    separator = " ... "
    tail = max_chars // 3
    return content[: max_chars - tail - len(separator)] + separator + content[-tail:]


def clean_history(history: list | None, channel: str, max_messages: int | None = None) -> list[dict]:
    """Server-side sanitisation of client/Deepgram-supplied history, which is
    UNTRUSTED: only user/assistant roles with string content survive, control
    characters are dropped, the list is capped to the last
    conversation.history_messages messages and each is truncated to
    conversation.history_max_chars characters. An over-long ASSISTANT message keeps its
    head and its tail (an offer or a question usually closes it, and "yes" is answered
    against that), still within the character cap. `max_messages` overrides the cap (lookups walk back further)."""
    if not history:
        return []
    max_messages = max_messages or channel_setting("conversation.history_messages", channel, 6)
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
            kept.append({"role": message["role"], "content": _truncate(content, max_chars, keep_tail=message["role"] == "assistant")})
    return kept[-max_messages:]


def _contains_phrase(normalized_text: str, phrases) -> bool:
    padded = f" {normalized_text} "
    return any(p and f" {normalize(p)} " in padded for p in phrases or [])


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _offer_sentence(assistant_text: str) -> str | None:
    """The last sentence of an assistant turn that contains the offer: one with an offer marker ("would you like", "want me
    to", "i can also"), else one ending in a question mark, else the last sentence."""
    sentences = [x.strip() for x in _SENTENCE_SPLIT.split(assistant_text.strip()) if x.strip()]
    if not sentences:
        return None
    markers = [m.lower() for m in memo_config("smalltalk.offer_markers", []) or []]
    for sentence in reversed(sentences):
        if any(m in sentence.lower().replace("\u2019", "'") for m in markers):
            return sentence
    for sentence in reversed(sentences):
        if sentence.rstrip().endswith("?"):
            return sentence
    return sentences[-1]


def _has_identity_marker(norm: str) -> bool:
    """"i am ...", "as a ..." introduce who the user IS; they mark a follow-up only in a SHORT utterance
    (retrieval.context.identity_max_words), where they refer back to the topic ("i am an intern"). In a longer
    sentence ("I am relocating to another city, what does the policy say") they are just words of a fresh question."""
    return (len(norm.split()) <= memo_config("retrieval.context.identity_max_words", 6)
            and _contains_phrase(norm, memo_config("retrieval.context.identity_markers", [])))


def _context_weights(current: str, channel: str) -> tuple[float, float]:
    """(standalone_weight, ctx_weight) for a contextualised turn. A follow-up marker ("that", "what about", ...) or a very
    short utterance (fewer than retrieval.context.ctx_heavy_below_words words) shifts the weights toward the contextualised
    query; a short identity statement ("i am an intern") uses retrieval.context.weights_identity (equal); anything else
    (merely SHORT, no marker) leans on the standalone query. Pairs are app_config objects {"standalone": x, "ctx": y};
    a missing key falls back to retrieval.context.weight_standalone / weight_ctx."""
    norm = normalize(current)
    if (_contains_phrase(norm, memo_config("retrieval.context.followup_markers", []))
            or len(norm.split()) < memo_config("retrieval.context.ctx_heavy_below_words", 3)):
        key = "retrieval.context.weights_with_marker"
    elif _has_identity_marker(norm):
        key = "retrieval.context.weights_identity"
    else:
        key = "retrieval.context.weights_without_marker"
    pair = memo_config(key, {}) or {}
    standalone = pair.get("standalone", memo_config("retrieval.context.weight_standalone", 1.0))
    ctx = pair.get("ctx", memo_config("retrieval.context.weight_ctx", 1.0))
    return float(standalone), float(ctx)


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
    follow_up = (len(norm.split()) < max_words
                 or _contains_phrase(norm, get_app_config("retrieval.context.followup_markers", []))
                 or _has_identity_marker(norm))
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
    expansion_weight: float | None = None       # None -> retrieval.expansion.weight
    expansion_candidate_n: int | None = None    # None -> retrieval.expansion.candidate_n
    standalone_query: str | None = None         # the bare utterance, searched next to the contextualised retrieval_query
    standalone_weight: float = 1.0
    ctx_weight: float = 1.0


def _affirmation_plan(current: str, cleaned: list[dict], channel: str) -> TurnPlan | None:
    """A bare "yes" after an assistant offer ("Want me to go through the work-from-home option?"): the main retrieval query
    is the OFFER sentence of the previous assistant turn and the last user message rides along as a second query fused in
    (one batched embedding, one retrieval, no extra call). Embedding "offer + user message" as ONE string let the long user
    message drown the offer (measured: the work-from-home chunks fell out of the top 3), so the two are fused with
    retrieval.context.affirmation_user_weight / affirmation_candidate_n instead."""
    if not (is_affirmation(current) and previous_turn_invites_reply(cleaned)):
        return None
    users = [m["content"] for m in cleaned if m["role"] == "user"]
    offer = _offer_sentence(cleaned[-1]["content"])
    if not users or not offer:
        return None
    return TurnPlan(cleaned, offer, True, False, [users[-1]],
                    float(memo_config("retrieval.context.affirmation_user_weight", 1.0)),
                    int(memo_config("retrieval.context.affirmation_candidate_n", 5)))


def plan_turn(current: str, history: list | None, channel: str) -> TurnPlan:
    """Everything the pipeline decides about a turn BEFORE retrieval, deterministically."""
    cleaned = clean_history(history, channel)
    affirmation = _affirmation_plan(current, cleaned, channel)
    if affirmation is not None:
        return affirmation
    query, contextualized = build_retrieval_query(current, cleaned, channel)
    advisory = detect_advisory(current, cleaned)
    plan = TurnPlan(cleaned, query, contextualized, advisory,
                    select_expansions(current, cleaned, channel) if advisory else [])
    if contextualized and memo_config("retrieval.context.dual_query.enabled", True):
        # the contextualised text is searched next to the bare utterance (one batched encode, one retrieve() call)
        plan.standalone_query = current
        plan.standalone_weight, plan.ctx_weight = _context_weights(current, channel)
    return plan


from config.runtime import (  # noqa: F401  (kept here: channels use common.memo_config)
    clear_config_memo,
    memo_config,
)

# ---- text normalisation ---------------------------------------------------------------------------------

_APOSTROPHES = re.compile(r"['\u2019`]")
_NON_WORD = re.compile(r"[^a-z0-9]+")
_REPEATS = re.compile(r"(.)\1{2,}")


def normalize(text: str) -> str:
    """Lowercase, drop apostrophes (what's -> whats), turn other punctuation into
    spaces, collapse whitespace."""
    return _NON_WORD.sub(" ", _APOSTROPHES.sub("", text.lower())).strip()


# ---- small talk: phrase sets live in app_config (smalltalk.phrases.<channel>) ----------------------------

SMALLTALK_ADDRESS_WORDS = {"there", "bot", "assistant", "team", "everyone", "all", "buddy"}
_LEADABLE = {"greeting", "how_are_you", "capabilities"}  # categories that may follow a leading greeting


class PhraseMatcher:
    """Whole-utterance small-talk matcher over the channel's phrase sets (category -> list of normalized
    phrases, from app_config via the memo). A turn matches only if ALL of it (<= max_words words, after
    collapsing runs of 3+ repeated letters: "heyyyy" -> "heyy") is one phrase, optionally with a trailing
    address word ("hi there") or, for greeting / how_are_you / capabilities, a leading greeting ("hi how
    are you"). A real question is never swallowed."""

    def __init__(self, channel: str, max_words: int = 6):
        self.channel, self.max_words = channel, max_words
        self._source = None
        self._index: dict[str, str] = {}
        self._greetings: set[str] = set()

    def _refresh(self) -> None:
        phrases = memo_config(f"smalltalk.phrases.{self.channel}", {}) or {}
        if phrases is self._source:
            return
        index: dict[str, str] = {}
        for category, items in phrases.items():
            for phrase in items:
                index.setdefault(_REPEATS.sub(r"\1\1", normalize(phrase)), category)
        self._index, self._greetings, self._source = index, {p for p, c in index.items() if c == "greeting"}, phrases

    def match(self, text: str) -> str | None:
        norm = _REPEATS.sub(r"\1\1", normalize(text))
        words = norm.split()
        if not words or len(words) > self.max_words:
            return None
        self._refresh()
        candidates = [norm]
        if len(words) > 1 and words[-1] in SMALLTALK_ADDRESS_WORDS:
            candidates.append(" ".join(words[:-1]))
        for base in candidates:
            if base in self._index:
                return self._index[base]
            parts = base.split()
            for i in range(1, len(parts)):
                lead, rest = " ".join(parts[:i]), " ".join(parts[i:])
                if lead in self._greetings and self._index.get(rest) in _LEADABLE:
                    return self._index[rest]
        return None


def is_affirmation(text: str) -> bool:
    """A bare yes-like utterance (smalltalk.affirmations)."""
    norm = _REPEATS.sub(r"\1\1", normalize(text))
    return norm in {_REPEATS.sub(r"\1\1", normalize(a)) for a in memo_config("smalltalk.affirmations", []) or []}


def previous_turn_invites_reply(history: list | None) -> bool:
    """True when the LAST history message is an assistant message that ends with a question mark or
    contains an offer (smalltalk.offer_markers: "would you like", "want me to", "i can also"). Uses the
    raw, untruncated history so the end of a long answer is seen."""
    if not history:
        return False
    last = history[-1]
    if not isinstance(last, dict) or last.get("role") != "assistant" or not isinstance(last.get("content"), str):
        return False
    text = last["content"].strip().lower().replace("\u2019", "'")
    if text.rstrip(" \t\n\"')*_\u201d").endswith("?"):
        return True
    return any(m.lower() in text for m in memo_config("smalltalk.offer_markers", []) or [])


def smalltalk_category(matcher: PhraseMatcher, text: str, history: list | None) -> str | None:
    """matcher.match(), except that a bare affirmation after an assistant question or offer is NOT small
    talk: it goes through the normal pipeline (with history) so the offer is followed up."""
    category = matcher.match(text)
    if category and is_affirmation(text) and previous_turn_invites_reply(history):
        return None
    return category


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


# ---- source questions ("where is this from") ---------------------------------------------------------------

REFUSAL_PREFIX = "i couldn't find"


def expand_informal(normalized: str, mapping: dict) -> str:
    """Apply an informal-form map (u -> you, docs -> documents, rn -> dropped ...) to a NORMALIZED string, longest
    keys first, on word boundaries; an empty replacement drops the word. Shared by the catalog and source matchers."""
    for key in sorted(mapping, key=len, reverse=True):
        k = normalize(key)
        if k:
            normalized = re.sub(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", f" {mapping[key]} ", normalized)
    return " ".join(normalized.split())


def _source_form(text: str, mapping: dict) -> str:
    return expand_informal(normalize(text), mapping)


def _cfg(key: str):
    """memo_config with the code default, so a key added after the database was seeded still works (no DB read per turn)."""
    from config.runtime import DEFAULT_APP_CONFIG

    return memo_config(key, DEFAULT_APP_CONFIG[key][0])


@lru_cache(maxsize=32)
def _normalized_set(items_json: str, mapping_json: str) -> frozenset:
    mapping = json.loads(mapping_json)
    return frozenset(_source_form(i, mapping) for i in json.loads(items_json))


def _forms(key: str, mapping: dict) -> frozenset:
    """The normalized/informal-expanded forms of an editable list; cached per (list, map) content, so the cost is paid
    once per config change rather than on every turn."""
    return _normalized_set(json.dumps(_cfg(key)), json.dumps(mapping, sort_keys=True))


def _has_phrase(joined: str, phrase: str) -> bool:
    return f" {phrase} " in f" {joined} "


def _noun_lookup(words: list[str], nouns: frozenset, wh: frozenset, tail: frozenset) -> bool:
    """'what section is that in', 'which part of the handbook', 'what page number', 'what section of the HR policies did
    you get it from': the utterance STARTS with what/which, then (optionally exact/specific) a section/page noun, then
    only words from lookup.tail_words - after an optional 'of <up to 2 words>'. 'what section of the policy covers sick
    leave' fails that tail test and so goes to the normal pipeline."""
    if len(words) < 2 or words[0] not in wh - {"where", "from"}:
        return False
    i = 1
    while i < len(words) and words[i] in {"exact", "specific", "particular"}:
        i += 1
    if i >= len(words) or words[i] not in nouns:
        return False
    rest = words[i + 1:]
    if rest and rest[0] == "of":
        rest = rest[1:]
        if rest and rest[0] in {"the", "this", "that", "your", "our", "these", "those"}:
            rest = rest[1:]
        taken = 0
        while rest and taken < 2 and rest[0] not in tail:
            rest, taken = rest[1:], taken + 1
    return all(w in tail for w in rest)


def lookup_intent(text: str) -> str | None:
    """'source' | 'section' | 'page' | None: is this WHOLE utterance (<= source.max_words words, filler words stripped at
    either end, informal forms expanded) a question about where the previous answer came from? Fixed markers
    (source.markers + source.markers.extra, section.markers, page.markers) and the editable what/you/verb/anaphor pattern
    combination. Pure in-process config reads: no DB, network or LLM call."""
    norm = normalize(text)
    if not norm or len(norm.split()) > int(_cfg("source.max_words")) + 4:  # cheap bound before any config work
        return None
    mapping = memo_config("catalog.normalize_map", {}) or {}
    fillers = {normalize(w) for w in memo_config("source.filler_words", []) or []}
    words = _source_form(text, mapping).split()
    while words and words[0] in fillers:
        words.pop(0)
    while words and words[-1] in fillers:
        words.pop()
    if not words or len(words) > int(_cfg("source.max_words")):
        return None
    joined = " ".join(words)
    if joined in _forms("page.markers", mapping):
        return "page"
    if joined in _forms("section.markers", mapping):
        return "section"
    if joined in _forms("source.markers", mapping) or joined in _forms("source.markers.extra", mapping):
        return "source"
    # pattern rules: vetoed by any lookup.exclude_words ("where can I find the leave form")
    if {w for w in words} & {normalize(x) for x in _cfg("lookup.exclude_words")}:
        return None
    wh = frozenset(normalize(x) for x in _cfg("lookup.wh_words"))
    tail = frozenset(normalize(x) for x in _cfg("lookup.tail_words"))
    if _noun_lookup(words, frozenset(normalize(x) for x in _cfg("page.words")), wh, tail):
        return "page"
    if _noun_lookup(words, frozenset(normalize(x) for x in _cfg("section.words")), wh, tail):
        return "section"
    word_set = set(words)
    if (word_set & wh and word_set & {normalize(x) for x in _cfg("lookup.you_tokens")}
            and word_set & {normalize(x) for x in _cfg("lookup.source_verbs")}
            and any(_has_phrase(joined, normalize(a)) for a in _cfg("lookup.anaphors"))):
        return "source"
    return None


def is_source_question(text: str) -> bool:
    """Kept for callers that only care about a plain source question."""
    return lookup_intent(text) == "source"


def previous_exchange(history: list[dict], skip=None) -> tuple[int, int] | None:
    """(user_index, assistant_index) of the exchange a lookup refers to, in CLEANED history: the most recent exchange that
    produced an ANSWER. Earlier source / section / page questions are walked past, and so are replies that never had a
    document (small talk, catalog) when `skip(user_text, history_before_it)` says so (returns a truthy label). If only
    such replies exist, the most recent of them is returned so the caller can say "that didn't come from a policy
    document". None when there is no assistant message to refer to."""
    end = len(history) - 1
    fallback = None
    while end >= 0:
        a = next((i for i in range(end, -1, -1) if history[i]["role"] == "assistant"), None)
        if a is None:
            break
        u = next((i for i in range(a - 1, -1, -1) if history[i]["role"] == "user"), None)
        if u is None:
            break
        if lookup_intent(history[u]["content"]):
            end = u - 1
            continue
        if skip is not None and skip(history[u]["content"], history[:u]):
            fallback = fallback or (u, a)
            end = u - 1
            continue
        return u, a
    return fallback


# ---- section names for speech ------------------------------------------------------------------------------

_NUMBERING = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")
_TRAILING_PUNCT = re.compile(r"[\s:;,.?!]+$")
_SMALL_WORDS = {"of", "and", "the", "in", "to", "for", "on", "or", "a", "an", "at", "if"}


def clean_section_name(raw: str | None, title: str, spoken: bool) -> str | None:
    """A heading as it reads aloud: leading numbering ("3.0", "4.2.") and trailing punctuation dropped, an ALL-CAPS heading
    title-cased (lower-cased when spoken), '&' read as 'and'. None for a heading that is empty, a table-row label
    (lookup.skip_section_patterns), equal to the document title, or too generic to mean anything alone
    (lookup.generic_sections)."""
    if not raw or not raw.strip():
        return None
    if any(re.search(p, raw.strip(), re.IGNORECASE) for p in _cfg("lookup.skip_section_patterns")):
        return None
    name = _TRAILING_PUNCT.sub("", _NUMBERING.sub("", raw.strip())).strip()
    if not name or name.lower() == title.strip().lower():
        return None
    if name.lower() in {g.strip().lower() for g in _cfg("lookup.generic_sections")}:
        return None
    if name.isupper():
        name = name.lower() if spoken else " ".join(w if (i == 0 or w.lower() not in _SMALL_WORDS) else w.lower() for i, w in enumerate(name.title().split()))
    return name.replace("&", "and")


def _best_section(section_path: str | None, title: str, spoken: bool) -> str | None:
    """The most specific usable heading of a chunk's section_path, walking up from the leaf (a generic leaf such as
    '4.0 Policy Statement' gives way to its parent '16. Teleworking Policy'); the document-title segment is never used."""
    parts = [p for p in (section_path or "").split(" > ")]
    for part in reversed(parts[1:] if parts and parts[0].strip().lower() == title.strip().lower() else parts):
        name = clean_section_name(part, title, spoken)
        if name:
            return name
    return None


def format_location_reply(results: list[dict], templates: dict, spoken: bool, intent: str) -> str:
    """Section and page replies, templated (lookup.templates.*): the cleaned section names of the previous answer's retrieved
    chunks, distinct, in rank order, at most lookup.max_sections.<channel> in total, each document named once. A section
    reply with several documents adds one 'also' sentence per further document; a page reply names the first document
    only. When no clean section exists the title-only forms are used. A template that fails to format falls back to the
    seeded default, so a bad edit cannot break a turn."""
    from config.runtime import DEFAULT_LOOKUP_TEMPLATES
    from config.topics import clean_title

    channel = "voice" if spoken else "text"
    defaults = DEFAULT_LOOKUP_TEMPLATES[channel]

    def render(key: str, **fields) -> str:
        try:
            return templates[key].format(**fields)
        except (KeyError, IndexError, ValueError):
            return defaults[key].format(**fields)

    docs: list[dict] = []
    for r in results:
        doc = next((d for d in docs if d["document_id"] == r["document_id"]), None)
        if doc is None:
            raw = r["document_title"]
            doc = {"document_id": r["document_id"], "name": (clean_title(raw) or display_title(raw)) if spoken else display_title(raw),
                   "raw": raw, "sections": []}
            docs.append(doc)
        section = _best_section(r["section_path"], r["document_title"], spoken)
        if section and section not in doc["sections"]:
            doc["sections"].append(section)
    docs = docs[:3]
    if not docs:
        return render("not_found")
    budget = max(1, int(_cfg(f"lookup.max_sections.{channel}")))
    with_sections = [d for d in docs if d["sections"]]
    if intent == "page":
        with_sections = with_sections[:1]
    if not with_sections:
        key = "page_title_only" if intent == "page" else "section_title_only"
        return render(key, title=_join_and([d["name"] for d in docs]))
    sentences = []
    for d in with_sections:
        if budget <= 0:
            break
        names = d["sections"][:budget]
        budget -= len(names)
        key = ("page" if intent == "page" else "section" if not sentences else "section_also")
        sentences.append(render(key, section=_join_and(names), title=d["name"]))
    return " ".join(sentences)


def lookup_reply(query: str, history: list | None, channel: str, smalltalk_match, retrieve_for_plan, no_match_base: str) -> dict | None:
    """A source / section / page question when the history has an earlier assistant answer: recompute the retrieval of the
    answer it refers to and name where it came from; else None (the caller falls through to the normal pipeline).
    Returns {"text", "chunk_ids", "timing", "type"} with type source_lookup | section_lookup | page_lookup.

    The exchange is the last one that produced an answer (earlier lookups, small talk and catalog replies are walked
    past). Its retrieval is rebuilt by the functions a normal turn uses: plan_turn over only the history that PRECEDED it,
    then the channel's own retrieve (`retrieve_for_plan`): 1 embedding and 1 retrieval, 0 LLM calls. If that answer was a
    refusal, a shortcut or never had a document (small talk, catalog), a templated "didn't come from a policy document"
    line is returned and nothing is retrieved."""
    if not history:
        return None
    intent = lookup_intent(query)
    if intent is None:
        return None
    from channels.catalog import is_catalog_question

    def skippable(user_text: str, before: list) -> str | None:
        return "smalltalk" if smalltalk_match(user_text, before) else "catalog" if is_catalog_question(user_text, before) else None

    # a conversation can hold several lookups in a row (source, page, section ...), so the walk-back reaches further than the
    # 6 messages a normal turn sees; the retrieval itself is planned from the usual capped window before the answered turn
    cleaned = clean_history(history, channel, int(_cfg("lookup.history_messages")))
    exchange = previous_exchange(cleaned, skippable)
    if exchange is None:
        return None
    user_index, assistant_index = exchange
    previous_query, previous_answer = cleaned[user_index]["content"], cleaned[assistant_index]["content"]
    spoken = channel == "voice"
    started = time.monotonic()
    if (skippable(previous_query, cleaned[:user_index]) or previous_answer.lower().startswith(REFUSAL_PREFIX)
            or previous_answer.startswith(no_match_base)):
        results, text_key = [], "none"
    else:
        results = retrieve_for_plan(plan_turn(previous_query, cleaned[:user_index], channel))
        text_key = None
    if intent == "source":
        templates = memo_config(f"source.templates.{channel}", {}) or {}
        from config.runtime import DEFAULT_SOURCE_TEMPLATES

        text = (templates.get("none") or DEFAULT_SOURCE_TEMPLATES[channel]["none"]) if text_key else format_source_reply(results, templates, spoken=spoken)
    else:
        templates = _cfg(f"lookup.templates.{channel}") or {}
        from config.runtime import DEFAULT_LOOKUP_TEMPLATES

        text = (templates.get("none") or DEFAULT_LOOKUP_TEMPLATES[channel]["none"]) if text_key else format_location_reply(results, templates, spoken, intent)
    chunk_ids = [r["chunk_id"] for r in results]
    logging.getLogger(f"hrchatbot.{channel}").info("turn type=%s_lookup chunks=%d", intent, len(chunk_ids))
    return {"text": text, "chunk_ids": chunk_ids, "type": f"{intent}_lookup",
            "timing": {"retrieval_time_ms": (time.monotonic() - started) * 1000}}


def display_title(title: str) -> str:
    """Readable form of a document title for a reply (the stored title is never changed): an ALL-CAPS
    title is title-cased."""
    return title.title() if title.isupper() else title


def _join_and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _leaf_section(section_path: str | None, title: str) -> str | None:
    leaf = (section_path or "").split(" > ")[-1].strip()
    return leaf if leaf and leaf != title else None


def format_source_reply(results: list[dict], templates: dict, spoken: bool) -> str:
    """Templated reply naming up to 3 distinct documents in rank order. Text may add the section name for a
    single document when every retrieved chunk of it sits in the same section; voice names the document
    only. `spoken` titles are lowercased the way config/topics.py does for topic names. A template that
    fails to format falls back to the seeded default, so a bad edit cannot break a turn."""
    from config.runtime import DEFAULT_SOURCE_TEMPLATES
    from config.topics import clean_title

    channel_defaults = DEFAULT_SOURCE_TEMPLATES["voice" if spoken else "text"]

    def render(key: str, **fields) -> str:
        try:
            return templates[key].format(**fields)
        except (KeyError, IndexError, ValueError):
            return channel_defaults[key].format(**fields)

    docs: list[dict] = []
    for r in results:
        if all(d["document_id"] != r["document_id"] for d in docs):
            docs.append({"document_id": r["document_id"], "title": r["document_title"], "sections": []})
        next(d for d in docs if d["document_id"] == r["document_id"])["sections"].append(_leaf_section(r["section_path"], r["document_title"]))
    docs = docs[:3]
    if not docs:
        return render("not_found")
    names = [(clean_title(d["title"]) or display_title(d["title"])) if spoken else display_title(d["title"]) for d in docs]
    if len(docs) > 1:
        return render("many", titles=_join_and(names))
    sections = set(docs[0]["sections"])
    if not spoken and "one_with_section" in templates and len(sections) == 1 and None not in sections:
        return render("one_with_section", title=names[0], section=next(iter(sections)))
    return render("one", title=names[0])
