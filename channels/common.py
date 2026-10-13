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


def _truncate(content: str, max_chars: int, keep_tail: bool) -> str:
    if len(content) <= max_chars:
        return content
    if not keep_tail:
        return content[:max_chars]
    separator = " ... "
    tail = max_chars // 3
    return content[: max_chars - tail - len(separator)] + separator + content[-tail:]


def clean_history(history: list | None, channel: str) -> list[dict]:
    """Server-side sanitisation of client/Deepgram-supplied history, which is
    UNTRUSTED: only user/assistant roles with string content survive, control
    characters are dropped, the list is capped to the last
    conversation.history_messages messages and each is truncated to
    conversation.history_max_chars characters. An over-long ASSISTANT message keeps its
    head and its tail (an offer or a question usually closes it, and "yes" is answered
    against that), still within the character cap."""
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
            kept.append({"role": message["role"], "content": _truncate(content, max_chars, keep_tail=message["role"] == "assistant")})
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


from config.runtime import clear_config_memo, memo_config  # noqa: E402,F401  (kept here: channels use common.memo_config)


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

SOURCE_MAX_WORDS = 8
REFUSAL_PREFIX = "i couldn't find"


def is_source_question(text: str) -> bool:
    """Whole-utterance match (<= 8 words) against source.markers."""
    norm = normalize(text)
    if not norm or len(norm.split()) > SOURCE_MAX_WORDS:
        return False
    return norm in {normalize(m) for m in memo_config("source.markers", []) or []}


def previous_exchange(history: list[dict]) -> tuple[int, int] | None:
    """(user_index, assistant_index) of the most recent answered exchange in CLEANED history: the last
    assistant message and the user message before it. An exchange that was itself a source question is
    skipped, so asking "source?" twice still refers to the real answer. None when there is no
    assistant message to refer to."""
    end = len(history) - 1
    while end >= 0:
        a = next((i for i in range(end, -1, -1) if history[i]["role"] == "assistant"), None)
        if a is None:
            return None
        u = next((i for i in range(a - 1, -1, -1) if history[i]["role"] == "user"), None)
        if u is None:
            return None
        if is_source_question(history[u]["content"]):
            end = u - 1
            continue
        return u, a
    return None


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
