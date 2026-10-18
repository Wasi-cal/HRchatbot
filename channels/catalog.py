"""Deterministic "catalog" intent: questions about WHICH documents / topics the assistant covers
("what docs do u have access to rn", "which policies do you cover") get a templated reply built from the
cached title list. 0 LLM, 0 retrieval, 0 embedding calls and no DB / network call on the turn itself.

Behaviour-free helpers shared by channels/text.py and channels/voice.py (each channel supplies its own
templates and limits). Precedence in the pipeline: small talk, source lookup, catalog, normal turn.

A question matches only if ALL of these hold, so anything with another content word falls through
untouched to the normal pipeline:
  * the whole utterance is at most catalog.max_words words (12);
  * after expanding informal forms (catalog.normalize_map: u->you, docs->documents, rn->drop ...) and removing
    stopwords, EVERY remaining token is in catalog.vocab (the second-person words you / your are allowed);
  * it contains a second-person token (you / your);
  * it contains a noun from catalog.nouns, or equals one of the multi-word entries ("what do you know").
It also does not fire when it could be the answer to the assistant's pending offer or question: the previous
assistant turn invited a reply AND this utterance does not itself read as a question or command.
"""
import re

from channels import common

SECOND_PERSON = {"you", "your", "yours"}
# An utterance starting with one of these is itself a question/command, not an answer to the bot's offer.
QUESTION_STARTERS = {"what", "which", "list", "show", "tell", "give", "can", "do", "are", "is", "how", "where", "who", "whats"}
# Cheap pre-filter on the RAW text: a catalog question addresses the assistant, so most turns skip every config read.
_ADDRESSES_ASSISTANT = re.compile(r"\b(you|your|yours|u|ur)\b", re.IGNORECASE)


def is_catalog_question(text: str, history: list | None = None) -> bool:
    if not _ADDRESSES_ASSISTANT.search(text):
        return False
    if not common.memo_config("catalog.enabled", True):
        return False
    norm = common.normalize(text)
    words = norm.split()
    if not words or len(words) > common.memo_config("catalog.max_words", 12):
        return False
    expanded = common.expand_informal(norm, common.memo_config("catalog.normalize_map", {}) or {})
    stopwords = {common.normalize(w) for w in common.memo_config("catalog.stopwords", []) or []}
    tokens = [t for t in expanded.split() if t not in stopwords]
    vocab = {common.normalize(v) for v in common.memo_config("catalog.vocab", []) or []}
    if not tokens or any(t not in vocab and t not in SECOND_PERSON for t in tokens):
        return False
    if not SECOND_PERSON & set(tokens):
        return False
    nouns = [common.normalize(n) for n in common.memo_config("catalog.nouns", []) or []]
    if not ({n for n in nouns if " " not in n} & set(tokens) or " ".join(expanded.split()) in {n for n in nouns if " " in n}):
        return False
    return not (common.previous_turn_invites_reply(history) and words[0] not in QUESTION_STARTERS)


def _join_and(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def build_reply(titles: list[str], topics: list[str], templates: dict, *, spoken: bool, max_titles: int,
                max_title_words: int, defaults: dict) -> str:
    """Templated catalog reply. Text names up to max_titles titles ("and N more" after that) plus one line of
    example topics when the topic list exists; voice names at most max_titles SHORT spoken names (long titles
    are skipped, never read aloud). No reply ends with a question or an offer. A broken template falls back to
    the seeded default, so a bad edit cannot break a turn."""
    from config.topics import clean_title

    def render(key: str, **fields) -> str:
        try:
            return templates[key].format(**fields)
        except (KeyError, IndexError, ValueError):
            return defaults[key].format(**fields)

    if not titles:
        return render("empty")
    n = len(titles)
    if spoken:
        names = [clean_title(t) or common.display_title(t) for t in titles]
        short = [x for x in names if x and len(x.split()) <= max_title_words]
        shown = short[:max_titles] or names[:max_titles]
        parts = [render("main", n=n, names=_join_and(shown)), render("closing")]
        return " ".join(parts)
    display = [common.display_title(t) for t in titles]
    shown = display[:max_titles]
    listing = ", ".join(shown[:-1]) + (" and " if len(shown) > 1 else "") + shown[-1] if shown else ""
    if n > len(shown):
        listing = f"{', '.join(shown)} " + render("more", k=n - len(shown))
    parts = [render("main", n=n, list=listing)]
    # Example topics: prefer ones the listed titles do not already spell out (they come from the "and N more").
    shown_text = " ".join(shown).lower()
    fresh = [t for t in topics if t.lower() not in shown_text] or topics
    if topics:
        parts.append(render("topics", topics=_join_and(fresh[:3])))
    parts.append(render("closing"))
    return " ".join(parts)
