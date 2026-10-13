"""Compatibility router for eval tooling and the clarification helpers.

The pipelines are NOT here: text and voice are fully separate modules
(channels/text.py, channels/voice.py), each owning its retrieval ->
applicability check -> generation flow. ask_and_answer(channel=...) only
dispatches to them; api/chat.py and api/voice.py call the channel modules
directly.
"""
from channels import get_channel
from channels.common import detect_blocking_tag_types  # noqa: F401  (re-export)
from channels.text import CLARIFYING_QUESTIONS  # noqa: F401  (resolve_clarification uses the question wording)


def ask_and_answer_stream(
    query: str,
    user_attributes: dict | None = None,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    channel: str = "text",
    history: list | None = None,
):
    """Streaming core of ask_and_answer(): see channels/text.py answer_stream()."""
    yield from get_channel(channel).answer_stream(
        query, user_attributes, top_k=top_k, candidate_n=candidate_n, k=k,
        confidence_threshold=confidence_threshold, history=history,
    )


def ask_and_answer(
    query: str,
    user_attributes: dict | None = None,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    channel: str = "text",
    history: list | None = None,
) -> dict:
    """Returns {"type": "needs_clarification", ...} or {"type": "answer", ...}: see
    channels/text.py answer()."""
    return get_channel(channel).answer(
        query, user_attributes, top_k=top_k, candidate_n=candidate_n, k=k,
        confidence_threshold=confidence_threshold, history=history,
    )
