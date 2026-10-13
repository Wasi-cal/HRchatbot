"""Compatibility router for eval tooling.

The pipelines are NOT here: text and voice are fully separate modules
(channels/text.py, channels/voice.py), each owning its retrieve -> generate flow.
ask_and_answer(channel=...) only dispatches to them; api/chat.py and api/voice.py
call the channel modules directly.

There is no applicability check or clarifying question any more: every turn
retrieves once and generates once (see docs/applicability_layer_removed.md).
user_attributes is accepted and ignored so older callers keep working.
"""
from channels import get_channel


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
        query, top_k=top_k, candidate_n=candidate_n, k=k,
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
    """Returns {"type": "answer", "answer", "shortcut", "timing", "chunk_ids", ...}: see
    channels/text.py answer()."""
    return get_channel(channel).answer(
        query, top_k=top_k, candidate_n=candidate_n, k=k,
        confidence_threshold=confidence_threshold, history=history,
    )
