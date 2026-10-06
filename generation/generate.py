"""Retrieval-to-generation core for the HR RAG voice chatbot.

Sits on top of vectorstore.retrieve.retrieve(): takes a query, retrieves
the top fused chunks, and streams a short, spoken-language-appropriate
answer from an OpenAI chat model grounded only in those chunks.

No STT/TTS integration and no serving/API layer here - this module ends
at a working, streaming, timed generate_answer() function, testable via
text queries (see eval/run_regression.py and eval/run_quality_eval.py).

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
import time

from openai import OpenAI

from config.runtime import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_GENERATION_MODEL,
    get_active_provider_config,
)
from vectorstore.retrieve import retrieve

NO_MATCH_RESPONSE = (
    "I don't have information on that. You should check with HR directly to get a "
    "reliable answer."
)

SYSTEM_PROMPT = """You are a voice assistant answering employee questions about HR policy, out loud, over a phone or voice interface. Your response will be read aloud by text-to-speech, not displayed as text.

Rules:
- Answer ONLY using the retrieved policy excerpts given to you below. Never use outside knowledge about HR policy, labor law, or general company practices - if the excerpts don't say it, you don't know it.
- Speak naturally in two or three short, conversational sentences (roughly 50 words at most) - lead with the direct answer and leave out secondary details unless asked; long answers are tedious to listen to. Do not use bullet points, numbered lists, headings, or any other visual formatting - it will not survive being read aloud.
- Never say things like "according to the handbook", "section 5.3 says", or mention any document title, section name, or page number. Just answer directly, as if you already knew it.
- If the excerpts don't actually contain an answer to the question, say plainly that you don't have that information and suggest checking with HR directly. Do not guess, infer, or fill gaps with plausible-sounding information.
- If the excerpts show the answer depends on something you can't tell from the question - like the employee's location, employment type, or tenure - say that briefly (for example: "that can depend on your location or employment type, so let me know which applies or check with HR to be sure") instead of silently assuming one answer.
"""

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


def get_generation_model() -> str:
    """Model name from the active category='generation' provider_configs
    row (cached for config.runtime.CACHE_TTL_SECONDS; seeded by
    config.runtime.seed_runtime_config)."""
    return get_active_provider_config("generation").get("model") or DEFAULT_GENERATION_MODEL


def get_confidence_threshold() -> float:
    value = get_active_provider_config("generation").get("confidence_threshold")
    return DEFAULT_CONFIDENCE_THRESHOLD if value is None else float(value)


def _build_messages(query: str, chunks: list[dict]) -> list[dict]:
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
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def generate_answer(
    query: str,
    top_k: int | None = None,
    candidate_n: int | None = None,
    k: int | None = None,
    confidence_threshold: float | None = None,
    user_attributes: dict | None = None,
):
    """Retrieves grounding chunks for query, then streams a generated
    answer. A generator yielding dict events:

        {"type": "token", "text": str}
            One streamed piece of the answer, in order.
        {"type": "done", "answer": str, "shortcut": bool, "timing": dict,
         "chunk_ids": list[str]}
            Always the final event. "chunk_ids" are the retrieved chunks'
            string ids used as grounding (empty on a shortcut). "answer" is the full accumulated
            text. "timing" has "retrieval_time_ms" always, plus
            "time_to_first_token_ms" and "total_generation_time_ms" when
            a real generation call was made (shortcut=False).

    top_k/candidate_n/k are forwarded to retrieve(); confidence_threshold
    overrides the configured generation.confidence_threshold for this call only.
    top_k/candidate_n/k default to the runtime app_config values (see retrieve()).
    user_attributes (see access_control/user_attributes.py) is forwarded
    to retrieve() unchanged - omitting it (the default) keeps this
    function's behavior exactly as it was before access control existed.
    """
    threshold = get_confidence_threshold() if confidence_threshold is None else confidence_threshold

    retrieval_start = time.monotonic()
    results = retrieve(query, top_k=top_k, candidate_n=candidate_n, k=k, user_attributes=user_attributes)
    retrieval_time_ms = (time.monotonic() - retrieval_start) * 1000

    top_score = results[0]["fused_score"] if results else 0.0
    if not results or top_score < threshold:
        yield {"type": "token", "text": NO_MATCH_RESPONSE}
        yield {
            "type": "done",
            "answer": NO_MATCH_RESPONSE,
            "shortcut": True,
            "timing": {"retrieval_time_ms": retrieval_time_ms},
            "chunk_ids": [],
        }
        return

    model_name = get_generation_model()
    client = get_client()
    messages = _build_messages(query, results)

    generation_start = time.monotonic()
    first_token_time = None
    pieces = []

    stream = client.chat.completions.create(model=model_name, messages=messages, stream=True)
    for event in stream:
        delta = event.choices[0].delta.content
        if not delta:
            continue
        if first_token_time is None:
            first_token_time = time.monotonic()
        pieces.append(delta)
        yield {"type": "token", "text": delta}

    generation_end = time.monotonic()
    answer = "".join(pieces)
    timing = {
        "retrieval_time_ms": retrieval_time_ms,
        "time_to_first_token_ms": (
            (first_token_time - generation_start) * 1000 if first_token_time is not None else None
        ),
        "total_generation_time_ms": (generation_end - generation_start) * 1000,
    }
    yield {
        "type": "done",
        "answer": answer,
        "shortcut": False,
        "timing": timing,
        "chunk_ids": [r["chunk_id"] for r in results],
    }


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
