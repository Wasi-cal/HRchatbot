"""Deepgram Voice Agent "think" endpoint (bring-your-own-LLM).

Deepgram's cloud calls this endpoint (configured as think.endpoint.url in
the Settings message - see api/voice_settings.py) each time the user
finishes speaking. Per Deepgram's docs the endpoint must behave like
OpenAI's Chat Completions API (provider.type "open_ai" + a custom
endpoint). The docs do NOT spell out the exact per-turn request body, so
this endpoint is deliberately tolerant:
  - reads only `messages` (OpenAI shape: role + string-or-parts content),
    ignores system/tool messages, `tools`, `model`, temperature, etc.
  - streams SSE chat.completion.chunk events unless the request says
    "stream": false explicitly (the docs say the agent consumes SSE).
Verify against a real Deepgram session before relying on those guesses.

The endpoint is STATELESS: everything - whether this turn answers a
clarifying question, which question, and which attributes were already
resolved - is re-derived from the message history Deepgram sends.
Clarifying questions are recognised by a fixed spoken lead-in plus one of
a closed set of per-attribute question wordings (no hidden token: TTS
would read it aloud).
"""
import json
import logging
import os
import re
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from access_control.ask_and_answer import ask_and_answer_stream
from access_control.resolve_clarification import resolve_clarification
from api.query_log import log_query

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

DEFAULT_ATTRIBUTES = {"country": "India"}

CLARIFY_LEAD_IN = "Quick check before I answer - "

# Closed set of spoken wordings per attribute, lowercase. Index 0 is the
# first ask; later indexes are slight rephrasings used when a reply
# wasn't usable. Matching (see parse_clarification) is by suffix, so none
# may be a suffix of another variant.
QUESTION_VARIANTS = {
    "country": ["which country are you based in?", "which country do you work from?"],
    "employment_type": [
        "are you a full-time employee or a contractor?",
        "just to confirm, are you a full-time employee, or a contractor?",
    ],
}
_GENERIC_ASK_RE = re.compile(r"(?:could you tell me your|what is your) ([a-z ]+)\?$")
_GENERIC_TEMPLATES = ["could you tell me your {}?", "what is your {}?"]

APOLOGY = "Sorry, I ran into a problem looking that up. Please try again in a moment."

THINK_ENDPOINT_SECRET = os.environ.get("THINK_ENDPOINT_SECRET")


def _variants(tag_type: str) -> list[str]:
    return QUESTION_VARIANTS.get(tag_type) or [t.format(tag_type.replace("_", " ")) for t in _GENERIC_TEMPLATES]


def clarifying_question(tag_type: str, variant: int = 0) -> str:
    variants = _variants(tag_type)
    return CLARIFY_LEAD_IN + variants[variant % len(variants)]


def parse_clarification(text: str) -> tuple[str, int] | None:
    """If `text` is one of our clarifying questions, returns
    (blocking_tag_type, variant_index); else None."""
    text = text.strip()
    if not text.startswith(CLARIFY_LEAD_IN):
        return None
    rest = text[len(CLARIFY_LEAD_IN):].strip().lower()
    for tag_type, variants in QUESTION_VARIANTS.items():
        for i, v in enumerate(variants):
            if rest.endswith(v):
                return tag_type, i
    m = _GENERIC_ASK_RE.search(rest)
    if m:
        tag_type = m.group(1).strip().replace(" ", "_")
        return tag_type, 0 if rest.startswith("could") else 1
    return None


# --------------------------------------------------------------------------
# History analysis
# --------------------------------------------------------------------------

def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return ""


def _turns(messages: list[dict]) -> list[tuple[str, str]]:
    turns = []
    for m in messages:
        role = m.get("role")
        text = _content_text(m.get("content")).strip()
        if role in ("user", "assistant") and text:
            turns.append((role, text))
    return turns


def _clarif_at(turns, i):
    return parse_clarification(turns[i][1]) if 0 <= i < len(turns) and turns[i][0] == "assistant" else None


def _original_query(turns, a: int) -> str | None:
    """Original blocked query for the clarifying question at index a: the
    user message before it - walking back past earlier clarification
    replies, since one query can trigger several clarifications in a row."""
    u = a - 1
    if u < 0 or turns[u][0] != "user":
        return None
    while _clarif_at(turns, u - 1):
        u -= 2
        if u < 0 or turns[u][0] != "user":
            return None
    return turns[u][1]


_resolve_cache: dict[tuple[str, str], str | None] = {}


def _resolve_value(tag_type: str, reply: str, original: str | None) -> str | None:
    key = (tag_type, reply)
    if key not in _resolve_cache:
        merged = resolve_clarification(original or "", tag_type, reply, {})
        _resolve_cache[key] = merged.get(tag_type) if merged else None
    return _resolve_cache[key]


def attributes_from_history(turns) -> dict:
    """Replays past clarification Q&A pairs (excluding the current, final
    user turn) to recover attributes already resolved this session. A
    pair counts as resolved unless the assistant immediately re-asked the
    same attribute. Unlike a stored session, this re-runs the (cached)
    extraction per past pair - cheap, and keeps the endpoint stateless."""
    last = len(turns) - 1
    pairs = []
    for a in range(len(turns) - 2):  # need a reply at a+1 < last
        parsed = _clarif_at(turns, a)
        if not parsed or turns[a + 1][0] != "user" or a + 1 >= last:
            continue
        nxt = _clarif_at(turns, a + 2)
        if nxt and nxt[0] == parsed[0]:
            continue  # re-asked: the reply was unusable
        pairs.append((parsed[0], turns[a + 1][1], _original_query(turns, a)))

    attrs = {}
    if pairs:
        with ThreadPoolExecutor(max_workers=len(pairs)) as ex:
            values = list(ex.map(lambda p: _resolve_value(*p), pairs))
        for (tag_type, _r, _o), value in zip(pairs, values):
            if value:
                attrs[tag_type] = value
    return attrs


# --------------------------------------------------------------------------
# Turn handling
# --------------------------------------------------------------------------

def _turn_pieces(turns, record: dict):
    """Generator of text pieces to speak for the final user turn. Fills
    `record` with what to log (query, attributes, outcome)."""
    current = turns[-1][1]
    attrs = {**DEFAULT_ATTRIBUTES, **attributes_from_history(turns)}
    record["user_attributes"] = attrs

    query = current
    pending = _clarif_at(turns, len(turns) - 2)
    original = _original_query(turns, len(turns) - 2) if pending else None
    if pending and original:
        tag_type, prev_variant = pending
        merged = resolve_clarification(original, tag_type, current, attrs)
        if merged is None:
            # Unusable reply: ask again, worded differently.
            record.update(query_text=original, response_type="clarification", blocking_tag_type=tag_type)
            yield clarifying_question(tag_type, prev_variant + 1)
            return
        attrs = {**DEFAULT_ATTRIBUTES, **merged}
        record["user_attributes"] = attrs
        query = original
    record["query_text"] = query

    for event in ask_and_answer_stream(query, attrs):
        if event["type"] == "needs_clarification":
            record.update(response_type="clarification", blocking_tag_type=event["blocking_tag_type"])
            yield clarifying_question(event["blocking_tag_type"])
            return
        if event["type"] == "token":
            record.setdefault("pieces", []).append(event["text"])
            yield event["text"]
        else:  # done
            record.update(
                response_type="refusal_shortcut" if event["shortcut"] else "answer",
                shortcut_fired=event["shortcut"],
                timing=event["timing"],
                chunk_ids_used=event.get("chunk_ids", []),
            )


def _logged_pieces(turns):
    """Wraps _turn_pieces: swallows errors into a spoken apology and logs
    the turn exactly once when the stream ends (or is cut off by a
    barge-in), never letting logging break the response."""
    record: dict = {}
    try:
        yield from _turn_pieces(turns, record)
    except Exception:
        logger.exception("think turn failed")
        record = {}
        yield APOLOGY
    finally:
        if record.get("query_text") and record.get("response_type"):
            pieces = record.pop("pieces", None)
            if record["response_type"] == "clarification":
                record["answer_text"] = None
            elif pieces:
                record["answer_text"] = "".join(pieces)
            record.pop("pieces", None)
            log_query(source="voice", **record)


# --------------------------------------------------------------------------
# OpenAI-compatible endpoint
# --------------------------------------------------------------------------

class ThinkRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    messages: list[dict]
    stream: bool | None = None


def _check_auth(authorization: str | None):
    if THINK_ENDPOINT_SECRET and not secrets.compare_digest(
        authorization or "", f"Bearer {THINK_ENDPOINT_SECRET}"
    ):
        raise HTTPException(401, "Invalid or missing bearer token")


def _chunk(completion_id, model, created, delta, finish=None) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@router.post("/v1/chat/completions")
def think(req: ThinkRequest, authorization: str | None = Header(None)):
    _check_auth(authorization)
    turns = _turns(req.messages)
    if not turns or turns[-1][0] != "user":
        raise HTTPException(400, "messages must end with a non-empty user message")

    completion_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    model = str(req.model_extra.get("model") or "hr-rag") if req.model_extra else "hr-rag"
    pieces = _logged_pieces(turns)

    if req.stream is False:
        text = "".join(pieces)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        })

    def sse():
        yield _chunk(completion_id, model, created, {"role": "assistant", "content": ""})
        for piece in pieces:
            yield _chunk(completion_id, model, created, {"content": piece})
        yield _chunk(completion_id, model, created, {}, finish="stop")
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
