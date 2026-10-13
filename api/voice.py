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
import hashlib
import json
import logging
import os
import re
import secrets
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

import httpx
from dotenv import dotenv_values
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from access_control.resolve_clarification import resolve_clarification
from channels import voice as voice_channel
from api.query_log import log_query
from api.voice_settings import ENV_PATH_DEFAULT, build_agent_settings
from cache.redis_cache import MISS, get_cached, set_cached

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

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


def _original_query_index(turns, a: int) -> int | None:
    """Index of the original blocked query for the clarifying question at index a: the
    user message before it - walking back past earlier clarification replies, since one
    query can trigger several clarifications in a row."""
    u = a - 1
    if u < 0 or turns[u][0] != "user":
        return None
    while _clarif_at(turns, u - 1):
        u -= 2
        if u < 0 or turns[u][0] != "user":
            return None
    return u


def _original_query(turns, a: int) -> str | None:
    u = _original_query_index(turns, a)
    return None if u is None else turns[u][1]


def conversation_history(turns, end: int) -> list[dict]:
    """Prior chat turns for the answer, from Deepgram's messages: everything before index
    `end` (the user turn being answered), minus plumbing - the synthetic leading
    {"role": "user", "content": "-"} and the greeting that follows it, and every
    clarifying question with the reply to it. Caps and sanitising happen later
    (channels/common.py clean_history)."""
    start = 0
    if turns and turns[0] == ("user", "-"):
        start = 2 if len(turns) > 1 and turns[1][0] == "assistant" else 1
    history = []
    i = start
    while i < end:
        role, text = turns[i]
        if role == "assistant" and parse_clarification(text):
            i += 2  # skip the question and the user's reply to it
            continue
        history.append({"role": role, "content": text})
        i += 1
    return history


NEGATIVE_TTL_SECONDS = 300


def _resolve_value(tag_type: str, reply: str, original: str | None) -> str | None:
    """Classification of one clarification reply, memoized in Redis with
    no expiry: a given (attribute, reply text) always classifies the same
    way, and Redis (unlike a per-process dict) is shared across workers
    and survives restarts. If Redis is down this just recomputes."""
    key = "clarif:" + hashlib.sha256(f"{tag_type}\0{reply}".encode()).hexdigest()
    cached = get_cached(key)
    if cached is not MISS:
        return cached
    merged = resolve_clarification(original or "", tag_type, reply, {})
    value = merged.get(tag_type) if merged else None
    # Only successful classifications are permanent. resolve_clarification()
    # also returns None when the LLM call itself failed, and caching that
    # forever would poison the pair, so "no usable value" gets a short TTL.
    set_cached(key, value, ttl=None if value else NEGATIVE_TTL_SECONDS)
    return value


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
    pending = _clarif_at(turns, len(turns) - 2)

    # Small talk first: before any history replay, retrieval, DB read or LLM
    # work. Bypassed when the previous assistant turn was a clarifying question.
    category = voice_channel.smalltalk_match(current, after_clarification=pending is not None)
    if category:
        text = voice_channel.smalltalk_reply(category)
        record.update(query_text=current, response_type="smalltalk", pieces=[text], contextualized=False, advisory=False)
        yield text
        return

    attrs = {**voice_channel.default_attributes(), **attributes_from_history(turns)}
    record["user_attributes"] = attrs

    query = current
    original_index = _original_query_index(turns, len(turns) - 2) if pending else None
    original = None if original_index is None else turns[original_index][1]
    # The user turn being answered is the original blocked query after a clarification
    # round, else the final turn; history is everything before it (minus plumbing).
    history = conversation_history(turns, original_index if (pending and original) else len(turns) - 1)
    if pending and original:
        tag_type, prev_variant = pending
        merged = resolve_clarification(original, tag_type, current, attrs)
        if merged is None:
            # Unusable reply: ask again, worded differently.
            record.update(query_text=original, response_type="clarification", blocking_tag_type=tag_type,
                          contextualized=False, advisory=False)
            yield clarifying_question(tag_type, prev_variant + 1)
            return
        attrs = {**voice_channel.default_attributes(), **merged}
        record["user_attributes"] = attrs
        query = original
    record["query_text"] = query

    for event in voice_channel.answer_stream(query, attrs, history=history):
        if event["type"] == "needs_clarification":
            record.update(response_type="clarification", blocking_tag_type=event["blocking_tag_type"],
                          contextualized=event.get("contextualized"), advisory=event.get("advisory"))
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
                prompt_version=event.get("prompt_version"),
                contextualized=event.get("contextualized"),
                advisory=event.get("advisory"),
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
# Settings + browser token
# --------------------------------------------------------------------------

DEEPGRAM_GRANT_URL = "https://api.deepgram.com/v1/auth/grant"


@router.get("/settings")
def agent_settings():
    """Deepgram Settings message, rebuilt on every call (no caching). The
    tunnel script rewrites PUBLIC_BASE_URL in .env while this server is
    running, so .env is re-read here rather than trusting the process
    environment captured at startup. provider_configs values are read
    fresh through config.runtime (<=30s cache; PATCH writes through)."""
    base = dotenv_values(ENV_PATH_DEFAULT).get("PUBLIC_BASE_URL") or os.environ.get("PUBLIC_BASE_URL")
    try:
        # think_secret is the one this process actually enforces.
        return build_agent_settings(public_base_url=base, think_secret=THINK_ENDPOINT_SECRET or "")
    except ValueError as exc:
        raise HTTPException(503, str(exc))


@router.post("/token")
def mint_token():
    """Mints a short-lived Deepgram token via POST /v1/auth/grant so the
    long-lived DEEPGRAM_API_KEY never leaves the server (the key needs
    Member role or higher).

    ASSUMPTION: a token only has to be valid when the WebSocket handshake
    happens, not for the whole session. Checked once empirically: an idle
    session held open 110s on a 60s token without dropping. If sessions
    ever drop around expiry (DEEPGRAM_TOKEN_TTL_SECONDS, default 60), the
    token needs periodic reissue or a longer TTL. Not verified with
    long mic conversations.

    No auth on this endpoint: anyone who can reach it can mint tokens
    (same dev-only gap as the rest of the API - see api/main.py)."""
    key = os.environ.get("DEEPGRAM_API_KEY")
    if not key or key == "changeme":
        raise HTTPException(503, "DEEPGRAM_API_KEY is not configured on the server")
    # Must be an int: Deepgram silently ignores a float (e.g. 60.0) and uses its 30s default.
    ttl = int(float(os.environ.get("DEEPGRAM_TOKEN_TTL_SECONDS", 60)))
    try:
        r = httpx.post(DEEPGRAM_GRANT_URL, headers={"Authorization": f"Token {key}"},
                       json={"ttl_seconds": ttl}, timeout=10)
    except httpx.HTTPError as exc:
        raise HTTPException(502, f"Could not reach Deepgram: {exc}")
    if r.status_code != 200:
        raise HTTPException(502, f"Deepgram /v1/auth/grant returned {r.status_code}: {r.text[:300]}")
    data = r.json()
    return {"token": data["access_token"], "expires_in": data.get("expires_in", ttl)}


# --------------------------------------------------------------------------
# DEBUG-ONLY raw request logging
# --------------------------------------------------------------------------

def _debug_raw_enabled() -> bool:
    return os.environ.get("VOICE_DEBUG_LOG_RAW_REQUEST", "").strip().lower() in ("1", "true", "yes", "on")


async def _debug_log_raw_request(request: Request):
    """DEBUG ONLY - off unless VOICE_DEBUG_LOG_RAW_REQUEST=1. Logs the
    complete, unredacted request body Deepgram sends to the think
    endpoint on every call (it contains the user's transcribed speech
    and conversation history, i.e. personal data - never leave this on
    outside of a debugging session). Header NAMES are logged, not values,
    so the bearer secret isn't written out. Exists to resolve what Deepgram
    actually sends (body shape; whether `stream` is present)."""
    if not _debug_raw_enabled():
        return
    raw = await request.body()
    try:
        body = json.loads(raw)
        summary = (
            f"top-level keys={sorted(body)} | stream={body.get('stream', '<ABSENT>')!r} | "
            f"roles={[m.get('role') for m in body.get('messages', [])]} | "
            f"tools={'present' if body.get('tools') or body.get('functions') else 'absent'}"
        ) if isinstance(body, dict) else "body is not a JSON object"
    except ValueError:
        summary = "body is not valid JSON"
    line = (f"[VOICE_DEBUG_LOG_RAW_REQUEST] {summary}\n  headers={sorted(request.headers.keys())}\n"
            f"  raw body: {raw.decode('utf-8', 'replace')}")
    print(line, flush=True)
    if path := os.environ.get("VOICE_DEBUG_LOG_FILE"):
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


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
def think(req: ThinkRequest, authorization: str | None = Header(None), _debug=Depends(_debug_log_raw_request)):
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
