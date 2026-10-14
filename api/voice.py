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

The endpoint is STATELESS: every turn is derived from the message history
Deepgram sends. A turn is: small-talk match, else retrieve once + generate
once and stream (channels/voice.py); a "where is this from" turn is a templated source lookup. There is no clarifying question, no
applicability filtering and no per-user attributes - see
docs/applicability_layer_removed.md.
"""
import asyncio
import json
import logging
import os
import secrets
import time
import uuid

import anyio
import httpx
from dotenv import dotenv_values
from fastapi import APIRouter, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

from api.query_log import log_query
from channels import common
from channels import voice as voice_channel
from api.voice_settings import ENV_PATH_DEFAULT, build_agent_settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/voice", tags=["voice"])

APOLOGY = "Sorry, I ran into a problem looking that up. Please try again in a moment."

THINK_ENDPOINT_SECRET = os.environ.get("THINK_ENDPOINT_SECRET")


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


def conversation_history(turns) -> list[dict]:
    """Prior chat turns for the answer, from Deepgram's messages: everything before the
    final user turn (the one being answered), minus the synthetic leading
    {"role": "user", "content": "-"} and the greeting that follows it. Caps and
    sanitising happen later (channels/common.py clean_history)."""
    start = 0
    if turns and turns[0] == ("user", "-"):
        start = 2 if len(turns) > 1 and turns[1][0] == "assistant" else 1
    return [{"role": role, "content": text} for role, text in turns[start:-1]]


# --------------------------------------------------------------------------
# Turn handling
# --------------------------------------------------------------------------

def _turn_pieces(turns, record: dict, handle: common.TurnHandle):
    """Generator of text pieces to speak for the final user turn. Fills
    `record` with what to log (query, outcome)."""
    current = turns[-1][1]
    history = conversation_history(turns)

    # Small talk first: before any retrieval, DB read or LLM work. (A bare "yes" after the agent's
    # question or offer is not small talk: it falls through so the offer is followed up.)
    category = voice_channel.smalltalk_match(current, history)
    if category:
        text = voice_channel.smalltalk_reply(category)
        record.update(query_text=current, response_type="smalltalk", pieces=[text], contextualized=False, advisory=False)
        yield text
        return

    # "Where is this from": 0 LLM calls, one re-retrieval of the previous answer's turn.
    source = voice_channel.source_reply(current, history)
    if source:
        record.update(query_text=current, response_type="source_lookup", pieces=[source["text"]],
                      chunk_ids_used=source["chunk_ids"], timing=source["timing"], contextualized=False, advisory=False)
        yield source["text"]
        return

    # "What documents do you have": templated from the cached title list, 0 LLM / retrieval / embedding calls.
    catalog_answer = voice_channel.catalog_reply(current, history)
    if catalog_answer:
        record.update(query_text=current, response_type="catalog", pieces=[catalog_answer["text"]], contextualized=False, advisory=False)
        yield catalog_answer["text"]
        return

    record["query_text"] = current
    for event in voice_channel.answer_stream(current, history=history, handle=handle):
        if event["type"] == "token":
            record.setdefault("pieces", []).append(event["text"])
            yield event["text"]
        else:  # done
            record.update(
                response_type="refusal_shortcut" if event["shortcut"] else ("error" if handle.info.get("fallback") else "answer"),
                shortcut_fired=event["shortcut"],
                timing=event["timing"],
                chunk_ids_used=event.get("chunk_ids", []),
                prompt_version=event.get("prompt_version"),
                contextualized=event.get("contextualized"),
                advisory=event.get("advisory"),
            )


def _logged_pieces(turns, handle: common.TurnHandle, record: dict):
    """Wraps _turn_pieces: swallows errors into a spoken apology, logs the turn exactly once when the stream ends,
    and records an ABORTED turn (the client disconnected mid-turn, e.g. Deepgram barge-in) instead of losing it.
    Never lets logging break the response. `record` is the caller's dict so the endpoint can read the outcome."""
    try:
        yield from _turn_pieces(turns, record, handle)
    except (common.TurnCancelled, GeneratorExit):
        record.update(response_type="aborted", query_text=record.get("query_text") or turns[-1][1])
    except Exception:
        logger.exception("think turn failed")
        keep = record.get("query_text")
        record.clear()
        if keep:
            record.update(query_text=keep, response_type="error")
        yield APOLOGY
    finally:
        if record.get("query_text") and record.get("response_type"):
            pieces = record.pop("pieces", None)
            if pieces:
                record["answer_text"] = "".join(pieces)
            if record["response_type"] == "aborted":
                record.setdefault("timing", {"time_to_first_token_ms": handle.info.get("ttft_ms")})
            row = {k: v for k, v in record.items() if k != "pieces"}
            log_query(source="voice", **row)


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


_END = object()
_think_logger = logging.getLogger("hrchatbot.voice")
_TYPE_BY_RESPONSE = {"answer": "answer", "refusal_shortcut": "shortcut", "smalltalk": "smalltalk", "source_lookup": "source_lookup",
                     "catalog": "catalog", "aborted": "aborted", "error": "error"}


def _log_think(started: float, kind: str, disconnected: bool, handle: common.TurnHandle | None = None, **extra) -> None:
    """ONE INFO line per think request: type (answer | source_lookup | smalltalk | catalog | shortcut | aborted | error),
    duration_ms and client_disconnected; generation turns add the OpenAI request id, attempts, time to response headers,
    time to first token and token usage (all captured from the same call - no extra request)."""
    fields = {"type": kind, "duration_ms": round((time.monotonic() - started) * 1000), "client_disconnected": disconnected}
    info = (handle.info if handle else {}) or {}
    if info.get("attempts"):
        fields.update(request_id=info.get("request_id"), attempts=info.get("attempts"),
                      headers_ms=None if info.get("headers_ms") is None else round(info["headers_ms"]),
                      ttft_ms=None if info.get("ttft_ms") is None else round(info["ttft_ms"]),
                      chunks=info.get("chunks"), prompt_tokens=info.get("prompt_tokens"), cached_tokens=info.get("cached_tokens"),
                      completion_tokens=info.get("completion_tokens"), fallback=info.get("fallback"))
    fields.update(extra)
    _think_logger.info("think " + " ".join(f"{k}={v}" for k, v in fields.items()))


def _next_piece(iterator):
    return next(iterator, _END)


@router.post("/v1/chat/completions")
async def think(req: ThinkRequest, authorization: str | None = Header(None), _debug=Depends(_debug_log_raw_request)):
    started = time.monotonic()
    try:
        _check_auth(authorization)
        turns = _turns(req.messages)
        if not turns or turns[-1][0] != "user":
            raise HTTPException(400, "messages must end with a non-empty user message")
    except HTTPException as exc:
        _log_think(started, "error", False, status=exc.status_code)
        raise

    completion_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
    model = str(req.model_extra.get("model") or "hr-rag") if req.model_extra else "hr-rag"
    handle, record = common.TurnHandle(), {}
    pieces = _logged_pieces(turns, handle, record)

    if req.stream is False:
        text = await anyio.to_thread.run_sync(lambda: "".join(pieces))
        _log_think(started, _TYPE_BY_RESPONSE.get(record.get("response_type"), "error"), False, handle)
        return JSONResponse({
            "id": completion_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        })

    async def sse():
        """Streams the turn. Each next piece is produced in a worker thread; if the client disconnects (Deepgram aborts a
        turn on barge-in) the await is cancelled at once, the upstream OpenAI stream is closed via the handle - so the
        provider stops generating - and the turn is logged as aborted."""
        disconnected = True
        try:
            yield _chunk(completion_id, model, created, {"role": "assistant", "content": ""})
            while True:
                piece = await anyio.to_thread.run_sync(_next_piece, pieces, abandon_on_cancel=True)
                if piece is _END:
                    break
                yield _chunk(completion_id, model, created, {"content": piece})
            disconnected = False  # every token was produced; what remains is only the stop chunk
            yield _chunk(completion_id, model, created, {}, finish="stop")
            yield "data: [DONE]\n\n"
        finally:
            if disconnected:
                handle.cancel()
                try:
                    pieces.close()  # finalises the turn (aborted row) if no worker thread is inside it right now
                except ValueError:
                    pass  # a worker thread is mid-next(): it sees handle.cancelled and finalises itself
                _log_think(started, "aborted", True, handle, tokens_streamed_so_far=handle.info.get("chunks", 0))
            else:
                _log_think(started, _TYPE_BY_RESPONSE.get(record.get("response_type"), "error"), False, handle)

    return StreamingResponse(sse(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})
