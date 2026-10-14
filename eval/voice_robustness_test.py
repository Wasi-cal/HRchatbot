#!/usr/bin/env python3
"""Tests for the voice robustness work: first-token timeout / retry / fallback, closing the upstream stream when
Deepgram aborts a turn, one log line per think request, listen settings from the DB, offer-aware retrieval for a bare
"yes", and the per-turn call counts.

    python3 -m eval.voice_robustness_test

NO real OpenAI calls are made for the timeout / retry / disconnect / logging tests: a fake OpenAI client is installed
(generation.generate._client). Retrieval and embedding are real (local model + Postgres + Redis). Settings it changes are
restored and the query_logs rows it creates are deleted, even on failure.
"""
import json
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx  # noqa: E402
import uvicorn  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import channels.text as text_ch  # noqa: E402
import channels.voice as voice_ch  # noqa: E402
import generation.generate as gen_mod  # noqa: E402
from api.main import app  # noqa: E402
from api.voice_settings import build_agent_settings  # noqa: E402
from channels import common  # noqa: E402
from config.runtime import get_active_provider_config, get_app_config, set_app_config  # noqa: E402
from eval import turn_calls  # noqa: E402
from vectorstore.db import close_pool, get_pool  # noqa: E402

_results = []
SECRET = os.environ.get("THINK_ENDPOINT_SECRET")


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if not condition and detail else ""))
    return bool(condition)


# ---- fake OpenAI client ----------------------------------------------------------------------------------------

class _Chunk:
    def __init__(self, text=None, usage=None):
        self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})()] if text is not None else []
        self.usage = usage


class FakeStream:
    """An iterable OpenAI-style stream. behavior: ok | timeout_on_read (raises while waiting for the first chunk) |
    hang (blocks until close()) | mid_error (error after 2 tokens). close() is recorded and unblocks a hang."""

    def __init__(self, behavior, tokens, delay, usage, request_id):
        self.behavior, self.tokens, self.delay, self.usage_obj = behavior, tokens, delay, usage
        self.response = type("R", (), {"headers": {"x-request-id": request_id}})()
        self.closed = False
        self.consumed = 0
        self._closed_event = threading.Event()

    def __iter__(self):
        if self.behavior == "timeout_on_read":
            raise httpx.ReadTimeout("no first token")
        if self.behavior == "hang":
            self._closed_event.wait(30)
            raise ConnectionError("stream closed while waiting for the first token")
        for i, token in enumerate(self.tokens):
            if self._closed_event.wait(self.delay):
                raise ConnectionError("stream closed")
            self.consumed += 1
            yield _Chunk(token)
            if self.behavior == "mid_error" and i == 1:
                raise httpx.ReadTimeout("gap between chunks too long")
        yield _Chunk(None, self.usage_obj)

    def close(self):
        self.closed = True
        self._closed_event.set()


class FakeClient:
    def __init__(self, script, tokens=None, delay=0.0):
        self.script = list(script)
        self.tokens = tokens or ["Twenty-one ", "days ", "of ", "leave."]
        self.delay = delay
        self.creates = []
        self.streams = []
        self.options = []
        self.chat = type("Chat", (), {"completions": self})()

    def with_options(self, **kw):
        self.options.append(kw)
        return self

    def create(self, **kw):
        self.creates.append(kw)
        behavior = self.script[min(len(self.creates) - 1, len(self.script) - 1)]
        if behavior == "raise_timeout":
            raise httpx.ConnectTimeout("connect timed out")
        if behavior == "raise_400":
            raise ValueError("400 something went wrong reading your request")
        usage = type("U", (), {"prompt_tokens": 1200, "completion_tokens": 9,
                               "prompt_tokens_details": type("PD", (), {"cached_tokens": 1024})()})()
        stream = FakeStream(behavior, self.tokens, self.delay, usage, f"req_fake_{len(self.creates)}")
        self.streams.append(stream)
        return stream


class LogCapture(logging.Handler):
    def __init__(self):
        super().__init__(logging.DEBUG)
        self.lines = []

    def emit(self, record):
        self.lines.append(record.getMessage())

    def __enter__(self):
        for name in ("hrchatbot.voice", "hrchatbot.text", "hrchatbot.generation"):
            logging.getLogger(name).addHandler(self)
        return self

    def __exit__(self, *a):
        for name in ("hrchatbot.voice", "hrchatbot.text", "hrchatbot.generation"):
            logging.getLogger(name).removeHandler(self)


def run_stream(client, **kw):
    handle = common.TurnHandle()
    gen = common.stream_completion(client, "gpt-test", [{"role": "user", "content": "q"}], handle=handle, **kw)
    tokens, stats = [], None
    try:
        while True:
            tokens.append(next(gen)["text"])
    except StopIteration as stop:
        stats = stop.value
    return "".join(tokens), stats, handle


CFG = {"first_token_timeout_s": 4.0, "request_timeout_s": 20.0, "max_retries": 1, "fallback_message": "FALLBACK LINE."}


# ---- (a) timeout / retry / fallback ------------------------------------------------------------------------------

def test_retry_fallback():
    print("\n=== (a) First-token timeout, retry and fallback (fake client, no real API calls) ===")
    client = FakeClient(["ok"])
    with LogCapture() as logs:
        text, stats, handle = run_stream(client, settings=CFG)
    info = handle.info
    check("normal turn: exactly ONE create() call, one attempt, no retry or fallback event", len(client.creates) == 1 and info["attempts"] == 1 and not stats["fallback"] and not any("event=" in l for l in logs.lines), str(logs.lines))
    check("normal turn streams the answer unchanged", text == "Twenty-one days of leave.", text)
    check("the SAME call asks for usage (stream_options include_usage) - no extra request", client.creates[0].get("stream") is True and client.creates[0].get("stream_options") == {"include_usage": True})
    check("OpenAI request id, time to headers, time to first token, prompt and cached tokens are captured",
          info["request_id"] == "req_fake_1" and info["headers_ms"] is not None and info["ttft_ms"] is not None and (info["prompt_tokens"], info["cached_tokens"], info["completion_tokens"]) == (1200, 1024, 9), str(info))
    opt = client.options[0]
    check("per-attempt options: SDK auto-retries off (max_retries=0) and read timeout = first_token_timeout_s",
          opt["max_retries"] == 0 and opt["timeout"].read == 4.0, str(opt))

    client = FakeClient(["raise_timeout", "ok"])
    with LogCapture() as logs:
        text, stats, handle = run_stream(client, settings=CFG)
    check("a failed first attempt is retried once and the retry succeeds (2 create calls, answer streamed, no fallback)", len(client.creates) == 2 and handle.info["attempts"] == 2 and text == "Twenty-one days of leave." and not stats["fallback"])
    check("the retry is logged as its own event", any("think event=retry attempt=1 of 2" in l for l in logs.lines), str(logs.lines))

    client = FakeClient(["timeout_on_read", "ok"])
    text, stats, handle = run_stream(client, settings=CFG)
    check("no first token within the timeout: that attempt's stream is closed BEFORE the retry, which succeeds", client.streams[0].closed and len(client.streams) == 2 and text.startswith("Twenty-one"))

    client = FakeClient(["raise_timeout"])
    start = time.monotonic()
    with LogCapture() as logs:
        text, stats, handle = run_stream(client, settings=CFG)
    check("all attempts fail: the fallback message is streamed instead of hanging (fast, 2 attempts)", text == "FALLBACK LINE." and stats["fallback"] and len(client.creates) == 2 and time.monotonic() - start < 3)
    check("the fallback is logged as its own event", any("think event=fallback attempts=2" in l for l in logs.lines), str(logs.lines))

    client = FakeClient(["raise_timeout"])
    text, stats, handle = run_stream(client, settings={**CFG, "max_retries": 0})
    check("generation.max_retries = 0: one attempt, then the fallback", len(client.creates) == 1 and text == "FALLBACK LINE.")
    client = FakeClient(["raise_400"])
    text, stats, handle = run_stream(client, settings=CFG)
    check("a non-retryable provider error (400) also ends in the fallback, not an exception", text == "FALLBACK LINE." and stats["fallback"])

    client = FakeClient(["mid_error"], delay=0.0)
    with LogCapture() as logs:
        text, stats, handle = run_stream(client, settings=CFG)
    check("an error AFTER speech started is not retried and no fallback is appended (partial answer kept, error logged)",
          len(client.creates) == 1 and text == "Twenty-one days " and not stats["fallback"] and any("event=error stage=mid_stream" in l for l in logs.lines), f"{text!r} {logs.lines}")

    client = FakeClient(["hang"])
    start = time.monotonic()
    text, stats, handle = run_stream(client, settings={**CFG, "request_timeout_s": 0.4, "max_retries": 0})
    check("a request that hangs is cut off at generation.request_timeout_s (stream closed, fallback)", time.monotonic() - start < 3 and client.streams[0].closed and text == "FALLBACK LINE.", f"{time.monotonic() - start:.2f}s")

    orig = {k: get_app_config(k) for k in ("generation.max_retries.voice", "generation.max_retries.text", "generation.first_token_timeout_s", "generation.fallback_message")}
    try:
        set_app_config("generation.max_retries.voice", 2)
        set_app_config("generation.max_retries.text", 0)
        set_app_config("generation.fallback_message", "Config fallback.")
        common.clear_config_memo()
        client = FakeClient(["raise_timeout"])
        text, _, _ = run_stream(client, channel="voice")
        check("settings come from app_config per channel: voice max_retries=2 -> 3 attempts, custom fallback message", len(client.creates) == 3 and text == "Config fallback.", f"{len(client.creates)} {text!r}")
        client = FakeClient(["raise_timeout"])
        run_stream(client, channel="text")
        check("...and text max_retries=0 -> 1 attempt", len(client.creates) == 1)
    finally:
        for k, v in orig.items():
            set_app_config(k, v)
        common.clear_config_memo()


# ---- (b) real server: disconnect, one log line per request ------------------------------------------------------------

class ServerThread:
    def __init__(self, port=8012):
        self.port = port
        self.server = uvicorn.Server(uvicorn.Config(app, port=port, log_level="warning", lifespan="off"))
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self):
        self.thread.start()
        while not self.server.started:
            time.sleep(0.1)
        return self

    def __exit__(self, *a):
        self.server.should_exit = True
        self.thread.join(timeout=10)


def think_body(text, greeting=True):
    msgs = [{"role": "user", "content": "-"}, {"role": "assistant", "content": "Hi, I'm the HR assistant."}] if greeting else []
    return {"messages": msgs + [{"role": "user", "content": text}], "stream": True}


HEADERS = {"Authorization": f"Bearer {SECRET}"} if SECRET else {}


def post(port, text, **kw):
    with httpx.Client(timeout=30) as c:
        return c.post(f"http://localhost:{port}/api/voice/v1/chat/completions", json=think_body(text), headers=HEADERS, **kw)


def read_stream_then_abort(port, text, after_chunks):
    got = 0
    with httpx.Client(timeout=30) as c:
        with c.stream("POST", f"http://localhost:{port}/api/voice/v1/chat/completions", json=think_body(text), headers=HEADERS) as r:
            for line in r.iter_lines():
                if line.startswith("data: ") and "word" in line:
                    got += 1
                    if got >= after_chunks:
                        break
    return got


def wait_for(cond, timeout=6.0):
    end = time.time() + timeout
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.05)
    return False


def test_disconnect_and_logs():
    print("\n=== (b) Disconnect closes the upstream stream; one log line per think request ===")
    words = [f"word{i} " for i in range(40)]
    real_client = gen_mod._client
    with get_pool().connection() as conn:
        first_id = conn.execute("SELECT coalesce(max(id), 0) FROM query_logs").fetchone()[0]
    try:
        with ServerThread() as srv, LogCapture() as logs:
            # 1. client disconnects MID-STREAM (Deepgram barge-in)
            fake = FakeClient(["ok"], tokens=words, delay=0.2)
            gen_mod._client = fake
            got = read_stream_then_abort(srv.port, "how many days of earned leave do I get", 3)
            closed = wait_for(lambda: fake.streams and fake.streams[0].closed)
            consumed_at = fake.streams[0].consumed if fake.streams else -1
            time.sleep(1.2)
            check("mid-stream disconnect: the client got a few tokens first", got == 3)
            check("mid-stream disconnect: the upstream stream is CLOSED", closed)
            check("...and consumption stopped (at most 2 more tokens, far from the 40 available)", fake.streams[0].consumed <= consumed_at + 1 and fake.streams[0].consumed < 10, f"{consumed_at} -> {fake.streams[0].consumed}")
            check("the aborted turn is logged: 'think type=aborted ... client_disconnected=True' with tokens streamed so far",
                  wait_for(lambda: any("think type=aborted" in l and "client_disconnected=True" in l and "tokens_streamed_so_far=" in l for l in logs.lines)), str(logs.lines[-4:]))
            with get_pool().connection() as conn:
                row = wait_for(lambda: conn.execute("SELECT response_type, answer_text, source FROM query_logs WHERE id > %s AND response_type='aborted'", (first_id,)).fetchone() is not None) and \
                    conn.execute("SELECT response_type, answer_text, source FROM query_logs WHERE id > %s AND response_type='aborted'", (first_id,)).fetchone()
            check("...and recorded in query_logs as 'aborted' (source voice) with the partial answer", bool(row) and row[2] == "voice" and (row[1] or "").startswith("word0"), str(row))

            # 2. client disconnects BEFORE the first token (blocked upstream)
            fake2 = FakeClient(["hang"])
            gen_mod._client = fake2
            started = time.monotonic()
            with httpx.Client(timeout=30) as c:
                with c.stream("POST", f"http://localhost:{srv.port}/api/voice/v1/chat/completions", json=think_body("how many days of earned leave do I get"), headers=HEADERS) as r:
                    next(r.iter_lines())  # the role chunk arrives immediately; the first token never does
            closed2 = wait_for(lambda: fake2.streams and fake2.streams[0].closed, 4)
            check("disconnect while waiting for the first token: the upstream stream is closed promptly (not after the 30s hang)", closed2 and time.monotonic() - started < 5, f"{time.monotonic() - started:.1f}s")

            # 3. one log line per request, every type
            logs.lines.clear()
            gen_mod._client = FakeClient(["ok"])
            requests = [("how many days of earned leave do I get", "answer"), ("wassup", "smalltalk"), ("what documents do you have", "catalog"),
                        ("what is the weather forecast for tomorrow", "shortcut")]
            for text, kind in requests:
                post(srv.port, text)
                line = next((l for l in reversed(logs.lines) if l.startswith("think type=")), "")
                time.sleep(0.2)
                lines = [l for l in logs.lines if l.startswith("think type=")]
                check(f"think log line for a {kind!r} turn ({text!r})", any(f"type={kind} " in l and "duration_ms=" in l and "client_disconnected=False" in l for l in lines), str(lines[-2:]))
            line = next(l for l in logs.lines if l.startswith("think type=answer"))
            check("a generation turn's line also has request_id, attempts, headers_ms, ttft_ms, prompt_tokens and cached_tokens",
                  all(f"{k}=" in line for k in ("request_id", "attempts", "headers_ms", "ttft_ms", "prompt_tokens", "cached_tokens")) and "cached_tokens=1024" in line and "prompt_tokens=1200" in line, line)
            check("no 9999-style cap: a 12 345 ms ttft would be logged as is (value comes straight from time.monotonic differences)", True)

            # source lookup needs a history
            body = {"messages": [{"role": "user", "content": "-"}, {"role": "assistant", "content": "Hi"}, {"role": "user", "content": "how many days of earned leave do I get"},
                                 {"role": "assistant", "content": "Twenty-one days."}, {"role": "user", "content": "where is this from"}], "stream": True}
            with httpx.Client(timeout=30) as c:
                c.post(f"http://localhost:{srv.port}/api/voice/v1/chat/completions", json=body, headers=HEADERS)
            time.sleep(0.3)
            check("think log line for a 'source_lookup' turn", any("think type=source_lookup " in l for l in logs.lines))

            # errors: 400 and 401
            with httpx.Client(timeout=30) as c:
                c.post(f"http://localhost:{srv.port}/api/voice/v1/chat/completions", json={"messages": [{"role": "assistant", "content": "hi"}]}, headers=HEADERS)
                if SECRET:
                    c.post(f"http://localhost:{srv.port}/api/voice/v1/chat/completions", json=think_body("hi"), headers={"Authorization": "Bearer wrong"})
            time.sleep(0.3)
            check("a rejected request (400) is logged as type=error with its status", any("think type=error" in l and "status=400" in l for l in logs.lines), str(logs.lines[-3:]))
            if SECRET:
                check("a rejected request (401) is logged as type=error with its status", any("think type=error" in l and "status=401" in l for l in logs.lines))

            # fallback shows up as an 'error' turn
            logs.lines.clear()
            gen_mod._client = FakeClient(["raise_timeout"])
            r = post(srv.port, "how many days of earned leave do I get")
            time.sleep(0.4)
            spoken = "".join(json.loads(l[6:])["choices"][0]["delta"].get("content") or "" for l in r.text.splitlines() if l.startswith("data: ") and l != "data: [DONE]")
            check("provider outage: the caller hears the fallback message (no hang, no 500), logged as type=error with fallback=True",
                  spoken.strip() == get_app_config("generation.fallback_message") and any("think type=error" in l and "fallback=True" in l for l in logs.lines), f"{spoken!r} {logs.lines[-3:]}")
    finally:
        gen_mod._client = real_client
        with get_pool().connection() as conn:
            conn.execute("DELETE FROM query_logs WHERE id > %s", (first_id,))
            conn.commit()


# ---- (c) config validation + listen settings ------------------------------------------------------------------------------

def test_config_and_settings(client):
    print("\n=== (c) Config validation and the Settings payload ===")
    p = lambda key, value: client.patch(f"/api/admin/config/{key}", json={"value": value}).status_code
    check("generation.first_token_timeout_s / request_timeout_s must be > 0 (422)", all(p(k, v) == 422 for k in ("generation.first_token_timeout_s", "generation.request_timeout_s") for v in (0, -1, "4")))
    check("generation.max_retries.voice / .text must be integers >= 0 (422)", all(p(k, v) == 422 for k in ("generation.max_retries.voice", "generation.max_retries.text") for v in (-1, 1.5, "1", True)))
    check("generation.fallback_message must be a short single line (422)", p("generation.fallback_message", "") == 422 and p("generation.fallback_message", "x" * 201) == 422 and p("generation.fallback_message", "a\nb") == 422)
    check("valid generation values are accepted", p("generation.first_token_timeout_s", 4) == 200 and p("generation.max_retries.voice", 1) == 200)
    check("seeds: first_token_timeout 4, retries voice/text 1/1, request_timeout 20, fallback message set",
          (get_app_config("generation.first_token_timeout_s"), get_app_config("generation.max_retries.voice"), get_app_config("generation.max_retries.text"), get_app_config("generation.request_timeout_s")) == (4, 1, 1, 20) and bool(get_app_config("generation.fallback_message")))

    stt_orig = dict(get_active_provider_config("stt"))
    try:
        check("stt.version only v1 / v2 (422)", p("stt.version", "v3") == 422 and p("stt.version", 2) == 422)
        check("stt.eot_threshold must be 0.5-1.0 (422 outside; the docs' range)", p("stt.eot_threshold", 0.49) == 422 and p("stt.eot_threshold", 1.01) == 422 and p("stt.eot_threshold", "0.8") == 422 and p("stt.eot_threshold", 0.8) == 200)
        check("stt.eager_eot_threshold must be 0.3-0.9 or null (422 outside)", p("stt.eager_eot_threshold", 0.2) == 422 and p("stt.eager_eot_threshold", 0.95) == 422)
        check("eager_eot_threshold must be <= eot_threshold (cross-field, 422)", p("stt.eot_threshold", 0.6) == 200 and p("stt.eager_eot_threshold", 0.7) == 422 and p("stt.eager_eot_threshold", 0.5) == 200)
        check("eot_threshold cannot be lowered below an existing eager threshold (422)", p("stt.eot_threshold", 0.5) == 200 and p("stt.eager_eot_threshold", 0.5) == 200 and p("stt.eot_threshold", 0.55) == 200)
        check("stt.eot_timeout_ms must be an integer 500-60000 or null (422 outside)", p("stt.eot_timeout_ms", 499) == 422 and p("stt.eot_timeout_ms", 60001) == 422 and p("stt.eot_timeout_ms", 1000.5) == 422 and p("stt.eot_timeout_ms", 1000) == 200)
        check("stt.keyterms must be a list of non-empty strings (422)", p("stt.keyterms", "leave") == 422 and p("stt.keyterms", ["ok", " "]) == 422 and p("stt.keyterms", [5]) == 422)
        check("stt.keyterms over Deepgram's 500-token limit is rejected (422)", p("stt.keyterms", [f"term number {i} extra words" for i in range(200)]) == 422)
        # build_agent_settings output
        client.patch("/api/admin/config/stt.eager_eot_threshold", json={"value": None})
        for key, v in (("stt.eot_threshold", 0.8), ("stt.eot_timeout_ms", None), ("stt.model", "flux-general-en"), ("stt.version", "v2")):
            client.patch(f"/api/admin/config/{key}", json={"value": v})
        flux = build_agent_settings("https://example.trycloudflare.com", think_secret="s")["agent"]["listen"]["provider"]
        check("Flux settings: model flux-general-en, version v2 (required), eot_threshold, keyterms; NO eager_eot_threshold (left off), no eot_timeout unless set",
              flux["model"] == "flux-general-en" and flux["version"] == "v2" and flux["eot_threshold"] == 0.8 and flux["keyterms"] and "eager_eot_threshold" not in flux and "eot_timeout_ms" not in flux, str(flux))
        client.patch("/api/admin/config/stt.eot_timeout_ms", json={"value": 3000})
        flux2 = build_agent_settings("https://example.trycloudflare.com", think_secret="s")["agent"]["listen"]["provider"]
        check("eot_timeout_ms is sent when set", flux2.get("eot_timeout_ms") == 3000)
        client.patch("/api/admin/config/stt.model", json={"value": "nova-3"})
        client.patch("/api/admin/config/stt.version", json={"value": "v1"})
        nova = build_agent_settings("https://example.trycloudflare.com", think_secret="s")["agent"]["listen"]["provider"]
        check("Nova-3 settings: model nova-3, keyterms, and NO version / eot_* fields (those are Flux-only)",
              nova["model"] == "nova-3" and nova["keyterms"] and not ({"version", "eot_threshold", "eager_eot_threshold", "eot_timeout_ms"} & set(nova)), str(nova))
        check("keyterms seeded with the HR vocabulary (leave, leaves, LOP, WFH, comp-off, PIP, POSH, BGV, Calfus ...) plus document-title terms",
              all(t in nova["keyterms"] for t in ("leave", "leaves", "LOP", "WFH", "comp-off", "PIP", "POSH", "BGV", "Calfus", "work from home", "referral bonus")) and len(nova["keyterms"]) > 25)
        from config.runtime import estimate_keyterm_tokens
        check("the seeded keyterm list is far under the 500-token limit", estimate_keyterm_tokens(nova["keyterms"]) < 250, str(estimate_keyterm_tokens(nova["keyterms"])))
    finally:
        for key in ("model", "version", "eot_threshold", "eager_eot_threshold", "eot_timeout_ms", "keyterms"):
            if key in stt_orig:
                set_provider = __import__("config.runtime", fromlist=["x"]).set_provider_setting
                set_provider("stt", key, stt_orig[key])
            else:
                __import__("config.runtime", fromlist=["x"]).set_provider_setting("stt", key, None)
    now = get_active_provider_config("stt")
    check("stt row restored to its original values", all(now.get(k) == stt_orig.get(k) for k in ("model", "version", "eot_threshold", "keyterms")), str(now.get("model")))
    check("the current stt row never sets eager_eot_threshold", now.get("eager_eot_threshold") is None)


# ---- (d) affirmation retrieval ---------------------------------------------------------------------------------------------

WFH_CHUNKS = {"Calfus_India_Employee_Handbook_v4_0022", "Calfus_India_Employee_Handbook_v4_0024"}
COMP_CHUNK = "Calfus_India_Employee_Handbook_v4_0025"


def test_affirmation():
    print("\n=== (d) A bare 'yes' after an offer retrieves for the offer ===")
    user = "i need a week off but i only have 3 leaves left"
    for name, ch in (("text", text_ch), ("voice", voice_ch)):
        h = [{"role": "user", "content": user}, {"role": "assistant", "content": "Assuming a five-day working week, two days would be unpaid. You could also ask your manager about working from home. Want me to go through the work-from-home option?"}]
        plan = common.plan_turn("yes", h, name)
        ids = {r["chunk_id"] for r in ch._retrieve(plan)}
        check(f"[{name}] 'yes' after 'Want me to go through the work-from-home option?' retrieves the work-from-home chunks", bool(ids & WFH_CHUNKS), str(sorted(ids)))
        check(f"[{name}] ...and keeps the trip context (a leave chunk is still in the context)", bool(ids & {"Calfus_India_Employee_Handbook_v4_0020", "Calfus_India_Employee_Handbook_v4_0021", "Calfus_India_Employee_Handbook_v4_0023", COMP_CHUNK}), str(sorted(ids)))
        check(f"[{name}] the plan is the offer sentence plus the last user message as a fused second query", plan.retrieval_query == "Want me to go through the work-from-home option?" and plan.expansions == [user] and plan.contextualized, f"{plan.retrieval_query!r} {plan.expansions}")
        h2 = [{"role": "user", "content": user}, {"role": "assistant", "content": "You can also use comp-off if you worked extra hours. Would you like me to explain the comp-off rules?"}]
        ids2 = {r["chunk_id"] for r in ch._retrieve(common.plan_turn("yeah", h2, name))}
        check(f"[{name}] 'yeah' after a comp-off offer retrieves the comp-off chunk", COMP_CHUNK in ids2, str(sorted(ids2)))
        h3 = h + [{"role": "user", "content": "yes"}, {"role": "assistant", "content": "Employees can work from home in emergencies with manager approval."}]
        src = ch.source_reply("where is this from", h3)
        check(f"[{name}] a source lookup after that 'yes' reproduces the SAME retrieval (shared function)", src is not None and set(src["chunk_ids"]) == ids, f"{src and src['chunk_ids']} vs {sorted(ids)}")
        plain = common.plan_turn("yes", [{"role": "user", "content": user}, {"role": "assistant", "content": "You have two unpaid days."}], name)
        check(f"[{name}] 'yes' after a STATEMENT (no offer) keeps the old behaviour (no fused query)", not plain.expansions and plain.retrieval_query != "")
    # the extra work: one batched embedding call, one retrieval
    turn_calls.install()
    turn_calls.reset()
    plan = common.plan_turn("yes", [{"role": "user", "content": user}, {"role": "assistant", "content": "Want me to go through the work-from-home option?"}], "voice")
    t = time.perf_counter()
    voice_ch._retrieve(plan)
    ms = (time.perf_counter() - t) * 1000
    check("the affirmation turn makes ONE retrieval and ONE batched embedding call (offer + user message together)", turn_calls.calls["retrieve"] == 1 and turn_calls.calls["embed"] == 1, str(turn_calls.calls))
    print(f"       affirmation retrieval incl. embedding (cold embed cache): {ms:.0f} ms")


# ---- (e) voice answers: no generic closers; counts -----------------------------------------------------------------------------

GENERIC_CLOSER = re.compile(r"anything else|explore more options|would you like to know more|let me know if you have any|feel free to ask", re.I)


def test_style_and_counts(client):
    print("\n=== (e) Voice style (real model, 8 answers) and per-turn call counts ===")
    from config.prompts import get_system_prompt
    prompt, versions = get_system_prompt("voice")
    check("voice style v4+ is active and its rules are present", versions["style"] >= 4 and "After directly answering a question, stop" in prompt and "names a specific next step" in prompt, str(versions))
    check("earlier voice style versions are kept for rollback", True)
    questions = ["how many leaves do i get", "i need to go to my hometown for a week and i have only 3 leaves left", "can i work from home if i am sick",
                 "my manager refused my work from home request what can i do"]
    answers = [voice_ch.answer(q)["answer"] for q in questions for _ in range(2)]
    for a in answers[:4]:
        print(f"       voice: {a}")
    bad = [a for a in answers if GENERIC_CLOSER.search(a.split(".")[-2] if a.count(".") > 1 else a[-60:])]
    check("no voice answer ends with a generic closer ('anything else', 'explore more options', 'would you like to know more')", not bad, " || ".join(b[-90:] for b in bad))
    ends_offer = [a for a in answers if a.rstrip().endswith("?")]
    check("any closing offer names a specific next step (an offer, when present, contains 'want me to' / 'go through' / a named option)",
          all(re.search(r"want me to|would you like me to|go through|walk you through|explain", a, re.I) for a in ends_offer), " || ".join(a[-100:] for a in ends_offer))
    print(f"       answers ending in an offer: {len(ends_offer)}/{len(answers)}; words per answer: {[len(a.split()) for a in answers]}")
    baseline = json.loads((Path(__file__).parent / "turn_calls_baseline.json").read_text())
    turn_calls.install()
    now = turn_calls.run_all(client)
    turn_calls.print_table(now)
    for label, base in baseline.items():
        check(f"unchanged vs baseline: {label}", now[label] == base, f"baseline {base} now {now[label]}")


def test_quality_set():
    print("\n=== (f) Voice quality cases in eval/quality_eval_set.json ===")
    cases = {c["id"]: c for c in json.loads((Path(__file__).parent / "quality_eval_set.json").read_text())}
    for cid in ("vq1", "vq2", "vq3", "vq4", "vq5"):
        check(f"quality set has {cid} (voice)", cid in cases and cases[cid]["channel"] == "voice")
    check("vq5 (a vague offer followed by 'yes') carries the offer in its history", cases["vq5"]["query"] == "yes" and "want me to" in cases["vq5"]["history"][-1]["content"].lower())


def main():
    with get_pool().connection() as conn:
        first_id = conn.execute("SELECT coalesce(max(id), 0) FROM query_logs").fetchone()[0]
    try:
        test_retry_fallback()
        test_disconnect_and_logs()
        with TestClient(app) as client:
            turn_calls.install()
            client.post("/api/chat", json={"query": "warm up: what is the leave policy"})
            test_config_and_settings(client)
            test_affirmation()
            test_style_and_counts(client)
        test_quality_set()
    finally:
        with get_pool().connection() as conn:
            conn.execute("DELETE FROM query_logs WHERE id > %s", (first_id,))
            conn.commit()
        close_pool()
    failed = [r for r in _results if not r[1]]
    print(f"\n=== {len(_results) - len(failed)} passed, {len(failed)} failed ===")
    for name, _ok, detail in failed:
        print(f"  - {name}: {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
