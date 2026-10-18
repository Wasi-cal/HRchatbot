#!/usr/bin/env python3
"""Connection handling and timeouts of the OpenAI generation path (fake clients, no real model calls).

    python3 -m eval.stall_fix_test

Covers: split timeouts (connect / first token / stream gap), no explicit close of a fully read stream (its connection goes back
to the pool), close on failure / cancel, SDK retries off, the single process-wide client lifecycle (build, idempotent get, warm-up,
close, FastAPI lifespan), and config validation. Per-turn call counts and the retrieval baseline are covered by voice_robustness_test
and run_regression.
"""
import os
import sys
import threading
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi.testclient import TestClient

import generation.generate as gen_mod
from api.main import app
from channels import common
from config.runtime import get_app_config, set_app_config
from vectorstore.db import close_pool

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if not ok and detail else ""))


class _Chunk:
    def __init__(self, text=None):
        self.choices = [type("C", (), {"delta": type("D", (), {"content": text})()})()] if text is not None else []
        self.usage = None


class Stream:
    def __init__(self, behavior="ok"):
        self.behavior, self.closed = behavior, False
        self.request = type("Rq", (), {"extensions": {"timeout": {"connect": 3.0, "read": 2.0, "write": 10.0, "pool": 5.0}}})()
        self.response = type("R", (), {"headers": {"x-request-id": "req_x"}, "request": self.request})()
        self._ev = threading.Event()

    def __iter__(self):
        if self.behavior == "hang":
            self._ev.wait(10)
            raise ConnectionError("closed")
        for t in ("a ", "b ", "c"):
            yield _Chunk(t)

    def close(self):
        self.closed = True
        self._ev.set()


class Client:
    def __init__(self, behaviors):
        self.behaviors, self.streams, self.options = list(behaviors), [], []
        self.chat = type("Chat", (), {"completions": self})()

    def with_options(self, **kw):
        self.options.append(kw)
        return self

    def create(self, **kw):
        s = Stream(self.behaviors[min(len(self.streams), len(self.behaviors) - 1)])
        self.streams.append(s)
        return s


CFG = {"first_token_timeout_s": 2.0, "connect_timeout_s": 3.0, "stream_gap_timeout_s": 6.0, "request_timeout_s": 20.0, "max_retries": 1, "fallback_message": "FALLBACK."}


def run(client, **kw):
    handle = common.TurnHandle()
    gen = common.stream_completion(client, "m", [{"role": "user", "content": "q"}], handle=handle, **kw)
    out = []
    try:
        while True:
            out.append(next(gen)["text"])
    except StopIteration as stop:
        return "".join(out), stop.value, handle


def main():
    # ---- timeouts and closing
    c = Client(["ok"])
    text, stats, h = run(c, settings=CFG)
    t = c.options[0]["timeout"]
    check("per-attempt timeouts: connect 3.0 (generation.connect_timeout_s), first-token/read 2.0, SDK retries off", (t.connect, t.read) == (3.0, 2.0) and c.options[0]["max_retries"] == 0, str(c.options[0]))
    check("after the response starts the read timeout is relaxed to the stream-gap timeout (6.0), so a mid-answer pause is not cut at 2.0 s",
          c.streams[0].request.extensions["timeout"]["read"] == 6.0, str(c.streams[0].request.extensions))
    check("a fully read stream is NOT closed explicitly (its connection returns to the pool for reuse)", not c.streams[0].closed and text == "a b c")
    c = Client(["hang", "ok"])
    timer = threading.Timer(0.3, lambda: c.streams[0].close())  # the first attempt never produces a token; closing it simulates the read timeout
    timer.start()
    text, stats, h = run(c, settings=CFG)
    check("an attempt that fails before the first token is closed and retried once; the retry's answer is streamed", c.streams[0].closed and len(c.streams) == 2 and text == "a b c" and h.info["attempts"] == 2, str((len(c.streams), h.info)))
    c = Client(["hang"])
    text, stats, h = run(c, settings={**CFG, "max_retries": 0, "request_timeout_s": 0.3})
    check("a hung request is cut at the request deadline: stream closed, fallback streamed", c.streams[0].closed and text == "FALLBACK." and stats["fallback"])
    handle = common.TurnHandle()
    c = Client(["hang"])
    gen = common.stream_completion(c, "m", [{"role": "user", "content": "q"}], handle=handle, settings=CFG)
    th = threading.Thread(target=lambda: list(_drain(gen)), daemon=True)
    th.start()
    time.sleep(0.2)
    handle.cancel()
    th.join(3)
    check("cancel (client disconnect) closes the upstream stream", c.streams and c.streams[0].closed)

    # ---- single process-wide client lifecycle
    saved = gen_mod._client
    try:
        gen_mod._client = None
        built = gen_mod.build_client()
        pool = built._client._transport._pool
        check("build_client: SDK-internal retries off (they must not stack with ours)", built.max_retries == 0)
        check("build_client: keepalive_expiry comes from app_config generation.keepalive_expiry_s", pool._keepalive_expiry == float(get_app_config("generation.keepalive_expiry_s", 5.0)), str(pool._keepalive_expiry))
        check("build_client: base URL is OpenAI's API", str(built.base_url).startswith("https://api.openai.com"))
        built.close()
        gen_mod._client = None
        a, b = gen_mod.get_client(), gen_mod.get_client()
        check("get_client returns ONE process-wide client", a is b)

        class Models:
            def __init__(self, fail=False):
                self.calls, self.fail = 0, fail

            def retrieve(self, name):
                self.calls += 1
                if self.fail:
                    raise ConnectionError("no network")

        class Fake:
            def __init__(self, fail=False):
                self.models, self.closed = Models(fail), False

            def close(self):
                self.closed = True

        f = Fake()
        gen_mod._client = f
        check("init_client(warm_up=True) sends exactly ONE lightweight non-LLM request (models.retrieve) and reuses the client", gen_mod.init_client(True) is f and f.models.calls == 1)
        gen_mod.init_client(False)
        check("init_client(warm_up=False) sends none", f.models.calls == 1)
        bad = Fake(fail=True)
        gen_mod._client = bad
        gen_mod.init_client(True)
        check("a failing warm-up is logged and ignored (never blocks startup)", bad.models.calls == 1)
        gen_mod.close_client()
        check("close_client closes the client and clears the singleton", bad.closed and gen_mod._client is None)

        # lifespan: startup creates and warms ONE client, shutdown closes it
        probe = Fake()
        original_get = gen_mod.get_client
        created = []

        def counting_get():
            if gen_mod._client is None:
                gen_mod._client = probe
                created.append(1)
            return gen_mod._client

        gen_mod.get_client = counting_get
        gen_mod._client = None
        if os.environ.get("OPENAI_API_KEY"):
            with TestClient(app) as client:
                check("FastAPI lifespan creates the process-wide client once at startup and warms it up (one non-LLM request)", len(created) == 1 and probe.models.calls == 1, f"{created} {probe.models.calls}")
                client.get("/health")
            check("FastAPI shutdown closes the client", probe.closed and gen_mod._client is None)
        gen_mod.get_client = original_get
    finally:
        gen_mod._client = saved

    # ---- config
    with TestClient(app) as client:
        p = lambda k, v: client.patch(f"/api/admin/config/{k}", json={"value": v}).status_code
        orig = {k: get_app_config(k) for k in ("generation.first_token_timeout_s", "generation.connect_timeout_s", "generation.stream_gap_timeout_s", "generation.keepalive_expiry_s", "generation.client_warmup")}
        try:
            check("timeouts and keep-alive must be numbers > 0 (422)", all(p(k, v) == 422 for k in ("generation.first_token_timeout_s", "generation.connect_timeout_s", "generation.stream_gap_timeout_s", "generation.keepalive_expiry_s") for v in (0, -1, "2")))
            check("client_warmup must be a boolean (422)", p("generation.client_warmup", "yes") == 422)
            check("valid values accepted", p("generation.connect_timeout_s", 3.0) == 200 and p("generation.client_warmup", True) == 200)
        finally:
            for k, v in orig.items():
                if v is not None:
                    set_app_config(k, v)
    seeds = {k: get_app_config(k) for k in orig}
    check("seeds: first-token 2.0, connect 3.0, stream gap 6.0, keep-alive 5.0, warm-up on", (seeds["generation.first_token_timeout_s"], seeds["generation.connect_timeout_s"], seeds["generation.stream_gap_timeout_s"], seeds["generation.keepalive_expiry_s"], seeds["generation.client_warmup"]) == (2.0, 3.0, 6.0, 5.0, True), str(seeds))
    close_pool()
    failed = results.count(False)
    print(f"\n=== {len(results) - failed} passed, {failed} failed ===")
    sys.exit(1 if failed else 0)


def _drain(gen):
    try:
        for _ in gen:
            pass
    except Exception:  # noqa: BLE001, S110
        pass
    return []


if __name__ == "__main__":
    main()
