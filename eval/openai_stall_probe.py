#!/usr/bin/env python3
"""Standalone probe for intermittent first-token stalls on the OpenAI path. Changes nothing in the app.

    PYTHONPATH=<dir with h2> python3 -m eval.openai_stall_probe --label p1 [--calls 30]
    python3 -m eval.openai_stall_probe --label real --real-prompt CONFIG [--calls 10]

Configurations (each has its OWN client / connection pool; all are fired at the same moment after each idle gap, so network drift
hits every configuration equally; gaps cycle 3, 10, 20, 30 s):
  a_new_client      a new client (new connection) per call
  b_shared_default  one shared client, httpx default keep-alive (expiry 5 s)
  c_shared_short    one shared client, keepalive_expiry 2 s
  d_shared_http2    one shared client over HTTP/2 (needs the h2 package)
  e_shared_ipv4     one shared client bound to IPv4 (local_address 0.0.0.0)
  f_shared_long     one shared client, keepalive_expiry 120 s
Each call is a streamed chat completion with max_tokens 5. Recorded per call: connect (TCP incl. DNS), TLS, time to response
headers, time to first token, whether the connection was reused. The SDK's own retries are off (max_retries=0) and the timeout is
generous (30 s) so stalls are measured instead of aborted.
"""
import argparse
import json
import os
import socket
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import openai

MODEL = "gpt-4o-mini"
GAPS = [3, 10, 20, 30]
RESULTS = Path(__file__).parent / "results"


class TracingTransport(httpx.HTTPTransport):
    """Records httpcore trace events of each request in a thread-local record."""
    local = threading.local()

    def handle_request(self, request):
        rec = {"t0": time.monotonic(), "events": {}}
        TracingTransport.local.rec = rec

        def trace(name, info):
            rec["events"].setdefault(name, time.monotonic())

        request.extensions = {**request.extensions, "trace": trace}
        return super().handle_request(request)


def make_http_client(kind):
    limits = httpx.Limits(max_connections=100, max_keepalive_connections=20)
    kw = {"timeout": httpx.Timeout(30.0, connect=10.0)}
    if kind == "c_shared_short":
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=2.0)
    elif kind == "f_shared_long":
        limits = httpx.Limits(max_connections=100, max_keepalive_connections=20, keepalive_expiry=120.0)
    transport = TracingTransport(limits=limits, http2=(kind == "d_shared_http2"), local_address="0.0.0.0" if kind == "e_shared_ipv4" else None)
    return httpx.Client(transport=transport, **kw)


class Config:
    def __init__(self, name, key):
        self.name, self.key = name, key
        self.shared = None if name == "a_new_client" else openai.OpenAI(api_key=key, max_retries=0, http_client=make_http_client(name))

    def call(self, messages, max_tokens):
        client = self.shared or openai.OpenAI(api_key=self.key, max_retries=0, http_client=make_http_client(self.name))
        dns0 = time.monotonic()
        try:
            socket.getaddrinfo("api.openai.com", 443, socket.AF_INET, socket.SOCK_STREAM)
        except OSError:
            pass
        dns_ms = (time.monotonic() - dns0) * 1000
        t0 = time.monotonic()
        first = None
        err = None
        rec = None
        processing_ms = None
        try:
            stream = client.chat.completions.create(model=MODEL, messages=messages, max_tokens=max_tokens, stream=True)
            rec = getattr(TracingTransport.local, "rec", None)
            processing_ms = stream.response.headers.get("openai-processing-ms")  # OpenAI's own server-side time (queue + model start)
            for chunk in stream:
                if first is None and chunk.choices and chunk.choices[0].delta.content:
                    first = (time.monotonic() - t0) * 1000
            stream.close()
        except Exception as exc:  # noqa: BLE001
            err = type(exc).__name__
        rec = rec or getattr(TracingTransport.local, "rec", None) or {"t0": t0, "events": {}}
        ev, base = rec["events"], rec["t0"]
        ms = lambda a, b: round((ev[b] - ev[a]) * 1000, 1) if a in ev and b in ev else None
        out = {
            "config": self.name, "dns_ms": round(dns_ms, 1),
            "connect_ms": ms("connection.connect_tcp.started", "connection.connect_tcp.complete"),
            "tls_ms": ms("connection.start_tls.started", "connection.start_tls.complete"),
            "headers_ms": next((round((ev[k] - base) * 1000, 1) for k in ("http11.receive_response_headers.complete", "http2.receive_response_headers.complete") if k in ev), None),
            "first_token_ms": None if first is None else round(first, 1),
            "reused": "connection.connect_tcp.started" not in ev,
            "openai_processing_ms": None if processing_ms is None else float(processing_ms),
            "error": err,
        }
        if self.shared is None:
            client.close()
        return out


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, round(q * (len(values) - 1)))] if values else None


def summarize(records):
    table = {}
    for group_name, keyf in (("config", lambda r: (r["config"],)), ("config+gap", lambda r: (r["config"], r["gap"]))):
        groups = {}
        for r in records:
            groups.setdefault(keyf(r), []).append(r)
        for key, rs in sorted(groups.items()):
            ft = [r["first_token_ms"] if r["first_token_ms"] is not None else 30000 for r in rs]  # an error / no token counts as a 30 s stall
            row = {"n": len(rs), "p50": round(pct(ft, .5)), "p95": round(pct(ft, .95)), "max": round(max(ft)),
                   "over2s": sum(1 for x in ft if x > 2000), "over3s": sum(1 for x in ft if x > 3000), "errors": sum(1 for r in rs if r["error"]),
                   "reused%": round(100 * sum(1 for r in rs if r["reused"]) / len(rs))}
            hs = [r["headers_ms"] for r in rs if r["headers_ms"] is not None]
            cs = [r["connect_ms"] for r in rs if r["connect_ms"] is not None]
            ts = [r["tls_ms"] for r in rs if r["tls_ms"] is not None]
            row.update(headers_p50=round(statistics.median(hs)) if hs else None, connect_p50=round(statistics.median(cs)) if cs else None, tls_p50=round(statistics.median(ts)) if ts else None)
            table["|".join(map(str, key))] = row
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument("--configs", default="a_new_client,b_shared_default,c_shared_short,d_shared_http2,e_shared_ipv4,f_shared_long")
    ap.add_argument("--real-prompt", help="use the real ~2,500-token voice prompt (system prompt + 3 excerpts) with this one configuration")
    ap.add_argument("--gap", type=int, help="fixed idle gap in seconds (default: cycle 3, 10, 20, 30)")
    args = ap.parse_args()
    key = os.environ["OPENAI_API_KEY"]
    names = [args.real_prompt] if args.real_prompt else args.configs.split(",")
    configs = [Config(n, key) for n in names]
    if args.real_prompt:
        from channels import common
        from config.prompts import get_system_prompt
        from vectorstore.db import get_pool
        with get_pool().connection() as conn:
            chunks = [{"chunk_id": cid, "section_path": sp, "text": t, "document_title": title, "applies_to": ""} for cid, sp, t, title in conn.execute(
                "SELECT c.chunk_id, c.section_path, c.text, d.document_title FROM chunks c JOIN documents d ON d.id=c.document_id "
                "WHERE d.status='active' AND c.chunk_id LIKE 'Calfus_India_Employee_Handbook_v4_002%' ORDER BY c.chunk_id LIMIT 4").fetchall()]
        messages = common.build_messages("how many days of earned leave do I get", chunks, get_system_prompt("voice")[0])
        max_tokens = 40
        print("real prompt chars:", sum(len(m["content"]) for m in messages))
    else:
        messages, max_tokens = [{"role": "user", "content": "Say OK."}], 5
    for c in configs:  # warm-up (not counted): DNS, first connection, SDK imports
        c.call(messages, max_tokens)
    records = []
    started = time.time()
    for i in range(args.calls):
        gap = args.gap or GAPS[i % len(GAPS)]
        time.sleep(gap)
        with ThreadPoolExecutor(len(configs)) as pool:
            for r in pool.map(lambda c: c.call(messages, max_tokens), configs):
                r.update(gap=gap, call=i + 1)
                records.append(r)
        print(f"call {i + 1}/{args.calls} gap {gap}s: " + " ".join(f"{r['config'][:1]}={r['first_token_ms']}" for r in records[-len(configs):]), flush=True)
    table = summarize(records)
    # rough cost: tiny prompt ~12 tokens in + ~5 out, or the real prompt; gpt-4o-mini $0.15 / $0.60 per 1M tokens
    in_tok = (sum(len(m["content"]) for m in messages) / 4) if args.real_prompt else 12
    out_tok = max_tokens
    n_calls = len(records) + len(configs)
    cost = n_calls * (in_tok * 0.15 + out_tok * 0.60) / 1e6
    out = {"label": args.label, "calls": args.calls, "elapsed_s": round(time.time() - started), "estimated_cost_usd": round(cost, 4), "table": table, "records": records}
    (RESULTS / f"stall_probe_{args.label}.json").write_text(json.dumps(out, indent=2))
    print(f"PROBE-DONE elapsed {out['elapsed_s']} s, {n_calls} calls, estimated cost ${cost:.4f}")
    for k, v in table.items():
        print("PROBE", k.ljust(26), v)


if __name__ == "__main__":
    main()
