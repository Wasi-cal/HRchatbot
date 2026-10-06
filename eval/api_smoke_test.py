#!/usr/bin/env python3
"""End-to-end smoke test for the API (server must be running: ./run_api.sh).

    python3 -m eval.api_smoke_test [--base-url http://localhost:8000] [--upload "hr docs/Travel Policy_Ver1.0.pdf"]

Exercises: normal answer, clarification + resolution, shortcut refusal,
document upload, document list, restrict toggle, usage summary/recent.
Query-dependent steps depend on the loaded corpus; steps that can't
trigger their scenario print SKIP/WARN instead of failing hard.
Restricting is reverted at the end. Equivalent curl commands are in
api/README.md.
"""
import argparse
import json
import sys
from pathlib import Path

import httpx

failures = 0


def check(label: str, ok: bool, detail=""):
    global failures
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" - {detail}" if detail else ""))
    failures += 0 if ok else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--upload", default="hr docs/Travel Policy_Ver1.0.pdf")
    ap.add_argument("--answer-query", default="What is the parental leave policy?")
    ap.add_argument("--clarify-query", default="How many days of leave do I get?")
    ap.add_argument("--clarify-replies", default='{"country": "I work in India", "employment_type": "I\'m a full-time employee"}',
                    help="JSON map of blocking_tag_type -> natural-language reply")
    ap.add_argument("--shortcut-query", default="best pizza topping")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.base_url, timeout=300)
    chat = lambda **body: c.post("/api/chat", json=body)

    before = c.get("/api/admin/usage/summary", params={"days": 0}).json()["total_queries"]

    print("1. Fresh query -> answer (user fully specified)")
    r = chat(query=args.answer_query, user_attributes={"country": "India", "employment_type": "full-time employee"})
    print("    ", r.json())
    check("200 + type answer", r.status_code == 200 and r.json()["type"] in ("answer", "refusal_shortcut"))

    print("2. Fresh query -> clarification, then resolve")
    r = chat(query=args.clarify_query)
    body = r.json()
    print("    ", body)
    if body.get("type") != "needs_clarification":
        print("  [SKIP] query didn't trigger clarification on this corpus; pass --clarify-query")
    else:
        reply = json.loads(args.clarify_replies).get(body["blocking_tag_type"], "I'm not sure")
        r2 = chat(query=reply, user_attributes=body.get("user_attributes"),
                  pending_clarification={"original_query": body["original_query"],
                                         "blocking_tag_type": body["blocking_tag_type"]})
        print("    ", r2.json())
        resolved = r2.json()["type"] in ("answer", "refusal_shortcut") or (
            r2.json()["type"] == "needs_clarification" and r2.json()["blocking_tag_type"] != body["blocking_tag_type"])
        check("reply resolved (answer, or moved on to a different attribute)", resolved)
        check("merged attributes echoed", bool((r2.json().get("user_attributes") or {}).get(body["blocking_tag_type"])))
        r3 = chat(query="banana purple 42", user_attributes=body.get("user_attributes"),
                  pending_clarification={"original_query": body["original_query"],
                                         "blocking_tag_type": body["blocking_tag_type"]})
        print("    unusable reply ->", r3.json())
        check("unusable reply re-asks", r3.json()["type"] == "needs_clarification")

    print("3. Shortcut refusal")
    r = chat(query=args.shortcut_query, user_attributes={"country": "India", "employment_type": "full-time employee"})
    print("    ", r.json())
    if r.json()["type"] == "refusal_shortcut":
        check("type refusal_shortcut", True)
    else:
        print("  [WARN] did not hit shortcut; pass --shortcut-query with a nonsense query")

    print("4. Upload")
    path = Path(args.upload)
    with path.open("rb") as f:
        r = c.post("/api/admin/documents/upload", files={"file": (path.name, f)})
    print("    ", r.status_code, r.json())
    check("upload ok", r.status_code == 200)
    doc_id = r.json().get("document_id") if r.status_code == 200 else None

    print("5. List reflects upload")
    docs = c.get("/api/admin/documents").json()
    mine = next((d for d in docs if d["id"] == doc_id), None)
    check("uploaded doc listed", mine is not None, mine and f"{mine['title']} status={mine['status']} chunks={mine['chunk_count']}")

    print("6. Restrict toggle")
    if doc_id:
        r = c.patch(f"/api/admin/documents/{doc_id}/restrict", json={"is_restricted": True})
        check("PATCH returns is_restricted=true", r.json().get("is_restricted") is True)
        mine = next(d for d in c.get("/api/admin/documents").json() if d["id"] == doc_id)
        check("list reflects restriction", mine["is_restricted"] is True)
        c.patch(f"/api/admin/documents/{doc_id}/restrict", json={"is_restricted": False})
        mine = next(d for d in c.get("/api/admin/documents").json() if d["id"] == doc_id)
        check("restriction reverted", mine["is_restricted"] is False)

    print("7. Usage")
    s = c.get("/api/admin/usage/summary", params={"days": 0}).json()
    print("    ", s)
    check("summary counted this run's queries", s["total_queries"] > before)
    recent = c.get("/api/admin/usage/recent", params={"limit": 5}).json()
    print("    ", [(x["response_type"], x["query_text"][:40]) for x in recent])
    check("recent newest-first", len(recent) > 0 and recent == sorted(recent, key=lambda x: x["created_at"], reverse=True))

    print(f"\n{'ALL PASSED' if not failures else f'{failures} FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
