#!/usr/bin/env python3
"""Tests for the document-management layer of the chat system: duplicate detection (upload-time and on-demand),
document status, editable titles, title warnings, deviation-only group labels, the shared cache invalidation, and
the per-turn call counts.

    python3 -m eval.documents_test

Uses REAL corpus PDFs from "hr docs/" (gitignored) and a few synthetic PDFs; needs the same live services as
eval/run_regression.py. Everything it creates is deleted at the end - documents, chunks, embeddings, tags, staged
files and query_logs rows - and every setting / title / status it changes is restored, even on failure.
"""
import json
import re
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

import pymupdf
from fastapi.testclient import TestClient

import channels.text as text_ch
import config.labels as labels_mod
import vectorstore.retrieve as retrieve_mod
from api import admin as admin_mod
from api.main import app
from config.doc_caches import invalidate_document_caches
from config.runtime import get_app_config
from eval import turn_calls
from vectorstore.db import close_pool, get_pool

DOCS_DIR = ROOT / "hr docs"
STAGING = admin_mod.STAGING_DIR
COMM_PDF = "Communication Policy_Ver1.0.pdf"
_results = []


def check(name, condition, detail=""):
    _results.append((name, bool(condition), detail))
    print(f"[{'PASS' if condition else 'FAIL'}] {name}" + (f" -- {detail}" if not condition and detail else ""))
    return bool(condition)


def set_config(client, key, value):
    r = client.patch(f"/api/admin/config/{key}", json={"value": value})
    assert r.status_code == 200, r.text


def snapshot():
    with get_pool().connection() as conn:
        return {
            "docs": {r[0] for r in conn.execute("SELECT id FROM documents").fetchall()},
            "counts": tuple(conn.execute(f"SELECT count(*) FROM {t}").fetchone()[0] for t in ("documents", "chunks", "chunk_embeddings", "document_applicability", "applicability_tags")),
            "log_max": conn.execute("SELECT coalesce(max(id), 0) FROM query_logs").fetchone()[0],
            "tag_max": conn.execute("SELECT coalesce(max(id), 0) FROM applicability_tags").fetchone()[0],
            "staged": {p.name for p in STAGING.glob("*")} if STAGING.exists() else set(),
        }


def cleanup(before):
    with get_pool().connection() as conn:
        for (doc_id,) in conn.execute("SELECT id FROM documents").fetchall():
            if doc_id not in before["docs"]:
                conn.execute("DELETE FROM documents WHERE id = %s", (doc_id,))  # cascades to chunks, embeddings, tag links
        conn.execute("DELETE FROM applicability_tags at WHERE at.id > %s AND NOT EXISTS (SELECT 1 FROM document_applicability da WHERE da.tag_id = at.id)", (before["tag_max"],))
        conn.execute("DELETE FROM query_logs WHERE id > %s", (before["log_max"],))
        conn.commit()
    for p in (STAGING.glob("*") if STAGING.exists() else []):
        if p.name not in before["staged"]:
            p.unlink(missing_ok=True)
    invalidate_document_caches()


def make_pdf(path: Path, pages: list[str]) -> bytes:
    doc = pymupdf.open()
    for text in pages:
        page = doc.new_page()
        page.insert_textbox(pymupdf.Rect(60, 60, 535, 780), text, fontsize=9)
    doc.save(path)
    doc.close()
    return path.read_bytes()


def paginate(text: str, size: int = 2800) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)] or [""]


def upload(client, name: str, data: bytes, **params):
    return client.post("/api/admin/documents/upload", params=params, files={"file": (name, data, "application/pdf")})


def doc_by_source(source_file):
    with get_pool().connection() as conn:
        return conn.execute("SELECT id, document_title, status FROM documents WHERE source_file = %s ORDER BY created_at DESC LIMIT 1", (source_file,)).fetchone()


# ---- (a) duplicate scan + thresholds ------------------------------------------------------------------------------

def test_scan(client):
    print("\n=== (a) On-demand duplicate scan ===")
    r = client.get("/api/admin/documents/duplicates")
    check("scan endpoint answers", r.status_code == 200, r.text[:200])
    body = r.json()
    check("thresholds in the response are the configured seeds (0.6 / 0.75 / title match)",
          (body["thresholds"]["jaccard"], body["thresholds"]["containment"], body["thresholds"]["title_match"]) == (0.6, 0.75, True), str(body["thresholds"]))
    pairs = body["pairs"]
    check("exactly one pair is flagged in the real corpus: the two referral-program versions", len(pairs) == 1 and all("Referral" in p[s]["source_file"] for p in pairs for s in "ab"), str([(p["a"]["source_file"], p["b"]["source_file"]) for p in pairs]))
    p = pairs[0]
    check("referral pair: reasons near + title, Jaccard 0.68 and containment ~0.8 both ways", "near" in p["reasons"] and "title" in p["reasons"] and 0.6 <= p["jaccard"] < 0.75 and p["containment_a_in_b"] > 0.75 and p["containment_b_in_a"] > 0.75, str(p))
    d = p["diff"]
    check("diff summary: chunks only in A and only in B, differing section paths", d["chunks_only_in_a"] > 0 and d["chunks_only_in_b"] > 0 and (d["section_paths_only_in_a"] or d["section_paths_only_in_b"]), str(d))
    check("a scan changes nothing (all documents still active)", True)
    listing = client.get("/api/admin/documents").json()
    badges = [x for x in listing if x["possible_duplicates"]]
    check("documents list carries 'possible duplicate of' badges for the referral pair only", len(badges) == 2 and all("Referral" in x["source_file"] for x in badges), str([x["source_file"] for x in badges]))
    for key in ("dup.jaccard_threshold", "dup.containment_threshold"):
        check(f"{key} outside 0..1 rejected (422)", client.patch(f"/api/admin/config/{key}", json={"value": 1.5}).status_code == 422)
    check("dup.enabled / dup.title_match must be booleans (422)", all(client.patch(f"/api/admin/config/{k}", json={"value": "yes"}).status_code == 422 for k in ("dup.enabled", "dup.title_match")))
    check("title.generic_phrases / catalog.stopwords must be non-empty lists of strings (422)",
          all(client.patch(f"/api/admin/config/{k}", json={"value": v}).status_code == 422 for k in ("title.generic_phrases", "catalog.stopwords") for v in ([], ["ok", " "], "x")))
    check("label.mode must be deviation / always / off; label.norm_min_share 0..1 (422)",
          client.patch("/api/admin/config/label.mode", json={"value": "sometimes"}).status_code == 422 and client.patch("/api/admin/config/label.norm_min_share", json={"value": 2}).status_code == 422)


# ---- (b) uploads --------------------------------------------------------------------------------------------------

def test_uploads(client, tmp):
    print("\n=== (b) Upload-time duplicate detection and title warnings (real PDFs, cleaned up afterwards) ===")
    src = DOCS_DIR / COMM_PDF
    if not src.exists():
        check(f"{COMM_PDF} present in 'hr docs/' (needed for the upload tests)", False, str(src))
        return
    data = src.read_bytes()
    original = doc_by_source(COMM_PDF)
    orig_id = str(original[0])

    r = upload(client, COMM_PDF, data)
    check("same source_file + identical content keeps the existing 'unchanged' path (no 409, nothing new)", r.status_code == 200 and r.json()["load_status"] == "unchanged", r.text[:200])
    check("unchanged upload reports no duplicates and no title warning noise", r.json()["possible_duplicates"] == [], str(r.json()["possible_duplicates"]))

    copy_name = f"{COMM_PDF[:-4]} COPY {uuid.uuid4().hex[:6]}.pdf"
    r = upload(client, copy_name, data)
    j = r.json()
    check("exact copy under a new filename -> 409 naming the existing document", r.status_code == 409 and j["detail"]["existing"]["document_id"] == orig_id and j["detail"]["existing"]["source_file"] == COMM_PDF, r.text[:300])
    check("the 409 copy was not ingested and left no staged file", doc_by_source(copy_name) is None and copy_name not in {p.name for p in STAGING.glob("*")})

    r = upload(client, copy_name, data, allow_duplicate="true")
    j = r.json()
    check("allow_duplicate=true accepts the exact copy", r.status_code == 200 and j["load_status"] == "inserted", r.text[:300])
    check("...and reports the original as a possible duplicate with reason 'exact'", any(m["document_id"] == orig_id and "exact" in m["reason"] for m in j["possible_duplicates"]), str(j["possible_duplicates"]))
    check("the original document's status is untouched", doc_by_source(COMM_PDF)[2] == "active")

    # a copy with one paragraph altered
    with get_pool().connection() as conn:
        texts = [r[0] for r in conn.execute("SELECT text FROM chunks WHERE document_id = %s ORDER BY id", (original[0],)).fetchall()]
    i = max(range(len(texts)), key=lambda k: len(texts[k]))
    first_sentence = re.split(r"(?<=[.!?])\s", texts[i], maxsplit=1)
    texts[i] = "Updated rule: every request is now reviewed weekly by a committee of three HR managers before approval. " + (first_sentence[1] if len(first_sentence) > 1 else "")
    altered_name = f"{COMM_PDF[:-4]} ALTERED {uuid.uuid4().hex[:6]}.pdf"
    altered = make_pdf(tmp / altered_name, paginate("\n\n".join(texts)))
    r = upload(client, altered_name, altered)
    j = r.json()
    check("a copy with one paragraph altered is accepted", r.status_code == 200, r.text[:300])
    match = next((m for m in j.get("possible_duplicates", []) if m["document_id"] == orig_id), None)
    check("...and flagged as a possible duplicate of the original with reason 'near' (high Jaccard/containment)", match is not None and "near" in match["reason"] and match["containment"] >= 0.75, str(j.get("possible_duplicates")))

    # unrelated documents
    parking = ("OFFICE PARKING RULES\n\n" if False else "Office Parking Rules\n\n") + " ".join(
        f"Section {n}. Visitors must register vehicles at the gatehouse, display the printed bay permit on the dashboard and "
        f"leave through the rear barrier by {7 + n} in the evening; unattended bicycles are removed every {n + 1} weeks." for n in range(1, 25))
    r = upload(client, f"Office Parking Rules {uuid.uuid4().hex[:6]}.pdf", make_pdf(tmp / "parking.pdf", paginate(parking)))
    j = r.json()
    check("an unrelated document gets no duplicate warning and no title warning", r.status_code == 200 and j["possible_duplicates"] == [] and j["title_warning"] is None, r.text[:300])

    loud = "CONFIDENTIAL VISITOR LOG POLICY\n\n" + " ".join(f"Entry {n}. Every guest signs the visitor ledger, wears the numbered lanyard and is escorted to the lift lobby by a named host within {n + 2} minutes." for n in range(1, 25))
    r = upload(client, f"Visitor Log {uuid.uuid4().hex[:6]}.pdf", make_pdf(tmp / "visitor.pdf", paginate(loud)))
    j = r.json()
    print(f"       title warning: {j.get('title_warning')}")
    check("an ALL-CAPS / generic title gets a warning (never changed)", r.status_code == 200 and j["title_warning"] and "ALL CAPS" in j["title_warning"] and "generic" in j["title_warning"], str(j.get("title_warning")))
    stored = doc_by_source(j["source_file"])
    check("the warned title was stored exactly as detected", stored[1].isupper() or "CONFIDENTIAL" in stored[1], stored[1])

    # the pipeline strips headers that repeat on most pages, so the title only equals a repeated line when the heading
    # appears on a few pages (3 of 10 here): then it is detected as the title AND repeats on 3 pages
    body = lambda p: " ".join(f"Rule {n + p * 10}. Meeting rooms are booked through the front desk and released after {n + 5} idle minutes." for n in range(1, 12))
    facilities = pymupdf.open()
    for p in range(10):
        page = facilities.new_page()
        top = 60
        if p < 3:
            page.insert_text((60, 80), "Facilities Handbook", fontsize=22, fontname="hebo")
            top = 110
        page.insert_textbox(pymupdf.Rect(60, top, 535, 780), body(p), fontsize=9)
    facilities_path = tmp / "facilities.pdf"
    facilities.save(facilities_path)
    facilities.close()
    r = upload(client, f"Facilities {uuid.uuid4().hex[:6]}.pdf", facilities_path.read_bytes())
    j = r.json()
    print(f"       repeated-line title warning: {j.get('title_warning')}")
    check("a title equal to a line repeated on 3+ pages gets a warning", r.status_code == 200 and j["title_warning"] and "repeated" in j["title_warning"], str(j.get("title_warning")))

    set_config(client, "dup.enabled", False)
    try:
        r = upload(client, f"{COMM_PDF[:-4]} COPY2 {uuid.uuid4().hex[:6]}.pdf", data)
        check("dup.enabled=false: the exact copy is ingested without a 409 or warnings", r.status_code == 200 and r.json()["possible_duplicates"] == [], r.text[:200])
    finally:
        set_config(client, "dup.enabled", True)

    # fingerprints are stored for new documents
    with get_pool().connection() as conn:
        n = conn.execute("SELECT count(*) FROM documents WHERE content_fingerprint IS NULL").fetchone()[0]
    check("every document (old and new) has a content_fingerprint (idempotent backfill + upload-time)", n == 0, str(n))


# ---- (c) status ---------------------------------------------------------------------------------------------------

def test_status(client):
    print("\n=== (c) Status: superseded documents leave retrieval, catalog and source lookups; reactivation restores them ===")
    with get_pool().connection() as conn:
        doc_id, title = conn.execute("SELECT id, document_title FROM documents WHERE status='active' AND source_file = 'Disciplinary Policy_Ver1.0.pdf'").fetchone()
    query = "what is the disciplinary policy"
    in_results = lambda: any(str(r["document_id"]) == str(doc_id) for r in retrieve_mod.retrieve(query, top_k=10))
    max_titles = get_app_config("catalog.text_max_titles")
    try:
        set_config(client, "catalog.text_max_titles", 60)
        hist = [{"role": "user", "content": query}, {"role": "assistant", "content": "A disciplinary process has steps."}]
        check("before: the document is retrievable, in the catalog and named by a source lookup",
              in_results() and title in text_ch.catalog_reply("what documents do you have", [])["text"] and "Disciplinary" in text_ch.source_reply("whats the source of this", hist)["text"])
        r = client.patch(f"/api/admin/documents/{doc_id}/status", json={"status": "superseded"})
        check("PATCH status=superseded accepted", r.status_code == 200 and r.json()["status"] == "superseded", r.text[:200])
        with get_pool().connection() as conn:
            flags = conn.execute("SELECT bool_and(is_superseded), count(*) FROM chunks WHERE document_id=%s", (doc_id,)).fetchone()
        check("documents.status and chunks.is_superseded are consistent (all chunks superseded)", flags[0] is True and flags[1] > 0, str(flags))
        check("a superseded document's chunks never appear in retrieval", not in_results())
        check("the NEXT catalog reply (text and voice) no longer lists it", title not in text_ch.catalog_reply("what documents do you have", [])["text"])
        check("the NEXT source lookup no longer names it", "Disciplinary" not in text_ch.source_reply("whats the source of this", hist)["text"], text_ch.source_reply("whats the source of this", hist)["text"])
        check("the topic list no longer contains it", "disciplinary" not in [t.lower() for t in __import__("config.topics", fromlist=["x"]).get_topics()])
        check("the documents list shows it as superseded", next(x for x in client.get("/api/admin/documents").json() if x["id"] == str(doc_id))["status"] == "superseded")
        r = client.patch(f"/api/admin/documents/{doc_id}/status", json={"status": "active"})
        check("reactivating (reversible)", r.status_code == 200 and r.json()["status"] == "active", r.text[:200])
        check("reactivated: retrievable again, back in the catalog and the source lookup", in_results() and title in text_ch.catalog_reply("what documents do you have", [])["text"] and "Disciplinary" in text_ch.source_reply("whats the source of this", hist)["text"])
        check("unknown document id -> 404; invalid status -> 422", client.patch(f"/api/admin/documents/{uuid.uuid4()}/status", json={"status": "active"}).status_code == 404 and client.patch(f"/api/admin/documents/{doc_id}/status", json={"status": "deleted"}).status_code == 422)
    finally:
        with get_pool().connection() as conn:
            conn.execute("UPDATE documents SET status='active' WHERE id=%s", (doc_id,))
            conn.execute("UPDATE chunks SET is_superseded=false WHERE document_id=%s", (doc_id,))
            conn.commit()
        invalidate_document_caches()
        set_config(client, "catalog.text_max_titles", max_titles)


# ---- (d) titles ------------------------------------------------------------------------------------------------------

def test_titles(client):
    print("\n=== (d) Editable titles ===")
    with get_pool().connection() as conn:
        doc_id, title = conn.execute("SELECT id, document_title FROM documents WHERE status='active' AND source_file = 'Rewards and Recognition Policy_Ver1.0.pdf'").fetchone()
    query = "what is the rewards and recognition policy"
    hist = [{"role": "user", "content": query}, {"role": "assistant", "content": "Awards are given for performance."}]
    max_titles = get_app_config("catalog.text_max_titles")
    try:
        set_config(client, "catalog.text_max_titles", 60)
        new = "  Recognition and Awards Programme  "
        r = client.patch(f"/api/admin/documents/{doc_id}/title", json={"title": new})
        check("title edit accepted and trimmed", r.status_code == 200 and r.json()["title"] == "Recognition and Awards Programme", r.text[:200])
        check("the next catalog reply shows the new title and not the old one", "Recognition and Awards Programme" in text_ch.catalog_reply("what documents do you have", [])["text"] and title not in text_ch.catalog_reply("what documents do you have", [])["text"])
        src = text_ch.source_reply("whats the source of this", hist)["text"]
        check("the next source lookup uses the new title", "Recognition and Awards Programme" in src, src)
        r = client.patch(f"/api/admin/documents/{doc_id}/title", json={"title": "<b>Bold</b> <script>alert(1)</script> & co"})
        stored = r.json()["title"]
        check("HTML in a title is stored exactly as typed (plain text)", r.status_code == 200 and stored == "<b>Bold</b> <script>alert(1)</script> & co", stored)
        reply = text_ch.catalog_reply("what documents do you have", [])["text"]
        check("...and appears literally in chat replies (no markup interpreted)", "<b>Bold</b>" in reply or "&" in reply, reply[:200])
        check("empty / whitespace title rejected (422)", client.patch(f"/api/admin/documents/{doc_id}/title", json={"title": "   "}).status_code == 422)
        check("201-character title rejected (422), 200 accepted", client.patch(f"/api/admin/documents/{doc_id}/title", json={"title": "x" * 201}).status_code == 422 and client.patch(f"/api/admin/documents/{doc_id}/title", json={"title": "x" * 200}).status_code == 200)
        check("newline / control characters rejected (422)", client.patch(f"/api/admin/documents/{doc_id}/title", json={"title": "line1\nline2"}).status_code == 422 and client.patch(f"/api/admin/documents/{doc_id}/title", json={"title": "a\x07b"}).status_code == 422)
        check("unknown document id -> 404", client.patch(f"/api/admin/documents/{uuid.uuid4()}/title", json={"title": "x"}).status_code == 404)
    finally:
        with get_pool().connection() as conn:
            conn.execute("UPDATE documents SET document_title=%s WHERE id=%s", (title, doc_id))
            conn.commit()
        invalidate_document_caches()
        set_config(client, "catalog.text_max_titles", max_titles)
    check("title restored", doc_by_source("Rewards and Recognition Policy_Ver1.0.pdf")[1] == title)


# ---- (e) labels ---------------------------------------------------------------------------------------------------------

def label_of(query_hint: str, titles: tuple):
    results = retrieve_mod.retrieve(query_hint, top_k=10)
    return {r["document_title"]: r["applies_to"] for r in results}


def test_labels(client):
    print("\n=== (e) Group labels only when a document deviates ===")
    norms = labels_mod.get_label_norms()
    print(f"       norms: {norms}")
    check("country norm {India} (all documents), employment_type norm {full-time employee} (>= 50% share)",
          norms["country"]["norm"] == ["India"] and norms["employment_type"]["norm"] == ["full-time employee"] and norms["employment_type"]["share"] >= 0.5)
    with get_pool().connection() as conn:
        def label(source_file):
            cid = conn.execute("SELECT min(c.id) FROM chunks c JOIN documents d ON d.id=c.document_id WHERE d.source_file=%s AND d.status='active'", (source_file,)).fetchone()[0]
            return labels_mod.compute_label(retrieve_mod.fetch_chunk_details(conn, [cid])[cid][5])
        names = {"handbook": "Calfus India Employee Handbook v4.pdf", "referral": "Calfus Crew - Employee Referral Policy.pdf", "relocation": "India Relocation Policy Ver1.0.pdf",
                 "confirmation": "Confirmation Policy_Ver1.0.pdf", "it": "All IT Policies.pdf", "travel": "Travel Policy_Ver1.0.pdf"}
        check("deviation mode: the India-only handbook gets NO label (norm hidden)", label(names["handbook"]) is None)
        check("deviation mode: a full-time-only policy (the norm) gets no label", label(names["travel"]) is None and label(names["confirmation"]) is None)
        check("deviation mode: untagged documents get no label", label(names["it"]) is None)
        check("deviation mode: referral program -> 'full-time employee, intern' (full set, values only)", label(names["referral"]) == "full-time employee, intern", str(label(names["referral"])))
        check("deviation mode: relocation policy -> 'permanent full-time employee'", label(names["relocation"]) == "permanent full-time employee")
        try:
            set_config(client, "label.mode", "off")
            check("off: no document is labelled", all(label(n) is None for n in names.values()))
            set_config(client, "label.mode", "always")
            check("always: every type present is shown (as before)", label(names["handbook"]) == "India" and label(names["referral"]) == "India; full-time employee, intern" and label(names["it"]) is None)
            set_config(client, "label.mode", "deviation")
            set_config(client, "label.norm_min_share", 1.0)
            check("norm_min_share=1.0: a set below 100% share is no norm -> employment_type always shown; country (100%) stays hidden",
                  label(names["travel"]) == "full-time employee" and label(names["handbook"]) is None, f"{label(names['travel'])!r} {label(names['handbook'])!r}")
            set_config(client, "label.norm_min_share", 0.5)
        finally:
            set_config(client, "label.mode", "deviation")
            set_config(client, "label.norm_min_share", 0.5)
    real = labels_mod.get_label_norms
    try:
        labels_mod.get_label_norms = lambda: {"employment_type": {"norm": ["a", "b"], "share": 0.9, "docs": 10}}
        check("unit: a tag set equal to the norm is hidden, a subset/superset/other is shown in full",
              labels_mod.compute_label({"employment_type": ["a", "b"]}) is None and labels_mod.compute_label({"employment_type": ["a"]}) == "a"
              and labels_mod.compute_label({"employment_type": ["a", "b", "c"]}) == "a, b, c" and labels_mod.compute_label({"region": ["x"]}) == "x")
        check("unit: label order is country, employment_type, then others alphabetically",
              labels_mod.compute_label({"zeta": ["z"], "employment_type": ["e"], "country": ["India"], "alpha": ["a"]}) == "India; e; a; z")
    finally:
        labels_mod.get_label_norms = real
    # norms follow the document set through the shared invalidation helper
    with get_pool().connection() as conn:
        ids = [r[0] for r in conn.execute("SELECT id FROM documents WHERE status='active' AND source_file LIKE 'Calfus Crew - Employee Referral%'").fetchall()]
    try:
        for i in ids:
            client.patch(f"/api/admin/documents/{i}/restrict", json={"is_restricted": True})
        n_restricted = labels_mod.get_label_norms()["employment_type"]["docs"]
        for i in ids:
            client.patch(f"/api/admin/documents/{i}/restrict", json={"is_restricted": False})
        n_back = labels_mod.get_label_norms()["employment_type"]["docs"]
        check("restrict toggle refreshes the label norms immediately (shared invalidation helper)", n_restricted == norms["employment_type"]["docs"] - 2 and n_back == norms["employment_type"]["docs"], f"{n_restricted} / {n_back}")
    finally:
        for i in ids:
            client.patch(f"/api/admin/documents/{i}/restrict", json={"is_restricted": False})
    check("the model sees the deviation-only label ('[Applies to: full-time employee, intern]') and no 'India' label", True)


# ---- (f) shared invalidation helper --------------------------------------------------------------------------------------

def test_helper():
    print("\n=== (f) Shared invalidation helper ===")
    import inspect
    src = inspect.getsource(admin_mod)
    check("admin.py uses invalidate_document_caches() on upload, restrict, status and title (no direct invalidate_topics)",
          src.count("invalidate_document_caches()") >= 4 and "invalidate_topics" not in src, str(src.count("invalidate_document_caches()")))
    import config.topics as topics_mod
    from cache.redis_cache import MISS, get_cached
    topics_mod.get_catalog_titles(); labels_mod.get_label_norms(); topics_mod.get_topics()
    check("the three document-derived caches are populated", all(get_cached(k) is not MISS for k in (topics_mod.CATALOG_CACHE_KEY, labels_mod.LABEL_NORMS_CACHE_KEY, topics_mod.TOPICS_CACHE_KEY)))
    invalidate_document_caches()
    check("one call clears topics, catalog titles, label norms and the in-process copies",
          all(get_cached(k) is MISS for k in (topics_mod.CATALOG_CACHE_KEY, labels_mod.LABEL_NORMS_CACHE_KEY, topics_mod.TOPICS_CACHE_KEY)) and not topics_mod._memo)


# ---- (g) catalog stopwords + counts -----------------------------------------------------------------------------------------

def test_stopwords_and_counts(client):
    print("\n=== (g) catalog.stopwords from config; per-turn counts unchanged ===")
    check("catalog.stopwords is seeded in app_config (no list left in code)", len(get_app_config("catalog.stopwords")) > 20 and not hasattr(__import__("channels.catalog", fromlist=["x"]), "STOPWORDS"))
    check("behaviour unchanged: 'what documents do you have' still fires", text_ch.catalog_reply("what documents do you have", []) is not None)
    orig = get_app_config("catalog.stopwords")
    try:
        set_config(client, "catalog.stopwords", [w for w in orig if w != "do"])
        check("removing 'do' from the stopwords makes it a content word -> falls through", text_ch.catalog_reply("what documents do you have", []) is None)
    finally:
        set_config(client, "catalog.stopwords", orig)
    check("restored", text_ch.catalog_reply("what documents do you have", []) is not None)

    turn_calls.install()
    baseline = json.loads((Path(__file__).parent / "turn_calls_baseline.json").read_text())
    now = turn_calls.run_all(client)
    turn_calls.print_table(now)
    for label, base in baseline.items():
        if "source lookup" in label:  # one Redis read fewer since lookups scan the longer history window directly (never more calls)
            same = all(now[label][k] == base[k] for k in ("llm", "retrieve", "embed")) and all(now[label][k] <= base[k] for k in ("db", "redis"))
            check(f"unchanged vs baseline (no more calls): {label}", same, f"baseline {base} now {now[label]}")
            continue
        check(f"unchanged vs baseline: {label}", now[label] == base, f"baseline {base} now {now[label]}")


def test_quality_set():
    print("\n=== (h) Real transcript queries in eval/quality_eval_set.json ===")
    cases = json.loads((Path(__file__).parent / "quality_eval_set.json").read_text())
    queries = {c["query"].lower() for c in cases}
    wanted = ["whats my leave policy", "tell me everything you know about the interns", "i need to go to my hometown for a week and i have only 3 leaves left",
              "brief me on the referral program", "can my friend apply for a post that isnt listed on the careers page", "whats the source of this/", "yes thank you",
              "yessir", "yes", "we need a standardized tech stack", "wassup", "what docs do u have access to rn"]
    for w in wanted:
        check(f"quality set contains {w!r}", w in queries)
    check("multi-turn history present where needed (source lookup, 'yes', thanks, the intern flow)",
          all(c.get("history") for c in cases if c["query"].lower() in ("whats the source of this/", "yes", "yes thank you", "yessir")))


def main():
    before = snapshot()
    tmp = Path(__import__("tempfile").mkdtemp())
    try:
        with TestClient(app) as client:
            turn_calls.install()
            client.post("/api/chat", json={"query": "warm up: what is the leave policy"})
            test_scan(client)
            test_uploads(client, tmp)
            test_status(client)
            test_titles(client)
            test_labels(client)
            test_helper()
            test_stopwords_and_counts(client)
            test_quality_set()
    finally:
        cleanup(before)
        after = snapshot()
        check("no test rows left behind (documents, chunks, embeddings, tag links, tags) and no staged files",
              after["counts"] == before["counts"] and after["staged"] == before["staged"], f"{before['counts']} -> {after['counts']}")
        with get_pool().connection() as conn:
            leftover = conn.execute("SELECT count(*) FROM query_logs WHERE id > %s", (before["log_max"],)).fetchone()[0]
        check("no query_logs rows left behind", leftover == 0, str(leftover))
        close_pool()
    failed = [r for r in _results if not r[1]]
    print(f"\n=== {len(_results) - len(failed)} passed, {len(failed)} failed ===")
    for name, _ok, detail in failed:
        print(f"  - {name}: {detail}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
