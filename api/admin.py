"""Admin endpoints: document upload/management and usage monitoring.

NO AUTH - see the security note in api/main.py.
"""
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Literal
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from psycopg.rows import dict_row
from pydantic import BaseModel

from access_control.extract_applicability import extract_and_store_for_document
from generation.generate import get_client, get_generation_model
from ingestion.pipeline import process_document
from vectorstore.db import get_pool
from vectorstore.embedder import ensure_embedding_model_registered
from vectorstore.load import load_document
from vectorstore.retrieve import _get_embedder

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

STAGING_DIR = Path(os.environ.get("STAGING_DIR", "staging"))
SUPPORTED_EXTENSIONS = {".pdf", ".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Ingestion + embedding is CPU/GPU heavy and writes to staging by
# filename, so uploads are processed one at a time.
_upload_lock = threading.Lock()


# --------------------------------------------------------------------------
# Document upload
# --------------------------------------------------------------------------

def _document_tags(conn, document_id) -> list[dict]:
    rows = conn.execute(
        """
        SELECT at.tag_type, at.tag_value
        FROM document_applicability da JOIN applicability_tags at ON at.id = da.tag_id
        WHERE da.document_id = %s ORDER BY at.tag_type, at.tag_value
        """,
        (document_id,),
    ).fetchall()
    return [{"tag_type": r[0], "tag_value": r[1]} for r in rows]


@router.post("/documents/upload")
def upload_document(file: UploadFile = File(...)):
    # Basename only: the name becomes documents.source_file, which is the
    # key for supersede-on-reupload, and must not carry path components.
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type; expected one of {sorted(SUPPORTED_EXTENSIONS)}")

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    staged = STAGING_DIR / filename

    with _upload_lock:
        written = 0
        with staged.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > MAX_UPLOAD_BYTES:
                    out.close()
                    staged.unlink(missing_ok=True)
                    raise HTTPException(413, f"File exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit")
                out.write(chunk)

        # extraction -> normalization -> structure -> chunking -> QA flags
        try:
            doc_json, stats = process_document(staged)
        except Exception as exc:
            logger.exception("ingestion failed for %s", filename)
            raise HTTPException(422, f"Ingestion failed: {exc}")
        if doc_json is None:
            raise HTTPException(422, f"Ingestion failed: {stats.get('error', 'unknown error')}")
        if not doc_json["chunks"]:
            raise HTTPException(422, {"detail": "No chunks could be extracted from this file",
                                       "extraction_errors": [str(e) for e in stats["errors"]]})

        # embedding -> load (supersede logic lives in load_document)
        pool = get_pool()
        embedder = _get_embedder()
        with pool.connection() as conn:
            model_id = ensure_embedding_model_registered(conn, embedder)
            conn.commit()
        try:
            loaded = load_document(pool, doc_json, embedder, model_id)
        except Exception as exc:
            logger.exception("load failed for %s", filename)
            raise HTTPException(500, f"Loading into the vector store failed: {exc}")

        # applicability extraction - skipped for an unchanged re-upload
        # (the active document, and its tags, are untouched).
        extraction_errors = [str(e) for e in stats["errors"]]
        ambiguous = False
        if loaded["status"] != "unchanged":
            try:
                with pool.connection() as conn:
                    outcome = extract_and_store_for_document(
                        conn, get_client(), get_generation_model(),
                        loaded["document_id"], doc_json["document_title"],
                    )
                ambiguous = outcome["ambiguous"]
            except Exception as exc:
                logger.exception("applicability extraction failed for %s", filename)
                extraction_errors.append(f"applicability extraction failed: {exc}")

        with pool.connection() as conn:
            tags = _document_tags(conn, loaded["document_id"])

    return {
        "document_id": loaded["document_id"],
        "source_file": filename,
        "load_status": loaded["status"],  # inserted | superseded | unchanged
        "superseded_document_id": loaded["superseded_document_id"],
        "chunks_created": loaded["chunks"],
        "structure_detection_status": stats["structure_status"],
        "review_reason": doc_json.get("review_reason"),
        "applicability_tags": tags,
        "applicability_ambiguous": ambiguous,
        "qa_flags": stats["flag_counts"],
        "extraction_errors": extraction_errors,
    }


# --------------------------------------------------------------------------
# Document management
# --------------------------------------------------------------------------

_DOCUMENT_SELECT = """
    SELECT d.id, d.document_title AS title, d.source_file, d.status, d.is_restricted,
           d.created_at AS uploaded_at,
           (SELECT count(*) FROM chunks c WHERE c.document_id = d.id) AS chunk_count,
           COALESCE((
               SELECT json_agg(json_build_object('tag_type', at.tag_type, 'tag_value', at.tag_value)
                               ORDER BY at.tag_type, at.tag_value)
               FROM document_applicability da JOIN applicability_tags at ON at.id = da.tag_id
               WHERE da.document_id = d.id
           ), '[]'::json) AS applicability_tags
    FROM documents d
"""


@router.get("/documents")
def list_documents():
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(_DOCUMENT_SELECT + " ORDER BY d.created_at DESC, d.id")
        return cur.fetchall()


class RestrictRequest(BaseModel):
    is_restricted: bool


@router.patch("/documents/{document_id}/restrict")
def restrict_document(document_id: UUID, body: RestrictRequest):
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "UPDATE documents SET is_restricted = %s, updated_at = now() WHERE id = %s",
            (body.is_restricted, document_id),
        )
        if cur.rowcount == 0:
            raise HTTPException(404, "Document not found")
        cur.execute(_DOCUMENT_SELECT + " WHERE d.id = %s", (document_id,))
        row = cur.fetchone()
        conn.commit()
        return row


# --------------------------------------------------------------------------
# Usage / monitoring
# --------------------------------------------------------------------------

@router.get("/usage/summary")
def usage_summary(
    days: int = Query(7, ge=0, description="Window in days; 0 = all time"),
    source: Literal["text", "voice"] | None = Query(None, description="Filter by request origin"),
):
    clauses, params = [], []
    if days:
        clauses.append("created_at >= now() - make_interval(days => %s)")
        params.append(days)
    if source:
        clauses.append("source = %s")
        params.append(source)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            f"""
            SELECT count(*) AS total,
                   count(*) FILTER (WHERE response_type = 'answer') AS answer,
                   count(*) FILTER (WHERE response_type = 'clarification') AS clarification,
                   count(*) FILTER (WHERE response_type = 'refusal_shortcut') AS refusal_shortcut,
                   avg(retrieval_time_ms) AS avg_retrieval_time_ms,
                   avg(ttft_ms) AS avg_ttft_ms,
                   avg(total_generation_time_ms) AS avg_total_generation_time_ms
            FROM query_logs {where}
            """,
            params,
        )
        r = cur.fetchone()

    total = r["total"]
    pct = lambda n: round(100.0 * n / total, 2) if total else 0.0
    rnd = lambda v: round(float(v), 2) if v is not None else None
    return {
        "window_days": days or None,
        "source": source,
        "total_queries": total,
        "by_response_type": {
            "answer": r["answer"],
            "clarification": r["clarification"],
            "refusal_shortcut": r["refusal_shortcut"],
        },
        "avg_retrieval_time_ms": rnd(r["avg_retrieval_time_ms"]),
        "avg_ttft_ms": rnd(r["avg_ttft_ms"]),
        "avg_total_generation_time_ms": rnd(r["avg_total_generation_time_ms"]),
        "shortcut_rate": pct(r["refusal_shortcut"]),
        "clarification_rate": pct(r["clarification"]),
    }


@router.get("/usage/recent")
def usage_recent(limit: int = Query(50, ge=1, le=500)):
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM query_logs ORDER BY created_at DESC, id DESC LIMIT %s", (limit,))
        return cur.fetchall()
