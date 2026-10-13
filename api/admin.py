"""Admin endpoints: document upload/management and usage monitoring.

NO AUTH - see the security note in api/main.py.
"""
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any, Literal
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from psycopg.rows import dict_row
from pydantic import BaseModel

from access_control.extract_applicability import extract_and_store_for_document
from config.runtime import set_app_config, set_provider_setting
from channels.common import clear_config_memo
from config.prompts import SLOTS, refresh_prompt_cache
from config.topics import invalidate_topics
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

        invalidate_topics()  # the active document set may have changed (inserted / superseded)

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
    invalidate_topics()  # restricted documents are not advertised as topics
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
                   count(*) FILTER (WHERE response_type = 'smalltalk') AS smalltalk,
                   count(*) FILTER (WHERE response_type = 'source_lookup') AS source_lookup,
                   count(*) FILTER (WHERE contextualized) AS contextualized,
                   count(*) FILTER (WHERE advisory) AS advisory,
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
            "smalltalk": r["smalltalk"],
            "source_lookup": r["source_lookup"],
        },
        "avg_retrieval_time_ms": rnd(r["avg_retrieval_time_ms"]),
        "avg_ttft_ms": rnd(r["avg_ttft_ms"]),
        "avg_total_generation_time_ms": rnd(r["avg_total_generation_time_ms"]),
        "shortcut_rate": pct(r["refusal_shortcut"]),
        "clarification_rate": pct(r["clarification"]),
        "contextualized_rate": pct(r["contextualized"]),
        "advisory_rate": pct(r["advisory"]),
    }


@router.get("/usage/recent")
def usage_recent(limit: int = Query(50, ge=1, le=500)):
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT * FROM query_logs ORDER BY created_at DESC, id DESC LIMIT %s", (limit,))
        return cur.fetchall()


# --------------------------------------------------------------------------
# Runtime config (app_config + provider_configs; secrets are env-only)
# --------------------------------------------------------------------------

def _pos_int(v):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise ValueError("must be a positive integer")
    return v


def _unit_interval(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
        raise ValueError("must be a number between 0 and 1")
    return float(v)


def _non_empty_str(v):
    if not isinstance(v, str) or not v.strip():
        raise ValueError("must be a non-empty string")
    return v.strip()


def _str_list(v):
    if not isinstance(v, list) or not v or any(not isinstance(x, str) or not x.strip() for x in v):
        raise ValueError("must be a non-empty list of non-empty strings")
    return [x.strip().lower() for x in v]


def _expansions(v):
    if not isinstance(v, list) or any(
        not isinstance(t, dict) or not isinstance(t.get("topic"), str) or not t["topic"].strip()
        or not isinstance(t.get("triggers"), list) or not t["triggers"]
        or not isinstance(t.get("queries"), list) or not t["queries"]
        or any(not isinstance(x, str) or not x.strip() for x in t["triggers"] + t["queries"])
        for t in v
    ):
        raise ValueError('must be a list of {"topic": str, "triggers": [str, ...], "queries": [str, ...]} objects')
    return v


_SMALLTALK_CATEGORIES = ("greeting", "how_are_you", "thanks", "goodbye", "capabilities", "ack", "are_you_there")


def _smalltalk_phrases(v):
    if not isinstance(v, dict) or not v or any(k not in _SMALLTALK_CATEGORIES for k in v):
        raise ValueError(f"must be an object with keys from {list(_SMALLTALK_CATEGORIES)}")
    out = {}
    for category, phrases in v.items():
        if not isinstance(phrases, list) or not phrases or any(not isinstance(p, str) or not p.strip() for p in phrases):
            raise ValueError(f"{category!r} must be a non-empty list of non-empty strings")
        out[category] = [p.strip().lower() for p in phrases]
    return out


_SOURCE_FIELDS = {"one": {"title"}, "one_with_section": {"title", "section"}, "many": {"titles"}, "none": set(), "not_found": set()}


def _source_templates(v):
    import string

    if not isinstance(v, dict) or any(k not in _SOURCE_FIELDS for k in v) or any(k not in v for k in ("one", "many", "none", "not_found")):
        raise ValueError(f"must be an object with keys one, many, none, not_found (and optionally one_with_section)")
    for key, template in v.items():
        if not isinstance(template, str) or not template.strip():
            raise ValueError(f"{key!r} must be a non-empty string")
        fields = {f for _, f, _, _ in string.Formatter().parse(template) if f is not None}
        if not fields <= _SOURCE_FIELDS[key]:
            raise ValueError(f"{key!r} may only use the placeholders {sorted(_SOURCE_FIELDS[key])}, found {sorted(fields)}")
    return {k: t.strip() for k, t in v.items()}


def _bool(v):
    if not isinstance(v, bool):
        raise ValueError("must be true or false")
    return v


def _non_neg_number(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
        raise ValueError("must be a number >= 0")
    return float(v)


# key -> validator. app_config keys have no table prefix; provider_configs
# keys are "<category>.<field>".
_APP_CONFIG_VALIDATORS = {
    "retrieval.top_k": _pos_int,
    "retrieval.candidate_n": _pos_int,
    "retrieval.rrf_k": _pos_int,
    "conversation.history_messages": _pos_int,
    "conversation.history_max_chars": _pos_int,
    "retrieval.context.max_words_followup": _pos_int,
    "retrieval.context.prior_user_turns": _pos_int,
    "retrieval.context.followup_markers": _str_list,
    "retrieval.advisory.markers": _str_list,
    "retrieval.advisory.max_chunks_per_doc": _pos_int,
    "smalltalk.phrases.text": _smalltalk_phrases,
    "smalltalk.phrases.voice": _smalltalk_phrases,
    "smalltalk.affirmations": _str_list,
    "smalltalk.offer_markers": _str_list,
    "source.markers": _str_list,
    "source.templates.text": _source_templates,
    "source.templates.voice": _source_templates,
    "retrieval.expansions": _expansions,
    "retrieval.expansion.enabled": _bool,
    "retrieval.expansion.weight": _non_neg_number,
    "retrieval.expansion.candidate_n": _pos_int,
    "retrieval.top_k.advisory.text": _pos_int,
    "retrieval.top_k.advisory.voice": _pos_int,
}
# Per-channel copies of the tunables (config/channel_settings.py): "<key>.<channel>".
for _channel in ("text", "voice"):
    _APP_CONFIG_VALIDATORS.update({
        f"retrieval.top_k.{_channel}": _pos_int,
        f"retrieval.candidate_n.{_channel}": _pos_int,
        f"retrieval.rrf_k.{_channel}": _pos_int,
        f"conversation.history_messages.{_channel}": _pos_int,
        f"conversation.history_max_chars.{_channel}": _pos_int,
        f"retrieval.context.max_words_followup.{_channel}": _pos_int,
        f"retrieval.context.prior_user_turns.{_channel}": _pos_int,
        f"retrieval.advisory.max_chunks_per_doc.{_channel}": _pos_int,
        f"retrieval.expansion.enabled.{_channel}": _bool,
        f"generation.model.{_channel}": _non_empty_str,
        f"generation.confidence_threshold.{_channel}": _unit_interval,
    })
_PROVIDER_VALIDATORS = {
    "generation.model": _non_empty_str,
    "generation.confidence_threshold": _unit_interval,
    "stt.model": _non_empty_str,
    "tts.voice": _non_empty_str,
    "tts.greeting": _non_empty_str,
}


@router.get("/config")
def get_config():
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT key, value, description, updated_at FROM app_config ORDER BY key")
        app_config = cur.fetchall()
        cur.execute(
            "SELECT category, provider_name, config, updated_at FROM provider_configs "
            "WHERE is_active AND category = ANY(%s) ORDER BY category",
            (["generation", "stt", "tts"],),
        )
        providers = cur.fetchall()
    return {
        "app_config (table: app_config)": app_config,
        "provider_configs (table: provider_configs, active rows)": providers,
    }


class ConfigUpdate(BaseModel):
    value: Any


@router.patch("/config/{key}")
def update_config(key: str, body: ConfigUpdate):
    """key is an app_config key (e.g. "retrieval.top_k") or a
    provider_configs setting as "<category>.<field>" (e.g.
    "generation.confidence_threshold"). Writes the DB row, then the cache."""
    validator = _PROVIDER_VALIDATORS.get(key) or _APP_CONFIG_VALIDATORS.get(key)
    if validator is None:
        raise HTTPException(404, {"detail": f"Unknown config key {key!r}",
                                  "valid_keys": sorted([*_APP_CONFIG_VALIDATORS, *_PROVIDER_VALIDATORS])})
    try:
        value = validator(body.value)
    except ValueError as exc:
        raise HTTPException(422, f"Invalid value for {key}: {exc} (got {body.value!r})")

    if key in _PROVIDER_VALIDATORS:
        category, field = key.split(".", 1)
        updated = set_provider_setting(category, field, value)
        if updated is None:
            raise HTTPException(404, f"No active provider_configs row for category {category!r}")
        return {"key": key, "value": value, "source": "provider_configs", "config": updated}
    if not set_app_config(key, value):
        raise HTTPException(404, f"app_config row {key!r} not found (run vectorstore.setup_db)")
    clear_config_memo()  # channels/common.py keeps a 30s in-process copy of a few config keys
    return {"key": key, "value": value, "source": "app_config"}


# --------------------------------------------------------------------------
# Prompts (prompt_templates; see config/prompts.py). Rows are immutable: an
# edit creates a new version and activates it, rollback re-activates an old one.
# Prompts must never contain secrets or credentials.
# --------------------------------------------------------------------------

MAX_PROMPT_CHARS = 6000
MAX_PROMPT_NOTE_CHARS = 200


@router.get("/prompts")
def list_prompts():
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, name, channel, version, content, is_active, created_at, created_by "
            "FROM prompt_templates WHERE channel IN ('text', 'voice') ORDER BY name, channel, version DESC"
        )
        rows = cur.fetchall()
    groups: dict[tuple[str, str], dict] = {}
    for r in rows:
        g = groups.setdefault((r["name"], r["channel"]),
                              {"name": r["name"], "channel": r["channel"], "active_version": None, "versions": []})
        if r["is_active"]:
            g["active_version"] = r["version"]
        g["versions"].append(r)
    return list(groups.values())


class PromptCreate(BaseModel):
    name: str
    channel: Literal["text", "voice"]
    content: str
    note: str | None = None


def _slot_lock(conn, name: str, channel: str) -> None:
    conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (f"prompt:{name}:{channel}",))


@router.post("/prompts")
def create_prompt_version(body: PromptCreate):
    """Creates version max+1 for an existing (name, channel) slot and activates
    it, atomically deactivating the previous active row."""
    if (body.name, body.channel) not in SLOTS:
        raise HTTPException(404, {"detail": f"Unknown prompt slot ({body.name!r}, {body.channel!r})",
                                  "valid_slots": [{"name": n, "channel": c} for n, c in SLOTS]})
    content = body.content.strip()
    if not content:
        raise HTTPException(422, "content must be non-empty")
    if len(content) > MAX_PROMPT_CHARS:
        raise HTTPException(422, f"content is {len(content)} chars; the maximum is {MAX_PROMPT_CHARS}")
    note = (body.note or "").strip() or None
    if note and len(note) > MAX_PROMPT_NOTE_CHARS:
        raise HTTPException(422, f"note is {len(note)} chars; the maximum is {MAX_PROMPT_NOTE_CHARS}")

    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        _slot_lock(conn, body.name, body.channel)
        cur.execute("UPDATE prompt_templates SET is_active = false WHERE name = %s AND channel = %s AND is_active",
                    (body.name, body.channel))
        cur.execute(
            "INSERT INTO prompt_templates (name, channel, version, content, is_active, created_by) "
            "SELECT %s, %s, COALESCE(MAX(version), 0) + 1, %s, true, %s FROM prompt_templates "
            "WHERE name = %s AND channel = %s "
            "RETURNING id, name, channel, version, is_active, created_at, created_by",
            (body.name, body.channel, content, note, body.name, body.channel),
        )
        row = cur.fetchone()
        conn.commit()
    refresh_prompt_cache()
    return row


@router.post("/prompts/{prompt_id}/activate")
def activate_prompt_version(prompt_id: int):
    """Makes an existing version the active one for its slot (rollback or re-activate)."""
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT name, channel FROM prompt_templates WHERE id = %s", (prompt_id,))
        slot = cur.fetchone()
        if slot is None:
            raise HTTPException(404, "Prompt version not found")
        _slot_lock(conn, slot["name"], slot["channel"])
        cur.execute("UPDATE prompt_templates SET is_active = false WHERE name = %s AND channel = %s AND is_active AND id <> %s",
                    (slot["name"], slot["channel"], prompt_id))
        cur.execute("UPDATE prompt_templates SET is_active = true WHERE id = %s "
                    "RETURNING id, name, channel, version, is_active, created_at, created_by", (prompt_id,))
        row = cur.fetchone()
        conn.commit()
    refresh_prompt_cache()
    return row
