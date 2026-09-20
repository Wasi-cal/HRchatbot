"""Per-document orchestration: extract -> normalize -> structure -> chunk -> QA."""
import re
from pathlib import Path

from .chunker import chunk_document
from .extract_docx import extract_docx
from .extract_pdf import extract_pdf
from .qaflags import apply_qa_flags
from .structure import (
    build_sections, detect_applicability, detect_document_title,
    detect_effective_date, evaluate_structure_status,
)
from .utils import find_repeated_lines, normalize_text, strip_headers_footers

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_]+")


def safe_id_prefix(stem: str) -> str:
    return _SAFE_ID_RE.sub("_", stem).strip("_")[:60] or "doc"


def process_document(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        blocks, page_raw_texts, errors = extract_pdf(path)
        repeated_lines = find_repeated_lines(page_raw_texts) if page_raw_texts else set()
    elif ext == ".docx":
        blocks, errors = extract_docx(path)
        repeated_lines = set()
    else:
        return None, {"error": f"unsupported file type: {ext}"}

    for b in blocks:
        text = b.get("text", "")
        if b["kind"] != "table":
            text = strip_headers_footers(text, repeated_lines)
        b["text"] = normalize_text(text, b["source_type"])

    blocks = [b for b in blocks if b.get("text", "").strip() or b["kind"] == "table"]

    document_title = detect_document_title(blocks, path.stem)
    effective_date = detect_effective_date(blocks)
    applicability = detect_applicability(blocks)

    sections = build_sections(blocks, document_title)
    status, reason = evaluate_structure_status(blocks, sections)

    id_prefix = safe_id_prefix(path.stem)
    chunks = chunk_document(sections, document_title, id_prefix)
    flag_counts = apply_qa_flags(chunks)

    for c in chunks:
        c.pop("_row_mismatch", None)

    doc_json = {
        "source_file": path.name,
        "document_title": document_title,
        "effective_date": effective_date,
        "applicability": applicability,
        "structure_detection_status": status,
        "chunks": chunks,
    }
    if status == "needs_manual_review":
        doc_json["review_reason"] = reason

    digital_pages = len({b["page"] for b in blocks if b["source_type"] == "digital_text" and b.get("page") is not None})
    ocr_pages = len({b["page"] for b in blocks if b["source_type"] == "ocr" and b.get("page") is not None})

    stats = {
        "errors": errors,
        "chunk_count": len(chunks),
        "flag_counts": dict(flag_counts),
        "structure_status": status,
        "digital_pages": digital_pages,
        "ocr_pages": ocr_pages,
    }
    return doc_json, stats
