"""Build a heading-hierarchy tree from ordered blocks and group content into
sections (the unit chunking operates on)."""
import re

from .ocr_utils import OCR_CONFIDENCE_THRESHOLD
from .utils import BARE_NUMBERED_ITEM_RE, NUMBERED_SECTION_RE, QA_PATTERN_RE, ANSWER_PATTERN_RE

_EFFECTIVE_DATE_RE = re.compile(
    r"(effective\s+date|version|ver\.?)\s*[:\-]?\s*([A-Za-z0-9 ,./]+)", re.IGNORECASE
)
_APPLICABILITY_RE = re.compile(r"(applicab(le|ility)\s*(to|:))\s*(.+)", re.IGNORECASE)


def build_sections(blocks, document_title):
    """Group ordered blocks into sections keyed by heading path.

    Returns list of sections: {path: [str,...], blocks: [content blocks],
    pages: set, source_types: set, ocr_confidences: [float]}.
    """
    sections = []
    # stack holds (level, text) tuples - keyed by level, not list position,
    # so a heading level "gap" (e.g. level-2 headings with no level-1
    # parent) doesn't corrupt sibling detection.
    stack = []
    current = {"path": [document_title], "blocks": [], "pages": set(), "source_types": set(), "ocr_confidences": []}
    sections.append(current)
    title_heading_skipped = False

    for block in blocks:
        if block["kind"] == "heading":
            # The heading that supplied document_title is already the path
            # root - skip its first occurrence so it isn't duplicated.
            if not title_heading_skipped and (block["level"] or 1) == 1 and block["text"] == document_title:
                title_heading_skipped = True
                continue
            level = block["level"] or 1
            stack = [(lv, txt) for lv, txt in stack if lv < level]
            stack.append((level, block["text"]))
            current = {
                "path": [document_title] + [txt for _, txt in stack],
                "blocks": [],
                "pages": set(),
                "source_types": set(),
                "ocr_confidences": [],
            }
            sections.append(current)
        else:
            current["blocks"].append(block)
            if block.get("page") is not None:
                current["pages"].add(block["page"])
            current["source_types"].add(block["source_type"])
            if block.get("ocr_confidence") is not None:
                current["ocr_confidences"].append(block["ocr_confidence"])

    return [s for s in sections if s["blocks"]]


def detect_document_title(blocks, fallback_name):
    for b in blocks:
        # A numbered-section heading ("1.1 OBJECTIVE") is a policy section,
        # not the document's own title - skip it.
        if (b["kind"] == "heading" and (b.get("level") or 1) == 1
                and len(b["text"]) < 120 and not NUMBERED_SECTION_RE.match(b["text"])
                and not BARE_NUMBERED_ITEM_RE.match(b["text"])):
            return b["text"]
    for b in blocks:
        if b["kind"] != "paragraph":
            continue
        first_line = b["text"].strip().splitlines()[0].strip() if b["text"].strip() else ""
        if first_line and len(first_line) < 120:
            return first_line
    return fallback_name


_DATE_OR_VERSION_VALUE_RE = re.compile(r"\d")


def detect_effective_date(blocks):
    for b in blocks[:40]:
        if b["kind"] == "table":
            continue
        text = b.get("text", "")
        m = _EFFECTIVE_DATE_RE.search(text)
        if m:
            value = m.group(2).strip().strip(",.")
            # Require the matched value to actually contain a digit (a date
            # or version number) - otherwise it's a false match on a label
            # like "Version Number" with no value attached.
            if value and _DATE_OR_VERSION_VALUE_RE.search(value) and len(value) < 40:
                return value
    return None


def detect_applicability(blocks):
    """DEPRECATED - kept for reference only, no longer called from
    pipeline.py (process_document() now always sets applicability=None).

    Root-cause diagnosis of why this regex-based approach was replaced:
    _APPLICABILITY_RE only matches the phrasing "applicable to ..." or
    "applicability: ...". Real policy documents in this corpus almost
    always phrase scope differently - "This policy applies to ...",
    "... is applicable for ..." - which this regex never matches, so it
    silently returns None for documents that do state a clear scope
    (e.g. the India referral policy's "This policy applies to all
    full-time employee/ Interns ..." sentence). Conversely, on the rare
    block where the exact phrase "applicable to" IS present, this
    function returns whatever text follows it within that SAME block
    only - block boundaries (a table cell, a line wrap) routinely cut
    that sentence short, producing noise like "1. Date of purchase" or
    a truncated fragment ending mid-sentence with no usable content.

    Superseded by access_control/extract_applicability.py, which runs a
    single LLM extraction pass per document over its title plus an
    opening window of chunk text pulled from the loaded database
    (not from this ingestion-time field), rather than pre-filtering by
    a regex match first.
    """
    for b in blocks:
        text = b.get("text", "")
        m = _APPLICABILITY_RE.search(text)
        if m:
            return {"raw": m.group(4).strip()}
    return None


def is_faq_style(blocks):
    q_count = sum(1 for b in blocks if b["kind"] != "table" and QA_PATTERN_RE.match(b.get("text", "").strip()))
    a_count = sum(1 for b in blocks if b["kind"] != "table" and ANSWER_PATTERN_RE.match(b.get("text", "").strip()))
    return q_count >= 2 and a_count >= 2


def evaluate_structure_status(blocks, sections):
    """Decide ok vs needs_manual_review, with a reason."""
    has_headings = any(b["kind"] == "heading" for b in blocks)
    ocr_confidences = [b["ocr_confidence"] for b in blocks if b.get("ocr_confidence") is not None]
    low_conf_ocr = [c for c in ocr_confidences if c < OCR_CONFIDENCE_THRESHOLD]

    if not blocks:
        return "needs_manual_review", "no extractable content found"

    if ocr_confidences and (len(low_conf_ocr) / len(ocr_confidences)) > 0.3:
        return "needs_manual_review", "significant portion of document is low-confidence OCR content"

    if not has_headings and len(sections) <= 1:
        return "needs_manual_review", "no heading hierarchy detected; structure could not be reliably determined"

    return "ok", None
