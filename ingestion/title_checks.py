"""Upload-time warnings about a detected document title. WARN ONLY: a title is never changed automatically (an
admin edits it with PATCH /api/admin/documents/{id}/title).

A title is flagged when it is
  * ALL CAPS,
  * generic (contains one of app_config title.generic_phrases, e.g. "document classification"), or
  * equal to a line that repeats on 3 or more pages of the source PDF (a running header or footer rather than a name).
"""
import re
from collections import defaultdict
from pathlib import Path

REPEAT_PAGES = 3


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def repeated_lines(pdf_path: Path, min_pages: int = REPEAT_PAGES) -> dict[str, int]:
    """{normalized line: number of pages it appears on} for lines on at least min_pages pages (PDF only)."""
    import pymupdf

    pages: dict[str, set] = defaultdict(set)
    with pymupdf.open(pdf_path) as pdf:
        for index, page in enumerate(pdf):
            for line in page.get_text().splitlines():
                key = _norm(line)
                if len(key) >= 3:
                    pages[key].add(index)
    return {line: len(p) for line, p in pages.items() if len(p) >= min_pages}


def title_warning(title: str | None, generic_phrases: list[str], pdf_path: Path | None = None) -> str | None:
    if not title or not title.strip():
        return None
    reasons = []
    letters = [c for c in title if c.isalpha()]
    if letters and all(c.isupper() for c in letters):
        reasons.append("the title is ALL CAPS")
    normalized = _norm(title)
    matched = [p for p in generic_phrases if _norm(p) and _norm(p) in normalized]
    if matched:
        reasons.append(f"the title looks generic (contains {matched[0]!r})")
    if pdf_path is not None and Path(pdf_path).suffix.lower() == ".pdf":
        try:
            repeated = repeated_lines(Path(pdf_path))
        except Exception:  # a warning must never fail an upload
            repeated = {}
        if normalized and normalized in repeated:
            reasons.append(f"the title equals a line repeated on {repeated[normalized]} pages of the source file (a running header?)")
    return ("Check this title: " + "; ".join(reasons) + ". It was not changed.") if reasons else None
