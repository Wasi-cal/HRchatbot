"""Shared helpers: text normalization, hashing, token estimation, header/footer stripping."""
import hashlib
import re
import unicodedata
from collections import Counter

MAX_CHUNK_TOKENS = 600

# Common OCR misread substitutions applied only to OCR-derived text, and only
# in contexts where they are unambiguous (standalone tokens), to avoid
# corrupting legitimate words.
_OCR_FIXES = [
    (re.compile(r"\bl\b"), "I"),          # lone lowercase "l" -> "I"
    (re.compile(r"(?<=[A-Za-z])0(?=[A-Za-z])"), "o"),  # "w0rk" -> "work"
    (re.compile(r"\|"), "I"),
    (re.compile(r"\s{2,}"), " "),
]

_HYPHEN_BREAK_RE = re.compile(r"(\w)-\s*\n\s*(\w)")
_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def normalize_unicode_whitespace(text: str) -> str:
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace(" ", " ").replace("​", "")
    text = _WHITESPACE_RE.sub(" ", text)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def dehyphenate(text: str) -> str:
    """Fix hyphenated line-break artifacts: 'bene-\\nfits' -> 'benefits'."""
    return _HYPHEN_BREAK_RE.sub(r"\1\2", text)


def clean_ocr_text(text: str) -> str:
    """Lightweight OCR cleanup pass - not a full correction model."""
    for pattern, repl in _OCR_FIXES:
        text = pattern.sub(repl, text)
    return text


def normalize_text(text: str, source_type: str = "digital_text") -> str:
    text = dehyphenate(text)
    text = normalize_unicode_whitespace(text)
    if source_type == "ocr":
        text = clean_ocr_text(text)
    return text.strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~1.3 tokens/word) - no tokenizer dependency."""
    words = text.split()
    return int(len(words) * 1.3) if words else 0


def find_repeated_lines(page_texts, min_repeat_ratio=0.6, max_line_len=120):
    """Given a list of raw page texts, find lines (typically first/last few
    lines of each page) that repeat across a majority of pages - these are
    treated as running headers/footers/watermarks and stripped.
    """
    if len(page_texts) < 3:
        return set()
    candidates = Counter()
    n_pages = len(page_texts)
    for text in page_texts:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        edge_lines = set(lines[:3]) | set(lines[-3:])
        for line in edge_lines:
            if 0 < len(line) <= max_line_len:
                candidates[line] += 1
    threshold = max(2, int(n_pages * min_repeat_ratio))
    return {line for line, count in candidates.items() if count >= threshold}


_PAGE_NUM_RE = re.compile(r"^\s*(page\s+)?\d+(\s*/\s*\d+)?\s*$", re.IGNORECASE)


def strip_headers_footers(text: str, repeated_lines: set) -> str:
    lines = text.splitlines()
    out = []
    for line in lines:
        stripped = line.strip()
        if stripped in repeated_lines:
            continue
        if _PAGE_NUM_RE.match(stripped):
            continue
        out.append(line)
    return "\n".join(out)


def is_garbled(text: str) -> bool:
    """Heuristic: high ratio of non-alphanumeric chars, or low word count
    relative to character count -> likely garbled OCR/extraction output."""
    text = text.strip()
    if not text:
        return False
    alnum = sum(c.isalnum() for c in text)
    ratio_alnum = alnum / len(text)
    words = text.split()
    avg_word_len = (sum(len(w) for w in words) / len(words)) if words else 0
    if ratio_alnum < 0.5:
        return True
    if len(text) > 40 and len(words) < len(text) / 25:
        return True
    if avg_word_len > 25:
        return True
    return False


NUMBERED_SECTION_RE = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,3})[.)]?\s+(\S.*)$")
QA_PATTERN_RE = re.compile(r"^(Q\s*[:.\-]|Q\d+[:.\-]|Question\s*[:.\-])", re.IGNORECASE)
ANSWER_PATTERN_RE = re.compile(r"^(A\s*[:.\-]|A\d+[:.\-]|Answer\s*[:.\-])", re.IGNORECASE)
