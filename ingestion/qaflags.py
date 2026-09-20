"""Automatic QA checks over produced chunks."""
from collections import Counter

from .ocr_utils import OCR_CONFIDENCE_THRESHOLD
from .utils import estimate_tokens, is_garbled

MIN_REASONABLE_TOKENS = 5


def apply_qa_flags(chunks):
    """Mutates each chunk in place to add 'qa_flags'. Returns a Counter of
    flag_type -> count for the run summary."""
    token_counts = [estimate_tokens(c["text"]) for c in chunks if c["chunk_type"] != "table"]
    median = _median(token_counts) if token_counts else 0

    flag_counts = Counter()

    for chunk in chunks:
        flags = []
        text = chunk["text"]

        if not text.strip():
            flags.append("empty")

        if chunk["chunk_type"] != "table" and text.strip():
            tokens = estimate_tokens(text)
            if tokens < MIN_REASONABLE_TOKENS or (median and (tokens < median * 0.15 or tokens > median * 4)):
                flags.append("length_outlier")

        # Markdown tables are naturally pipe/dash-heavy; table quality is
        # covered separately by table_conversion_issue.
        if chunk["chunk_type"] != "table" and text.strip() and is_garbled(text):
            flags.append("garbled_text")

        if chunk["chunk_type"] == "table" and chunk.pop("_row_mismatch", False):
            flags.append("table_conversion_issue")
        else:
            chunk.pop("_row_mismatch", None)

        if chunk["source_type"] == "ocr" and chunk.get("ocr_confidence") is not None:
            if chunk["ocr_confidence"] < OCR_CONFIDENCE_THRESHOLD:
                flags.append("low_ocr_confidence")

        chunk["qa_flags"] = flags
        for f in flags:
            flag_counts[f] += 1

    return flag_counts


def _median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return 0
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2
