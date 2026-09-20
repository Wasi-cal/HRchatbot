"""Turn sections into atomic, inspectable chunks per the v1 chunking rules."""
import re

from .utils import (
    MAX_CHUNK_TOKENS, content_hash, estimate_tokens, normalize_text,
    QA_PATTERN_RE, ANSWER_PATTERN_RE,
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _section_source_type_and_conf(blocks):
    has_ocr = any(b["source_type"] == "ocr" for b in blocks)
    if not has_ocr:
        return "digital_text", None
    confs = [b["ocr_confidence"] for b in blocks if b.get("ocr_confidence") is not None]
    return "ocr", (min(confs) if confs else None)

def _make_table_chunk(counter, id_prefix, section_path_str, block):
    return {
        "chunk_id": f"{id_prefix}_{counter:04d}",
        "section_path": section_path_str,
        "chunk_type": "table",
        "source_type": block["source_type"],
        "ocr_confidence": block.get("ocr_confidence"),
        "text": block["text"],
        "page_number": block.get("page"),
        "content_hash": content_hash(block["text"]),
        "_row_mismatch": block.get("row_mismatch", False),
    }

def _min_page(blocks):
    pages = [b["page"] for b in blocks if b.get("page") is not None]
    return min(pages) if pages else None

def chunk_document(sections, document_title, id_prefix):
    chunks = []
    counter = 0

    for section in sections:
        section_path_str = " > ".join(section["path"])
        blocks = section["blocks"]

        # FAQ sections are handled as a complete unit because the Q/A
        # boundaries depend on the ordering of all non-table blocks.
        if _looks_like_faq([b for b in blocks if b["kind"] != "table"]):
            prose_blocks = []
            for block in blocks:
                if block["kind"] == "table":
                    # Flush any Q/A content before the table so the original
                    # reading order is preserved.
                    if prose_blocks:
                        counter, faq_chunks = _chunk_faq(
                            prose_blocks,
                            section_path_str,
                            id_prefix,
                            counter,
                        )
                        chunks.extend(faq_chunks)
                        prose_blocks = []

                    counter += 1
                    chunks.append(_make_table_chunk(
                        counter,
                        id_prefix,
                        section_path_str,
                        block,
                    ))
                else:
                    prose_blocks.append(block)

            if prose_blocks:
                counter, faq_chunks = _chunk_faq(
                    prose_blocks,
                    section_path_str,
                    id_prefix,
                    counter,
                )
                chunks.extend(faq_chunks)

            continue

        # Normal prose/table sections.
        prose_blocks = []

        def flush_prose():
            nonlocal counter, prose_blocks

            if not prose_blocks:
                return

            source_type, ocr_conf = _section_source_type_and_conf(prose_blocks)
            full_text = normalize_text(
                "\n\n".join(b["text"] for b in prose_blocks),
                source_type,
            )
            page = _min_page(prose_blocks)

            if not full_text.strip():
                counter += 1
                chunks.append(
                    _make_prose_chunk(
                        counter,
                        id_prefix,
                        section_path_str,
                        "",
                        source_type,
                        ocr_conf,
                        page,
                    )
                )
                prose_blocks = []
                return

            if estimate_tokens(full_text) <= MAX_CHUNK_TOKENS:
                counter += 1
                chunks.append(
                    _make_prose_chunk(
                        counter,
                        id_prefix,
                        section_path_str,
                        full_text,
                        source_type,
                        ocr_conf,
                        page,
                    )
                )
            else:
                # section_path_str already contains document_title.
                header = section_path_str

                for piece in _split_into_token_budgets(
                    full_text,
                    MAX_CHUNK_TOKENS,
                ):
                    counter += 1
                    piece_text = f"{header}\n\n{piece}"

                    chunks.append(
                        _make_prose_chunk(
                            counter,
                            id_prefix,
                            section_path_str,
                            piece_text,
                            source_type,
                            ocr_conf,
                            page,
                        )
                    )

            prose_blocks = []

        for block in blocks:
            if block["kind"] == "table":
                # Emit the table exactly where it appeared in the source.
                flush_prose()

                counter += 1
                chunks.append(
                    _make_table_chunk(
                        counter,
                        id_prefix,
                        section_path_str,
                        block,
                    )
                )
            else:
                prose_blocks.append(block)

        flush_prose()

    return chunks

def _make_prose_chunk(counter, id_prefix, section_path_str, text, source_type, ocr_conf, page):
    return {
        "chunk_id": f"{id_prefix}_{counter:04d}",
        "section_path": section_path_str,
        "chunk_type": "prose",
        "source_type": source_type,
        "ocr_confidence": ocr_conf,
        "text": text,
        "page_number": page,
        "content_hash": content_hash(text),
    }


def _split_into_token_budgets(text, max_tokens):
    paragraphs = text.split("\n\n")
    pieces, current, current_tokens = [], [], 0

    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        if para_tokens > max_tokens:
            if current:
                pieces.append("\n\n".join(current))
                current, current_tokens = [], 0
            sentences = _SENTENCE_SPLIT_RE.split(para)
            buf, buf_tokens = [], 0
            for sent in sentences:
                st = estimate_tokens(sent)
                if buf_tokens + st > max_tokens and buf:
                    pieces.append(" ".join(buf))
                    buf, buf_tokens = [], 0
                buf.append(sent)
                buf_tokens += st
            if buf:
                pieces.append(" ".join(buf))
            continue

        if current_tokens + para_tokens > max_tokens and current:
            pieces.append("\n\n".join(current))
            current, current_tokens = [], 0
        current.append(para)
        current_tokens += para_tokens

    if current:
        pieces.append("\n\n".join(current))
    return pieces or [text]


def _looks_like_faq(blocks):
    q = sum(1 for b in blocks if QA_PATTERN_RE.match(b["text"].strip()))
    a = sum(1 for b in blocks if ANSWER_PATTERN_RE.match(b["text"].strip()))
    return q >= 1 and a >= 1 and (q + a) >= len(blocks) * 0.4


def _chunk_faq(blocks, section_path_str, id_prefix, counter):
    chunks = []
    pair_blocks = []

    def flush():
        nonlocal counter
        if not pair_blocks:
            return
        source_type, ocr_conf = _section_source_type_and_conf(pair_blocks)
        text = normalize_text("\n".join(b["text"] for b in pair_blocks), source_type)
        page = _min_page(pair_blocks)
        counter += 1
        chunks.append({
            "chunk_id": f"{id_prefix}_{counter:04d}",
            "section_path": section_path_str,
            "chunk_type": "qa_pair",
            "source_type": source_type,
            "ocr_confidence": ocr_conf,
            "text": text,
            "page_number": page,
            "content_hash": content_hash(text),
        })

    for b in blocks:
        if QA_PATTERN_RE.match(b["text"].strip()) and pair_blocks:
            flush()
            pair_blocks = [b]
        else:
            pair_blocks.append(b)
    flush()

    return counter, chunks
