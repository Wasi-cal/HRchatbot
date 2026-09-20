"""PDF extraction: per-page text-layer detection, layout-aware digital text,
digital table separation, and OCR routing for scanned/mixed pages."""
import logging
from collections import Counter

import pymupdf
from PIL import Image

from .ocr_utils import ocr_plain_text, try_extract_tables, upscale_if_low_dpi, MIN_OCR_DPI_TARGET
from .utils import BARE_NUMBERED_ITEM_RE, NUMBERED_SECTION_RE

import io

logger = logging.getLogger(__name__)

MIN_TEXT_LAYER_CHARS = 20
RENDER_DPI = 300
MIN_EMBEDDED_IMAGE_AREA_PTS = 100 * 100  # skip tiny icons/logos


def extract_pdf(path):
    """Returns (blocks, page_raw_texts, errors).

    blocks: ordered list of dicts:
      {kind: "heading"|"paragraph"|"table", text/markdown, level, page,
       source_type, ocr_confidence, row_mismatch}
    page_raw_texts: list of raw page text (digital pages only) for
      header/footer frequency detection.
    """
    errors = []
    blocks = []
    page_raw_texts = []
    try:
        doc = pymupdf.open(path)
    except Exception as exc:
        return [], [], [f"failed to open PDF: {exc}"]

    body_size = _estimate_body_font_size(doc)
    heading_sizes = _collect_heading_sizes(doc, body_size)

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            page_num = page_index + 1
            try:
                raw_text = page.get_text().strip()
            except Exception as exc:
                errors.append(f"page {page_num}: text extraction failed: {exc}")
                raw_text = ""

            has_text_layer = len(raw_text) >= MIN_TEXT_LAYER_CHARS

            if has_text_layer:
                page_raw_texts.append(raw_text)
                try:
                    page_blocks = _extract_digital_page(page, page_num, body_size, heading_sizes)
                    blocks.extend(page_blocks)
                except Exception as exc:
                    errors.append(f"page {page_num}: digital extraction failed: {exc}")

                try:
                    blocks.extend(_extract_embedded_images(doc, page, page_num))
                except Exception as exc:
                    errors.append(f"page {page_num}: embedded image OCR failed: {exc}")
            else:
                try:
                    blocks.extend(_extract_scanned_page(page, page_num))
                except Exception as exc:
                    errors.append(f"page {page_num}: OCR extraction failed: {exc}")
    finally:
        doc.close()
    blocks = _demote_numbered_list_runs(blocks)
    return blocks, page_raw_texts, errors


def _demote_numbered_list_runs(blocks):
    """A run of >=3 consecutive bare-numbered lines ("1. Verbal warning",
    "2. Corrective Actions", "3. Official written reprimand", ...) is a
    numbered procedural list, not a run of section headings - each one
    individually looks like a plausible short heading, but treating them
    as separate headings fragments one cohesive list into many near-empty
    sections. Collapse such runs into a single list-item paragraph."""
    out = []
    run = []

    def flush_run():
        if not run:
            return
        if len(run) >= 3:
            text = "\n".join(f"- {b['text']}" for b in run)
            pages = [b["page"] for b in run if b.get("page") is not None]
            out.append({
                "kind": "paragraph", "text": text, "level": None,
                "page": min(pages) if pages else None,
                "source_type": "digital_text", "ocr_confidence": None,
            })
        else:
            out.extend(run)
        run.clear()

    expected = None
    for b in blocks:
        bn = b.pop("bare_number", None)
        if bn is not None:
            if expected is None or bn != expected:
                flush_run()
            run.append(b)
            expected = bn + 1
        else:
            flush_run()
            expected = None
            out.append(b)
    flush_run()
    return out


def _estimate_body_font_size(doc):
    sizes = Counter()
    for page in doc:
        try:
            d = page.get_text("dict")
        except Exception:
            continue
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    txt = s.get("text", "").strip()
                    if txt:
                        sizes[round(s["size"], 1)] += len(txt)
    return sizes.most_common(1)[0][0] if sizes else 11.0


def _collect_heading_sizes(doc, body_size):
    candidates = Counter()
    for page in doc:
        try:
            d = page.get_text("dict")
        except Exception:
            continue
        for b in d.get("blocks", []):
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    size = round(s["size"], 1)
                    txt = s.get("text", "").strip()
                    if size > body_size * 1.12 and txt:
                        candidates[size] += 1
    sizes = sorted(candidates.keys(), reverse=True)
    return sizes[:4]  # map to heading levels 1..4


def _heading_level_for_size(size, heading_sizes, body_size):
    size = round(size, 1)
    for i, hs in enumerate(heading_sizes):
        if abs(size - hs) < 0.3:
            return i + 1
    if size > body_size * 1.12:
        return len(heading_sizes) + 1 if heading_sizes else 3
    return None


def _numbered_level(n_dots, max_size, heading_sizes, body_size):
    """Level for a dotted numbered heading ("1.1 OBJECTIVE", "1.0 Introduction"),
    combining font-size tier with dot-count depth.

    A consolidated multi-policy document (e.g. one PDF bundling 14 IT
    policies) styles each policy's chapter title in a visually distinct,
    larger font than its own "X.0"/"X.Y" subsections, which are the same
    size as body text. Pure dot-count is blind to this: "1.0 Introduction"
    (1 dot) and a bare "1. Acceptable Use Policy" chapter title both
    reduce to "depth 1", so without font-size input they collide as
    siblings and the chapter title - the very thing that disambiguates 14
    policies' otherwise-identical "1.0 Introduction" sections - gets
    dropped from the tree.

    If this heading itself carries a distinct font tier, that tier IS its
    level (a numbered chapter title is still a chapter title). Otherwise
    it has no visual distinction from body text - the same case as most
    single-tier documents - so nest it below every font-based tier found
    in the doc, using dot-count for depth within that shared tier.
    """
    size_level = _heading_level_for_size(max_size, heading_sizes, body_size)
    if size_level is not None:
        return size_level
    base = (len(heading_sizes) if heading_sizes else 0) + 1
    return base + min(n_dots - 1, 3)


def _bbox_overlaps(bbox, table_bboxes, thresh=0.4):
    x0, y0, x1, y1 = bbox
    area = max(0, x1 - x0) * max(0, y1 - y0)
    if area == 0:
        return False
    for tb in table_bboxes:
        tx0, ty0, tx1, ty1 = tb
        ix0, iy0 = max(x0, tx0), max(y0, ty0)
        ix1, iy1 = min(x1, tx1), min(y1, ty1)
        inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
        if inter / area > thresh:
            return True
    return False


def _pad_bbox(bbox, top=16, sides=5, bottom=5):
    """find_tables() sometimes returns a bbox anchored tightly to the
    detected grid lines, clipping the header row's label text which sits
    just above the first gridline (no visible borders in these docs -
    detection is heuristic column-alignment, not ruled lines). Pad the top
    generously so the overlap check below reliably excludes the whole
    table region, including its header row, from paragraph extraction."""
    x0, y0, x1, y1 = bbox
    return (x0 - sides, y0 - top, x1 + sides, y1 + bottom)


def _extract_digital_page(page, page_num, body_size, heading_sizes):
    blocks = []
    table_bboxes = []

    try:
        tf = page.find_tables()
        for table in tf.tables:
            try:
                rows = table.extract()
            except Exception:
                continue
            if not rows or not _looks_like_real_table(rows):
                continue
            markdown, mismatch = _rows_to_markdown(rows)
            blocks.append({
                "kind": "table",
                "text": markdown,
                "level": None,
                "page": page_num,
                "source_type": "digital_text",
                "ocr_confidence": None,
                "row_mismatch": mismatch,
            })
            table_bboxes.append(_pad_bbox(tuple(table.bbox)))
    except Exception as exc:
        logger.debug("find_tables failed on page %s: %s", page_num, exc)

    d = page.get_text("dict")
    for b in d.get("blocks", []):
        if "lines" not in b:
            continue
        bbox = b.get("bbox", (0, 0, 0, 0))
        if _bbox_overlaps(bbox, table_bboxes):
            continue

        for line in b["lines"]:
            spans = line.get("spans", [])
            if not spans:
                continue
            line_text = "".join(s.get("text", "") for s in spans).strip()
            if not line_text:
                continue
            # Ignore whitespace-only spans when sizing the line - Word-
            # exported PDFs often leave a trailing blank run at a different
            # (sometimes much larger) font size, which would otherwise be
            # mistaken for a real font-size heading signal.
            text_spans = [s for s in spans if s.get("text", "").strip()] or spans
            max_size = max(s["size"] for s in text_spans)
            is_bold = any(s.get("flags", 0) & 2**4 for s in text_spans)

            # Dotted section labels ("1.1 OBJECTIVE", "4.2 Parental Leave")
            # are a strong, unambiguous structural signal - no length limit
            # needed since the dots themselves disambiguate from prose.
            # Level combines dot-count with font-size tier (see
            # _numbered_level) so a chapter title and its own numbered
            # subsections don't collide onto the same level.
            m = NUMBERED_SECTION_RE.match(line_text)
            level = None
            bare_number = None
            if m:
                n_dots = m.group(1).count(".")
                if is_bold or len(line_text) < 100 or len(line_text.split()) < 12:
                    level = _numbered_level(n_dots, max_size, heading_sizes, body_size)
            else:
                # A bare number ("3", "12") is structurally ambiguous - it
                # could be a real chapter title, a table/legend cell
                # ("3 - Medium"), or one item of a numbered list ("3.
                # Official written reprimand") - all three share identical
                # surface syntax. Require an explicit delimiter + capitalized
                # title (BARE_NUMBERED_ITEM_RE) before considering it at
                # all, and even then only treat it as a heading if it
                # carries a font size distinctly larger than body text (a
                # real chapter title) - a bare-numbered list item is
                # visually indistinguishable from body prose, so it stays
                # a plain paragraph. Keep the bare_number tag regardless,
                # so a run of these can still be caught and grouped into a
                # single list by _demote_numbered_list_runs.
                bm = BARE_NUMBERED_ITEM_RE.match(line_text)
                if bm:
                    bare_number = int(bm.group(1))
                    size_level = _heading_level_for_size(max_size, heading_sizes, body_size)
                    if size_level is not None and len(line_text) < 100:
                        level = size_level
            if level is None and bare_number is None:
                level = _heading_level_for_size(max_size, heading_sizes, body_size)

            if level is not None and len(line_text) < 150:
                blocks.append({
                    "kind": "heading", "text": line_text, "level": level,
                    "page": page_num, "source_type": "digital_text",
                    "ocr_confidence": None, "bare_number": bare_number,
                })
            else:
                if blocks and blocks[-1]["kind"] == "paragraph" and blocks[-1]["page"] == page_num:
                    blocks[-1]["text"] += "\n" + line_text
                else:
                    blocks.append({
                        "kind": "paragraph", "text": line_text, "level": None,
                        "page": page_num, "source_type": "digital_text",
                        "ocr_confidence": None, "bare_number": bare_number,
                    })
    return blocks


def _looks_like_real_table(rows):
    """PyMuPDF's find_tables() sometimes misdetects a single heading line
    (e.g. '1.1 OBJECTIVE' split across columns) as a 1-row table. Filter
    those out so they fall through to normal heading/paragraph detection."""
    if len(rows) < 2:
        non_empty = sum(1 for r in rows for c in r if c and str(c).strip())
        if len(rows) == 1 and non_empty >= 4 and len(rows[0]) >= 4:
            return True
        return False
    return True


def _rows_to_markdown(rows):
    rows = [[("" if c is None else str(c)) for c in r] for r in rows]
    ncols = len(rows[0]) if rows else 0
    mismatch = any(len(r) != ncols for r in rows)
    if not rows:
        return "", False
    header, body = rows[0], rows[1:]
    lines = ["| " + " | ".join(h.replace("\n", " ").strip() for h in header) + " |",
             "| " + " | ".join(["---"] * ncols) + " |"]
    for r in body:
        r = (r + [""] * ncols)[:ncols]
        lines.append("| " + " | ".join(c.replace("\n", " ").strip() for c in r) + " |")
    return "\n".join(lines), mismatch


def _page_pixmap(page, dpi=RENDER_DPI):
    pix = page.get_pixmap(dpi=dpi)
    mode = "RGB" if pix.n < 4 else "RGBA"
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples).convert("RGB")

def _extract_scanned_page(page, page_num):
    blocks = []
    img = _page_pixmap(page, dpi=RENDER_DPI)

    tables = try_extract_tables(img)
    table_bboxes_px = []

    for t in tables:
        blocks.append({
            "kind": "table",
            "text": t["markdown"],
            "level": None,
            "page": page_num,
            "source_type": "ocr",
            "ocr_confidence": t["confidence"],
            "row_mismatch": t["row_mismatch"],
        })

        bbox = t.get("bbox")
        if bbox:
            table_bboxes_px.append(bbox)

    # Mask detected tables before plain OCR.
    # Otherwise the same table is extracted once as a table
    # and again as ordinary paragraph text.
    text_img = img.copy()

    if table_bboxes_px:
        from PIL import ImageDraw

        draw = ImageDraw.Draw(text_img)

        for bbox in table_bboxes_px:
            draw.rectangle(bbox, fill="white")

    text, confidence = ocr_plain_text(text_img)

    if text.strip():
        blocks.append({
            "kind": "paragraph",
            "text": text.strip(),
            "level": None,
            "page": page_num,
            "source_type": "ocr",
            "ocr_confidence": confidence,
        })

    return blocks

def _extract_embedded_images(doc, page, page_num):
    """Route embedded images within an otherwise-digital page to OCR
    (e.g. a digital cover page with a scanned signed appendix pasted in)."""
    blocks = []
    for img_info in page.get_images(full=True):
        xref = img_info[0]
        rects = page.get_image_rects(xref)
        for rect in rects:
            area = rect.width * rect.height
            if area < MIN_EMBEDDED_IMAGE_AREA_PTS:
                continue
            try:
                base = doc.extract_image(xref)
                raw_w, raw_h = base.get("width", 0), base.get("height", 0)
                pil_img = Image.open(io.BytesIO(base["image"])).convert("RGB")
            except Exception as exc:
                logger.debug("could not extract embedded image xref %s: %s", xref, exc)
                continue

            est_dpi = int(raw_w / (rect.width / 72)) if rect.width else RENDER_DPI
            pil_img = upscale_if_low_dpi(pil_img, est_dpi, MIN_OCR_DPI_TARGET)

            tables = try_extract_tables(pil_img)
            table_bboxes_px = []
            for t in tables:
                blocks.append({
                    "kind": "table", "text": t["markdown"], "level": None, "page": page_num,
                    "source_type": "ocr", "ocr_confidence": t["confidence"],
                    "row_mismatch": t["row_mismatch"],
                })
                bbox = t.get("bbox")
                if bbox:
                    table_bboxes_px.append(bbox)

            # Mask detected tables before plain OCR, otherwise the same
            # table is extracted once as a table and again as prose text.
            text_img = pil_img.copy()
            if table_bboxes_px:
                from PIL import ImageDraw
                draw = ImageDraw.Draw(text_img)
                for bbox in table_bboxes_px:
                    draw.rectangle(bbox, fill="white")

            text, confidence = ocr_plain_text(text_img)
            if text.strip():
                blocks.append({
                    "kind": "paragraph", "text": text.strip(), "level": None, "page": page_num,
                    "source_type": "ocr", "ocr_confidence": confidence,
                })
    return blocks
