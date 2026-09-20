"""PDF extraction: per-page text-layer detection, layout-aware digital text,
digital table separation, and OCR routing for scanned/mixed pages."""
import logging
from collections import Counter

import pymupdf
from PIL import Image

from .ocr_utils import ocr_plain_text, try_extract_tables, upscale_if_low_dpi, MIN_OCR_DPI_TARGET
from .utils import NUMBERED_SECTION_RE

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
    return blocks, page_raw_texts, errors


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


def _bbox_overlaps(bbox, table_bboxes, thresh=0.5):
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
            table_bboxes.append(tuple(table.bbox))
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
            max_size = max(s["size"] for s in spans)
            is_bold = any(s.get("flags", 0) & 2**4 for s in spans)

            # Numbered-section patterns ("1.1 OBJECTIVE", "4.2 Parental Leave")
            # are a stronger, more consistent structural signal than font-size
            # clustering, so they take priority when present.
            m = NUMBERED_SECTION_RE.match(line_text)
            level = None
            if m:
                n_dots = m.group(1).count(".")
                # A bare "2." (no dot) is structurally ambiguous - it could be
                # a top-level section number, or the 2nd item of a numbered
                # definition/glossary list ("2. Aggrieved Person: ...").
                # Require it to be short (a title), not a full sentence, to
                # count as a heading; dotted patterns ("1.1", "4.2") are
                # unambiguous section labels regardless of length.
                short_enough = len(line_text) < 40 if n_dots == 0 else len(line_text) < 100
                if short_enough and (is_bold or len(line_text.split()) < 12):
                    # "1.1", "1.2" siblings under an implicit top section both
                    # get 1 dot -> same level; "1.1.1" (2 dots) nests deeper.
                    depth = max(n_dots, 1)
                    level = min(depth, 4)
            if level is None:
                level = _heading_level_for_size(max_size, heading_sizes, body_size)

            if level is not None and len(line_text) < 150:
                blocks.append({
                    "kind": "heading", "text": line_text, "level": level,
                    "page": page_num, "source_type": "digital_text",
                    "ocr_confidence": None,
                })
            else:
                if blocks and blocks[-1]["kind"] == "paragraph" and blocks[-1]["page"] == page_num:
                    blocks[-1]["text"] += "\n" + line_text
                else:
                    blocks.append({
                        "kind": "paragraph", "text": line_text, "level": None,
                        "page": page_num, "source_type": "digital_text",
                        "ocr_confidence": None,
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
