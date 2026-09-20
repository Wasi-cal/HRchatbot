"""DOCX extraction: native heading/list styles, native tables, embedded
image OCR routing. DOCX has no fixed pagination, so page_number is None."""
import io
import logging
import re

from docx import Document
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph
from PIL import Image

from .ocr_utils import ocr_plain_text, try_extract_tables

logger = logging.getLogger(__name__)

_HEADING_STYLE_RE = re.compile(r"heading\s*(\d)", re.IGNORECASE)


def extract_docx(path):
    errors = []
    blocks = []
    try:
        doc = Document(path)
    except Exception as exc:
        return [], [f"failed to open DOCX: {exc}"]

    for child in doc.element.body.iterchildren():
        try:
            if child.tag == qn("w:p"):
                para = Paragraph(child, doc)
                blocks.extend(_handle_paragraph(doc, para))
            elif child.tag == qn("w:tbl"):
                table = Table(child, doc)
                blocks.append(_table_to_block(table))
        except Exception as exc:
            errors.append(f"element extraction failed: {exc}")

    return blocks, errors


def _handle_paragraph(doc, para):
    blocks = []
    text = para.text.strip()
    style_name = (para.style.name or "") if para.style else ""

    m = _HEADING_STYLE_RE.match(style_name)
    if text and m:
        level = min(int(m.group(1)), 4)
        blocks.append({
            "kind": "heading", "text": text, "level": level, "page": None,
            "source_type": "digital_text", "ocr_confidence": None,
        })
    elif text:
        is_list = "list" in style_name.lower() or para._p.pPr is not None and para._p.pPr.numPr is not None
        prefix = "- " if is_list else ""
        blocks.append({
            "kind": "paragraph", "text": prefix + text, "level": None, "page": None,
            "source_type": "digital_text", "ocr_confidence": None,
        })

    for run in para.runs:
        for drawing in run._element.findall(".//" + qn("w:drawing")):
            blocks.extend(_handle_inline_image(doc, drawing))
    return blocks

def _handle_inline_image(doc, drawing_el):
    blocks = []
    blips = drawing_el.findall(".//" + qn("a:blip"))

    for blip in blips:
        rId = blip.get(qn("r:embed"))
        if not rId:
            continue

        try:
            image_part = doc.part.related_parts[rId]
            pil_img = Image.open(
                io.BytesIO(image_part.blob)
            ).convert("RGB")
        except Exception as exc:
            logger.debug(
                "failed to load embedded docx image %s: %s",
                rId,
                exc,
            )
            continue

        if pil_img.width * pil_img.height < 150 * 150:
            continue

        tables = try_extract_tables(pil_img)
        table_bboxes = []

        for t in tables:
            blocks.append({
                "kind": "table",
                "text": t["markdown"],
                "level": None,
                "page": None,
                "source_type": "ocr",
                "ocr_confidence": t["confidence"],
                "row_mismatch": t["row_mismatch"],
            })

            if t.get("bbox"):
                table_bboxes.append(t["bbox"])

        # Mask tables before prose OCR to avoid extracting
        # table contents twice.
        text_img = pil_img.copy()

        if table_bboxes:
            from PIL import ImageDraw

            draw = ImageDraw.Draw(text_img)

            for bbox in table_bboxes:
                draw.rectangle(bbox, fill="white")

        text, confidence = ocr_plain_text(text_img)

        if text.strip():
            blocks.append({
                "kind": "paragraph",
                "text": text.strip(),
                "level": None,
                "page": None,
                "source_type": "ocr",
                "ocr_confidence": confidence,
            })

    return blocks


def _table_to_block(table):
    rows = [[cell.text.strip() for cell in row.cells] for row in table.rows]
    ncols = len(rows[0]) if rows else 0
    mismatch = any(len(r) != ncols for r in rows)
    if not rows:
        markdown = ""
    else:
        header, body = rows[0], rows[1:]
        lines = ["| " + " | ".join(h.replace("\n", " ") for h in header) + " |",
                 "| " + " | ".join(["---"] * ncols) + " |"]
        for r in body:
            r = (r + [""] * ncols)[:ncols]
            lines.append("| " + " | ".join(c.replace("\n", " ") for c in r) + " |")
        markdown = "\n".join(lines)
    return {
        "kind": "table", "text": markdown, "level": None, "page": None,
        "source_type": "digital_text", "ocr_confidence": None, "row_mismatch": mismatch,
    }
