"""OCR pipeline: table-structure-aware extraction first, then plain OCR fallback."""
import io
import logging

import pytesseract
from PIL import Image

logger = logging.getLogger(__name__)

OCR_CONFIDENCE_THRESHOLD = 0.65
MIN_OCR_DPI_TARGET = 300


def upscale_if_low_dpi(img: Image.Image, current_dpi: int, target_dpi: int = MIN_OCR_DPI_TARGET) -> Image.Image:
    """Low-DPI scans are a common cause of garbled OCR - upscale before OCR."""
    if current_dpi >= target_dpi:
        return img
    scale = target_dpi / max(current_dpi, 1)
    scale = min(scale, 3.0)
    new_size = (int(img.width * scale), int(img.height * scale))
    return img.resize(new_size, Image.LANCZOS)


def ocr_plain_text(img: Image.Image):
    """Run standard OCR once; return (text, confidence 0-1)."""
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
    )

    lines = {}
    confs = []

    for i, (text, conf) in enumerate(
        zip(data.get("text", []), data.get("conf", []))
    ):
        text = text.strip()

        try:
            conf = float(conf)
        except (TypeError, ValueError):
            continue

        if not text or conf < 0:
            continue

        key = (
            data["block_num"][i],
            data["par_num"][i],
            data["line_num"][i],
        )
        lines.setdefault(key, []).append(text)
        confs.append(conf)

    text = "\n".join(
        " ".join(words)
        for words in lines.values()
    )

    confidence = (
        sum(confs) / len(confs) / 100.0
        if confs
        else 0.0
    )

    return text.strip(), confidence

def try_extract_tables(img: Image.Image):
    """Run table-detection + table-structure-aware OCR on an image.

    Returns a list of dicts:
    {
        "markdown": str,
        "confidence": float,
        "row_mismatch": bool,
        "bbox": (x1, y1, x2, y2),
    }

    Empty list if no table-like region detected.
    """
    try:
        from img2table.document import Image as I2TImage
        from img2table.ocr import TesseractOCR
    except ImportError:
        logger.warning("img2table not available; skipping table-structure OCR")
        return []

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    buf.seek(0)

    try:
        doc = I2TImage(buf, detect_rotation=False)
        ocr = TesseractOCR(lang="eng")
        extracted = doc.extract_tables(ocr=ocr, implicit_rows=False, borderless_tables=False)
    except Exception as exc:
        logger.warning("img2table extraction failed: %s", exc)
        return []

    results = []
    for table in extracted:
        try:
            df = table.df
        except Exception:
            continue
        if df is None or df.empty:
            continue
        markdown, row_mismatch = dataframe_to_markdown(df)
        # img2table doesn't expose a direct per-table confidence; approximate
        # via plain-OCR confidence over the same crop region as a proxy.
        try:
            x1, y1, x2, y2 = (
                table.bbox.x1,
                table.bbox.y1,
                table.bbox.x2,
                table.bbox.y2,
            )
            crop = img.crop((x1, y1, x2, y2))
            _, conf = ocr_plain_text(crop)
        except Exception:
            conf = 0.5
            x1 = y1 = x2 = y2 = 0

        results.append({
            "markdown": markdown,
            "confidence": conf,
            "row_mismatch": row_mismatch,
            "bbox": (x1, y1, x2, y2),
        })
    return results


def dataframe_to_markdown(df):
    """Convert an img2table/pandas DataFrame to a markdown table.
    Returns (markdown_str, row_mismatch_flag)."""
    rows = df.fillna("").astype(str).values.tolist()
    header = [str(c) for c in df.columns]
    ncols = len(header)
    mismatch = any(len(r) != ncols for r in rows)
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * ncols) + " |"]
    for r in rows:
        r = (r + [""] * ncols)[:ncols]
        lines.append("| " + " | ".join(cell.replace("\n", " ").strip() for cell in r) + " |")
    return "\n".join(lines), mismatch
