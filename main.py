#!/usr/bin/env python3
"""HR RAG chatbot - v1 document ingestion + chunking pipeline.

Usage:
    python main.py <input_folder> <output_folder>

Reads PDF/DOCX HR documents from <input_folder>, extracts structure-aware
text (with OCR + table-structure-aware OCR fallback for scanned content),
chunks each document into inspectable JSON, and writes one JSON file per
source document plus an ingestion_summary.json to <output_folder>.

No embeddings, vector store, retrieval, or generation happen here.
"""
import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

from ingestion.pipeline import process_document

SUPPORTED_EXTENSIONS = {".pdf", ".docx"}


def main():
    parser = argparse.ArgumentParser(description="HR document ingestion + chunking pipeline (v1)")
    parser.add_argument("input_folder", type=Path, help="Folder containing HR documents (PDF/DOCX)")
    parser.add_argument("output_folder", type=Path, help="Folder to write chunk JSON files + summary")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING,
                         format="%(levelname)s %(name)s: %(message)s")

    if not args.input_folder.is_dir():
        print(f"Input folder not found: {args.input_folder}", file=sys.stderr)
        sys.exit(1)
    args.output_folder.mkdir(parents=True, exist_ok=True)

    files = sorted(
        p for p in args.input_folder.iterdir()
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    summary = {
        "documents_processed": 0,
        "documents_flagged_for_manual_review": [],
        "total_chunks_created": 0,
        "chunks_with_qa_flags": {},
        "ocr_pages_processed": 0,
        "digital_text_pages_processed": 0,
        "extraction_errors": [],
    }
    flag_totals = Counter()

    for f in files:
        print(f"Processing {f.name} ...")
        start = time.time()
        try:
            doc_json, stats = process_document(f)
        except Exception as exc:
            summary["extraction_errors"].append({"file": f.name, "error": str(exc)})
            print(f"  FAILED: {exc}")
            continue

        if doc_json is None:
            summary["extraction_errors"].append({"file": f.name, "error": stats.get("error", "unknown error")})
            continue

        out_path = args.output_folder / f"{f.stem}.json"
        out_path.write_text(json.dumps(doc_json, indent=2, ensure_ascii=False), encoding="utf-8")

        summary["documents_processed"] += 1
        summary["total_chunks_created"] += stats["chunk_count"]
        summary["ocr_pages_processed"] += stats["ocr_pages"]
        summary["digital_text_pages_processed"] += stats["digital_pages"]
        flag_totals.update(stats["flag_counts"])

        if stats["structure_status"] == "needs_manual_review":
            summary["documents_flagged_for_manual_review"].append(f.name)

        for err in stats["errors"]:
            summary["extraction_errors"].append({"file": f.name, "error": err})

        elapsed = time.time() - start
        print(f"  -> {stats['chunk_count']} chunks, status={stats['structure_status']}, "
              f"digital_pages={stats['digital_pages']}, ocr_pages={stats['ocr_pages']} ({elapsed:.1f}s)")

    summary["chunks_with_qa_flags"] = dict(flag_totals)
    summary_path = args.output_folder / "ingestion_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== Ingestion Summary ===")
    print(f"Documents processed: {summary['documents_processed']} (of {len(files)} found)")
    print(f"Documents flagged for manual review: {len(summary['documents_flagged_for_manual_review'])}")
    if summary["documents_flagged_for_manual_review"]:
        for name in summary["documents_flagged_for_manual_review"]:
            print(f"  - {name}")
    print(f"Total chunks created: {summary['total_chunks_created']}")
    print(f"Digital-text pages processed: {summary['digital_text_pages_processed']}")
    print(f"OCR pages processed: {summary['ocr_pages_processed']}")
    print("Chunks with QA flags:")
    if flag_totals:
        for flag, count in flag_totals.most_common():
            print(f"  - {flag}: {count}")
    else:
        print("  (none)")
    if summary["extraction_errors"]:
        print(f"Extraction errors: {len(summary['extraction_errors'])}")
        for e in summary["extraction_errors"][:10]:
            print(f"  - {e['file']}: {e['error']}")
    print(f"\nOutput written to: {args.output_folder}")


if __name__ == "__main__":
    main()
