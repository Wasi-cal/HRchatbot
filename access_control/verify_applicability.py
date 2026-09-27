#!/usr/bin/env python3
"""Prints every active document alongside its extracted applicability
tags, for manual spot-checking against the real corpus after running
access_control.extract_applicability.

    python3 -m access_control.verify_applicability
"""
from collections import defaultdict

from vectorstore.db import close_pool, get_pool


def main():
    pool = get_pool()
    with pool.connection() as conn:
        rows = conn.execute(
            """
            SELECT d.source_file, d.document_title, d.is_restricted, at.tag_type, at.tag_value
            FROM documents d
            LEFT JOIN document_applicability da ON da.document_id = d.id
            LEFT JOIN applicability_tags at ON at.id = da.tag_id
            WHERE d.status = 'active'
            ORDER BY d.source_file, at.tag_type, at.tag_value
            """
        ).fetchall()
    close_pool()

    titles = {}
    restricted = {}
    tags_by_doc = defaultdict(list)
    for source_file, title, is_restricted, tag_type, tag_value in rows:
        titles[source_file] = title
        restricted[source_file] = is_restricted
        if tag_type:
            tags_by_doc[source_file].append(f"{tag_type}={tag_value}")

    print(f"=== Applicability tags across {len(titles)} active documents ===\n")
    for source_file in sorted(titles):
        tags = tags_by_doc.get(source_file, [])
        tag_str = ", ".join(tags) if tags else "none - applies to everyone"
        flag = "  [RESTRICTED]" if restricted[source_file] else ""
        print(f"{source_file} ({titles[source_file]}){flag}")
        print(f"    {tag_str}")


if __name__ == "__main__":
    main()
