#!/usr/bin/env python3
"""Create the pgvector extension and all vector-store tables.

Runnable independently and safe to re-run (schema.sql uses IF NOT EXISTS
throughout):

    python3 -m vectorstore.setup_db
"""
from pathlib import Path

from vectorstore.db import close_pool, get_pool

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def main():
    sql = SCHEMA_PATH.read_text()
    pool = get_pool()
    with pool.connection() as conn:
        conn.execute(sql)
        conn.commit()
    print(f"Schema applied from {SCHEMA_PATH}")
    close_pool()


if __name__ == "__main__":
    main()
