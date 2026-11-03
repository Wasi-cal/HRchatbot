"""GET /api/policies - the policy library as an employee sees it: active, non-restricted documents only.

Read-only and deliberately light (one SELECT, no duplicate scan or tag aggregation), unlike /api/admin/documents,
which is the management view and is meant to sit behind an admin role.
"""
from fastapi import APIRouter
from psycopg.rows import dict_row

from vectorstore.db import get_pool

router = APIRouter(prefix="/api/policies", tags=["policies"])


@router.get("")
@router.get("/", include_in_schema=False)
def list_policies():
    with get_pool().connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            """
            SELECT d.id::text AS id, COALESCE(NULLIF(d.document_title, ''), d.source_file) AS title,
                   COALESCE(d.updated_at, d.created_at) AS updated_at
            FROM documents d
            WHERE d.status = 'active' AND NOT d.is_restricted
            ORDER BY title
            """
        )
        return cur.fetchall()
