"""Fire-and-forget query logging. Never raises into the caller."""
import logging

from psycopg.types.json import Jsonb

from vectorstore.db import get_pool

logger = logging.getLogger(__name__)


def log_query(
    *,
    query_text: str,
    response_type: str,
    answer_text: str | None = None,
    user_attributes: dict | None = None,
    blocking_tag_type: str | None = None,
    shortcut_fired: bool | None = None,
    timing: dict | None = None,
    chunk_ids_used: list | None = None,
) -> None:
    timing = timing or {}
    try:
        with get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO query_logs (
                    query_text, response_type, answer_text, user_attributes,
                    blocking_tag_type, shortcut_fired, retrieval_time_ms,
                    ttft_ms, total_generation_time_ms, chunk_ids_used
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    query_text, response_type, answer_text,
                    Jsonb(user_attributes) if user_attributes is not None else None,
                    blocking_tag_type, shortcut_fired,
                    timing.get("retrieval_time_ms"),
                    timing.get("time_to_first_token_ms"),
                    timing.get("total_generation_time_ms"),
                    Jsonb(chunk_ids_used) if chunk_ids_used is not None else None,
                ),
            )
            conn.commit()
    except Exception:
        logger.exception("query_logs insert failed (ignored)")
