"""Fire-and-forget query logging. Never raises into the caller.

The query_logs columns user_attributes and blocking_tag_type are no longer written (the
applicability layer is unwired); they stay in the table, nullable, so historical rows remain readable."""
import logging

from psycopg.types.json import Jsonb

from vectorstore.db import get_pool

logger = logging.getLogger(__name__)


def log_query(
    *,
    query_text: str,
    response_type: str,
    answer_text: str | None = None,
    shortcut_fired: bool | None = None,
    timing: dict | None = None,
    chunk_ids_used: list | None = None,
    source: str = "text",
    prompt_version: dict | None = None,
    contextualized: bool | None = None,
    advisory: bool | None = None,
) -> None:
    timing = timing or {}
    try:
        with get_pool().connection() as conn:
            conn.execute(
                """
                INSERT INTO query_logs (
                    query_text, response_type, answer_text, shortcut_fired, retrieval_time_ms,
                    ttft_ms, total_generation_time_ms, chunk_ids_used, source, prompt_version,
                    contextualized, advisory
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    query_text, response_type, answer_text, shortcut_fired,
                    timing.get("retrieval_time_ms"),
                    timing.get("time_to_first_token_ms"),
                    timing.get("total_generation_time_ms"),
                    Jsonb(chunk_ids_used) if chunk_ids_used is not None else None,
                    source,
                    Jsonb(prompt_version) if prompt_version is not None else None,
                    contextualized, advisory,
                ),
            )
            conn.commit()
    except Exception:
        logger.exception("query_logs insert failed (ignored)")
