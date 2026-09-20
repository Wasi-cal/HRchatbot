"""Postgres connection pooling for the vector store.

Uses psycopg3's standard ConnectionPool (psycopg_pool) so the application
holds a small set of persistent connections rather than opening a new one
per operation. All connection details come from environment variables -
never hardcoded - loaded from a local .env file if present (see
.env.example for the required names).
"""
import os

from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv()

_REQUIRED_ENV_VARS = ["PGHOST", "PGDATABASE", "PGUSER", "PGPASSWORD"]

_pool = None


def _conninfo() -> str:
    missing = [v for v in _REQUIRED_ENV_VARS if not os.environ.get(v)]
    if missing:
        raise RuntimeError(
            f"Missing required env var(s): {', '.join(missing)}. "
            f"Copy .env.example to .env and fill in real values."
        )
    return (
        f"host={os.environ['PGHOST']} "
        f"port={os.environ.get('PGPORT', '5432')} "
        f"dbname={os.environ['PGDATABASE']} "
        f"user={os.environ['PGUSER']} "
        f"password={os.environ['PGPASSWORD']}"
    )


def _configure_connection(conn):
    try:
        register_vector(conn)
    except Exception:
        # The pgvector extension/type doesn't exist yet - this happens on
        # a brand-new database before setup_db.py has run. Connections
        # opened after that will register successfully.
        pass


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=_conninfo(),
            min_size=int(os.environ.get("PG_POOL_MIN", 1)),
            max_size=int(os.environ.get("PG_POOL_MAX", 5)),
            configure=_configure_connection,
            open=True,
        )
    return _pool


def close_pool():
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None
