"""Thin JSON cache over Redis. Never raises: any Redis problem (down,
slow, bad data) degrades to "always miss", so callers fall through to
Postgres.

get_cached() returns the MISS sentinel on a miss, because None is a
legitimate cached value (e.g. "this reply had no usable value").

After a failure the client is skipped for BACKOFF_SECONDS so an outage
costs one short timeout per window instead of one per call.
"""
import json
import logging
import os
import time

import redis
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

KEY_PREFIX = "hrchatbot:"
BACKOFF_SECONDS = 5.0

MISS = object()

_client: redis.Redis | None = None
_url: str | None = None
_down_until = 0.0


def configure(url: str | None = None) -> None:
    """(Re)points the cache at `url` (default: $REDIS_URL). Mainly for tests."""
    global _client, _url, _down_until
    _url = url or os.environ.get("REDIS_URL", "redis://localhost:6380/0")
    _client = redis.Redis.from_url(_url, socket_connect_timeout=0.25, socket_timeout=0.25)
    _down_until = 0.0


def _get_client() -> redis.Redis | None:
    global _client
    if time.monotonic() < _down_until:
        return None
    if _client is None:
        configure()
    return _client


def _mark_down(exc: Exception) -> None:
    global _down_until
    _down_until = time.monotonic() + BACKOFF_SECONDS
    logger.warning("Redis unavailable, falling back to DB for %.0fs: %s", BACKOFF_SECONDS, exc)


def get_cached(key: str):
    client = _get_client()
    if client is None:
        return MISS
    try:
        raw = client.get(KEY_PREFIX + key)
        return MISS if raw is None else json.loads(raw)
    except Exception as exc:
        _mark_down(exc)
        return MISS


def set_cached(key: str, value, ttl: int | None = None) -> bool:
    """Stores JSON-serialisable `value`; ttl seconds, or None for no expiry."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.set(KEY_PREFIX + key, json.dumps(value), ex=ttl)
        return True
    except Exception as exc:
        _mark_down(exc)
        return False


def delete_cached(key: str) -> bool:
    """Drops `key` (cache invalidation). If Redis is down there is nothing
    to invalidate that a reader could see, since it also reads as a miss."""
    client = _get_client()
    if client is None:
        return False
    try:
        client.delete(KEY_PREFIX + key)
        return True
    except Exception as exc:
        _mark_down(exc)
        return False
