"""
app/services/cache.py
=====================
Redis-backed response cache for MaritimeMind AI.

Purpose: Eliminates repeated LLM inference latency for identical queries.
Especially useful during demos — cached queries return in < 100ms.

Cache key: SHA-256 of normalized query string (lowercase, stripped).
TTL: Configurable via settings.CACHE_TTL_SECONDS (default: 10 minutes).

Fails silently: if Redis is unavailable, the system falls back to
uncached operation without raising any error.
"""
import hashlib
import json
import logging
from typing import Any, Optional

from app.configs.config import settings

logger = logging.getLogger("maritimemind.cache")

_redis_client = None


def _get_redis():
    """Lazy singleton Redis client. Returns None if Redis is unavailable."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=True, socket_connect_timeout=1)
        client.ping()  # Verify connection
        _redis_client = client
        logger.info("Redis cache connected")
        return _redis_client
    except Exception as e:
        logger.warning(f"Redis unavailable — caching disabled: {e}")
        return None


def _cache_key(query: str) -> str:
    """Deterministic cache key from normalized query."""
    normalized = query.lower().strip()
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:24]
    return f"maritimemind:v1:{digest}"


def get_cached_response(query: str) -> Optional[dict]:
    """
    Retrieve a cached query response.

    Returns:
        Deserialized response dict if cache hit, else None.
    """
    client = _get_redis()
    if client is None:
        return None
    try:
        data = client.get(_cache_key(query))
        if data:
            logger.debug(f"Cache HIT for query: '{query[:50]}...'")
            return json.loads(data)
    except Exception as e:
        logger.debug(f"Cache read error: {e}")
    return None


def set_cached_response(query: str, response: dict, ttl: Optional[int] = None) -> None:
    """
    Store a query response in the cache.

    Args:
        query: The original query string.
        response: The serializable response dict to cache.
        ttl: Time-to-live in seconds. Defaults to settings.CACHE_TTL_SECONDS.
    """
    client = _get_redis()
    if client is None:
        return
    ttl = ttl or settings.CACHE_TTL_SECONDS
    try:
        client.setex(_cache_key(query), ttl, json.dumps(response, default=str))
        logger.debug(f"Cache SET for query: '{query[:50]}...' (TTL: {ttl}s)")
    except Exception as e:
        logger.debug(f"Cache write error: {e}")


def invalidate_cache(query: str) -> None:
    """Remove a specific query from the cache."""
    client = _get_redis()
    if client is None:
        return
    try:
        client.delete(_cache_key(query))
    except Exception:
        pass


def clear_all_cache() -> int:
    """Clear all MaritimeMind cache keys. Returns number of deleted keys."""
    client = _get_redis()
    if client is None:
        return 0
    try:
        keys = client.keys("maritimemind:v1:*")
        if keys:
            return client.delete(*keys)
    except Exception:
        pass
    return 0
