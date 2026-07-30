from __future__ import annotations

import logging
import threading
from typing import Any

from shared.config import get_settings
from shared.redis_client import get_redis_url

logger = logging.getLogger("shared.semantic_cache")

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
# Same default model shared/intent_router.py's embedding tier uses - reusing it means a query
# already embedded for intent routing and a query looked up in a semantic cache are directly
# comparable, and there's only ever one embedding model/API key to configure for both features.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_EMBEDDING_TIMEOUT = 5.0

# Redis COSINE distance, 0-2 (0 = identical, 2 = opposite) - see RedisVL's docs on
# SemanticCache(distance_threshold=...). Deliberately tighter than RedisVL's own example default
# (0.1) since a false-positive cache hit here silently returns a wrong answer for a different
# question - callers doing lower-stakes caching can always pass a looser threshold explicitly.
DEFAULT_DISTANCE_THRESHOLD = 0.05

_caches: dict[str, Any] = {}
_caches_lock = threading.Lock()
_embed_client = None
_embed_client_lock = threading.Lock()


def _embedding_model() -> str:
    return get_settings().get("INTENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL


def _get_embed_client():
    """Shared across every cache instance in this process - None (not raised) when
    DEEPINFRA_API_KEY isn't configured, same "degrade, don't crash" contract as
    shared/intent_router.py's _get_client()."""
    global _embed_client
    if _embed_client is not None:
        return _embed_client
    with _embed_client_lock:
        if _embed_client is not None:
            return _embed_client
        api_key = get_settings().get("DEEPINFRA_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI  # local import - keep this module importable without `openai`
        _embed_client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL, timeout=DEFAULT_EMBEDDING_TIMEOUT)
        return _embed_client


def _embed_one(text_input: str, **_kwargs) -> list[float]:
    """Matches the signature RedisVL's CustomVectorizer calls its embed function with
    (text, **kwargs) - see https://docs.redisvl.com/en/latest/user_guide/04_vectorizers.html#custom-vectorizers."""
    client = _get_embed_client()
    response = client.embeddings.create(model=_embedding_model(), input=[text_input])
    return response.data[0].embedding


def get_cache(name: str, distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD, ttl: int | None = None):
    """Returns a shared RedisVL SemanticCache for `name` (each distinct name gets its own Redis
    search index - use a stable, unique name per call site, e.g. "intent_classifier"), or None if
    caching isn't available right now. The first caller for a given `name` wins on
    distance_threshold/ttl for the life of the process; later calls just reuse the same instance.

    Always check for None - never assume a cache exists."""
    cache = _caches.get(name)
    if cache is not None or name in _caches:
        return cache

    with _caches_lock:
        if name in _caches:  # re-check inside the lock (double-checked locking)
            return _caches[name]

        if _get_embed_client() is None:
            logger.warning("semantic_cache[%s]: DEEPINFRA_API_KEY not configured - caching disabled", name)
            _caches[name] = None
            return None

        try:
            from redisvl.extensions.cache.llm import SemanticCache
            from redisvl.utils.vectorize import CustomVectorizer

            vectorizer = CustomVectorizer(_embed_one)
            cache = SemanticCache(
                name=name,
                redis_url=get_redis_url(),
                distance_threshold=distance_threshold,
                vectorizer=vectorizer,
                ttl=ttl,
            )
            logger.info(
                "semantic_cache[%s]: initialized (distance_threshold=%.3f, ttl=%s, model=%s)",
                name, distance_threshold, ttl, _embedding_model(),
            )
        except Exception:
            # Missing/incompatible redisvl package, or Redis itself unreachable - either way,
            # every call site should just behave as if caching were never enabled.
            logger.exception("semantic_cache[%s]: failed to initialize - caching disabled", name)
            cache = None

        _caches[name] = cache
        return cache


def cached_check(
    name: str, prompt: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD, ttl: int | None = None,
) -> str | None:
    """Best-effort cache lookup - returns the cached response string on a semantically similar
    hit, or None on a miss OR any failure. Never raises: callers should treat None exactly like a
    cache miss and fall through to calling the LLM."""
    cache = get_cache(name, distance_threshold=distance_threshold, ttl=ttl)
    if cache is None:
        return None
    try:
        results = cache.check(prompt=prompt, return_fields=["response"])
    except Exception:
        logger.exception("semantic_cache[%s]: check() failed - treating as a cache miss", name)
        return None
    if not results:
        return None
    return results[0].get("response")


def cached_store(
    name: str, prompt: str, response: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD, ttl: int | None = None,
) -> None:
    """Best-effort cache write - failures are logged and swallowed (the caller already has its
    real response by the time this runs; a cache-write failure must never affect what gets
    returned to the user)."""
    cache = get_cache(name, distance_threshold=distance_threshold, ttl=ttl)
    if cache is None:
        return
    try:
        cache.store(prompt=prompt, response=response)
    except Exception:
        logger.exception("semantic_cache[%s]: store() failed", name)
