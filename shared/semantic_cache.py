from __future__ import annotations

import logging
import threading
from typing import Any

from shared.config import get_settings
from shared.redis_client import get_redis_url

logger = logging.getLogger("shared.semantic_cache")

DEEPINFRA_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
DEFAULT_EMBEDDING_TIMEOUT = 5.0

DEFAULT_DISTANCE_THRESHOLD = 0.05

_caches: dict[str, Any] = {}
_caches_lock = threading.Lock()
_embed_client = None
_embed_client_lock = threading.Lock()


def _embedding_model() -> str:
    return get_settings().get("INTENT_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL) or DEFAULT_EMBEDDING_MODEL


def _get_embed_client():
    global _embed_client
    if _embed_client is not None:
        return _embed_client
    with _embed_client_lock:
        if _embed_client is not None:
            return _embed_client
        api_key = get_settings().get("DEEPINFRA_API_KEY")
        if not api_key:
            return None
        from openai import OpenAI
        _embed_client = OpenAI(api_key=api_key, base_url=DEEPINFRA_BASE_URL, timeout=DEFAULT_EMBEDDING_TIMEOUT)
        return _embed_client


def _embed_one(text_input: str, **_kwargs) -> list[float]:
    client = _get_embed_client()
    response = client.embeddings.create(model=_embedding_model(), input=[text_input])
    return response.data[0].embedding


def get_cache(name: str, distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD, ttl: int | None = None):
    cache = _caches.get(name)
    if cache is not None or name in _caches:
        return cache

    with _caches_lock:
        if name in _caches:
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
            logger.exception("semantic_cache[%s]: failed to initialize - caching disabled", name)
            cache = None

        _caches[name] = cache
        return cache


def cached_check(
    name: str, prompt: str,
    distance_threshold: float = DEFAULT_DISTANCE_THRESHOLD, ttl: int | None = None,
) -> str | None:
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
    cache = get_cache(name, distance_threshold=distance_threshold, ttl=ttl)
    if cache is None:
        return
    try:
        cache.store(prompt=prompt, response=response)
    except Exception:
        logger.exception("semantic_cache[%s]: store() failed", name)
