"""Redis access shared by api_service and worker_service.

Two distinct uses of the same Redis instance (see full_application_build_plan.md):
  - durable job queue, via arq (api_service enqueues, worker_service consumes)
  - ephemeral pub/sub, the live event relay for SSE streaming

`get_redis()` is a plain redis.asyncio client for pub/sub (publish from the
worker, subscribe from api_service's SSE endpoint). `get_arq_redis_settings()`
gives arq's own connection settings, used both to create the enqueueing pool
in api_service and as `WorkerSettings.redis_settings` in worker_service.
"""
from arq.connections import ArqRedis, RedisSettings, create_pool
from redis.asyncio import Redis

from shared.config import get_settings

_redis: Redis | None = None
_arq_pool: ArqRedis | None = None


def get_redis_url() -> str:
    return get_settings().get("REDIS_URL", "redis://localhost:6379/0")


def get_redis() -> Redis:
    global _redis
    if _redis is None:
        _redis = Redis.from_url(get_redis_url(), decode_responses=True)
    return _redis


def get_arq_redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(get_redis_url())


async def get_arq_pool() -> ArqRedis:
    global _arq_pool
    if _arq_pool is None:
        _arq_pool = await create_pool(get_arq_redis_settings())
    return _arq_pool


def investigation_channel(investigation_id: str) -> str:
    return f"investigation:{investigation_id}"


async def _close(conn) -> None:
    # redis-py renamed close() -> aclose() around v5; support either so this
    # doesn't break depending on exactly what got pip-installed.
    closer = getattr(conn, "aclose", None) or getattr(conn, "close", None)
    if closer is not None:
        await closer()


async def close_redis() -> None:
    global _redis, _arq_pool
    if _redis is not None:
        await _close(_redis)
        _redis = None
    if _arq_pool is not None:
        await _close(_arq_pool)
        _arq_pool = None
