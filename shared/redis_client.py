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
