import json
import logging
from collections.abc import Iterable
from typing import Any, Optional

from shared.config import get_settings
from shared.redis_client import get_redis

logger = logging.getLogger("shared.workspace_cache")

DEFAULT_FILE_CATALOG_TTL_SECONDS = 120
DEFAULT_DB_SUMMARIES_TTL_SECONDS = 120


def file_catalog_key(workspace_id: str) -> str:
    return f"ws:{workspace_id}:file_catalog:v1"


def db_summaries_key(workspace_id: str) -> str:
    return f"ws:{workspace_id}:db_summaries:v1"


def _resolve_ttl(setting_name: str, default_seconds: int) -> int:
    raw = get_settings().get(setting_name)
    if raw is None or raw == "":
        return default_seconds
    try:
        return int(raw)
    except (TypeError, ValueError):
        logger.warning("invalid %s=%r; using default %ds", setting_name, raw, default_seconds)
        return default_seconds


async def _read_json(key: str) -> Any:
    try:
        raw = await get_redis().get(key)
    except Exception:
        logger.warning("workspace_cache: redis GET failed for %s (falling back to DB)", key, exc_info=True)
        return None
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        logger.warning("workspace_cache: corrupt JSON for %s (ignoring cached value)", key, exc_info=True)
        return None


async def _write_json(key: str, payload: Any, ttl_seconds: int) -> None:
    if ttl_seconds <= 0:
        return
    try:
        encoded = json.dumps(payload, separators=(",", ":"))
        await get_redis().set(key, encoded, ex=ttl_seconds)
    except Exception:
        logger.warning("workspace_cache: redis SET failed for %s (continuing uncached)", key, exc_info=True)


async def _delete_keys(keys: Iterable[str]) -> None:
    unique_keys = [k for k in dict.fromkeys(keys) if k]
    if not unique_keys:
        return
    try:
        await get_redis().delete(*unique_keys)
    except Exception:
        logger.warning("workspace_cache: redis DELETE failed for %s", unique_keys, exc_info=True)


async def get_cached_file_catalog(workspace_id: str) -> Optional[list[dict]]:
    if not workspace_id or _resolve_ttl("WORKSPACE_CATALOG_CACHE_TTL_SECONDS", DEFAULT_FILE_CATALOG_TTL_SECONDS) <= 0:
        return None
    data = await _read_json(file_catalog_key(workspace_id))
    return data if isinstance(data, list) else None


async def set_cached_file_catalog(
    workspace_id: str, entries: list[dict], ttl_seconds: Optional[int] = None,
) -> None:
    if not workspace_id:
        return
    ttl = (
        ttl_seconds
        if ttl_seconds is not None
        else _resolve_ttl("WORKSPACE_CATALOG_CACHE_TTL_SECONDS", DEFAULT_FILE_CATALOG_TTL_SECONDS)
    )
    await _write_json(file_catalog_key(workspace_id), entries, ttl)


async def invalidate_file_catalog(workspace_id: str) -> None:
    if not workspace_id:
        return
    await _delete_keys([file_catalog_key(workspace_id)])


async def get_cached_db_summaries(workspace_id: str) -> Optional[list[dict]]:
    if (
        not workspace_id
        or _resolve_ttl("WORKSPACE_DB_SUMMARIES_CACHE_TTL_SECONDS", DEFAULT_DB_SUMMARIES_TTL_SECONDS) <= 0
    ):
        return None
    data = await _read_json(db_summaries_key(workspace_id))
    return data if isinstance(data, list) else None


async def set_cached_db_summaries(
    workspace_id: str, items: list[dict], ttl_seconds: Optional[int] = None,
) -> None:
    if not workspace_id:
        return
    ttl = (
        ttl_seconds
        if ttl_seconds is not None
        else _resolve_ttl("WORKSPACE_DB_SUMMARIES_CACHE_TTL_SECONDS", DEFAULT_DB_SUMMARIES_TTL_SECONDS)
    )
    await _write_json(db_summaries_key(workspace_id), items, ttl)


async def invalidate_db_summaries(workspace_id: str) -> None:
    if not workspace_id:
        return
    await _delete_keys([db_summaries_key(workspace_id)])


async def invalidate_db_summaries_many(workspace_ids: Iterable[str]) -> None:
    keys = [db_summaries_key(ws_id) for ws_id in workspace_ids if ws_id]
    await _delete_keys(keys)
