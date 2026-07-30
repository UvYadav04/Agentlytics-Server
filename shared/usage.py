from shared.admin import is_admin_email
from shared.config import get_settings
from shared.db import get_db
from shared.models.message import COLLECTION as MESSAGES
from shared.models.usage import COLLECTION as USAGE
from shared.models.usage import Usage


def _limit(key: str, default: int) -> int:
    return int(get_settings().get(key, str(default)) or default)


def messages_limit() -> int:
    return _limit("FREE_TIER_MESSAGES", 20)


def messages_per_chat_limit() -> int:
    return _limit("FREE_TIER_MESSAGES_PER_CHAT", 8)


def charts_limit() -> int:
    return _limit("FREE_TIER_CHARTS", 3)


def reports_limit() -> int:
    return _limit("FREE_TIER_REPORTS", 2)


def chats_limit() -> int:
    return _limit("FREE_TIER_CHATS", 2)


def workspaces_limit() -> int:
    return _limit("FREE_TIER_WORKSPACES", 2)


async def get_or_create_usage(user_id: str) -> Usage:
    db = get_db()
    doc = await db[USAGE].find_one({"user_id": user_id})
    if doc is not None:
        return Usage.from_mongo(doc)
    usage = Usage(user_id=user_id)
    await db[USAGE].insert_one(usage.to_mongo())
    return usage


async def has_message_capacity(user_id: str, email: str | None = None) -> bool:
    if is_admin_email(email):
        return True
    usage = await get_or_create_usage(user_id)
    return usage.messages_sent < messages_limit()


async def has_chat_message_capacity(chat_id: str, email: str | None = None) -> bool:
    if is_admin_email(email):
        return True
    count = await get_db()[MESSAGES].count_documents({"chat_id": chat_id})
    return count < messages_per_chat_limit()


async def has_chart_capacity(user_id: str, email: str | None = None) -> bool:
    if is_admin_email(email):
        return True
    usage = await get_or_create_usage(user_id)
    return usage.charts_created < charts_limit()


async def has_report_capacity(user_id: str, email: str | None = None) -> bool:
    if is_admin_email(email):
        return True
    usage = await get_or_create_usage(user_id)
    return usage.reports_created < reports_limit()


async def has_chat_creation_capacity(user_id: str, email: str | None = None) -> bool:
    if is_admin_email(email):
        return True
    usage = await get_or_create_usage(user_id)
    return usage.chats_created < chats_limit()


async def has_workspace_creation_capacity(user_id: str, email: str | None = None) -> bool:
    if is_admin_email(email):
        return True
    usage = await get_or_create_usage(user_id)
    return usage.workspaces_created < workspaces_limit()


async def increment_messages(user_id: str) -> None:
    await get_or_create_usage(user_id)
    await get_db()[USAGE].update_one({"user_id": user_id}, {"$inc": {"messages_sent": 1}})


async def increment_charts(user_id: str) -> None:
    await get_or_create_usage(user_id)
    await get_db()[USAGE].update_one({"user_id": user_id}, {"$inc": {"charts_created": 1}})


async def increment_reports(user_id: str) -> None:
    await get_or_create_usage(user_id)
    await get_db()[USAGE].update_one({"user_id": user_id}, {"$inc": {"reports_created": 1}})


async def increment_chats(user_id: str) -> None:
    await get_or_create_usage(user_id)
    await get_db()[USAGE].update_one({"user_id": user_id}, {"$inc": {"chats_created": 1}})


async def increment_workspaces(user_id: str) -> None:
    await get_or_create_usage(user_id)
    await get_db()[USAGE].update_one({"user_id": user_id}, {"$inc": {"workspaces_created": 1}})
