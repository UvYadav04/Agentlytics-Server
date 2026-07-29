"""Free-tier usage limits (Phase 6): 20 messages total, 3 charts/dashboards,
2 reports, 2 chats created, 2 workspaces created - per user, tracked in the
`usage` collection. On top of that, each individual chat is capped at 8
messages (checked against the `messages` collection directly, scoped by
chat_id, since that's a per-chat count rather than a lifetime per-user
counter - see has_chat_message_capacity below).

Checks happen before the gated action starts (POST /chats/{id}/messages
checks before enqueueing; POST /workspaces/{id}/chats and POST /workspaces
check before inserting the new Chat/Workspace doc); increments happen only
after the action actually succeeds (a cancelled/failed investigation, or a
chart/report that failed to generate, must not count against the limit).

Chart/report generation isn't its own enqueue step in this engine - the
orchestrator decides autonomously, mid-investigation, whether to call
generate_dashboard/generate_markdown_report as one of its tools. So unlike
messages, chart/report caps can't be checked "before enqueueing"; instead
worker_service checks the cap right before persisting each Chart/Report doc
it finds in the orchestrator's artifact_refs, and simply skips persisting
(doesn't create the Mongo doc / doesn't count it) if the user is already at
the cap - see worker_service/tasks/investigation.py.
"""
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
    """Per-chat cap (default 8), independent of the per-user lifetime caps
    above - counted directly off the `messages` collection (both roles)
    rather than a Usage counter, since it naturally resets to 0 for every
    new chat and needs no increment step of its own."""
    if is_admin_email(email):
        return True
    count = await get_db()[MESSAGES].count_documents({"chat_id": chat_id})
    return count < messages_per_chat_limit()


async def has_chart_capacity(user_id: str) -> bool:
    usage = await get_or_create_usage(user_id)
    return usage.charts_created < charts_limit()


async def has_report_capacity(user_id: str) -> bool:
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
