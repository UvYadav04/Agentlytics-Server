"""Free-tier usage limits (Phase 6): 20 messages, 4 charts/dashboards, 2
reports, per user, tracked in the `usage` collection.

Checks happen before the gated action starts (POST /chats/{id}/messages
checks before enqueueing); increments happen only after the action actually
succeeds (a cancelled/failed investigation, or a chart/report that failed to
generate, must not count against the limit).

Chart/report generation isn't its own enqueue step in this engine - the
orchestrator decides autonomously, mid-investigation, whether to call
generate_dashboard/generate_markdown_report as one of its tools. So unlike
messages, chart/report caps can't be checked "before enqueueing"; instead
worker_service checks the cap right before persisting each Chart/Report doc
it finds in the orchestrator's artifact_refs, and simply skips persisting
(doesn't create the Mongo doc / doesn't count it) if the user is already at
the cap - see worker_service/tasks/investigation.py.
"""
from shared.config import get_settings
from shared.db import get_db
from shared.models.usage import COLLECTION as USAGE
from shared.models.usage import Usage


def _limit(key: str, default: int) -> int:
    return int(get_settings().get(key, str(default)) or default)


def messages_limit() -> int:
    return _limit("FREE_TIER_MESSAGES", 20)


def charts_limit() -> int:
    return _limit("FREE_TIER_CHARTS", 4)


def reports_limit() -> int:
    return _limit("FREE_TIER_REPORTS", 2)


async def get_or_create_usage(user_id: str) -> Usage:
    db = get_db()
    doc = await db[USAGE].find_one({"user_id": user_id})
    if doc is not None:
        return Usage.from_mongo(doc)
    usage = Usage(user_id=user_id)
    await db[USAGE].insert_one(usage.to_mongo())
    return usage


async def has_message_capacity(user_id: str) -> bool:
    usage = await get_or_create_usage(user_id)
    return usage.messages_sent < messages_limit()


async def has_chart_capacity(user_id: str) -> bool:
    usage = await get_or_create_usage(user_id)
    return usage.charts_created < charts_limit()


async def has_report_capacity(user_id: str) -> bool:
    usage = await get_or_create_usage(user_id)
    return usage.reports_created < reports_limit()


async def increment_messages(user_id: str) -> None:
    await get_or_create_usage(user_id)
    await get_db()[USAGE].update_one({"user_id": user_id}, {"$inc": {"messages_sent": 1}})


async def increment_charts(user_id: str) -> None:
    await get_or_create_usage(user_id)
    await get_db()[USAGE].update_one({"user_id": user_id}, {"$inc": {"charts_created": 1}})


async def increment_reports(user_id: str) -> None:
    await get_or_create_usage(user_id)
    await get_db()[USAGE].update_one({"user_id": user_id}, {"$inc": {"reports_created": 1}})
