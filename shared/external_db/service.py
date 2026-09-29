import asyncio
import logging

from shared.external_db import mongodb, postgresql, queries
from shared.external_db.errors import ConnectionStringError, DatabaseInspectionError
from shared.external_db.models import InspectionResult
from shared.external_db.network import ensure_public_hosts
from shared.external_db.time_window import ResolvedWindow
from shared.external_db.uri import LABELS, ParsedConnection, parse_connection_string, scrub_secrets

logger = logging.getLogger("shared.external_db")

INSPECTION_TIMEOUT_SECONDS = 60
QUERY_BATCH_TIMEOUT_SECONDS = 90

_INSPECTORS = {"mongodb": mongodb.inspect, "postgresql": postgresql.inspect}

_AUTH_MARKERS = ("authentication failed", "auth failed", "bad auth", "password authentication failed")
_NETWORK_MARKERS = (
    "timed out", "timeout", "could not connect", "connection refused", "no route to host",
    "network is unreachable", "server selection", "name or service not known",
    "could not translate host name", "nodename nor servname",
)


def _friendly_error(db_type: str, exc: Exception, raw: str, parsed: ParsedConnection) -> str:
    label = LABELS[db_type]
    message = str(exc)
    lowered = message.lower()
    if any(marker in lowered for marker in _AUTH_MARKERS) or getattr(exc, "code", None) == 18:
        return "Authentication failed - check the username and password in the connection string."
    if "does not exist" in lowered and "database" in lowered:
        return "That database doesn't exist on the server - check the database name in the connection string."
    if "ssl" in lowered or "tls" in lowered or "certificate" in lowered:
        return f"A secure (TLS/SSL) connection to {label} couldn't be established. Check the SSL options in the connection string."
    if any(marker in lowered for marker in _NETWORK_MARKERS):
        if db_type == "mongodb":
            return (
                "Couldn't reach the MongoDB cluster. Check the cluster address, and in Atlas allow "
                "this server's IP address under Security > Network Access."
            )
        return (
            "Couldn't reach the PostgreSQL server. Check the host and port, and make sure the "
            "server accepts connections from this app's IP address."
        )
    scrubbed = scrub_secrets(message, raw, parsed)
    return f"{label} returned an error: {scrubbed}" if scrubbed else f"Couldn't connect to {label}."


async def inspect_database(db_type: str, connection_string: str) -> tuple[ParsedConnection, InspectionResult]:
    parsed = parse_connection_string(db_type, connection_string)
    await asyncio.to_thread(ensure_public_hosts, parsed)
    inspector = _INSPECTORS[db_type]
    raw = connection_string.strip()
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(inspector, raw, parsed), timeout=INSPECTION_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise DatabaseInspectionError(
            f"{LABELS[db_type]} took too long to respond. Try again, or connect to a smaller database."
        ) from None
    except (ConnectionStringError, DatabaseInspectionError):
        raise
    except Exception as exc:
        friendly = _friendly_error(db_type, exc, raw, parsed)
        logger.warning(
            "database inspection failed: db_type=%s host=%s error_type=%s",
            db_type, parsed.host_label, type(exc).__name__,
        )
        raise DatabaseInspectionError(friendly) from None
    return parsed, result


def _run_queries(db_type: str, raw: str, database, query_list: list, window):
    if db_type == "postgresql":
        return queries.run_postgres_queries(raw, query_list, window)
    return queries.run_mongo_queries(raw, database, query_list, window)


async def run_dashboard_queries(
    db_type: str, connection_string: str, database, query_list: list[dict], window: ResolvedWindow | None,
) -> list:
    parsed = parse_connection_string(db_type, connection_string)
    await asyncio.to_thread(ensure_public_hosts, parsed)
    raw = connection_string.strip()
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(_run_queries, db_type, raw, database or parsed.database, query_list, window),
            timeout=QUERY_BATCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        raise DatabaseInspectionError(f"{LABELS[db_type]} took too long to answer the dashboard queries.") from None
    except (ConnectionStringError, DatabaseInspectionError):
        raise
    except Exception as exc:
        friendly = _friendly_error(db_type, exc, raw, parsed)
        logger.warning(
            "dashboard queries failed: db_type=%s host=%s error_type=%s",
            db_type, parsed.host_label, type(exc).__name__,
        )
        raise DatabaseInspectionError(friendly) from None
