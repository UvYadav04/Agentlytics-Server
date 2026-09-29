import copy
import datetime
import decimal
import math
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import psycopg
from bson import Decimal128, ObjectId
from pymongo import MongoClient

from shared.external_db.errors import DatabaseInspectionError
from shared.external_db.time_window import ResolvedWindow

MAX_ROWS = 5000
STATEMENT_TIMEOUT_MS = 20000
MAX_SQL_LENGTH = 20000
PLACEHOLDERS = ("start", "end")

_SQL_FORBIDDEN_KEYWORDS = {
    "insert", "update", "delete", "merge", "drop", "alter", "create", "truncate", "grant", "revoke",
    "copy", "do", "vacuum", "reindex", "cluster", "set", "reset", "listen", "notify", "unlisten",
    "prepare", "execute", "deallocate", "discard", "into", "checkpoint",
}
_SQL_FORBIDDEN_FUNCTIONS = {
    "pg_read_file", "pg_read_binary_file", "pg_ls_dir", "pg_stat_file", "pg_sleep", "pg_sleep_for",
    "pg_sleep_until", "lo_import", "lo_export", "lo_get", "dblink", "dblink_exec", "pg_terminate_backend",
    "pg_cancel_backend", "set_config", "pg_reload_conf", "pg_rotate_logfile", "query_to_xml",
    "pg_logical_emit_message", "txid_current", "pg_advisory_lock", "pg_advisory_xact_lock",
}
_MONGO_ALLOWED_STAGES = {
    "$match", "$group", "$project", "$sort", "$limit", "$skip", "$unwind", "$addFields", "$set",
    "$unset", "$count", "$bucket", "$bucketAuto", "$facet", "$lookup", "$sortByCount",
    "$replaceRoot", "$replaceWith", "$densify", "$fill", "$setWindowFields", "$sample",
}
_MONGO_FORBIDDEN_OPERATORS = {
    "$out", "$merge", "$function", "$accumulator", "$where", "$currentOp", "$listSessions",
    "$listLocalSessions", "$planCacheStats", "$indexStats", "$collStats", "$unionWith", "$documents",
    "$changeStream", "$searchMeta", "$search",
}


class QueryValidationError(ValueError):
    pass


@dataclass
class QueryResult:
    columns: list[str] = field(default_factory=list)
    rows: list[dict] = field(default_factory=list)
    truncated: bool = False


def _strip_sql_literals(sql: str) -> str:
    if "$$" in sql or re.search(r"\$[A-Za-z_]*\$", sql):
        raise QueryValidationError("Dollar-quoted strings are not allowed.")
    if "--" in sql or "/*" in sql:
        raise QueryValidationError("SQL comments are not allowed.")
    out, i, n = [], 0, len(sql)
    while i < n:
        ch = sql[i]
        if ch in ("'", '"'):
            quote, j = ch, i + 1
            while j < n:
                if sql[j] == quote:
                    if j + 1 < n and sql[j + 1] == quote:
                        j += 2
                        continue
                    break
                j += 1
            if j >= n:
                raise QueryValidationError("The SQL has an unterminated quoted string.")
            out.append(" " if quote == "'" else "ident")
            i = j + 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def validate_sql(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise QueryValidationError("The query is empty.")
    cleaned = sql.strip().rstrip(";").strip()
    if len(cleaned) > MAX_SQL_LENGTH:
        raise QueryValidationError("The query is too long.")
    skeleton = _strip_sql_literals(cleaned)
    if ";" in skeleton:
        raise QueryValidationError("Only a single SQL statement is allowed.")
    words = re.findall(r"[a-z_][a-z0-9_]*", skeleton.lower())
    if not words or words[0] not in ("select", "with"):
        raise QueryValidationError("The query must be a single SELECT (or WITH ... SELECT) statement.")
    forbidden = sorted(set(words) & _SQL_FORBIDDEN_KEYWORDS)
    if forbidden:
        raise QueryValidationError(f"The query uses a disallowed keyword: {forbidden[0].upper()}.")
    functions = sorted(set(words) & _SQL_FORBIDDEN_FUNCTIONS)
    if functions:
        raise QueryValidationError(f"The query calls a disallowed function: {functions[0]}.")
    return cleaned


def sql_placeholders(sql: str) -> set[str]:
    skeleton = _strip_sql_literals(sql)
    return {name for name in PLACEHOLDERS if re.search(rf"(?<![:\w]):{name}\b", skeleton)}


def _bind_sql(sql: str) -> str:
    bound = sql.replace("%", "%%")
    for name in PLACEHOLDERS:
        bound = re.sub(rf"(?<![:\w]):{name}\b", f"%({name})s", bound)
    return bound


def _walk_mongo(value: Any, path: str) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise QueryValidationError(f"Invalid key at {path}.")
            if key in _MONGO_FORBIDDEN_OPERATORS:
                raise QueryValidationError(f"The pipeline uses a disallowed operator: {key}.")
            _walk_mongo(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_mongo(child, f"{path}[{index}]")


def validate_pipeline(query: dict) -> tuple[str, Optional[str], list]:
    if not isinstance(query, dict):
        raise QueryValidationError("The MongoDB query must be an object with collection and pipeline.")
    collection = query.get("collection")
    database = query.get("database")
    pipeline = query.get("pipeline")
    if not isinstance(collection, str) or not collection.strip() or collection.startswith("system."):
        raise QueryValidationError("The MongoDB query needs a valid collection name.")
    if database is not None and (not isinstance(database, str) or database in ("admin", "local", "config")):
        raise QueryValidationError("The MongoDB query has an invalid database name.")
    if not isinstance(pipeline, list):
        raise QueryValidationError("The MongoDB pipeline must be a list of stages.")
    for index, stage in enumerate(pipeline):
        if not isinstance(stage, dict) or len(stage) != 1:
            raise QueryValidationError(f"Pipeline stage {index} must be an object with exactly one operator.")
        operator = next(iter(stage))
        if operator not in _MONGO_ALLOWED_STAGES:
            raise QueryValidationError(f"Pipeline stage {operator} is not allowed.")
        if operator == "$lookup":
            spec = stage[operator]
            if not isinstance(spec, dict) or "db" in spec or not isinstance(spec.get("from"), str):
                raise QueryValidationError("$lookup must reference a collection in the same database.")
    _walk_mongo(pipeline, "pipeline")
    return collection, database, pipeline


def pipeline_placeholders(pipeline: list) -> set[str]:
    found: set[str] = set()

    def walk(value):
        if isinstance(value, dict):
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            for name in PLACEHOLDERS:
                if value == f"{{{{{name}}}}}":
                    found.add(name)

    walk(pipeline)
    return found


def _bind_pipeline(pipeline: list, params: dict) -> list:
    def walk(value):
        if isinstance(value, dict):
            return {key: walk(child) for key, child in value.items()}
        if isinstance(value, list):
            return [walk(child) for child in value]
        if isinstance(value, str):
            for name, bound in params.items():
                if value == f"{{{{{name}}}}}":
                    return bound
        return value

    return walk(copy.deepcopy(pipeline))


def validate_query(db_type: str, query: dict) -> dict:
    if db_type == "postgresql":
        sql = validate_sql((query or {}).get("sql"))
        return {"sql": sql}
    collection, database, pipeline = validate_pipeline(query)
    cleaned = {"collection": collection, "pipeline": pipeline}
    if database:
        cleaned["database"] = database
    return cleaned


def query_placeholders(db_type: str, query: dict) -> set[str]:
    if db_type == "postgresql":
        return sql_placeholders(query["sql"])
    return pipeline_placeholders(query["pipeline"])


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, decimal.Decimal):
        return float(value) if value.is_finite() else None
    if isinstance(value, Decimal128):
        converted = value.to_decimal()
        return float(converted) if converted.is_finite() else None
    if isinstance(value, datetime.datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=datetime.timezone.utc)
        return value.isoformat()
    if isinstance(value, (datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, (ObjectId, uuid.UUID)):
        return str(value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    return str(value)


def _flatten_mongo_row(document: dict) -> dict:
    row = dict(document)
    group_key = row.get("_id")
    if isinstance(group_key, dict):
        row.pop("_id")
        for key, value in group_key.items():
            row.setdefault(key, value)
    return {key: _json_safe(value) for key, value in row.items()}


def _params(window: Optional[ResolvedWindow]) -> dict:
    if window is None:
        return {}
    return {"start": window.start, "end": window.end}


def _require_params(db_type: str, query: dict, window: Optional[ResolvedWindow]) -> None:
    needed = query_placeholders(db_type, query)
    if needed and window is None:
        raise QueryValidationError("The query uses :start/:end but the dashboard has no time window.")


def run_postgres_queries(connection_string: str, queries: list[dict], window: Optional[ResolvedWindow]) -> list:
    results: list = []
    params = _params(window)
    with psycopg.connect(connection_string, connect_timeout=10, autocommit=False, application_name="DataAnalyzer") as conn:
        conn.read_only = True
        for query in queries:
            try:
                cleaned = validate_query("postgresql", query)
                _require_params("postgresql", cleaned, window)
                bound = _bind_sql(cleaned["sql"])
                wrapped = f"SELECT * FROM ({bound}) AS _dashboard_query LIMIT {MAX_ROWS + 1}"
                with conn.cursor() as cur:
                    cur.execute(f"SET LOCAL statement_timeout = {STATEMENT_TIMEOUT_MS}")
                    cur.execute(wrapped, params)
                    columns = [d.name for d in cur.description or []]
                    fetched = cur.fetchall()
                conn.rollback()
                truncated = len(fetched) > MAX_ROWS
                rows = [
                    {column: _json_safe(value) for column, value in zip(columns, record)}
                    for record in fetched[:MAX_ROWS]
                ]
                results.append(QueryResult(columns=columns, rows=rows, truncated=truncated))
            except QueryValidationError as exc:
                conn.rollback()
                results.append(exc)
            except psycopg.Error as exc:
                conn.rollback()
                message = (getattr(exc, "diag", None) and exc.diag.message_primary) or str(exc).splitlines()[0]
                results.append(DatabaseInspectionError(f"PostgreSQL error: {message}"))
    return results


def run_mongo_queries(
    connection_string: str, default_database: Optional[str], queries: list[dict], window: Optional[ResolvedWindow],
) -> list:
    results: list = []
    params = _params(window)
    client = MongoClient(
        connection_string, serverSelectionTimeoutMS=8000, connectTimeoutMS=8000,
        socketTimeoutMS=STATEMENT_TIMEOUT_MS + 5000, appname="DataAnalyzer", maxPoolSize=2, retryWrites=False,
    )
    try:
        client.admin.command("ping")
        for query in queries:
            try:
                cleaned = validate_query("mongodb", query)
                _require_params("mongodb", cleaned, window)
                database_name = cleaned.get("database") or default_database
                if not database_name:
                    raise QueryValidationError("The MongoDB query needs a database name.")
                pipeline = _bind_pipeline(cleaned["pipeline"], params) + [{"$limit": MAX_ROWS + 1}]
                cursor = client[database_name][cleaned["collection"]].aggregate(
                    pipeline, maxTimeMS=STATEMENT_TIMEOUT_MS, allowDiskUse=False,
                )
                documents = list(cursor)
                truncated = len(documents) > MAX_ROWS
                rows = [_flatten_mongo_row(doc) for doc in documents[:MAX_ROWS]]
                columns: list[str] = []
                for row in rows:
                    for key in row:
                        if key not in columns:
                            columns.append(key)
                results.append(QueryResult(columns=columns, rows=rows, truncated=truncated))
            except QueryValidationError as exc:
                results.append(exc)
            except Exception as exc:
                message = getattr(exc, "details", None) or {}
                text = message.get("errmsg") if isinstance(message, dict) else None
                results.append(DatabaseInspectionError(f"MongoDB error: {text or type(exc).__name__}"))
    finally:
        client.close()
    return results
