import datetime
import decimal
import uuid

from bson import Binary, Decimal128, ObjectId, Timestamp
from bson.regex import Regex
from pymongo import MongoClient
from pymongo.errors import OperationFailure, PyMongoError

from shared.external_db.errors import DatabaseInspectionError
from shared.external_db.models import MAX_FIELDS_PER_OBJECT, MAX_OBJECTS, InspectionResult
from shared.external_db.uri import ParsedConnection

SAMPLE_SIZE = 50
MAX_DEPTH = 4
MAX_DATABASES = 5
QUERY_TIMEOUT_MS = 8000
SYSTEM_DATABASES = {"admin", "local", "config"}
WRITE_ACTIONS = {
    "insert", "update", "remove", "createCollection", "dropCollection", "dropDatabase",
    "createIndex", "dropIndex", "renameCollectionSameDB", "collMod",
}

_TYPE_NAMES = [
    (bool, "bool"),
    (int, "int"),
    (float, "double"),
    (str, "string"),
    (ObjectId, "objectId"),
    (datetime.datetime, "date"),
    (Decimal128, "decimal"),
    (decimal.Decimal, "decimal"),
    (Binary, "binData"),
    (bytes, "binData"),
    (uuid.UUID, "uuid"),
    (Timestamp, "timestamp"),
    (Regex, "regex"),
    (dict, "object"),
    (list, "array"),
]


def _type_name(value) -> str:
    if value is None:
        return "null"
    for cls, name in _TYPE_NAMES:
        if isinstance(value, cls):
            return name
    return type(value).__name__


def _walk(value, path: str, depth: int, stats: dict, seen_in_doc: set) -> None:
    entry = stats.setdefault(path, {"types": set(), "count": 0})
    entry["types"].add(_type_name(value))
    if path not in seen_in_doc:
        seen_in_doc.add(path)
        entry["count"] += 1
    if depth >= MAX_DEPTH:
        return
    if isinstance(value, dict):
        for key, child in value.items():
            _walk(child, f"{path}.{key}", depth + 1, stats, seen_in_doc)
    elif isinstance(value, list):
        for item in value[:20]:
            if isinstance(item, dict):
                for key, child in item.items():
                    _walk(child, f"{path}[].{key}", depth + 1, stats, seen_in_doc)


def infer_fields(documents: list[dict]) -> list[dict]:
    stats: dict = {}
    for document in documents:
        seen: set = set()
        for key, value in document.items():
            _walk(value, key, 1, stats, seen)
    total = max(len(documents), 1)
    fields = [
        {
            "name": path,
            "type": " | ".join(sorted(entry["types"])),
            "presence": round(entry["count"] / total, 2),
            "primary_key": path == "_id",
        }
        for path, entry in stats.items()
    ]
    fields.sort(key=lambda f: (f["name"] != "_id", f["name"].count("."), f["name"]))
    return fields[:MAX_FIELDS_PER_OBJECT]


def _privilege_warnings(client: MongoClient) -> list[str]:
    try:
        status = client.admin.command({"connectionStatus": 1, "showPrivileges": True})
    except PyMongoError:
        return []
    privileges = status.get("authInfo", {}).get("authenticatedUserPrivileges", []) or []
    actions = {action for privilege in privileges for action in privilege.get("actions", [])}
    if actions & WRITE_ACTIONS:
        return ["This user can modify data. For safety, connect with a dedicated read-only user."]
    return []


def _databases(client: MongoClient, parsed: ParsedConnection) -> list[str]:
    if parsed.database:
        return [parsed.database]
    try:
        names = client.list_database_names()
    except OperationFailure:
        raise DatabaseInspectionError(
            "This user can't list databases. Add the database name to the connection string, "
            "e.g. mongodb+srv://user:password@cluster/mydatabase"
        )
    names = [n for n in names if n not in SYSTEM_DATABASES]
    if not names:
        raise DatabaseInspectionError(
            "No readable databases were found. Add the database name to the connection string."
        )
    return sorted(names)[:MAX_DATABASES]


def _sample(collection) -> list[dict]:
    try:
        return list(collection.aggregate([{"$sample": {"size": SAMPLE_SIZE}}], maxTimeMS=QUERY_TIMEOUT_MS))
    except OperationFailure:
        return list(collection.find({}, limit=SAMPLE_SIZE, max_time_ms=QUERY_TIMEOUT_MS))


def _count(collection):
    try:
        return int(collection.estimated_document_count(maxTimeMS=QUERY_TIMEOUT_MS))
    except PyMongoError:
        return None


def inspect(connection_string: str, parsed: ParsedConnection) -> InspectionResult:
    client = MongoClient(
        connection_string,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        socketTimeoutMS=20000,
        appname="DataAnalyzer",
        maxPoolSize=2,
        retryWrites=False,
    )
    try:
        client.admin.command("ping")
        server_version = client.server_info().get("version")
        database_names = _databases(client, parsed)
        result = InspectionResult(
            database=parsed.database or (database_names[0] if len(database_names) == 1 else None),
            server_version=server_version,
            warnings=_privilege_warnings(client),
        )
        for database_name in database_names:
            database = client[database_name]
            try:
                infos = list(database.list_collections(filter={"type": {"$in": ["collection", "view"]}}))
            except OperationFailure:
                raise DatabaseInspectionError(
                    f"This user can't read the '{database_name}' database. Grant it the read role."
                )
            for info in sorted(infos, key=lambda i: i.get("name", "")):
                name = info.get("name", "")
                if not name or name.startswith("system."):
                    continue
                if len(result.objects) >= MAX_OBJECTS:
                    result.truncated = True
                    break
                collection = database[name]
                documents = _sample(collection)
                result.objects.append({
                    "schema": database_name,
                    "name": name,
                    "kind": "view" if info.get("type") == "view" else "collection",
                    "row_count": _count(collection) if info.get("type") != "view" else None,
                    "sampled": len(documents),
                    "fields": infer_fields(documents),
                    "foreign_keys": [],
                })
        return result
    finally:
        client.close()
