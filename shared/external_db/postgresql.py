import psycopg

from shared.external_db.errors import DatabaseInspectionError
from shared.external_db.models import MAX_FIELDS_PER_OBJECT, MAX_OBJECTS, InspectionResult
from shared.external_db.uri import ParsedConnection

STATEMENT_TIMEOUT_MS = 15000

_SYSTEM_FILTER = """
    n.nspname NOT IN ('pg_catalog', 'information_schema')
    AND n.nspname NOT LIKE 'pg\\_toast%%'
    AND n.nspname NOT LIKE 'pg\\_temp%%'
"""

_RELATIONS_SQL = f"""
SELECT c.oid, n.nspname, c.relname, c.relkind,
       CASE WHEN c.reltuples < 0 THEN NULL ELSE c.reltuples::bigint END
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f')
  AND {_SYSTEM_FILTER}
  AND has_table_privilege(c.oid, 'SELECT')
ORDER BY n.nspname, c.relname
LIMIT %s
"""

_COLUMNS_SQL = """
SELECT a.attrelid, a.attname, format_type(a.atttypid, a.atttypmod), NOT a.attnotnull
FROM pg_attribute a
WHERE a.attrelid = ANY(%s) AND a.attnum > 0 AND NOT a.attisdropped
ORDER BY a.attrelid, a.attnum
"""

_PRIMARY_KEYS_SQL = """
SELECT con.conrelid, a.attname
FROM pg_constraint con
JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = ANY(con.conkey)
WHERE con.contype = 'p' AND con.conrelid = ANY(%s)
"""

_FOREIGN_KEYS_SQL = """
SELECT con.conrelid,
       ARRAY(SELECT a.attname FROM unnest(con.conkey) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.attnum ORDER BY k.ord),
       fn.nspname, fc.relname,
       ARRAY(SELECT a.attname FROM unnest(con.confkey) WITH ORDINALITY k(attnum, ord)
             JOIN pg_attribute a ON a.attrelid = con.confrelid AND a.attnum = k.attnum ORDER BY k.ord)
FROM pg_constraint con
JOIN pg_class fc ON fc.oid = con.confrelid
JOIN pg_namespace fn ON fn.oid = fc.relnamespace
WHERE con.contype = 'f' AND con.conrelid = ANY(%s)
"""

_WRITABLE_SQL = f"""
SELECT count(*)
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE c.relkind IN ('r', 'p')
  AND {_SYSTEM_FILTER}
  AND (has_table_privilege(c.oid, 'INSERT') OR has_table_privilege(c.oid, 'UPDATE')
       OR has_table_privilege(c.oid, 'DELETE') OR has_table_privilege(c.oid, 'TRUNCATE'))
"""

_KIND_NAMES = {"r": "table", "p": "table", "v": "view", "m": "materialized_view", "f": "foreign_table"}


def _privilege_warnings(cur) -> list[str]:
    cur.execute("SELECT rolsuper FROM pg_roles WHERE rolname = current_user")
    row = cur.fetchone()
    if row and row[0]:
        return ["This is a superuser account. For safety, connect with a dedicated read-only user."]
    cur.execute(_WRITABLE_SQL, ())
    writable = cur.fetchone()[0]
    if writable:
        return [
            f"This user can modify {writable} table(s). For safety, connect with a dedicated read-only user."
        ]
    return []


def inspect(connection_string: str, parsed: ParsedConnection) -> InspectionResult:
    conn = psycopg.connect(
        connection_string, connect_timeout=10, autocommit=True, application_name="DataAnalyzer",
    )
    with conn:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
            cur.execute("SET default_transaction_read_only = on")
            cur.execute("SELECT current_database(), current_setting('server_version')")
            database, server_version = cur.fetchone()
            warnings = _privilege_warnings(cur)

            cur.execute(_RELATIONS_SQL, (MAX_OBJECTS + 1,))
            relations = cur.fetchall()
            truncated = len(relations) > MAX_OBJECTS
            relations = relations[:MAX_OBJECTS]
            if not relations:
                raise DatabaseInspectionError(
                    "Connected, but this user can't read any tables. Grant it SELECT on the tables to analyze."
                )
            oids = [row[0] for row in relations]

            cur.execute(_COLUMNS_SQL, (oids,))
            columns: dict[int, list] = {}
            for relid, name, data_type, nullable in cur.fetchall():
                columns.setdefault(relid, []).append({"name": name, "type": data_type, "nullable": nullable})

            cur.execute(_PRIMARY_KEYS_SQL, (oids,))
            primary_keys: dict[int, set] = {}
            for relid, name in cur.fetchall():
                primary_keys.setdefault(relid, set()).add(name)

            cur.execute(_FOREIGN_KEYS_SQL, (oids,))
            foreign_keys: dict[int, list] = {}
            for relid, cols, ref_schema, ref_table, ref_cols in cur.fetchall():
                foreign_keys.setdefault(relid, []).append({
                    "columns": list(cols),
                    "references": f"{ref_schema}.{ref_table}",
                    "referenced_columns": list(ref_cols),
                })

    objects = []
    for oid, schema, name, kind, estimate in relations:
        pk = primary_keys.get(oid, set())
        fields = [{**col, "primary_key": col["name"] in pk} for col in columns.get(oid, [])]
        objects.append({
            "schema": schema,
            "name": name,
            "kind": _KIND_NAMES.get(kind, "table"),
            "row_count": int(estimate) if estimate is not None else None,
            "fields": fields[:MAX_FIELDS_PER_OBJECT],
            "foreign_keys": foreign_keys.get(oid, []),
        })

    return InspectionResult(
        database=database or parsed.database, server_version=server_version,
        objects=objects, warnings=warnings, truncated=truncated,
    )
