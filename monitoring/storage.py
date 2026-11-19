"""SQLite storage backend for monitoring traces and spans.

Uses WAL mode and connection timeouts to handle multi-worker Celery writes safely.
Stores spans and provides aggregations for latency, tokens, and component stats.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from monitoring.sanitizer import safe_json_dumps
from monitoring.schema import SpanRecord, SpanStatus, StepType

logger = logging.getLogger("monitoring.storage")

# Trace directory and SQLite DB resolution:
# 1. MONITORING_DB_PATH / MONITORING_TRACES_DIR environment variables
# 2. Container bound volume at /app/traces / /app/traces/traces.db
# 3. Default relative to Server/traces
def _resolve_traces_dir() -> Path:
    env_dir = os.environ.get("MONITORING_TRACES_DIR")
    if env_dir:
        p = Path(env_dir).resolve()
        if p.exists() or p.parent.exists():
            return p
    docker_dir = Path("/app/traces")
    if docker_dir.is_dir():
        return docker_dir.resolve()
    repo_dir = Path(__file__).resolve().parent.parent / "traces"
    return repo_dir.resolve()


def _resolve_db_path() -> Path:
    env_db = os.environ.get("MONITORING_DB_PATH")
    if env_db:
        p = Path(env_db).resolve()
        if p.exists() or p.parent.exists():
            return p
    docker_db = Path("/app/traces/traces.db")
    if docker_db.exists():
        return docker_db.resolve()
    traces_dir = _resolve_traces_dir()
    return (traces_dir / "traces.db").resolve()


TRACES_DIR = _resolve_traces_dir()
DB_PATH = _resolve_db_path()


class MonitoringStore:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or _resolve_db_path()
        self._ensure_dir_and_db()

    def _ensure_dir_and_db(self) -> None:
        """Create traces directory and initialize SQLite schema."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with self._get_connection() as conn:
                conn.execute("""
                CREATE TABLE IF NOT EXISTS spans (
                    span_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    parent_span_id TEXT,
                    step_name TEXT NOT NULL,
                    component TEXT NOT NULL,
                    step_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    start_time REAL NOT NULL,
                    end_time REAL NOT NULL,
                    duration_ms REAL NOT NULL,
                    ttft_ms REAL,
                    prompt_tokens INTEGER DEFAULT 0,
                    completion_tokens INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    inputs_json TEXT,
                    outputs_json TEXT,
                    metrics_json TEXT,
                    error_json TEXT,
                    tags_json TEXT,
                    created_at TEXT NOT NULL
                )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_trace ON spans(trace_id, start_time)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_parent ON spans(parent_span_id)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_component ON spans(component, start_time)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_step_type ON spans(step_type, start_time)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_spans_status ON spans(status)")
                conn.commit()
        except Exception:
            logger.exception("Failed to initialize monitoring storage at %s", self.db_path)

    def _get_connection(self) -> sqlite3.Connection:
        """Create a connection with WAL mode and generous busy timeout."""
        conn = sqlite3.connect(
            str(self.db_path),
            timeout=10.0,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        return conn

    def save_span(self, span: SpanRecord) -> None:
        """Insert or replace a span record."""
        metrics = span.metrics or {}
        prompt_tokens = metrics.get("prompt_tokens") or 0
        completion_tokens = metrics.get("completion_tokens") or 0
        total_tokens = metrics.get("total_tokens") or (prompt_tokens + completion_tokens)

        query = """
        INSERT INTO spans (
            span_id, trace_id, parent_span_id, step_name, component, step_type,
            status, start_time, end_time, duration_ms, ttft_ms,
            prompt_tokens, completion_tokens, total_tokens,
            inputs_json, outputs_json, metrics_json, error_json, tags_json, created_at
        ) VALUES (
            :span_id, :trace_id, :parent_span_id, :step_name, :component, :step_type,
            :status, :start_time, :end_time, :duration_ms, :ttft_ms,
            :prompt_tokens, :completion_tokens, :total_tokens,
            :inputs_json, :outputs_json, :metrics_json, :error_json, :tags_json, :created_at
        )
        ON CONFLICT(span_id) DO UPDATE SET
            status = excluded.status,
            end_time = excluded.end_time,
            duration_ms = excluded.duration_ms,
            ttft_ms = excluded.ttft_ms,
            prompt_tokens = excluded.prompt_tokens,
            completion_tokens = excluded.completion_tokens,
            total_tokens = excluded.total_tokens,
            outputs_json = excluded.outputs_json,
            metrics_json = excluded.metrics_json,
            error_json = excluded.error_json
        """

        params = {
            "span_id": span.span_id,
            "trace_id": span.trace_id,
            "parent_span_id": span.parent_span_id,
            "step_name": span.step_name,
            "component": span.component,
            "step_type": span.step_type,
            "status": span.status,
            "start_time": span.start_time,
            "end_time": span.end_time,
            "duration_ms": span.duration_ms,
            "ttft_ms": span.ttft_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "inputs_json": safe_json_dumps(span.inputs),
            "outputs_json": safe_json_dumps(span.outputs),
            "metrics_json": safe_json_dumps(span.metrics),
            "error_json": safe_json_dumps(span.error) if span.error else None,
            "tags_json": safe_json_dumps(span.tags),
            "created_at": span.created_at,
        }

        try:
            with self._get_connection() as conn:
                conn.execute(query, params)
                conn.commit()
        except Exception:
            logger.exception("Failed to save span %s for trace %s", span.span_id, span.trace_id)

    def get_trace_spans(self, trace_id: str) -> list[dict[str, Any]]:
        """Retrieve all spans for an investigation trace in execution order."""
        query = """
        SELECT * FROM spans 
        WHERE trace_id = ? 
           OR json_extract(tags_json, '$.investigation_id') = ?
        ORDER BY COALESCE(json_extract(tags_json, '$.call_order'), 999999) ASC, start_time ASC
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, (trace_id, trace_id)).fetchall()
                return [self._row_to_dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to query trace %s", trace_id)
            return []

    def get_trace_tree(self, trace_id: str) -> dict[str, Any]:
        """Build a hierarchical tree of spans for UI waterfall visualization."""
        spans = self.get_trace_spans(trace_id)
        if not spans:
            return {"trace_id": trace_id, "spans": [], "root": None}

        by_id = {s["span_id"]: {**s, "children": []} for s in spans}
        roots = []

        for s in spans:
            pid = s.get("parent_span_id")
            if pid and pid in by_id:
                by_id[pid]["children"].append(by_id[s["span_id"]])
            else:
                roots.append(by_id[s["span_id"]])

        return {
            "trace_id": trace_id,
            "total_spans": len(spans),
            "roots": roots,
            "spans": spans,
        }

    def get_traces_summary(
        self, limit: int = 50, offset: int = 0, status: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """List investigation traces with aggregated duration, tokens, and errors."""
        having_conds = [
            """NOT (
                COUNT(*) = 1 AND (
                    (SELECT s.step_type FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) IN ('llm', 'tool', 'database', 'retriever', 'reranker', 'router')
                    OR (SELECT s.step_name FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) LIKE 'deepinfra%'
                    OR (SELECT s.step_name FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) LIKE 'openai%'
                    OR (SELECT s.step_name FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) LIKE 'execute_%'
                    OR (SELECT s.step_name FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) LIKE 'inspect_%'
                )
            )"""
        ]
        params: list[Any] = []
        if status:
            if status.lower() == "ok":
                having_conds.append("error_count = 0")
            elif status.lower() in ("error", "failed"):
                having_conds.append("error_count > 0")

        where_clause = "HAVING " + " AND ".join(having_conds)

        query = f"""
        SELECT
            COALESCE(json_extract(t.tags_json, '$.investigation_id'), t.trace_id) as request_id,
            MIN(t.start_time) as start_time,
            MAX(t.end_time) as end_time,
            ROUND((MAX(t.end_time) - MIN(t.start_time)) * 1000, 2) as total_duration_ms,
            COUNT(*) as total_spans,
            SUM(CASE WHEN t.status = 'error' THEN 1 ELSE 0 END) as error_count,
            SUM(t.prompt_tokens) as total_prompt_tokens,
            SUM(t.completion_tokens) as total_completion_tokens,
            SUM(t.total_tokens) as total_tokens,
            MAX(t.created_at) as created_at,
            (SELECT s.step_name FROM spans s WHERE (s.trace_id = t.trace_id OR json_extract(s.tags_json, '$.investigation_id') = COALESCE(json_extract(t.tags_json, '$.investigation_id'), t.trace_id)) AND (s.parent_span_id IS NULL OR s.parent_span_id = '') LIMIT 1) as root_step_name,
            (SELECT s.component FROM spans s WHERE (s.trace_id = t.trace_id OR json_extract(s.tags_json, '$.investigation_id') = COALESCE(json_extract(t.tags_json, '$.investigation_id'), t.trace_id)) AND (s.parent_span_id IS NULL OR s.parent_span_id = '') LIMIT 1) as root_component,
            (SELECT s.inputs_json FROM spans s WHERE (s.trace_id = t.trace_id OR json_extract(s.tags_json, '$.investigation_id') = COALESCE(json_extract(t.tags_json, '$.investigation_id'), t.trace_id)) AND (s.inputs_json LIKE '%"objective"%' OR s.inputs_json LIKE '%"query"%' OR s.inputs_json LIKE '%"prompt"%') ORDER BY s.start_time ASC LIMIT 1) as query_inputs,
            (SELECT s.tags_json FROM spans s WHERE (s.trace_id = t.trace_id OR json_extract(s.tags_json, '$.investigation_id') = COALESCE(json_extract(t.tags_json, '$.investigation_id'), t.trace_id)) AND (s.tags_json LIKE '%"query"%' OR s.tags_json LIKE '%"objective"%') ORDER BY s.start_time ASC LIMIT 1) as query_tags
        FROM spans t
        GROUP BY request_id
        {where_clause}
        ORDER BY start_time DESC
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                results = []
                for row in rows:
                    d = dict(row)
                    d["trace_id"] = d.get("request_id") or d.get("trace_id")
                    step_name = d.get("root_step_name") or ""
                    component = d.get("root_component") or ""
                    inputs = {}
                    if d.get("query_inputs"):
                        try:
                            inputs = json.loads(d["query_inputs"])
                        except Exception:
                            pass
                    tags = {}
                    if d.get("query_tags"):
                        try:
                            tags = json.loads(d["query_tags"])
                        except Exception:
                            pass

                    # Extract natural query / objective
                    q = inputs.get("objective") or inputs.get("query") or inputs.get("question") or inputs.get("prompt") or tags.get("query") or tags.get("objective") or ""
                    if not q and "messages" in inputs:
                        msgs = inputs["messages"]
                        if msgs and isinstance(msgs, list):
                            q = msgs[-1].get("content", "") if isinstance(msgs[-1], dict) else str(msgs[-1])

                    # Strictly require a real user query: if empty or starts with "Execution of", it's an internal step
                    if not q:
                        continue

                    q_lower = q.lower().strip()
                    if (
                        q_lower.startswith("execution of")
                        or q_lower.startswith("execute_")
                        or q_lower.startswith("inspect_")
                        or q_lower.startswith("list_")
                        or q_lower.startswith("deepinfra")
                        or q_lower.startswith("openai")
                    ):
                        continue

                    # Classify intent & route based on Quarry architecture
                    if "tabular" in step_name or "tabular" in component:
                        intent = "Tabular Analysis"
                        route = "Tabular & DuckDB"
                    elif "database" in step_name or "database" in component:
                        intent = "Database Query"
                        route = "SQL Database Agent"
                    elif "document" in step_name or "document" in component or "rag" in component:
                        intent = "Document RAG"
                        route = "Document RAG"
                    elif "orchestrator" in step_name or "orchestrator" in component:
                        intent = "Multi-Agent Investigation"
                        route = "Orchestrator Pipeline"
                    elif "visualization" in step_name:
                        intent = "Visualization Engine"
                        route = "Tabular + Charts"
                    else:
                        intent = "Data Query"
                        route = "Quarry Engine"

                    uid = tags.get("user_id") or tags.get("user_email") or (
                        "usr_quarry_" + str(abs(hash(d["trace_id"])) % 9000 + 1000)
                    )

                    total_ms = d.get("total_duration_ms") or 0.0
                    err_count = d.get("error_count") or 0
                    if err_count > 0:
                        stat = "Error"
                    elif total_ms > 15000:
                        stat = "Timeout"
                    else:
                        stat = "OK"

                    prompt_toks = d.get("total_prompt_tokens") or 0
                    comp_toks = d.get("total_completion_tokens") or 0
                    cost_usd = round((prompt_toks * 0.0015 + comp_toks * 0.002) / 1000.0, 4)

                    created_iso = d.get("created_at") or ""
                    time_display = created_iso[11:19] if len(created_iso) >= 19 else ""

                    d["intent"] = intent
                    d["query"] = q[:160]
                    d["route"] = route
                    d["user_id"] = uid
                    d["status"] = stat
                    d["latency_s"] = round(total_ms / 1000.0, 2)
                    d["cost_usd"] = cost_usd
                    d["time_display"] = time_display

                    # Clean up internal join fields
                    d.pop("root_step_name", None)
                    d.pop("root_component", None)
                    d.pop("root_inputs", None)
                    d.pop("root_tags", None)
                    results.append(d)
                return results
        except Exception:
            logger.exception("Failed to query traces summary")
            return []

    def get_component_stats(self, since_hours: int = 24) -> list[dict[str, Any]]:
        """Aggregate performance and latency metrics by component and step type."""
        cutoff = time.time() - (since_hours * 3600)
        query = """
        SELECT
            step_type,
            component,
            COUNT(*) as total_calls,
            ROUND(AVG(duration_ms), 2) as avg_duration_ms,
            ROUND(MIN(duration_ms), 2) as min_duration_ms,
            ROUND(MAX(duration_ms), 2) as max_duration_ms,
            ROUND(AVG(ttft_ms), 2) as avg_ttft_ms,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
            ROUND(CAST(SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS FLOAT) / COUNT(*) * 100, 2) as error_rate_pct,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens
        FROM spans
        WHERE start_time >= ?
        GROUP BY step_type, component
        ORDER BY total_calls DESC
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, (cutoff,)).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to query component stats")
            return []

    def get_slowest_spans(self, limit: int = 20) -> list[dict[str, Any]]:
        """Find the slowest operations across all investigations."""
        query = """
        SELECT span_id, trace_id, step_name, component, step_type, duration_ms, status, created_at
        FROM spans
        ORDER BY duration_ms DESC
        LIMIT ?
        """
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, (limit,)).fetchall()
                return [dict(r) for r in rows]
        except Exception:
            logger.exception("Failed to query slowest spans")
            return []

    def get_dashboard_overview(self) -> dict[str, Any]:
        """Aggregate high-level overview metrics, timeline points, and component stats from real traces."""
        try:
            with self._get_connection() as conn:
                # 1. Trace durations, errors, and counts grouped by user request
                trace_query = """
                SELECT
                    COALESCE(json_extract(t.tags_json, '$.investigation_id'), t.trace_id) as request_id,
                    ROUND((MAX(t.end_time) - MIN(t.start_time)), 3) as duration_s,
                    SUM(CASE WHEN t.status = 'error' THEN 1 ELSE 0 END) as error_count,
                    substr(MIN(t.created_at), 1, 10) as day
                FROM spans t
                GROUP BY request_id
                HAVING 
                    NOT (
                        COUNT(*) = 1 AND (
                            (SELECT s.step_type FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) IN ('llm', 'tool', 'database', 'retriever', 'reranker', 'router')
                            OR (SELECT s.step_name FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) LIKE 'deepinfra%'
                            OR (SELECT s.step_name FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) LIKE 'openai%'
                            OR (SELECT s.step_name FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) LIKE 'execute_%'
                            OR (SELECT s.step_name FROM spans s WHERE s.trace_id = t.trace_id LIMIT 1) LIKE 'inspect_%'
                        )
                    )
                """
                rows = conn.execute(trace_query).fetchall()
                total_traces = len(rows)
                durations = []
                failed_traces = 0
                days_dict: dict[str, list[float]] = {}

                for r in rows:
                    dur = r["duration_s"]
                    if dur is not None and dur > 0:
                        durations.append(dur)
                        day = r["day"] or "Unknown"
                        days_dict.setdefault(day, []).append(dur)
                    if r["error_count"] > 0:
                        failed_traces += 1

                durations.sort()
                p50 = round(durations[int(len(durations) * 0.50)], 2) if durations else 0.0
                p95 = round(durations[int(len(durations) * 0.95)], 2) if durations else 0.0
                avg_dur = round(sum(durations) / len(durations), 2) if durations else 0.0
                error_rate_pct = round((failed_traces / total_traces) * 100, 2) if total_traces else 0.0

                # 2. Token counts & total spans
                span_query = """
                SELECT
                    COUNT(*) as total_spans,
                    SUM(prompt_tokens) as total_prompt_tokens,
                    SUM(completion_tokens) as total_completion_tokens,
                    SUM(total_tokens) as total_tokens,
                    SUM(CASE WHEN step_type = 'llm' THEN 1 ELSE 0 END) as llm_calls,
                    SUM(CASE WHEN step_type = 'database' THEN 1 ELSE 0 END) as database_calls
                FROM spans
                """
                s_row = conn.execute(span_query).fetchone()
                total_spans = s_row["total_spans"] or 0
                prompt_tokens = s_row["total_prompt_tokens"] or 0
                completion_tokens = s_row["total_completion_tokens"] or 0
                total_tokens = s_row["total_tokens"] or (prompt_tokens + completion_tokens)
                llm_calls = s_row["llm_calls"] or 0
                database_calls = s_row["database_calls"] or 0

                # 3. Timeline by day
                timeline = []
                for day in sorted(days_dict.keys()):
                    day_durs = sorted(days_dict[day])
                    cnt = len(day_durs)
                    d_p50 = round(day_durs[int(cnt * 0.50)], 2) if day_durs else 0.0
                    d_p95 = round(day_durs[int(cnt * 0.95)], 2) if day_durs else 0.0
                    try:
                        dt = datetime.strptime(day, "%Y-%m-%d")
                        label = dt.strftime("%b %d")
                    except Exception:
                        label = day
                    timeline.append({
                        "time": label,
                        "date": day,
                        "count": cnt,
                        "p50": d_p50,
                        "p95": d_p95,
                    })

                metrics = {
                    "totalTraces": total_traces,
                    "totalSpans": total_spans,
                    "p50LatencySec": p50,
                    "p95LatencySec": p95,
                    "avgLatencySec": avg_dur,
                    "errorRatePct": error_rate_pct,
                    "failedTracesCount": failed_traces,
                    "totalTokens": total_tokens,
                    "inputTokens": prompt_tokens,
                    "outputTokens": completion_tokens,
                    "llmCallsCount": llm_calls,
                    "databaseCallsCount": database_calls,
                }

                return {
                    "metrics": metrics,
                    "timeline": timeline,
                    "componentStats": self.get_component_stats(since_hours=720),
                    "slowestSpans": self.get_slowest_spans(limit=20),
                    "recentTraces": self.get_traces_summary(limit=50),
                }
        except Exception:
            logger.exception("Failed to build dashboard overview")
            return {
                "metrics": {},
                "timeline": [],
                "componentStats": [],
                "slowestSpans": [],
                "recentTraces": [],
            }

    def delete_trace(self, trace_id: str) -> bool:
        """Delete all spans associated with a given trace_id or investigation_id."""
        return self.delete_traces([trace_id]) > 0

    def delete_traces(self, trace_ids: list[str]) -> int:
        """Delete all spans associated with multiple trace IDs or investigation IDs."""
        if not trace_ids:
            return 0
        deleted_count = 0
        try:
            with self._get_connection() as conn:
                for chunk_start in range(0, len(trace_ids), 50):
                    chunk = trace_ids[chunk_start:chunk_start + 50]
                    placeholders = ",".join("?" for _ in chunk)
                    query = f"""
                    DELETE FROM spans
                    WHERE trace_id IN ({placeholders})
                       OR json_extract(tags_json, '$.investigation_id') IN ({placeholders})
                    """
                    cursor = conn.execute(query, chunk + chunk)
                    deleted_count += cursor.rowcount
                conn.commit()

            # Clean up any exported investigation files
            inv_dir = TRACES_DIR / "investigations"
            if inv_dir.exists():
                for tid in trace_ids:
                    f = inv_dir / f"{tid}.json"
                    if f.exists():
                        try:
                            f.unlink()
                        except Exception:
                            pass

            return deleted_count
        except Exception:
            logger.exception("Failed to delete traces: %s", trace_ids)
            return 0

    def export_trace_to_file(self, trace_id: str) -> Optional[Path]:
        """Save a complete investigation trace as an isolated JSON file in traces/investigations/."""
        tree = self.get_trace_tree(trace_id)
        if not tree.get("spans"):
            return None

        inv_dir = TRACES_DIR / "investigations"
        inv_dir.mkdir(parents=True, exist_ok=True)
        file_path = inv_dir / f"{trace_id}.json"

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(tree, f, indent=2, default=str)
            return file_path
        except Exception:
            logger.exception("Failed to export trace JSON to %s", file_path)
            return None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        d = dict(row)
        for key in ("inputs_json", "outputs_json", "metrics_json", "error_json", "tags_json"):
            if d.get(key):
                try:
                    d[key.replace("_json", "")] = json.loads(d[key])
                except Exception:
                    d[key.replace("_json", "")] = d[key]
            else:
                d[key.replace("_json", "")] = None
            d.pop(key, None)
        return d


_global_store: Optional[MonitoringStore] = None


def get_monitoring_store() -> MonitoringStore:
    """Singleton getter for the monitoring SQLite store."""
    global _global_store
    if _global_store is None:
        _global_store = MonitoringStore()
    return _global_store

