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
from pathlib import Path
from typing import Any, Optional

from monitoring.sanitizer import safe_json_dumps
from monitoring.schema import SpanRecord, SpanStatus, StepType

logger = logging.getLogger("monitoring.storage")

# Default trace directory inside Server/traces
_DEFAULT_BASE_DIR = Path(__file__).resolve().parent.parent / "traces"
TRACES_DIR = Path(os.environ.get("MONITORING_TRACES_DIR") or _DEFAULT_BASE_DIR).resolve()
DB_PATH = Path(os.environ.get("MONITORING_DB_PATH") or (TRACES_DIR / "traces.db")).resolve()


class MonitoringStore:
    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
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
        query = "SELECT * FROM spans WHERE trace_id = ? ORDER BY start_time ASC"
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, (trace_id,)).fetchall()
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
        where_clause = ""
        params: list[Any] = []
        if status:
            where_clause = "WHERE status = ?"
            params.append(status)

        query = f"""
        SELECT
            trace_id,
            MIN(start_time) as start_time,
            MAX(end_time) as end_time,
            ROUND((MAX(end_time) - MIN(start_time)) * 1000, 2) as total_duration_ms,
            COUNT(*) as total_spans,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) as error_count,
            SUM(prompt_tokens) as total_prompt_tokens,
            SUM(completion_tokens) as total_completion_tokens,
            SUM(total_tokens) as total_tokens,
            MAX(created_at) as created_at
        FROM spans
        {where_clause}
        GROUP BY trace_id
        ORDER BY start_time DESC
        LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [dict(r) for r in rows]
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

