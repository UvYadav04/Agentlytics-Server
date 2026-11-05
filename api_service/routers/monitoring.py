"""Monitoring and Observability API router.

Exposes endpoints to query trace trees, performance metrics,
component latency breakdowns, and aggregated statistics.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from api_service.deps import get_current_user
from monitoring.storage import get_monitoring_store

logger = logging.getLogger("api_service.monitoring")

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/traces")
async def list_traces(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    status_filter: Optional[str] = Query(default=None, alias="status"),
    user=Depends(get_current_user),
) -> dict[str, Any]:
    """List investigation traces with aggregated durations, token metrics, and error counts."""
    store = get_monitoring_store()
    traces = store.get_traces_summary(limit=limit, offset=offset, status=status_filter)
    return {
        "limit": limit,
        "offset": offset,
        "count": len(traces),
        "traces": traces,
    }


@router.get("/traces/{trace_id}")
async def get_trace_detail(
    trace_id: str,
    user=Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve full hierarchical waterfall trace tree for a specific investigation."""
    store = get_monitoring_store()
    tree = store.get_trace_tree(trace_id)
    if not tree.get("spans"):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace with ID '{trace_id}' not found.",
        )
    return tree


@router.get("/stats")
async def get_stats(
    since_hours: int = Query(default=24, ge=1, le=720),
    user=Depends(get_current_user),
) -> dict[str, Any]:
    """Get aggregated latency, TTFT, token usage, and error rates grouped by component and step type."""
    store = get_monitoring_store()
    stats = store.get_component_stats(since_hours=since_hours)
    return {
        "since_hours": since_hours,
        "stats": stats,
    }


@router.get("/slowest")
async def get_slowest(
    limit: int = Query(default=20, ge=1, le=100),
    user=Depends(get_current_user),
) -> dict[str, Any]:
    """Retrieve the slowest spans across all traces."""
    store = get_monitoring_store()
    slowest = store.get_slowest_spans(limit=limit)
    return {
        "limit": limit,
        "spans": slowest,
    }


@router.post("/traces/{trace_id}/export")
async def export_trace(
    trace_id: str,
    user=Depends(get_current_user),
) -> dict[str, Any]:
    """Export the trace tree to an isolated JSON file in traces/investigations/."""
    store = get_monitoring_store()
    exported_path = store.export_trace_to_file(trace_id)
    if not exported_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Trace with ID '{trace_id}' not found or empty.",
        )
    return {
        "trace_id": trace_id,
        "exported_file": str(exported_path),
        "status": "exported",
    }

