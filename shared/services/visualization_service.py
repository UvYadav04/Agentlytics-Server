import logging
import os
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from shared import usage
from shared.models.chart import COLLECTION as CHARTS, Chart
from shared.storage import build_chart_key, get_bucket_name, get_s3_client, new_file_id

logger = logging.getLogger("shared.services.visualization")


@dataclass
class StagedChart:
    chart_id: str
    workspace_id: str
    title: str
    chart_type: str
    location: str  # local path to rendered HTML file
    artifact_file_id: Optional[str] = None
    spec: Optional[dict] = None
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class VisualizationService:
    """
    Dedicated service to handle chart creation and local staging.
    Keeps full rendered chart payloads/HTML out of the LLM context window.
    Returns only lightweight metadata (chart_id, type, title) to the agent.
    When persistence is needed, fetches full data by chart_id and saves to S3/MongoDB.
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = output_dir
        self._staged_charts: dict[str, StagedChart] = {}
        os.makedirs(output_dir, exist_ok=True)

    def stage_chart(
        self,
        workspace_id: str,
        title: str,
        chart_type: str,
        location: str,
        chart_id: Optional[str] = None,
        artifact_file_id: Optional[str] = None,
        spec: Optional[dict] = None,
        **extra_meta: Any,
    ) -> dict:
        """
        Stage a locally generated chart by ID.
        Returns ONLY compact metadata to keep the LLM context window small.
        """
        cid = chart_id or f"chart_{chart_type}_{uuid.uuid4().hex[:10]}"
        staged = StagedChart(
            chart_id=cid,
            workspace_id=workspace_id,
            title=title or "Untitled chart",
            chart_type=chart_type,
            location=location,
            artifact_file_id=artifact_file_id,
            spec=spec,
            metadata=extra_meta,
        )
        self._staged_charts[cid] = staged
        logger.info("Staged chart %s (%s) at %s", cid, title, location)

        # Lean metadata returned to the caller / LLM turn
        return {
            "chart_id": cid,
            "title": staged.title,
            "chart_type": staged.chart_type,
            "artifact_file_id": staged.artifact_file_id,
            "status": "staged",
        }

    def get_by_id(self, chart_id: str) -> Optional[StagedChart]:
        """Fetch full staged chart record by its unique ID."""
        return self._staged_charts.get(chart_id)

    def list_staged(self, workspace_id: Optional[str] = None) -> list[StagedChart]:
        """List all staged charts, optionally filtered by workspace."""
        if workspace_id:
            return [c for c in self._staged_charts.values() if c.workspace_id == workspace_id]
        return list(self._staged_charts.values())

    async def persist_chart(
        self,
        chart_id: str,
        db,
        workspace_id: str,
        message_id: str,
        user_id: str,
        email: Optional[str] = None,
    ) -> Optional[str]:
        """
        Persist a single staged chart to S3 and MongoDB.
        Enforces user usage quotas and cleans up local temporary files.
        """
        staged = self.get_by_id(chart_id)
        if staged is None:
            logger.warning("persist_chart: chart_id %s not found in staged charts", chart_id)
            return None

        ref = staged.location
        if not isinstance(ref, str) or not os.path.isfile(ref):
            logger.warning("persist_chart: local file %s missing for chart %s", ref, chart_id)
            return None

        if not await usage.has_chart_capacity(user_id, email):
            logger.info("persist_chart: user %s reached chart capacity", user_id)
            return None

        s3 = get_s3_client()
        bucket = get_bucket_name()
        persisted_id = new_file_id()
        key = build_chart_key(workspace_id, persisted_id)

        try:
            s3.upload_file(ref, bucket, key, ExtraArgs={"ContentType": "text/html"})
            chart_doc = Chart(
                id=persisted_id,
                workspace_id=workspace_id,
                message_id=message_id,
                title=staged.title,
                storage_key=key,
            )
            await db[CHARTS].insert_one(chart_doc.to_mongo())
            await usage.increment_charts(user_id)
            logger.info("Successfully persisted chart %s -> %s (key=%s)", chart_id, persisted_id, key)
            return persisted_id
        except Exception:
            logger.exception("Failed to upload/persist chart %s (%s)", chart_id, ref)
            return None
        finally:
            shutil.rmtree(os.path.dirname(ref), ignore_errors=True)
            self._staged_charts.pop(chart_id, None)

    async def persist_all(
        self,
        db,
        workspace_id: str,
        message_id: str,
        user_id: str,
        email: Optional[str] = None,
    ) -> list[str]:
        """Persist all currently staged charts to S3/MongoDB."""
        saved_ids: list[str] = []
        for cid in list(self._staged_charts.keys()):
            saved_id = await self.persist_chart(cid, db, workspace_id, message_id, user_id, email)
            if saved_id:
                saved_ids.append(saved_id)
        return saved_ids

    def clear(self) -> None:
        """Clear all staged charts from memory."""
        self._staged_charts.clear()

