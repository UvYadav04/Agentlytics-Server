import logging
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("shared.services.table")

_SAFE_IDENT_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _safe_segment(val: str, name: str = "segment") -> str:
    cleaned = str(val).strip()
    if not cleaned or not _SAFE_IDENT_RE.match(cleaned):
        raise ValueError(f"Invalid {name}: {val!r}")
    return cleaned


@dataclass
class StagedTable:
    file_id: str
    workspace_id: str
    name: str
    row_count: int
    columns: list[str]
    dtypes: dict[str, str] = field(default_factory=dict)
    column_kinds: dict[str, str] = field(default_factory=dict)
    preview: list[dict] = field(default_factory=list)
    file_path: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class TableArtifactService:
    """
    Dedicated service to handle intermediate tables, DataFrames, and Parquet data.
    Keeps large tabular dataframes and raw row dictionaries out of the LLM context window.
    Returns only lightweight metadata (file_id, name, columns, small preview) to the agent.
    When persistence or export is needed, fetches full data by file_id from the parquet store.
    """

    def __init__(self, root_dir: str = "data/parquet"):
        self.root_dir = root_dir
        self._staged_tables: dict[str, StagedTable] = {}
        os.makedirs(root_dir, exist_ok=True)

    def _parquet_path(self, workspace_id: str, file_id: str) -> str:
        safe_ws = _safe_segment(workspace_id, "workspace_id")
        safe_fid = _safe_segment(file_id, "file_id")
        return os.path.join(self.root_dir, safe_ws, f"{safe_fid}.parquet")

    def stage_dataframe(
        self,
        workspace_id: str,
        name: str,
        df: pd.DataFrame,
        file_id: Optional[str] = None,
    ) -> dict:
        """
        Save a DataFrame to parquet storage and stage its metadata.
        Returns ONLY compact metadata for the LLM context.
        """
        fid = file_id or f"table_{uuid.uuid4().hex[:12]}"
        full_path = self._parquet_path(workspace_id, fid)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)

        # De-duplicate any duplicated column names
        cleaned_df = df.copy()
        if cleaned_df.columns.duplicated().any():
            counts: dict[str, int] = {}
            new_cols = []
            for col in cleaned_df.columns:
                counts[col] = counts.get(col, 0) + 1
                new_cols.append(col if counts[col] == 1 else f"{col}__{counts[col]}")
            cleaned_df.columns = new_cols

        cleaned_df.to_parquet(full_path, index=False)

        columns = [str(c) for c in cleaned_df.columns]
        dtypes = {str(k): str(v) for k, v in cleaned_df.dtypes.items()}
        preview = cleaned_df.head(3).to_dict(orient="records")

        staged = StagedTable(
            file_id=fid,
            workspace_id=workspace_id,
            name=name or fid,
            row_count=len(cleaned_df),
            columns=columns,
            dtypes=dtypes,
            preview=preview,
            file_path=full_path,
        )
        self._staged_tables[fid] = staged
        logger.info("Staged table %s (%s) with %d rows at %s", fid, name, len(cleaned_df), full_path)

        return {
            "file_id": fid,
            "name": staged.name,
            "row_count": staged.row_count,
            "columns": staged.columns,
            "preview": staged.preview,
            "status": "staged",
        }

    def register_artifact(self, file_id: str, metadata: dict, workspace_id: str = "default") -> None:
        """Register an artifact that was persisted externally (e.g. by Python sandbox)."""
        full_path = self._parquet_path(workspace_id, file_id)
        self._staged_tables[file_id] = StagedTable(
            file_id=file_id,
            workspace_id=workspace_id,
            name=metadata.get("name") or file_id,
            row_count=metadata.get("row_count", 0),
            columns=metadata.get("columns", []),
            dtypes=metadata.get("dtypes", {}),
            column_kinds=metadata.get("column_kinds", {}),
            preview=metadata.get("preview", []),
            file_path=full_path if os.path.isfile(full_path) else None,
        )

    def get_by_id(self, file_id: str, workspace_id: str = "default") -> Optional[pd.DataFrame]:
        """Load full DataFrame by file ID."""
        path = self._parquet_path(workspace_id, file_id)
        if os.path.isfile(path):
            return pd.read_parquet(path)
        logger.warning("get_by_id: table %s not found on disk at %s", file_id, path)
        return None

    def get_metadata(self, file_id: str) -> Optional[StagedTable]:
        """Fetch metadata for a staged table."""
        return self._staged_tables.get(file_id)

    def exists(self, file_id: str, workspace_id: str = "default") -> bool:
        """Check whether a table's parquet file exists on disk."""
        path = self._parquet_path(workspace_id, file_id)
        return os.path.isfile(path)

    def export_csv(
        self,
        file_id: str,
        workspace_id: str = "default",
        output_dir: str = "data/reports",
        name: Optional[str] = None,
    ) -> str:
        """Export staged table to a CSV file."""
        df = self.get_by_id(file_id, workspace_id)
        if df is None:
            raise FileNotFoundError(f"Table '{file_id}' not found for export")
        folder = os.path.join(output_dir, name or f"export_{file_id}_{uuid.uuid4().hex[:6]}")
        os.makedirs(folder, exist_ok=True)
        csv_path = os.path.join(folder, "data.csv")
        df.to_csv(csv_path, index=False)
        return csv_path

    def clear(self) -> None:
        """Clear in-memory registry of staged tables."""
        self._staged_tables.clear()

