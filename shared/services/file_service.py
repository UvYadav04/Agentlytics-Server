import io
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal, Optional

import pandas as pd

from shared.file_preview import _cache, is_sample_key
from shared.storage import get_bucket_name, get_s3_client

logger = logging.getLogger("shared.services.file")


@dataclass
class FileEntry:
    file_id: str
    filename: str
    file_type: str
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    size_bytes: int = 0
    output_ref: Optional[str] = None
    row_count: Optional[int] = None
    page_count: Optional[int] = None
    columns: Optional[list[str]] = None
    tags: Optional[list[str]] = None
    storage_key: Optional[str] = None


class FileSystemService:
    """
    Unified workspace file system service.
    Provides standard operations to:
    - Get a file by id: get_by_id(file_id)
    - Get a file by name: get_by_name(filename)
    - Find files by prefix: find_by_prefix(prefix)
    - Read file contents: read_file(file_id, mode="dataframe"|"text"|"bytes")
    - Filter and list files: list_files(filters)
    """

    def __init__(self, catalog: Any = None, db: Any = None, storage_root: str = "data/parquet"):
        self.storage_root = storage_root
        self.db = db
        self._entries: dict[str, FileEntry] = {}

        if catalog is not None:
            self._import_from_catalog(catalog)

    def _import_from_catalog(self, catalog: Any) -> None:
        """Import existing entries from a FileCatalog instance."""
        entries_dict = getattr(catalog, "entries", {})
        for fid, raw in entries_dict.items():
            self._entries[fid] = FileEntry(
                file_id=raw.file_id,
                filename=raw.filename,
                file_type=raw.file_type,
                uploaded_at=getattr(raw, "uploaded_at", datetime.now(timezone.utc)),
                size_bytes=getattr(raw, "size_bytes", 0),
                output_ref=getattr(raw, "output_ref", None),
                row_count=getattr(raw, "row_count", None),
                page_count=getattr(raw, "page_count", None),
                columns=getattr(raw, "columns", None),
                tags=getattr(raw, "tags", None),
            )

    def add_entry(self, entry: FileEntry) -> None:
        """Register or update a file entry."""
        self._entries[entry.file_id] = entry

    def get_by_id(self, file_id: str) -> Optional[FileEntry]:
        """Fetch file metadata by its unique file_id."""
        return self._entries.get(file_id)

    def get_by_name(self, filename: str, exact: bool = False) -> list[FileEntry]:
        """
        Fetch file(s) by filename.
        If exact=True, checks exact string equality.
        If exact=False, checks case-insensitive equality or suffix.
        """
        target = filename.strip()
        matches = []
        for e in self._entries.values():
            if exact and e.filename == target:
                matches.append(e)
            elif not exact and (e.filename.lower() == target.lower() or target.lower() in e.filename.lower()):
                matches.append(e)
        return matches

    def find_by_prefix(self, prefix: str) -> list[FileEntry]:
        """
        Find files by prefix matching on filename or file_id.
        Useful for searching 'sales_', 'table_', 'chart_', etc.
        """
        pref = prefix.strip().lower()
        return [
            e for e in self._entries.values()
            if e.filename.lower().startswith(pref) or e.file_id.lower().startswith(pref)
        ]

    def list_files(self, filters: Optional[dict] = None) -> list[FileEntry]:
        """
        List queryable files matching optional filters:
        - name_contains: substring match
        - file_type: list or single type
        - min_rows / max_rows
        - tags: list of tags to require
        - uploaded_after / uploaded_before
        """
        filters = filters or {}
        results = []

        for entry in self._entries.values():
            if "name_contains" in filters and filters["name_contains"].lower() not in entry.filename.lower():
                continue
            if "file_type" in filters:
                types = filters["file_type"] if isinstance(filters["file_type"], (list, tuple)) else [filters["file_type"]]
                if entry.file_type not in types:
                    continue
            if "min_rows" in filters and (entry.row_count or 0) < filters["min_rows"]:
                continue
            if "max_rows" in filters and (entry.row_count or 0) > filters["max_rows"]:
                continue
            if "tags" in filters:
                entry_tags = entry.tags or []
                if not any(tag in entry_tags for tag in filters["tags"]):
                    continue
            results.append(entry)

        return results

    def read_file(
        self,
        file_id: str,
        workspace_id: str = "default",
        mode: Literal["dataframe", "text", "bytes"] = "dataframe",
    ) -> Any:
        """
        Read file contents in the requested format:
        - 'dataframe': loads parquet table into a pandas DataFrame
        - 'text': loads as string (from local parquet preview, text file, or S3)
        - 'bytes': loads raw file bytes
        """
        parquet_path = os.path.join(self.storage_root, workspace_id, f"{file_id}.parquet")

        if os.path.isfile(parquet_path):
            if mode == "dataframe":
                return pd.read_parquet(parquet_path)
            elif mode == "bytes":
                with open(parquet_path, "rb") as f:
                    return f.read()
            elif mode == "text":
                df = pd.read_parquet(parquet_path)
                return df.to_string(max_rows=50)

        # Fallback to S3 object if storage_key is present on the entry
        entry = self.get_by_id(file_id)
        if entry and entry.storage_key:
            data = _cache.get(entry.storage_key)
            if data is None:
                s3 = get_s3_client()
                bucket = get_bucket_name()
                resp = s3.get_object(Bucket=bucket, Key=entry.storage_key)
                data = resp["Body"].read()
                _cache.put(entry.storage_key, data)

            if mode == "bytes":
                return data
            elif mode == "text":
                return data.decode("utf-8", errors="replace")
            elif mode == "dataframe":
                if entry.file_type == "csv":
                    return pd.read_csv(io.BytesIO(data))
                elif entry.file_type in ("xlsx", "xls"):
                    return pd.read_excel(io.BytesIO(data))

        raise FileNotFoundError(f"File '{file_id}' could not be located in local storage or S3.")

    def exists(self, file_id: str, workspace_id: str = "default") -> bool:
        """Check if file exists in catalog and has reachable data."""
        if file_id not in self._entries:
            return False
        parquet_path = os.path.join(self.storage_root, workspace_id, f"{file_id}.parquet")
        if os.path.isfile(parquet_path):
            return True
        entry = self._entries[file_id]
        return bool(entry.storage_key)

