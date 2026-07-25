from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "files"

FileStatus = Literal["pending_upload", "processing", "ready", "failed", "cancelled"]


class File(MongoModel):
    workspace_id: str
    filename: str
    file_type: str  # extension without the dot, e.g. "csv", "pdf"
    storage_key: str  # R2 object key for the raw uploaded file
    size_bytes: Optional[int] = None
    status: FileStatus = "pending_upload"
    uploaded_at: datetime = Field(default_factory=utcnow)
    error: Optional[str] = None

    # Populated once ingestion completes - this is the shallow catalog the
    # orchestrator reads from (see agent_tools_specification.md Section 1.4).
    # For tabular data (csv/json/xlsx-table/pdf-table): the artifact's file_id - the same id as
    # this doc's own _id for a main file, or the table's own id for an extracted table (see
    # analyzerEngine/sandbox/path_resolver.py; physical location is always derived as
    # {workspace_id}/{file_id}.parquet, never stored as a path). For PDF/TXT main entries: a
    # vector-store pointer ("workspace_{id}"), not a file_id. "" for an xlsx workbook's own
    # main entry (no queryable data of its own - see xlsx_ingestor.py).
    output_ref: Optional[str] = None
    schema_summary: Optional[dict] = None
    row_count: Optional[int] = None
    page_count: Optional[int] = None
    columns: Optional[list[str]] = None
    # One entry per table docling's hybrid PDF pipeline extracted (see
    # ingestion/storage/local_store.py + PDFIngestor._extract_tables) - each
    # becomes its own FileCatalogEntry (file_type="table") when the worker
    # rebuilds the catalog for an investigation. Empty for csv/json files.
    extracted_tables: list[dict] = Field(default_factory=list)
