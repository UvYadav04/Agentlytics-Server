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
    output_ref: Optional[str] = None  # local parquet path (worker-local disk) or vector collection ref
    schema_summary: Optional[dict] = None
    row_count: Optional[int] = None
    page_count: Optional[int] = None
    columns: Optional[list[str]] = None
    # One entry per table docling's hybrid PDF pipeline extracted (see
    # ingestion/storage/local_store.py + PDFIngestor._extract_tables) - each
    # becomes its own FileCatalogEntry (file_type="table") when the worker
    # rebuilds the catalog for an investigation. Empty for csv/json files.
    extracted_tables: list[dict] = Field(default_factory=list)
