from datetime import datetime
from typing import Literal, Optional

from pydantic import Field

from shared.models.base import MongoModel, utcnow

COLLECTION = "files"

FileStatus = Literal["pending_upload", "processing", "ready", "failed", "cancelled"]


class File(MongoModel):
    workspace_id: str
    filename: str
    file_type: str  
    storage_key: str
    size_bytes: Optional[int] = None
    status: FileStatus = "pending_upload"
    uploaded_at: datetime = Field(default_factory=utcnow)
    error: Optional[str] = None

    output_ref: Optional[str] = None
    schema_summary: Optional[dict] = None
    row_count: Optional[int] = None
    page_count: Optional[int] = None
    columns: Optional[list[str]] = None
    extracted_tables: list[dict] = Field(default_factory=list)

    # Set once, at presign time, from whatever the client sent - all files picked in the same
    # "Upload" click share one batch_id. Used by worker_service/tasks/ingestion.py to roll back
    # already-ingested siblings if a later file in the same batch fails ingestion because the
    # vector store rejected it for being too large.
    batch_id: Optional[str] = None

    # True for the sample dummy.csv/dummy.xlsx/dummy.pdf files seeded by shared/dummy_files.py
    # into a workspace that has no files of its own yet, so a brand-new user always has something
    # to ask questions about without uploading anything first. Never shown/used once the workspace
    # has at least one real (non-dummy) file - see list_files and _build_catalog.
    dummy: bool = False
