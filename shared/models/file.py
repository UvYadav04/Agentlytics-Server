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
