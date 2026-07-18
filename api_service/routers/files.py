import os

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from api_service.deps import get_current_user, get_owned_file, get_owned_workspace
from shared.db import get_db
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.models.user import User
from shared.redis_client import get_arq_pool
from shared.storage import build_upload_key, delete_object, new_file_id, presign_put

router = APIRouter(tags=["files"])


class FileOut(BaseModel):
    id: str
    workspace_id: str
    filename: str
    file_type: str
    size_bytes: int | None
    status: str
    uploaded_at: str
    error: str | None
    row_count: int | None
    page_count: int | None


class PresignRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int | None = None


class PresignResponse(BaseModel):
    file_id: str
    upload_url: str
    storage_key: str


def _out(f: File) -> FileOut:
    return FileOut(
        id=f.id,
        workspace_id=f.workspace_id,
        filename=f.filename,
        file_type=f.file_type,
        size_bytes=f.size_bytes,
        status=f.status,
        uploaded_at=f.uploaded_at.isoformat(),
        error=f.error,
        row_count=f.row_count,
        page_count=f.page_count,
    )


@router.post("/workspaces/{workspace_id}/files/presign", response_model=PresignResponse)
async def presign_upload(
    workspace_id: str, body: PresignRequest, user: User = Depends(get_current_user)
):
    await get_owned_workspace(workspace_id, user)

    file_id = new_file_id()
    ext = os.path.splitext(body.filename)[1].lstrip(".").lower()
    storage_key = build_upload_key(workspace_id, file_id, body.filename)

    file = File(
        id=file_id,
        workspace_id=workspace_id,
        filename=body.filename,
        file_type=ext,
        storage_key=storage_key,
        size_bytes=body.size_bytes,
        status="pending_upload",
    )
    await get_db()[FILES].insert_one(file.to_mongo())

    upload_url = presign_put(storage_key, content_type=body.content_type)
    return PresignResponse(file_id=file_id, upload_url=upload_url, storage_key=storage_key)


@router.post("/files/{file_id}/confirm", response_model=FileOut)
async def confirm_upload(file_id: str, user: User = Depends(get_current_user)):
    file = await get_owned_file(file_id, user)

    await get_db()[FILES].update_one({"_id": file.id}, {"$set": {"status": "processing", "error": None}})
    file.status = "processing"
    file.error = None

    pool = await get_arq_pool()
    await pool.enqueue_job("run_ingestion", file_id=file.id)

    return _out(file)


@router.post("/files/{file_id}/cancel", response_model=FileOut)
async def cancel_upload(file_id: str, user: User = Depends(get_current_user)):
    file = await get_owned_file(file_id, user)

    await get_db()[FILES].update_one({"_id": file.id}, {"$set": {"status": "cancelled"}})
    file.status = "cancelled"

    # Best-effort cleanup of whatever was (partially) uploaded.
    try:
        delete_object(file.storage_key)
    except Exception:
        pass

    return _out(file)


@router.get("/workspaces/{workspace_id}/files", response_model=list[FileOut])
async def list_files(workspace_id: str, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    cursor = get_db()[FILES].find({"workspace_id": workspace_id}).sort("uploaded_at", 1)
    docs = await cursor.to_list(length=500)
    return [_out(File.from_mongo(d)) for d in docs]


@router.delete("/files/{file_id}")
async def delete_file(file_id: str, user: User = Depends(get_current_user)):
    file = await get_owned_file(file_id, user)

    try:
        delete_object(file.storage_key)
    except Exception:
        pass
    if file.output_ref:
        # Parquet output living in R2 under the engine's own key scheme -
        # best-effort, ingestion R2 store is keyed the same as storage_key's
        # bucket so delete_object works for it too.
        try:
            delete_object(file.output_ref)
        except Exception:
            pass

    await get_db()[FILES].delete_one({"_id": file.id})
    return {"ok": True}
