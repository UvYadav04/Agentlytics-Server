import logging
import os
import time

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from api_service.deps import get_current_user, get_owned_file, get_owned_workspace
from shared.db import get_db
from shared.dummy_files import ensure_dummy_files, mark_real_upload
from shared.job_timing import now_iso
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.models.user import User
from shared.redis_client import get_arq_pool
from shared.storage import (
    build_upload_key,
    delete_object,
    get_bucket_name,
    get_s3_client,
    new_file_id,
    presign_put,
)
from shared.upload_limits import describe_limit, is_supported_upload_extension, max_size_bytes

logger = logging.getLogger("api.files")

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
    dummy: bool = False
    pages_done: int | None = None
    pages_total: int | None = None
    source_url: str | None = None


class PresignRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int | None = None
    batch_id: str | None = None
    source_url: str | None = None


class PresignResponse(BaseModel):
    file_id: str
    upload_url: str
    storage_key: str


class FailRequest(BaseModel):
    error: str | None = None


class ReplacePresignRequest(BaseModel):
    filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int | None = None
    source_url: str | None = None


class PatchCsvRequest(BaseModel):
    content: str


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
        dummy=f.dummy,
        pages_done=f.pages_done,
        pages_total=f.pages_total,
        source_url=f.source_url,
    )


_INGESTION_RESET_FIELDS = {
    "output_ref": None,
    "schema_summary": None,
    "row_count": None,
    "page_count": None,
    "columns": None,
    "extracted_tables": [],
    "pages_done": None,
    "pages_total": None,
}


@router.post("/workspaces/{workspace_id}/files/presign", response_model=PresignResponse)
async def presign_upload(
    workspace_id: str, body: PresignRequest, user: User = Depends(get_current_user)
):
    await get_owned_workspace(workspace_id, user)

    ext = os.path.splitext(body.filename)[1].lstrip(".").lower()

    if not is_supported_upload_extension(ext):
        raise HTTPException(
            status_code=415,
            detail=f".{ext} files aren't supported. CSV, XLSX, PDF, and TXT files can be uploaded.",
        )

    limit = max_size_bytes(ext)
    if limit is not None and body.size_bytes is not None and body.size_bytes > limit:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{describe_limit(ext)} - this file is "
                f"{body.size_bytes / (1024 * 1024):.1f}MB."
            ),
        )

    file_id = new_file_id()
    storage_key = build_upload_key(workspace_id, file_id, body.filename)

    file = File(
        id=file_id,
        workspace_id=workspace_id,
        filename=body.filename,
        file_type=ext,
        storage_key=storage_key,
        size_bytes=body.size_bytes,
        status="pending_upload",
        batch_id=body.batch_id,
        source_url=body.source_url,
    )
    await get_db()[FILES].insert_one(file.to_mongo())

    upload_url = presign_put(storage_key)
    return PresignResponse(file_id=file_id, upload_url=upload_url, storage_key=storage_key)


@router.post("/files/{file_id}/confirm", response_model=FileOut)
async def confirm_upload(file_id: str, user: User = Depends(get_current_user)):
    request_received_at = now_iso()
    t0 = time.perf_counter()
    logger.info("confirm_upload: request arrived at %s (file_id=%s, user_id=%s)",
                request_received_at, file_id, user.id)

    file = await get_owned_file(file_id, user)

    await get_db()[FILES].update_one({"_id": file.id}, {"$set": {"status": "processing", "error": None}})
    file.status = "processing"
    file.error = None
    logger.info("confirm_upload: file %s marked processing at +%.1fms",
                file.id, (time.perf_counter() - t0) * 1000)

    if not file.dummy:
        await mark_real_upload(get_db(), file.workspace_id)

    pool = await get_arq_pool()
    job = await pool.enqueue_job("run_ingestion", file_id=file.id, requested_at=request_received_at)
    logger.info(
        "confirm_upload: file %s enqueued as arq job %s at +%.1fms total",
        file.id, getattr(job, "job_id", None), (time.perf_counter() - t0) * 1000,
    )

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


@router.post("/files/{file_id}/fail", response_model=FileOut)
async def fail_upload(file_id: str, body: FailRequest, user: User = Depends(get_current_user)):
    file = await get_owned_file(file_id, user)
    error_message = (body.error or "Upload failed")[:500]

    await get_db()[FILES].update_one(
        {"_id": file.id}, {"$set": {"status": "failed", "error": error_message}}
    )
    file.status = "failed"
    file.error = error_message

    try:
        delete_object(file.storage_key)
    except Exception:
        pass

    return _out(file)


@router.post("/files/{file_id}/replace/presign", response_model=PresignResponse)
async def replace_file_presign(
    file_id: str, body: ReplacePresignRequest, user: User = Depends(get_current_user)
):
    file = await get_owned_file(file_id, user)
    if file.dummy:
        raise HTTPException(status_code=400, detail="Sample files can't be replaced.")

    ext = os.path.splitext(body.filename)[1].lstrip(".").lower()
    if not is_supported_upload_extension(ext):
        raise HTTPException(
            status_code=415,
            detail=f".{ext} files aren't supported. CSV, XLSX, PDF, and TXT files can be uploaded.",
        )

    limit = max_size_bytes(ext)
    if limit is not None and body.size_bytes is not None and body.size_bytes > limit:
        raise HTTPException(
            status_code=413,
            detail=(
                f"{describe_limit(ext)} - this file is "
                f"{body.size_bytes / (1024 * 1024):.1f}MB."
            ),
        )

    new_storage_key = build_upload_key(file.workspace_id, file.id, body.filename)
    previous_storage_key = file.storage_key if file.storage_key != new_storage_key else None

    update = {
        "filename": body.filename,
        "file_type": ext,
        "storage_key": new_storage_key,
        "size_bytes": body.size_bytes,
        "status": "pending_upload",
        "error": None,
        "batch_id": None,
        "previous_storage_key": previous_storage_key,
    }
    if body.source_url is not None:
        update["source_url"] = body.source_url

    await get_db()[FILES].update_one({"_id": file.id}, {"$set": update})

    upload_url = presign_put(new_storage_key)
    return PresignResponse(file_id=file.id, upload_url=upload_url, storage_key=new_storage_key)


@router.post("/files/{file_id}/replace/confirm", response_model=FileOut)
async def replace_file_confirm(file_id: str, user: User = Depends(get_current_user)):
    file = await get_owned_file(file_id, user)
    if file.dummy:
        raise HTTPException(status_code=400, detail="Sample files can't be replaced.")

    if file.previous_storage_key:
        try:
            delete_object(file.previous_storage_key)
        except Exception:
            pass

    update = dict(_INGESTION_RESET_FIELDS)
    update.update({"status": "processing", "error": None, "previous_storage_key": None})
    await get_db()[FILES].update_one({"_id": file.id}, {"$set": update})
    for key, value in update.items():
        setattr(file, key, value)

    pool = await get_arq_pool()
    await pool.enqueue_job("run_ingestion", file_id=file.id, requested_at=now_iso())

    return _out(file)


@router.patch("/files/{file_id}/csv", response_model=FileOut)
async def patch_csv_file(file_id: str, body: PatchCsvRequest, user: User = Depends(get_current_user)):
    file = await get_owned_file(file_id, user)
    if file.dummy:
        raise HTTPException(status_code=400, detail="Sample files can't be edited.")
    if file.file_type != "csv":
        raise HTTPException(status_code=400, detail="Only CSV files can be edited directly.")

    data = body.content.encode("utf-8")
    limit = max_size_bytes("csv")
    if limit is not None and len(data) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"{describe_limit('csv')} - this file is {len(data) / (1024 * 1024):.1f}MB.",
        )

    get_s3_client().put_object(
        Bucket=get_bucket_name(), Key=file.storage_key, Body=data, ContentType="text/csv",
    )

    update = dict(_INGESTION_RESET_FIELDS)
    update.update({"status": "processing", "error": None, "size_bytes": len(data)})
    await get_db()[FILES].update_one({"_id": file.id}, {"$set": update})
    for key, value in update.items():
        setattr(file, key, value)

    pool = await get_arq_pool()
    await pool.enqueue_job("run_ingestion", file_id=file.id, requested_at=now_iso())

    return _out(file)


@router.get("/workspaces/{workspace_id}/files", response_model=list[FileOut])
async def list_files(workspace_id: str, user: User = Depends(get_current_user)):
    await get_owned_workspace(workspace_id, user)
    db = get_db()
    await ensure_dummy_files(db, workspace_id)
    cursor = db[FILES].find({"workspace_id": workspace_id}).sort("uploaded_at", 1)
    docs = await cursor.to_list(length=500)
    # Dummy files are only ever shown as a fallback - the instant the workspace has a real
    # (non-dummy) file, hide the samples entirely rather than listing both side by side.
    real_docs = [d for d in docs if not d.get("dummy")]
    docs = real_docs or docs
    return [_out(File.from_mongo(d)) for d in docs]


@router.delete("/files/{file_id}")
async def delete_file(file_id: str, user: User = Depends(get_current_user)):
    file = await get_owned_file(file_id, user)
    if file.dummy:
        raise HTTPException(status_code=400, detail="Sample files can't be deleted.")

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
