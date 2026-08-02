import asyncio
import logging
import os
import shutil
import tempfile


from worker_service import engine_bootstrap  
from analyzerEngine.ingestion.manager import IngestionManager

from shared.db import get_db
from shared.job_timing import log_job_finished, log_job_picked_up
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.storage import delete_object, get_bucket_name, get_s3_client
from shared.upload_limits import describe_limit, max_size_bytes

logger = logging.getLogger("worker.ingestion")


async def _mark_failed(db, file: File, error: str) -> None:
    logger.warning("ingestion failed for file %s: %s", file.id, error)
    await db[FILES].update_one({"_id": file.id}, {"$set": {"status": "failed", "error": error}})


async def run_ingestion(ctx, file_id: str, requested_at: str | None = None) -> None:
    picked_up_at = log_job_picked_up(logger, ctx, "run_ingestion", requested_at=requested_at, file_id=file_id)
    status_for_log = "unknown"
    db = get_db()
    file: File | None = None
    try:
        doc = await db[FILES].find_one({"_id": file_id})
        if doc is None:
            logger.warning("run_ingestion: file %s no longer exists, skipping", file_id)
            status_for_log = "skipped_missing"
            return

        file = File.from_mongo(doc)
        if file.status == "cancelled":
            logger.info("run_ingestion: file %s was cancelled before processing started", file_id)
            status_for_log = "skipped_cancelled"
            return

        tmp_dir = tempfile.mkdtemp(prefix="ingest_")
        local_path = os.path.join(tmp_dir, file.filename)
        s3 = get_s3_client()
        bucket = get_bucket_name()

        try:
            try:
                s3.download_file(bucket, file.storage_key, local_path)
            except Exception as exc:
                await _mark_failed(db, file, f"Failed to download uploaded file from storage: {exc}")
                status_for_log = "failed_download"
                return

            # Re-check the actual downloaded size against the same cap presign_upload enforces -
            # that check only trusts whatever size_bytes the client reported at presign time, so
            # this catches a client that lied (or, since size_bytes is optional, omitted it
            # entirely and skipped the check altogether).
            limit = max_size_bytes(file.file_type)
            if limit is not None and os.path.getsize(local_path) > limit:
                actual_mb = os.path.getsize(local_path) / (1024 * 1024)
                await _mark_failed(
                    db, file, f"{describe_limit(file.file_type)} - this file is {actual_mb:.1f}MB.",
                )
                status_for_log = "failed_too_large"
                return

            manager = IngestionManager(storage=ctx["storage"], vector_store=ctx["vector_store"])

            result = await asyncio.to_thread(manager.ingest_file, local_path, file.workspace_id, file.id)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True   )

        if result.status == "failed":
            await _mark_failed(db, file, "; ".join(result.errors) if result.errors else "Ingestion failed")
            status_for_log = "failed"
            return

        schema_summary = result.schema_summary or {}
        update = {
            "status": "ready",
            "output_ref": result.output_ref,
            "schema_summary": schema_summary,
            "row_count": result.row_count,
            "page_count": schema_summary.get("page_count"),
            "columns": schema_summary.get("columns"),
            "extracted_tables": result.extracted_tables or [],
            "error": "; ".join(result.errors) if result.errors else None,
        }
        await db[FILES].update_one({"_id": file.id}, {"$set": update})
        logger.info("ingestion complete for file %s (status=%s)", file.id, result.status)
        status_for_log = result.status
        try:
            await asyncio.to_thread(delete_object, file.storage_key)
            logger.info("run_ingestion: deleted raw upload from S3 for file %s (key=%s)",
                        file.id, file.storage_key)
        except Exception:
            logger.exception("run_ingestion: failed to delete raw upload from S3 for file %s (key=%s)",
                              file.id, file.storage_key)
    except Exception as exc:
        # Belt-and-suspenders on top of ingest_file's own try/except (manager.py): anything else
        # unexpected in this function - IngestionManager construction, the "mark ready" update,
        # etc. - previously propagated straight out to arq uncaught, leaving the File stuck at
        # status="processing" forever (arq would retry/eventually give up on the *job*, but
        # nothing ever told the File doc itself). Always mark it failed instead.
        logger.exception("run_ingestion: unexpected failure for file %s", file_id)
        status_for_log = "failed_unexpected"
        if file is not None:
            await _mark_failed(db, file, f"Unexpected error during ingestion: {exc}")
    finally:
        log_job_finished(logger, "run_ingestion", picked_up_at, status=status_for_log, file_id=file_id)
