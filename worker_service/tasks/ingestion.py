import asyncio
import logging
import os
import shutil
import tempfile


from worker_service import engine_bootstrap
from analyzerEngine.ingestion.manager import IngestionManager
from analyzerEngine.sandbox.path_resolver import InvalidArtifactIdError, get_parquet_path

from shared.db import get_db
from shared.job_timing import log_job_finished, log_job_picked_up, now_iso
from shared.models.dashboard import COLLECTION as DASHBOARDS
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.storage import delete_object, get_bucket_name, get_s3_client
from shared.upload_limits import describe_limit, max_size_bytes

logger = logging.getLogger("worker.ingestion")


async def _mark_failed(db, file: File, error: str) -> None:
    logger.warning("ingestion failed for file %s: %s", file.id, error)
    await db[FILES].update_one(
        {"_id": file.id},
        {"$set": {"status": "failed", "error": error, "pages_done": None, "pages_total": None}},
    )


def _make_progress_reporter(db, loop, file_id: str):
    async def _update(done: int, total: int) -> None:
        try:
            await db[FILES].update_one(
                {"_id": file_id}, {"$set": {"pages_done": done, "pages_total": total}},
            )
        except Exception:
            logger.exception("run_ingestion: failed to write progress for file %s", file_id)

    def _report(done: int, total: int) -> None:
        try:
            asyncio.run_coroutine_threadsafe(_update(done, total), loop)
        except Exception:
            logger.exception("run_ingestion: failed to schedule progress update for file %s", file_id)

    return _report


def _delete_parquet_output(storage, workspace_id: str, artifact_id: str | None) -> None:
    if not artifact_id:
        return
    try:
        path = get_parquet_path(storage.root_dir, workspace_id, artifact_id)
    except InvalidArtifactIdError:
        return
    try:
        storage.delete(path)
    except Exception:
        logger.exception("rollback: failed to delete parquet output %s", path)


async def _clear_vectors(vector_store, file_id: str) -> None:
    try:
        old_chunks = await asyncio.to_thread(vector_store.get_by_filter, {"file_id": file_id})
        old_chunk_ids = [c.chunk_id for c in old_chunks]
        if old_chunk_ids:
            await asyncio.to_thread(vector_store.delete, old_chunk_ids)
    except Exception:
        logger.exception("failed to clear vector store entries for file %s", file_id)


def _cleanup_targets(file: File) -> tuple[list[str], bool]:
    pending = file.pending_cleanup or {}
    refs = [file.output_ref, *(pending.get("output_refs") or [])]
    refs += [t.get("output_ref") for t in (file.extracted_tables or [])]
    refs += [t.get("output_ref") for t in (pending.get("extracted_tables") or [])]
    has_previous = bool(file.output_ref or file.extracted_tables or pending)
    return sorted({ref for ref in refs if ref and not ref.startswith("workspace_")}), has_previous


def _schema_warning(previous: dict | None, file_id: str, columns: list | None, tables: list) -> str | None:
    if not previous:
        return None
    current: dict = {}
    if columns:
        current[file_id] = [str(c) for c in columns]
    for table in tables or []:
        if table.get("file_id") and isinstance(table.get("columns"), list):
            current[table["file_id"]] = [str(c) for c in table["columns"]]
    missing_tables = [key for key in previous if key != file_id and key not in current]
    missing_columns: list[str] = []
    for key, old_columns in previous.items():
        if key in current:
            missing_columns += [c for c in old_columns if c not in set(current[key])]
    parts = []
    if missing_columns:
        shown = ", ".join(sorted(set(missing_columns))[:8])
        more = len(set(missing_columns)) - 8
        parts.append(f"Columns no longer present: {shown}{f' (+{more} more)' if more > 0 else ''}.")
    if missing_tables:
        parts.append(f"{len(missing_tables)} table(s) from the previous version are gone.")
    if not parts:
        return None
    return " ".join(parts) + " Charts and dashboards that used them may need updating."


async def _refresh_dependent_dashboards(ctx, db, file_id: str) -> int:
    docs = await db[DASHBOARDS].find(
        {"real_time": True, "file_ids": file_id, "source": {"$nin": ["postgresql", "mongodb"]}},
        {"_id": 1},
    ).to_list(length=100)
    for doc in docs:
        try:
            await ctx["redis"].enqueue_job("refresh_dashboard", dashboard_id=doc["_id"], requested_at=now_iso())
        except Exception:
            logger.exception("failed to enqueue dashboard refresh for %s after file %s changed", doc["_id"], file_id)
    return len(docs)


async def purge_file_artifacts(ctx, workspace_id: str, file_id: str, cleanup: dict | None = None) -> None:
    cleanup = cleanup or {}
    storage = ctx["storage"]
    refs = [*(cleanup.get("output_refs") or []), *(t.get("output_ref") for t in cleanup.get("extracted_tables") or [])]
    for ref in {r for r in refs if r}:
        _delete_parquet_output(storage, workspace_id, ref)
    await _clear_vectors(ctx["vector_store"], file_id)
    logger.info("purge_file_artifacts: cleaned %d parquet output(s) and vectors for deleted file %s", len(refs), file_id)


async def _rollback_batch_siblings(ctx, db, file: File) -> list[str]:
    if not file.batch_id:
        return []

    siblings = await db[FILES].find({
        "batch_id": file.batch_id,
        "workspace_id": file.workspace_id,
        "status": "ready",
        "_id": {"$ne": file.id},
    }).to_list(length=200)

    if not siblings:
        return []

    storage = ctx["storage"]
    vector_store = ctx["vector_store"]
    rolled_back = []

    for doc in siblings:
        sibling = File.from_mongo(doc)
        try:
            chunks = await asyncio.to_thread(vector_store.get_by_filter, {"file_id": sibling.id})
            chunk_ids = [c.chunk_id for c in chunks]
            if chunk_ids:
                await asyncio.to_thread(vector_store.delete, chunk_ids)
        except Exception:
            logger.exception("rollback: failed to clear vector store entries for file %s", sibling.id)

        if sibling.file_type in ("csv", "xlsx"):
            _delete_parquet_output(storage, sibling.workspace_id, sibling.output_ref)
        for table in sibling.extracted_tables or []:
            _delete_parquet_output(storage, sibling.workspace_id, table.get("output_ref"))

        try:
            await asyncio.to_thread(delete_object, sibling.storage_key)
        except Exception:
            pass

        await db[FILES].delete_one({"_id": sibling.id})
        rolled_back.append(sibling.filename)
        logger.warning(
            "rollback: removed file %s (%s) from batch %s after sibling %s failed with "
            "vector_store_size_exceeded",
            sibling.id, sibling.filename, file.batch_id, file.id,
        )

    return rolled_back


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

        storage = ctx["storage"]
        vector_store = ctx["vector_store"]

        stale_refs, has_previous = _cleanup_targets(file)
        if has_previous:
            await _clear_vectors(vector_store, file.id)
            for ref in stale_refs:
                _delete_parquet_output(storage, file.workspace_id, ref)
            await db[FILES].update_one({"_id": file.id}, {"$set": {"pending_cleanup": None}})
            logger.info(
                "run_ingestion: cleared previous outputs for file %s (%d parquet output(s))",
                file.id, len(stale_refs),
            )

        tmp_dir = tempfile.mkdtemp(prefix="ingest_")
        local_path = os.path.join(tmp_dir, file.filename)
        s3 = get_s3_client()
        bucket = get_bucket_name()

        try:
            if file.dummy:
                sample_path = os.path.join(
                    engine_bootstrap.ENGINE_DIR, "sample_data", file.filename,
                )
                try:
                    shutil.copy(sample_path, local_path)
                except Exception as exc:
                    await _mark_failed(db, file, f"Failed to load sample file: {exc}")
                    status_for_log = "failed_download"
                    return
            else:
                try:
                    s3.download_file(bucket, file.storage_key, local_path)
                except Exception as exc:
                    await _mark_failed(db, file, f"Failed to download uploaded file from storage: {exc}")
                    status_for_log = "failed_download"
                    return

            limit = max_size_bytes(file.file_type)
            if limit is not None and os.path.getsize(local_path) > limit:
                actual_mb = os.path.getsize(local_path) / (1024 * 1024)
                await _mark_failed(
                    db, file, f"{describe_limit(file.file_type)} - this file is {actual_mb:.1f}MB.",
                )
                status_for_log = "failed_too_large"
                return

            manager = IngestionManager(storage=storage, vector_store=vector_store)

            progress_reporter = _make_progress_reporter(db, asyncio.get_running_loop(), file.id)
            result = await asyncio.to_thread(
                manager.ingest_file, local_path, file.workspace_id, file.id,
                progress_callback=progress_reporter,
            )
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True   )

        if result.status == "failed":
            error_message = "; ".join(result.errors) if result.errors else "Ingestion failed"
            if result.error_kind == "vector_store_size_exceeded":
                rolled_back = await _rollback_batch_siblings(ctx, db, file)
                if rolled_back:
                    error_message += (
                        f" - this file was too large to index, so the other file(s) uploaded "
                        f"alongside it ({', '.join(rolled_back)}) were removed too. Please "
                        f"re-upload them separately from this one."
                    )
                status_for_log = "failed_vector_store_size"
            else:
                status_for_log = "failed"
            await _mark_failed(db, file, error_message)
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
            "pages_done": None,
            "pages_total": None,
            "warning": _schema_warning(
                file.previous_columns, file.id, schema_summary.get("columns"), result.extracted_tables or [],
            ),
            "previous_columns": None,
        }
        await db[FILES].update_one({"_id": file.id}, {"$set": update})
        logger.info("ingestion complete for file %s (status=%s)", file.id, result.status)
        status_for_log = result.status
        if has_previous:
            refreshed = await _refresh_dependent_dashboards(ctx, db, file.id)
            if refreshed:
                logger.info("run_ingestion: queued refresh for %d dashboard(s) using file %s", refreshed, file.id)
    except Exception as exc:
        logger.exception("run_ingestion: unexpected failure for file %s", file_id)
        status_for_log = "failed_unexpected"
        if file is not None:
            await _mark_failed(db, file, f"Unexpected error during ingestion: {exc}")
    finally:
        log_job_finished(logger, "run_ingestion", picked_up_at, status=status_for_log, file_id=file_id)
