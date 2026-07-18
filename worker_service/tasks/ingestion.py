"""arq job: run_ingestion(ctx, file_id).

Pulls the raw uploaded file from R2, hands it to the engine's
IngestionManager (Parquet output written to the worker's local disk via
LocalParquetStore - see engine_bootstrap.PARQUET_ROOT for why this isn't
R2 - PDF chunks written to the Chroma vector store), and updates the File
document's status/catalog fields in Mongo.

No Redis pub/sub event streaming here - that's the Investigation flow
(tasks/investigation.py). File status is polled/refetched by the frontend
via GET /workspaces/{id}/files.
"""
import asyncio
import logging
import os
import shutil
import tempfile

# Must run before any `from ingestion...` / `from vectordb...` import below -
# this is what makes analyzerEngine's top-level-style imports resolve.
from worker_service import engine_bootstrap  # noqa: F401

from analyzerEngine.ingestion.manager import IngestionManager
from analyzerEngine.ingestion.storage.local_store import LocalParquetStore
from analyzerEngine.vectordb.chroma_store import ChromaVectorStore

from shared.db import get_db
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.storage import get_bucket_name, get_s3_client

logger = logging.getLogger("worker.ingestion")


async def _mark_failed(db, file: File, error: str) -> None:
    logger.warning("ingestion failed for file %s: %s", file.id, error)
    await db[FILES].update_one({"_id": file.id}, {"$set": {"status": "failed", "error": error}})


async def run_ingestion(ctx, file_id: str) -> None:
    db = get_db()
    doc = await db[FILES].find_one({"_id": file_id})
    if doc is None:
        logger.warning("run_ingestion: file %s no longer exists, skipping", file_id)
        return

    file = File.from_mongo(doc)
    if file.status == "cancelled":
        logger.info("run_ingestion: file %s was cancelled before processing started", file_id)
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
            return

        storage = LocalParquetStore(root_dir=engine_bootstrap.PARQUET_ROOT)
        vector_store = ChromaVectorStore()
        manager = IngestionManager(storage=storage, vector_store=vector_store)

        # ingest_file() is fully synchronous (pandas/docling/chromadb calls) -
        # run it off the event loop so it doesn't block other jobs or the
        # worker's health checks.
        result = await asyncio.to_thread(manager.ingest_file, local_path, file.workspace_id, file.id)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    if result.status == "failed":
        await _mark_failed(db, file, "; ".join(result.errors) if result.errors else "Ingestion failed")
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
        # status == "partial" (e.g. scanned PDF) still counts as ready, but
        # keep the warning visible rather than silently dropping it.
        "error": "; ".join(result.errors) if result.errors else None,
    }
    await db[FILES].update_one({"_id": file.id}, {"$set": update})
    logger.info("ingestion complete for file %s (status=%s)", file.id, result.status)
