import asyncio
import logging
import os
import shutil
import uuid

from pymongo.errors import DuplicateKeyError

from analyzerEngine.sandbox.path_resolver import InvalidArtifactIdError, get_parquet_path
from analyzerEngine.tools.orchestrator.file_catalog import is_tabular_output_ref
from analyzerEngine.vectordb.schema import ChunkRecord

from shared.db import get_db
from shared.dummy_files import DUMMY_FILES, TEMPLATE_WORKSPACE_ID, dummy_file_id
from shared.job_timing import log_job_finished, log_job_picked_up
from shared.models.file import COLLECTION as FILES
from shared.models.file import File

logger = logging.getLogger("worker.dummy_files")


def _copy_parquet(storage, src_workspace_id: str, src_ref: str, dst_workspace_id: str, dst_ref: str) -> bool:
    try:
        src_path = get_parquet_path(storage.root_dir, src_workspace_id, src_ref)
        dst_path = get_parquet_path(storage.root_dir, dst_workspace_id, dst_ref)
    except InvalidArtifactIdError:
        logger.exception("clone_dummy_files: invalid artifact id copying %s -> %s", src_ref, dst_ref)
        return False
    if not os.path.exists(src_path):
        logger.warning("clone_dummy_files: template parquet missing at %s, skipping", src_path)
        return False
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    shutil.copy2(src_path, dst_path)
    return True


async def clone_dummy_files(ctx, workspace_id: str, requested_at: str | None = None) -> None:
    """The expensive part of the 3 sample files - Docling/LlamaParse PDF parsing, table
    extraction, pandas parsing - already happened exactly once, into TEMPLATE_WORKSPACE_ID (see
    shared/dummy_files.py's _ensure_template_ingested). Every other empty workspace just needs a
    workspace-scoped COPY of that already-computed output: the parquet bytes on disk, and the
    already-chunked text re-upserted into the vector store under new ids (embeddings are cheap to
    recompute - that's not the part being optimized away here). No PDF parsing, no LLM calls, no
    pandas - just file copies and a vector upsert, so this finishes in well under a second."""
    picked_up_at = log_job_picked_up(
        logger, ctx, "clone_dummy_files", requested_at=requested_at, workspace_id=workspace_id,
    )
    status_for_log = "unknown"
    db = get_db()
    try:
        existing = await db[FILES].count_documents({"workspace_id": workspace_id}, limit=1)
        if existing:
            logger.info("clone_dummy_files: workspace %s already has files, nothing to do", workspace_id)
            status_for_log = "skipped_not_empty"
            return

        template_docs = await db[FILES].find({
            "workspace_id": TEMPLATE_WORKSPACE_ID, "status": "ready",
        }).to_list(length=len(DUMMY_FILES))
        if len(template_docs) < len(DUMMY_FILES):
            # Template hasn't finished ingesting yet (or hasn't started) - nothing to clone yet.
            # The next ensure_dummy_files call for this workspace (next list_files poll, next
            # message) will enqueue another clone_dummy_files and pick it up once it's ready.
            logger.info(
                "clone_dummy_files: template not ready yet (%d/%d files), will retry later",
                len(template_docs), len(DUMMY_FILES),
            )
            status_for_log = "skipped_template_not_ready"
            return

        storage = ctx["storage"]
        vector_store = ctx["vector_store"]
        cloned = 0

        for doc in template_docs:
            template = File.from_mongo(doc)
            new_id = dummy_file_id(workspace_id, template.filename)

            table_ref_map: dict[str, str] = {}
            new_extracted_tables = []
            for index, table in enumerate(template.extracted_tables or []):
                old_ref = table.get("output_ref") or ""
                if not old_ref:
                    continue
                new_ref = f"{new_id}_table_{index}"
                if not _copy_parquet(storage, TEMPLATE_WORKSPACE_ID, old_ref, workspace_id, new_ref):
                    continue
                table_ref_map[old_ref] = new_ref
                new_table = dict(table)
                new_table["file_id"] = new_ref
                new_table["output_ref"] = new_ref
                new_extracted_tables.append(new_table)

            if is_tabular_output_ref(template.output_ref or ""):
                if _copy_parquet(storage, TEMPLATE_WORKSPACE_ID, template.output_ref, workspace_id, new_id):
                    new_output_ref = new_id
                else:
                    new_output_ref = ""
            elif template.file_type in ("pdf", "txt"):
                new_output_ref = f"workspace_{workspace_id}"
            else:
                new_output_ref = ""

            try:
                template_chunks = await asyncio.to_thread(
                    vector_store.get_by_filter, {"file_id": template.id}
                )
            except Exception:
                logger.exception("clone_dummy_files: failed to read template chunks for %s", template.id)
                template_chunks = []

            if template_chunks:
                new_chunks = []
                for chunk in template_chunks:
                    new_meta = dict(chunk.metadata)
                    old_table_ref = new_meta.get("table_ref")
                    if old_table_ref and old_table_ref in table_ref_map:
                        new_meta["table_ref"] = table_ref_map[old_table_ref]
                    new_chunks.append(ChunkRecord(
                        chunk_id=f"{new_id}_{uuid.uuid4().hex[:8]}",
                        file_id=new_id,
                        workspace_id=workspace_id,
                        text=chunk.text,
                        metadata=new_meta,
                    ))
                try:
                    await asyncio.to_thread(vector_store.upsert, new_chunks)
                except Exception:
                    logger.exception("clone_dummy_files: failed to upsert cloned chunks for %s", new_id)

            new_file = File(
                id=new_id,
                workspace_id=workspace_id,
                filename=template.filename,
                file_type=template.file_type,
                storage_key=f"__sample__/{template.filename}",
                size_bytes=template.size_bytes,
                status="ready",
                output_ref=new_output_ref,
                schema_summary=template.schema_summary,
                row_count=template.row_count,
                page_count=template.page_count,
                columns=template.columns,
                extracted_tables=new_extracted_tables,
                dummy=True,
            )
            try:
                await db[FILES].insert_one(new_file.to_mongo())
                cloned += 1
            except DuplicateKeyError:
                continue

        logger.info("clone_dummy_files: cloned %d/%d sample file(s) into workspace %s",
                    cloned, len(DUMMY_FILES), workspace_id)
        status_for_log = "success"
    except Exception:
        logger.exception("clone_dummy_files: unexpected failure for workspace %s", workspace_id)
        status_for_log = "failed_unexpected"
    finally:
        log_job_finished(logger, "clone_dummy_files", picked_up_at, status=status_for_log, workspace_id=workspace_id)
