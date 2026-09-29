import asyncio
import logging
import os

from worker_service import engine_bootstrap  # noqa: F401

from analyzerEngine.ingestion.manager import IngestionManager
from analyzerEngine.ingestion.storage.local_store import LocalParquetStore
from analyzerEngine.vectordb.chroma_store import ChromaVectorStore

from shared.db import close_client, get_db
from shared.dummy_files import DUMMY_FILES, TEMPLATE_WORKSPACE_ID, dummy_file_id
from shared.models.file import COLLECTION as FILES
from shared.models.file import File

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("seed_dummy_template")


async def main() -> None:
    db = get_db()
    storage = LocalParquetStore(root_dir=engine_bootstrap.PARQUET_ROOT)
    vector_store = ChromaVectorStore()
    manager = IngestionManager(storage=storage, vector_store=vector_store)

    ok = 0
    for spec in DUMMY_FILES:
        filename = spec["filename"]
        sample_path = os.path.join(engine_bootstrap.ENGINE_DIR, "sample_data", filename)

        if not os.path.isfile(sample_path):
            logger.error("missing %s - put the real file at %s and re-run", filename, sample_path)
            continue

        file_id = dummy_file_id(TEMPLATE_WORKSPACE_ID, filename)
        result = manager.ingest_file(sample_path, TEMPLATE_WORKSPACE_ID, file_id)

        if result.status == "failed":
            logger.error("failed to ingest %s: %s", filename, "; ".join(result.errors or []))
            continue

        schema_summary = result.schema_summary or {}
        file = File(
            id=file_id,
            workspace_id=TEMPLATE_WORKSPACE_ID,
            filename=filename,
            file_type=spec["file_type"],
            storage_key=f"__sample__/{filename}",
            status="ready",
            output_ref=result.output_ref,
            schema_summary=schema_summary,
            row_count=result.row_count,
            page_count=schema_summary.get("page_count"),
            columns=schema_summary.get("columns"),
            extracted_tables=result.extracted_tables or [],
            dummy=True,
        )
        await db[FILES].replace_one({"_id": file.id}, file.to_mongo(), upsert=True)
        logger.info("seeded %s (status=%s, id=%s)", filename, result.status, file_id)
        ok += 1

    await close_client()
    logger.info("done: %d/%d sample file(s) seeded under workspace_id=%s",
                ok, len(DUMMY_FILES), TEMPLATE_WORKSPACE_ID)


if __name__ == "__main__":
    asyncio.run(main())
