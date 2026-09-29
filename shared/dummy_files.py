import logging

from pymongo.errors import DuplicateKeyError

from shared.job_timing import now_iso
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.models.workspace import COLLECTION as WORKSPACES
from shared.redis_client import get_arq_pool

logger = logging.getLogger("shared.dummy_files")

DUMMY_STORAGE_PREFIX = "__sample__"

TEMPLATE_WORKSPACE_ID = "__dummy_template__"

DUMMY_FILES = [
    {"filename": "dummy.csv", "file_type": "csv"},
    {"filename": "dummy.xlsx", "file_type": "xlsx"},
    {"filename": "dummy.pdf", "file_type": "pdf"},
]


def dummy_file_id(workspace_id: str, filename: str) -> str:
    return f"dummy-{workspace_id}-{filename.replace('.', '-')}"


async def _ensure_template_ingested(db) -> None:
    existing = await db[FILES].count_documents({"workspace_id": TEMPLATE_WORKSPACE_ID}, limit=1)
    if existing:
        return

    pool = await get_arq_pool()
    for spec in DUMMY_FILES:
        file = File(
            id=dummy_file_id(TEMPLATE_WORKSPACE_ID, spec["filename"]),
            workspace_id=TEMPLATE_WORKSPACE_ID,
            filename=spec["filename"],
            file_type=spec["file_type"],
            storage_key=f"{DUMMY_STORAGE_PREFIX}/{spec['filename']}",
            status="pending_upload",
            dummy=True,
        )
        try:
            await db[FILES].insert_one(file.to_mongo())
        except DuplicateKeyError:
            continue
        except Exception:
            logger.exception("_ensure_template_ingested: failed to create template File doc for %s", spec["filename"])
            continue
        await pool.enqueue_job("run_ingestion", file_id=file.id, requested_at=now_iso())
    logger.info("_ensure_template_ingested: template ingestion kicked off (workspace=%s)", TEMPLATE_WORKSPACE_ID)


async def ensure_dummy_files(db, workspace_id: str) -> None:
    if workspace_id == TEMPLATE_WORKSPACE_ID:
        return

    ws_doc = await db[WORKSPACES].find_one({"_id": workspace_id}, {"has_uploaded_file": 1})
    if ws_doc and ws_doc.get("has_uploaded_file"):
        return

    existing = await db[FILES].count_documents({"workspace_id": workspace_id}, limit=1)
    if existing:
        return

    await _ensure_template_ingested(db)

    pool = await get_arq_pool()
    await pool.enqueue_job("clone_dummy_files", workspace_id=workspace_id, requested_at=now_iso())


async def mark_real_upload(db, workspace_id: str) -> None:
    result = await db[WORKSPACES].update_one(
        {"_id": workspace_id, "has_uploaded_file": {"$ne": True}},
        {"$set": {"has_uploaded_file": True}},
    )
    if result.modified_count:
        await db[FILES].delete_many({"workspace_id": workspace_id, "dummy": True})
