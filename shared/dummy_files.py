import logging

from pymongo.errors import DuplicateKeyError

from shared.job_timing import now_iso
from shared.models.file import COLLECTION as FILES
from shared.models.file import File
from shared.redis_client import get_arq_pool

logger = logging.getLogger("shared.dummy_files")

DUMMY_STORAGE_PREFIX = "__sample__"

# Not a real Workspace doc - no user owns it, get_owned_workspace never matches it, it's purely
# an internal storage namespace. The 3 sample files are fully ingested (parsed, chunked, table-
# extracted, embedded) exactly ONCE under this id, ever. Every empty user workspace after that
# gets a cheap CLONE of that already-computed result (see worker_service/tasks/dummy_files.py)
# rather than re-running the whole ingestion pipeline again.
TEMPLATE_WORKSPACE_ID = "__dummy_template__"

DUMMY_FILES = [
    {"filename": "dummy.csv", "file_type": "csv"},
    {"filename": "dummy.xlsx", "file_type": "xlsx"},
    {"filename": "dummy.pdf", "file_type": "pdf"},
]


def dummy_file_id(workspace_id: str, filename: str) -> str:
    """Deterministic id shared by the template docs (under TEMPLATE_WORKSPACE_ID) and every
    per-workspace clone (see worker_service/tasks/dummy_files.py) - lets both rely on Mongo's
    unique-_id constraint instead of a lock to stay safe under concurrent callers."""
    return f"dummy-{workspace_id}-{filename.replace('.', '-')}"


async def _ensure_template_ingested(db) -> None:
    """Idempotent - only actually does anything the very first time this is ever called across
    the app's lifetime. Deterministic ids mean a second concurrent caller's insert_one just hits
    a DuplicateKeyError and is ignored, same race-avoidance as ensure_dummy_files below."""
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
    """No-op the instant the workspace has ANY File doc at all (real or already-cloned dummy) -
    cheap enough to call from every list_files/send_message request. Otherwise makes sure the
    shared template has been (or is being) ingested, then enqueues a fast clone of it into this
    workspace - see worker_service/tasks/dummy_files.py's clone_dummy_files. If the template
    itself isn't ready yet (only possible for whichever workspace happens to be first ever), the
    clone job just no-ops and the next ensure_dummy_files call (list_files polls, next message)
    picks it up once the template has finished."""
    if workspace_id == TEMPLATE_WORKSPACE_ID:
        return

    existing = await db[FILES].count_documents({"workspace_id": workspace_id}, limit=1)
    if existing:
        return

    await _ensure_template_ingested(db)

    pool = await get_arq_pool()
    await pool.enqueue_job("clone_dummy_files", workspace_id=workspace_id, requested_at=now_iso())
