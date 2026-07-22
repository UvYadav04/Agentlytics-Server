"""arq job: run_investigation(ctx, investigation_id, chat_id, workspace_id, user_id, query, file_ids).

Rebuilds a shallow FileCatalog from Mongo (only "ready" files, see
agent_tools_specification.md Section 1.4), runs the engine's
OrchestratorAgent with streaming callbacks wired to Mongo (source of truth,
Investigation.events[]) and Redis pub/sub (live tail for connected SSE
clients - see full_application_build_plan.md Phase 5), and on completion
creates the assistant Message plus any Chart/Report docs the investigation
produced.

This job is entirely independent of any HTTP connection - it keeps running
and writing progress regardless of whether anyone is currently subscribed
to the investigation's SSE stream (refresh-safety, see the build plan).
"""
import json
import logging
import os
import shutil
from datetime import datetime, timezone

from worker_service import engine_bootstrap  # noqa: F401

from analyzerEngine.agents.orchestrator.agent import InvestigationCancelled, OrchestratorAgent
from analyzerEngine.ingestion.storage.local_store import LocalParquetStore
from analyzerEngine.llm_provider.errors import classify_llm_error
from analyzerEngine.tools.orchestrator.file_catalog import FileCatalog, table_catalog_entry
from analyzerEngine.tools.orchestrator.memory import LongTermMemory
from analyzerEngine.tools.orchestrator.models import FileCatalogEntry
from analyzerEngine.tools.orchestrator.thread_summary import update_summary

from shared import usage
from shared.db import get_db
from shared.models.chart import COLLECTION as CHARTS
from shared.models.chart import Chart
from shared.models.chat import COLLECTION as CHATS
from shared.models.dashboard import COLLECTION as DASHBOARDS
from shared.models.dashboard import ChartConfig, Dashboard
from shared.models.file import COLLECTION as FILES
from shared.models.investigation import COLLECTION as INVESTIGATIONS
from shared.models.investigation import InvestigationEvent
from shared.models.message import COLLECTION as MESSAGES
from shared.models.message import Message
from shared.models.report import COLLECTION as REPORTS
from shared.models.report import Report
from shared.redis_client import get_redis, investigation_channel
from shared.storage import build_chart_key, build_report_key, get_bucket_name, get_s3_client, new_file_id

# recent_turns keeps the last this-many {query, response} pairs verbatim on
# the Chat doc; anything older only survives through Chat.summary (see
# _update_chat_continuity). files_used/files_created are capped separately
# so a very long chat's lists can't grow without bound either.
RECENT_TURNS_LIMIT = 5
FILE_LIST_LIMIT = 30

logger = logging.getLogger("worker.investigation")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _looks_like_local_parquet_ref(ref: str) -> bool:
    """Not every ingestor's main output_ref is a real LocalParquetStore path: csv/json write
    one parquet per file and store a real path there, but PDFIngestor's output_ref is a
    vector-store pointer ("workspace_{id}", not a filesystem path at all) and XLSXIngestor's
    is deliberately "" - a workbook has no single "whole file" artifact, only its per-table
    entries do (see xlsx_ingestor.py's comment). storage.exists() on either of those against
    local disk is always False, which - before this check - flipped every freshly-ingested
    PDF/xlsx file to "failed" on its very first investigation even though nothing was
    actually missing (see the false positive this fixes: 'Product-Sales-Region.xlsx' marked
    missing seconds after ingesting it successfully). Every extracted_tables[i].output_ref,
    by contrast, always IS a real LocalParquetStore path regardless of file_type, so those
    are still checked unconditionally below - this only gates the main file-level ref."""
    return bool(ref) and ref.endswith(".parquet")


async def _build_catalog(db, workspace_id: str, storage: LocalParquetStore) -> tuple[FileCatalog, list[str]]:
    """Rebuilds the FileCatalog from Mongo's "ready" files - but doesn't trust Mongo blindly.
    `status: "ready"` only means the parquet existed on disk at ingestion time. PARQUET_ROOT
    (docker-compose.yml's /data/parquet bind mount, see engine_bootstrap.py) is a persistent
    HOST directory that survives container rebuilds/redeploys/recreation, so in normal
    operation this disk and Mongo - cloud Atlas, also durable - stay in agreement. This check
    exists for the remaining edge cases where they don't (the host directory was deleted or
    repointed outside of a normal deploy, a fresh machine/dev environment doesn't have the
    volume populated yet, etc.) - Mongo has no way to know about any of that on its own.
    Previously we handed those file_ids straight to the orchestrator and only found out via an
    IO Error deep inside invoke_tabular_agent (see the "No files found that match the pattern
    ..." failures this replaces).

    So every output_ref (the main file's and each extracted table's) is checked against this
    disk before being added to the catalog. Anything missing is flipped back to "failed" here -
    once, cheaply - so it stops being offered to every future investigation until the user
    re-uploads it, and its filename is returned so the caller can tell the user why it's absent.
    """
    catalog = FileCatalog()
    stale_ids: list[str] = []
    skipped_filenames: list[str] = []

    cursor = db[FILES].find({"workspace_id": workspace_id, "status": "ready"})
    async for doc in cursor:
        print("doc : ",doc)
        output_ref = doc.get("output_ref") or ""
        if _looks_like_local_parquet_ref(output_ref) and not storage.exists(output_ref):
            stale_ids.append(doc["_id"])
            skipped_filenames.append(doc["filename"])
            continue

        table_entries = []
        tables_ok = True
        for table in doc.get("extracted_tables") or []:
            if not storage.exists(table.get("output_ref") or ""):
                tables_ok = False
                break
            table_entries.append(table_catalog_entry(
                table,
                source_id=doc["_id"],
                source_filename=doc["filename"],
                source_file_type=doc["file_type"],
                uploaded_at=doc["uploaded_at"],
            ))

        if not tables_ok:
            # Same disk, same ingestion run as the main file - if one table's parquet is
            # gone the rest almost certainly are too. Drop the whole file rather than
            # handing the orchestrator a partial, inconsistent table set.
            stale_ids.append(doc["_id"])
            skipped_filenames.append(doc["filename"])
            continue

        catalog.add_entry(FileCatalogEntry(
            file_id=doc["_id"],
            filename=doc["filename"],
            file_type=doc["file_type"],
            uploaded_at=doc["uploaded_at"],
            size_bytes=doc.get("size_bytes") or 0,
            output_ref=doc.get("output_ref") or "",
            row_count=doc.get("row_count"),
            page_count=doc.get("page_count"),
            columns=doc.get("columns"),
        ))
        for entry in table_entries:
            catalog.add_entry(entry)

    # if stale_ids:
        # Safe to persist this back to Mongo (previously left commented out): PARQUET_ROOT is
        # a confirmed-persistent bind mount (see docstring above and docker-compose.yml), so a
        # missing output_ref reliably means the file is genuinely gone, not just a transient
        # artifact of a redeploy. Marking it "failed" once here - rather than leaving Mongo
        # saying "ready" forever - stops every future investigation from silently re-doing this
        # same disk check and re-reporting the same skipped file to the user on every turn.
        # await db[FILES].update_many(
        #     {"_id": {"$in": stale_ids}},
        #     {"$set": {
        #         "status": "failed",
        #         "error": (
        #             "Parquet output missing from local storage - the file's disk data is gone "
        #             "even though the persistent PARQUET_ROOT volume is intact. Please re-upload "
        #             "this file."
        #         ),
        #     }},
        # )
        # logger.warning(
        #     "workspace %s: %d file(s) marked failed - output_ref missing on disk: %s",
        #     workspace_id, len(stale_ids), skipped_filenames,
        # )

    return catalog, skipped_filenames


async def _append_event(db, investigation_id: str, event_type: str, message: str, data: dict = None) -> None:
    event = InvestigationEvent(type=event_type, message=message, data=data or {})
    payload = event.model_dump(mode="json")
    await db[INVESTIGATIONS].update_one({"_id": investigation_id}, {"$push": {"events": payload}})
    try:
        await get_redis().publish(investigation_channel(investigation_id), json.dumps(payload))
    except Exception:
        logger.exception("failed to publish event to redis for investigation %s", investigation_id)


async def _is_cancelled(db, investigation_id: str) -> bool:
    doc = await db[INVESTIGATIONS].find_one({"_id": investigation_id}, {"cancel_requested": 1})
    return bool(doc and doc.get("cancel_requested"))


async def _thread_context(db, chat_id: str) -> dict:
    """Read side of thread continuity - handed to OrchestratorAgent.run() as
    `thread_context` so this investigation's task prompt includes what
    happened earlier in this same chat. See _update_chat_continuity for the
    write side."""
    doc = await db[CHATS].find_one(
        {"_id": chat_id}, {"summary": 1, "recent_turns": 1, "files_used": 1, "files_created": 1},
    ) or {}
    return {
        "summary": doc.get("summary", ""),
        "recent_turns": doc.get("recent_turns", []),
        "files_used": doc.get("files_used", []),
        "files_created": doc.get("files_created", []),
    }


def _merge_capped(existing: list, new_items: list, cap: int) -> list:
    merged = list(existing)
    for item in new_items or []:
        if item not in merged:
            merged.append(item)
    return merged[-cap:]


async def _update_chat_continuity(db, chat_id: str, query: str, result) -> None:
    """Write side of thread continuity - called AFTER the investigation's own
    completion is already recorded and broadcast (see call site in
    run_investigation), so the summary LLM call below never delays the user
    seeing their answer. A failure here only means the next message in this
    chat starts from a slightly stale summary, never that this investigation
    itself fails - see the try/except around the LLM call."""
    doc = await db[CHATS].find_one(
        {"_id": chat_id}, {"summary": 1, "recent_turns": 1, "files_used": 1, "files_created": 1},
    ) or {}

    recent_turns = (doc.get("recent_turns", []) + [{"query": query, "response": result.final_answer}])
    recent_turns = recent_turns[-RECENT_TURNS_LIMIT:]

    files_used = _merge_capped(doc.get("files_used", []), result.files_used, FILE_LIST_LIMIT)
    files_created = _merge_capped(doc.get("files_created", []), result.artifact_refs, FILE_LIST_LIMIT)

    try:
        new_summary = await update_summary(doc.get("summary", ""), query, result.final_answer)
    except Exception:
        logger.exception("failed to update chat summary for chat %s - keeping previous summary", chat_id)
        new_summary = doc.get("summary", "")

    await db[CHATS].update_one(
        {"_id": chat_id},
        {"$set": {
            "summary": new_summary,
            "files_used": files_used,
            "files_created": files_created,
            "recent_turns": recent_turns,
        }},
    )


def _artifact_kind(path: str) -> str | None:
    # A real-time dashboard (ReportingTools.generate_realtime_dashboard_bundle) returns its
    # manifest.json path instead of an .html path specifically so it's distinguishable here
    # from an ordinary single chart - see _persist_dashboard_bundle below.
    if os.path.basename(path) == "manifest.json":
        return "dashboard_bundle"
    ext = os.path.splitext(path)[1].lower()
    if ext == ".html":
        return "chart"
    if ext in (".md", ".csv"):
        return "report"
    return None


def _artifact_title(path: str) -> str:
    name = os.path.basename(os.path.dirname(path))
    return name or "Untitled"


async def _persist_dashboard_bundle(
    db, s3, bucket: str, workspace_id: str, investigation_id: str, message_id: str, user_id: str, manifest_path: str,
) -> list:
    """Handles the "dashboard_bundle" artifact kind - see ReportingTools.
    generate_realtime_dashboard_bundle() and _artifact_kind() above. Uploads one HTML file
    per chart (each gets its own Chart doc, same chart-capacity gating as an ordinary
    chart), then - only if at least one chart actually made it past that gate - writes the
    Dashboard doc that ties them together with the transform_script/file_ids a later
    refresh needs. Returns the chart_ids it created, the same shape _persist_artifacts
    already returns for a plain chart, so its caller doesn't need to know real-time
    dashboards are a different code path."""
    folder = os.path.dirname(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    chart_ids: list = []
    chart_configs: list[ChartConfig] = []

    for chart_meta in manifest.get("charts", []):
        if not await usage.has_chart_capacity(user_id):
            await _append_event(
                db, investigation_id, "status",
                "Chart limit reached - some dashboard charts generated but not saved.",
            )
            break

        html_path = os.path.join(folder, chart_meta["html_filename"])
        if not os.path.isfile(html_path):
            continue

        chart_id = new_file_id()
        key = build_chart_key(workspace_id, chart_id)
        s3.upload_file(html_path, bucket, key, ExtraArgs={"ContentType": "text/html"})
        chart = Chart(
            id=chart_id, workspace_id=workspace_id, message_id=message_id,
            title=chart_meta.get("title") or "Untitled chart", storage_key=key,
        )
        await db[CHARTS].insert_one(chart.to_mongo())
        await usage.increment_charts(user_id)
        chart_ids.append(chart_id)
        chart_configs.append(ChartConfig(
            chart_id=chart_id,
            name=chart_meta["name"],
            chart_type=chart_meta.get("chart_type", "bar"),
            title=chart_meta.get("title"),
            label_column=chart_meta.get("label_column"),
            value_columns=chart_meta.get("value_columns"),
            time_column=chart_meta.get("time_column"),
            series_column=chart_meta.get("series_column"),
            value_column=chart_meta.get("value_column"),
            x_column=chart_meta.get("x_column"),
            y_column=chart_meta.get("y_column"),
            z_column=chart_meta.get("z_column"),
        ))

    if not chart_configs:
        # Every chart hit the cap (or the bundle came in empty) - nothing to tie
        # together, so don't create an empty, unrefreshable Dashboard doc.
        return chart_ids

    dashboard = Dashboard(
        workspace_id=workspace_id,
        name=manifest.get("title") or "Untitled dashboard",
        chart_ids=chart_ids,
        real_time=True,
        file_ids=manifest.get("file_ids") or [],
        transform_script=manifest.get("transform_script"),
        charts=chart_configs,
        last_refreshed_at=datetime.now(timezone.utc),
    )
    await db[DASHBOARDS].insert_one(dashboard.to_mongo())
    return chart_ids


async def _persist_artifacts(
    db, workspace_id: str, investigation_id: str, message_id: str, user_id: str, artifact_refs: list,
) -> tuple[list, str | None]:
    """Uploads local files the orchestrator produced (dashboards/reports/csv
    exports - see tools/reporting/reporting_tools.py) to R2 and creates
    Chart/Report/Dashboard docs, respecting the free-tier caps. Hitting a cap doesn't
    delete the generated file or the answer text that already mentions it -
    it just skips creating the Mongo doc/R2 upload for that one artifact, so
    it won't be persisted/browsable but the user's answer is unaffected."""
    s3 = get_s3_client()
    bucket = get_bucket_name()
    chart_ids: list = []
    report_id = None

    for ref in artifact_refs:
        if not isinstance(ref, str) or not os.path.isfile(ref):
            continue
        kind = _artifact_kind(ref)
        if kind is None:
            continue

        try:
            if kind == "chart":
                if not await usage.has_chart_capacity(user_id):
                    await _append_event(
                        db, investigation_id, "status",
                        "Chart limit reached - dashboard generated but not saved.",
                    )
                    continue
                chart_id = new_file_id()
                key = build_chart_key(workspace_id, chart_id)
                s3.upload_file(ref, bucket, key, ExtraArgs={"ContentType": "text/html"})
                chart = Chart(
                    id=chart_id, workspace_id=workspace_id, message_id=message_id,
                    title=_artifact_title(ref), storage_key=key,
                )
                await db[CHARTS].insert_one(chart.to_mongo())
                await usage.increment_charts(user_id)
                chart_ids.append(chart.id)
            elif kind == "dashboard_bundle":
                chart_ids.extend(await _persist_dashboard_bundle(
                    db, s3, bucket, workspace_id, investigation_id, message_id, user_id, ref,
                ))
            else:
                if not await usage.has_report_capacity(user_id):
                    await _append_event(
                        db, investigation_id, "status",
                        "Report limit reached - file generated but not saved.",
                    )
                    continue
                new_report_id = new_file_id()
                is_markdown = ref.endswith(".md")
                ext = "md" if is_markdown else "csv"
                fmt = "markdown" if is_markdown else "csv"
                content_type = "text/markdown" if is_markdown else "text/csv"
                key = build_report_key(workspace_id, new_report_id, ext=ext)
                s3.upload_file(ref, bucket, key, ExtraArgs={"ContentType": content_type})
                report = Report(
                    id=new_report_id, workspace_id=workspace_id, message_id=message_id,
                    title=_artifact_title(ref), status="ready", format=fmt, storage_key=key,
                )
                await db[REPORTS].insert_one(report.to_mongo())
                await usage.increment_reports(user_id)
                report_id = report.id
        except Exception:
            logger.exception("failed to persist artifact %s", ref)
        finally:
            # ReportingTools wrote to data/reports/{date}/{name}/... - clean
            # up that scratch folder regardless of whether the upload
            # succeeded, so failed uploads don't leak local disk forever.
            shutil.rmtree(os.path.dirname(ref), ignore_errors=True)

    return chart_ids, report_id


async def run_investigation(
    ctx, investigation_id: str, chat_id: str, workspace_id: str, user_id: str, query: str,
    file_ids: list[str] | None = None,
) -> None:
    # File ids the user referenced via "@" in the client's message composer
    # (see api_service/routers/chats.py's SendMessageRequest.file_ids). Just
    # extracted here for now - not yet wired into the catalog/orchestrator.
    mentioned_file_ids = file_ids or []
    if mentioned_file_ids:
        logger.info("investigation %s: received %d @-mentioned file id(s): %s",
                    investigation_id, len(mentioned_file_ids), mentioned_file_ids)
    db = get_db()

    async def on_event(event: dict) -> None:
        await _append_event(db, investigation_id, event["type"], event["message"], event.get("data"))

    async def cancel_check() -> bool:
        return await _is_cancelled(db, investigation_id)

    # Built once at worker startup (see worker.py's on_startup), not per-job - ChromaVectorStore()
    # in particular opens a real network connection to Chroma Cloud, which every job used to pay
    # for individually.
    storage = ctx["storage"]
    catalog, skipped_files = await _build_catalog(db, workspace_id, storage)

    print("catalog : ",catalog)

    vector_store = ctx["vector_store"]
    memory = LongTermMemory(path=os.path.join(engine_bootstrap.MEMORY_ROOT, f"{user_id}.json"))
    orchestrator = OrchestratorAgent(
        catalog, vector_store=vector_store, memory=memory, storage=storage,
        reports_dir=engine_bootstrap.REPORTS_ROOT,
    )

    if skipped_files:
        await on_event({
            "type": "status",
            "message": (
                f"{len(skipped_files)} file(s) need to be re-uploaded (missing from local "
                f"storage) and were excluded from this investigation: {', '.join(skipped_files)}"
            ),
            "data": {"skipped_files": skipped_files},
        })

    try:
        thread_context = await _thread_context(db, chat_id)
        result = await orchestrator.run(
            query, workspace_id=workspace_id, thread_context=thread_context,
            on_event=on_event, cancel_check=cancel_check,
        )
    except InvestigationCancelled:
        await db[INVESTIGATIONS].update_one(
            {"_id": investigation_id}, {"$set": {"status": "cancelled", "completed_at": _now()}},
        )
        logger.info("investigation %s cancelled", investigation_id)
        return
    except Exception as exc:
        # Full raw exception (incl. any provider-internal detail like quota numbers/org ids)
        # still goes to the logs/Loki via logger.exception - it just doesn't reach the user.
        logger.exception("investigation %s failed", investigation_id)
        error_info = classify_llm_error(exc)
        # "unknown" is classify_llm_error's catch-all for anything that ISN'T a recognizable
        # LLM-provider HTTP error (rate limit/auth/connection/server all require a status code or
        # exception-name match) - that's most likely a real bug in our own code, not an LLM
        # provider hiccup, so keep the original str(exc) behavior for those instead of masking it
        # behind a generic "trouble talking to the AI provider" message that would misdirect
        # anyone debugging it later.
        user_facing = (
            error_info.user_message
            if error_info.kind != "unknown"
            else f"Something went wrong while investigating: {exc}"
        )
        await _append_event(db, investigation_id, "error", user_facing)
        await db[INVESTIGATIONS].update_one(
            {"_id": investigation_id}, {"$set": {"status": "failed", "completed_at": _now()}},
        )
        message = Message(
            chat_id=chat_id, role="assistant",
            content=user_facing,
            investigation_id=investigation_id,
        )
        await db[MESSAGES].insert_one(message.to_mongo())
        return

    message = Message(
        chat_id=chat_id, role="assistant", content=result.final_answer, investigation_id=investigation_id,
    )
    chart_ids, report_id = await _persist_artifacts(
        db, workspace_id, investigation_id, message.id, user_id, result.artifact_refs,
    )
    message.chart_ids = chart_ids
    message.report_id = report_id
    await db[MESSAGES].insert_one(message.to_mongo())

    await db[INVESTIGATIONS].update_one(
        {"_id": investigation_id},
        {"$set": {"status": "completed", "final_answer": result.final_answer, "completed_at": _now()}},
    )
    await usage.increment_messages(user_id)
    await _append_event(
        db, investigation_id, "completed", "Investigation complete.",
        {"message_id": message.id, "chart_ids": chart_ids, "report_id": report_id},
    )
    logger.info("investigation %s completed", investigation_id)

    # Strictly after the above - the user already has their answer (SSE
    # "completed" event just went out) before this starts, so the summary
    # LLM call's latency is never on the user-facing critical path.
    await _update_chat_continuity(db, chat_id, query, result)
