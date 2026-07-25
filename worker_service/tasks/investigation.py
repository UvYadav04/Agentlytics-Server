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
import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone

from worker_service import engine_bootstrap  # noqa: F401

from analyzerEngine.agents.orchestrator.agent import InvestigationCancelled, OrchestratorAgent
from analyzerEngine.ingestion.storage.local_store import LocalParquetStore
from analyzerEngine.llm_provider.errors import classify_llm_error
from analyzerEngine.sandbox.path_resolver import InvalidArtifactIdError, get_parquet_path
from analyzerEngine.tools.orchestrator.file_catalog import FileCatalog, is_tabular_output_ref, table_catalog_entry
from analyzerEngine.tools.orchestrator.memory import LongTermMemory
from analyzerEngine.tools.orchestrator.models import FileCatalogEntry
from analyzerEngine.tools.orchestrator.thread_summary import update_summary

from shared import usage
from shared.db import get_db
from shared.job_timing import log_job_finished, log_job_picked_up
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


def _artifact_exists(storage: LocalParquetStore, workspace_id: str, file_id: str) -> bool:
    """True when file_id resolves to a real parquet artifact under this workspace. Builds the
    path itself via get_parquet_path (workspace_id/file_id, both validated) rather than trusting
    a caller-supplied path - there's no path left in a Mongo doc to trust or mistrust anymore."""
    if not file_id:
        return False
    try:
        path = get_parquet_path(storage.root_dir, workspace_id, file_id)
    except InvalidArtifactIdError:
        return False
    return storage.exists(path)


async def _build_catalog(db, workspace_id: str, storage: LocalParquetStore) -> tuple[FileCatalog, list[str]]:

    catalog = FileCatalog()
    stale_ids: list[str] = []
    skipped_filenames: list[str] = []

    t0 = time.perf_counter()
    docs = await db[FILES].find({"workspace_id": workspace_id, "status": "ready"}).to_list(length=None)

    # Every ref (main file + every extracted table) that needs an existence check, deduped -
    # two files could theoretically share a table ref only in corrupted data, but deduping is
    # free and avoids doing the same stat() twice regardless.
    refs_to_check: set[str] = set()
    for doc in docs:
        output_ref = doc.get("output_ref") or ""
        if is_tabular_output_ref(output_ref):
            refs_to_check.add(output_ref)
        for table in doc.get("extracted_tables") or []:
            refs_to_check.add(table.get("output_ref") or "")

    existence_results = await asyncio.gather(
        *(asyncio.to_thread(_artifact_exists, storage, workspace_id, ref) for ref in refs_to_check)
    )
    exists_by_ref = dict(zip(refs_to_check, existence_results))
    logger.debug(
        "_build_catalog: checked %d file(s) on disk in %.1fms (workspace=%s)",
        len(refs_to_check), (time.perf_counter() - t0) * 1000, workspace_id,
    )

    for doc in docs:
        output_ref = doc.get("output_ref") or ""
        # is_tabular_output_ref filters out the "" xlsx-workbook-main sentinel and the
        # "workspace_{id}" PDF/TXT vector-store pointer - neither is a parquet artifact, so
        # there's nothing on disk to check for those.
        if is_tabular_output_ref(output_ref) and not exists_by_ref.get(output_ref, False):
            stale_ids.append(doc["_id"])
            skipped_filenames.append(doc["filename"])
            continue

        table_entries = []
        tables_ok = True
        for table in doc.get("extracted_tables") or []:
            if not exists_by_ref.get(table.get("output_ref") or "", False):
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
    file_ids: list[str] | None = None, requested_at: str | None = None,
) -> None:
    # First line, always - logs when this worker actually started running the job, plus
    # queue_wait/request_to_worker latency (requested_at comes from chats.py's send_message,
    # the moment the original HTTP request arrived - see shared/job_timing.py).
    picked_up_at = log_job_picked_up(
        logger, ctx, "run_investigation", requested_at=requested_at,
        investigation_id=investigation_id, chat_id=chat_id,
    )

    # Kick off this investigation's persistent sandbox container NOW, in the background, so its
    # ~2-3s Docker create + health-check wait (see sandbox/sandbox_manager.py) overlaps with the
    # catalog build and the orchestrator's own LLM calls below instead of being paid synchronously
    # whenever the Tabular Agent's first run_python() call eventually happens. SandboxManager.
    # get_or_create is keyed + locked by investigation_id, so that later call just finds (or
    # briefly waits on) this same in-flight/cached container - it never races to create a second
    # one. Fire-and-forget except for the `finally` block below, which always waits for this task
    # before releasing, so cleanup can never run concurrently with creation.
    #
    # ctx["sandbox_manager"] - built ONCE in worker.py's on_startup - not get_sandbox_manager()
    # called fresh here: the module-level singleton getter is keyed by import path
    # (`analyzerEngine.sandbox.sandbox_manager`, which this file uses, vs the bare
    # `sandbox.sandbox_manager` analyzerEngine's own internal modules use, resolve to two
    # different sys.modules entries with two independent singletons - see
    # sandbox_manager.get_manager's docstring). Using ctx's instance and threading it explicitly
    # into OrchestratorAgent below (which passes it all the way down to PythonSandbox) guarantees
    # this pre-warm, the real run_python() execution, and the release() call at the end of this
    # function all agree on the exact same SandboxManager - otherwise the pre-warmed container
    # sits unused in one singleton's cache while the real call creates a second one in the other.
    sandbox_manager = ctx["sandbox_manager"]
    prewarm_start = time.perf_counter()
    sandbox_prewarm_task = asyncio.create_task(
        asyncio.to_thread(sandbox_manager.get_or_create, investigation_id)
    )

    def _log_prewarm_result(task: asyncio.Task, _start=prewarm_start) -> None:
        if task.cancelled():
            return
        exc = task.exception()  # also marks the exception as "retrieved" - no asyncio warning
        if exc is not None:
            logger.warning(
                "investigation %s: sandbox pre-warm failed after %.1fms (the first run_python "
                "call will just create it synchronously instead, same as before this change): %s",
                investigation_id, (time.perf_counter() - _start) * 1000, exc,
            )
        else:
            logger.info(
                "investigation %s: sandbox pre-warmed in %.1fms (overlapped with catalog build "
                "and the orchestrator's own LLM calls, not added on top of them)",
                investigation_id, (time.perf_counter() - _start) * 1000,
            )

    sandbox_prewarm_task.add_done_callback(_log_prewarm_result)

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
    catalog_start = time.perf_counter()
    catalog, skipped_files = await _build_catalog(db, workspace_id, storage)
    logger.info(
        "investigation %s: catalog built in %.1fms (%d file(s), %d skipped)",
        investigation_id, (time.perf_counter() - catalog_start) * 1000,
        len(catalog.entries), len(skipped_files),
    )

    vector_store = ctx["vector_store"]
    memory = LongTermMemory(path=os.path.join(engine_bootstrap.MEMORY_ROOT, f"{user_id}.json"))
    orchestrator = OrchestratorAgent(
        catalog, vector_store=vector_store, memory=memory, storage=storage,
        reports_dir=engine_bootstrap.REPORTS_ROOT, investigation_id=investigation_id,
        sandbox_manager=sandbox_manager,
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
    finally:
        # Always wait for the pre-warm task kicked off at the top of this function to actually
        # finish - success or failure - before releasing anything below. Without this, a fast
        # investigation that never touched the Tabular Agent (so nothing else ever awaited this
        # task) could reach release() while the background thread is still mid-`docker run`,
        # racing container creation against its own teardown. The exception (if any) was already
        # logged by _log_prewarm_result's done-callback, so this is just a rendezvous, not a
        # second place that needs to report it.
        try:
            await sandbox_prewarm_task
        except Exception:
            pass

        # Runs on every exit path (completed/failed/cancelled/an exception this function itself
        # doesn't catch) - this investigation is over either way, so its persistent sandbox
        # container (see sandbox/sandbox_manager.py - one container per investigation_id, warm
        # across every invoke_tabular_agent call this run made) is released now rather than left
        # for the idle-timeout reaper to eventually notice. release() itself makes blocking
        # Docker SDK calls (container stop/remove), so it's pushed off the event loop the same
        # way PythonSandbox.run's own Docker calls already are elsewhere (see
        # dashboard_refresh.py's asyncio.to_thread note).
        try:
            await asyncio.to_thread(sandbox_manager.release, investigation_id)
        except Exception:
            logger.exception("failed to release sandbox for investigation %s", investigation_id)
        # Paired with log_job_picked_up above - total in-worker duration (excludes queue wait,
        # which was already logged separately). The specific outcome (completed/failed/
        # cancelled) is already logged with its own status a few lines up in each branch; this
        # is just the closing timestamp for the job as a whole.
        log_job_finished(logger, "run_investigation", picked_up_at, investigation_id=investigation_id, chat_id=chat_id)
