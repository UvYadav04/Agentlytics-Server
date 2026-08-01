import asyncio
import json
import logging
import os
import shutil
import time
from datetime import datetime, timezone

from worker_service import engine_bootstrap  # noqa: F401

from analyzerEngine.agents.document.agent import DocumentAgent
from analyzerEngine.agents.orchestrator.agent import InvestigationCancelled, OrchestratorAgent
from analyzerEngine.agents.tabular.agent import TabularAgent
from analyzerEngine.ingestion.storage.local_store import LocalParquetStore
from analyzerEngine.llm_provider.errors import classify_llm_error
from analyzerEngine.sandbox.path_resolver import InvalidArtifactIdError, get_parquet_path
from analyzerEngine.tools.document.metadata import build_document_metadata_brief
from analyzerEngine.tools.orchestrator.file_catalog import FileCatalog, is_tabular_output_ref, table_catalog_entry
from analyzerEngine.tools.orchestrator.memory import LongTermMemory
from analyzerEngine.tools.orchestrator.models import (
    FileCatalogEntry, FileRef, FinalResultCollector, OrchestratorResult,
)
from analyzerEngine.tools.orchestrator.thread_summary import analyze_turn
from analyzerEngine.tools.tabular.models import FileRef as TabularFileRef

from arq import Retry

from shared import usage
from shared.db import get_db
from shared.job_timing import log_job_finished, log_job_picked_up, now_iso
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

RECENT_TURNS_LIMIT = 5
FILE_LIST_LIMIT = 30

logger = logging.getLogger("worker.investigation")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _artifact_exists(storage: LocalParquetStore, workspace_id: str, file_id: str) -> bool:
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


async def update_chat_memory(
    ctx, chat_id: str, user_id: str, query: str, response: str,
    files_used: list, files_created: list, requested_at: str | None = None,
) -> None:
    db = get_db()

    try:
        doc = await db[CHATS].find_one(
            {"_id": chat_id}, {"summary": 1, "recent_turns": 1, "files_used": 1, "files_created": 1},
        ) or {}

        existing_turns = doc.get("recent_turns", [])
        combined_turns = existing_turns + [{"query": query, "response": response}]
        overflow_turns = combined_turns[:-RECENT_TURNS_LIMIT] if len(combined_turns) > RECENT_TURNS_LIMIT else []
        recent_turns = combined_turns[-RECENT_TURNS_LIMIT:]

        new_summary, new_preferences = await analyze_turn(
            doc.get("summary", ""), query, response, turns_to_fold=overflow_turns,
        )

        merged_files_used = _merge_capped(doc.get("files_used", []), files_used, FILE_LIST_LIMIT)
        merged_files_created = _merge_capped(doc.get("files_created", []), files_created, FILE_LIST_LIMIT)

        await db[CHATS].update_one(
            {"_id": chat_id},
            {"$set": {
                "summary": new_summary,
                "files_used": merged_files_used,
                "files_created": merged_files_created,
                "recent_turns": recent_turns,
            }},
        )

        if new_preferences:
            memory = LongTermMemory(path=os.path.join(engine_bootstrap.MEMORY_ROOT, f"{user_id}.json"))
            for fact in new_preferences:
                memory.remember(fact)
    except Exception as exc:
        logger.exception(
            "update_chat_memory failed for chat %s (job_try=%s) - retrying", chat_id, ctx.get("job_try"),
        )
        raise Retry(defer=ctx["job_try"] * 5) from exc

    log_job_finished(logger, "update_chat_memory", picked_up_at, chat_id=chat_id)


def _artifact_title(path: str) -> str:
    name = os.path.basename(os.path.dirname(path))
    return name or "Untitled"


async def _persist_dashboard_bundle(
    db, s3, bucket: str, workspace_id: str, investigation_id: str, message_id: str, user_id: str, manifest_path: str,
    email: str | None = None,
) -> list:

    folder = os.path.dirname(manifest_path)
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    chart_ids: list = []
    chart_configs: list[ChartConfig] = []

    for chart_meta in manifest.get("charts", []):
        if not await usage.has_chart_capacity(user_id, email):
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
            bins=chart_meta.get("bins"),
        ))

    if not chart_configs:
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
    db, workspace_id: str, investigation_id: str, message_id: str, user_id: str,
    chart_paths: list, artifacts: list, email: str | None = None,
) -> tuple[list, str | None]:
    s3 = get_s3_client()
    bucket = get_bucket_name()
    chart_ids: list = []
    report_id = None

    for chart in chart_paths:
        ref = chart.get("location")
        if not isinstance(ref, str) or not os.path.isfile(ref):
            continue
        try:
            if not await usage.has_chart_capacity(user_id, email):
                await _append_event(
                    db, investigation_id, "status",
                    "Chart limit reached - a chart was generated but not saved.",
                )
                continue
            chart_id = new_file_id()
            key = build_chart_key(workspace_id, chart_id)
            s3.upload_file(ref, bucket, key, ExtraArgs={"ContentType": "text/html"})
            chart_doc = Chart(
                id=chart_id, workspace_id=workspace_id, message_id=message_id,
                title=chart.get("title") or _artifact_title(ref), storage_key=key,
            )
            await db[CHARTS].insert_one(chart_doc.to_mongo())
            await usage.increment_charts(user_id)
            chart_ids.append(chart_id)
        except Exception:
            logger.exception("failed to persist chart %s", ref)
        finally:
            shutil.rmtree(os.path.dirname(ref), ignore_errors=True)

    for artifact in artifacts:
        ref = artifact.get("ref")
        kind = artifact.get("type")
        
        if kind not in ("report", "dashboard_bundle"):
            continue
        if not isinstance(ref, str) or not os.path.isfile(ref):
            continue

        try:
            if kind == "dashboard_bundle":
                chart_ids.extend(await _persist_dashboard_bundle(
                    db, s3, bucket, workspace_id, investigation_id, message_id, user_id, ref,
                    email,
                ))
            else:
                if not await usage.has_report_capacity(user_id, email):
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
            shutil.rmtree(os.path.dirname(ref), ignore_errors=True)

    return chart_ids, report_id


def _select_direct_route_files(catalog: FileCatalog, mentioned_file_ids: list, kind: str) -> list | None:
    """Files to hand a direct-routed Tabular/Document agent, or None if direct-routing isn't
    safe here - the caller falls back to the full Orchestrator in that case, same as if route
    had been None/"orchestrator" to begin with.

    Only two cases count as safe:
    - the user explicitly @-mentioned file_id(s) (SendMessageRequest.file_ids) that resolve to
      at least one eligible file of the right kind, or
    - the workspace has EXACTLY ONE eligible file, so there's nothing to disambiguate.

    Anything else (zero eligible files, or more than one with no explicit mention) is NOT
    direct-routed - guessing which file a request means is exactly the judgment call the
    Orchestrator exists for, and this router makes no attempt to replace that judgment."""
    if kind == "tabular":
        eligible = [e for e in catalog.browsable() if is_tabular_output_ref(e.output_ref)]
    else:
        eligible = [e for e in catalog.browsable() if e.file_type in ("pdf", "txt")]

    if not eligible:
        return None

    if mentioned_file_ids:
        matched = [e for e in eligible if e.file_id in mentioned_file_ids]
        return matched or None

    if len(eligible) == 1:
        return eligible

    return None


async def _run_tabular_direct(
    catalog: FileCatalog, storage: LocalParquetStore, chat_id: str, sandbox_manager,
    workspace_id: str, query: str, mentioned_file_ids: list, on_event, result_collector: FinalResultCollector,
    thread_context: dict | None = None, chart_capacity_checker=None,
) -> OrchestratorResult | None:
    entries = _select_direct_route_files(catalog, mentioned_file_ids, "tabular")
    if entries is None:
        return None

    assigned_files = [TabularFileRef(file_id=e.file_id, filename=e.filename) for e in entries]
    agent = TabularAgent(
        assigned_files, storage=storage, workspace_id=workspace_id,
        chat_id=chat_id, sandbox_manager=sandbox_manager, direct_route=True,
        reports_dir=engine_bootstrap.REPORTS_ROOT, chart_capacity_checker=chart_capacity_checker,
    )
    findings = await agent.run(query, constraints={}, on_event=on_event, thread_context=thread_context)
    result_collector.add_tabular_findings(
        findings, "invoke_tabular_agent", [e.file_id for e in entries],
    )
    return OrchestratorResult(
        final_answer=findings.summary, chart_paths=result_collector.chart_paths,
        artifacts=result_collector.artifacts, files_used=result_collector.files_used,
        open_questions=[], follow_up_questions=findings.follow_up_questions,
    )


async def _run_document_direct(
    catalog: FileCatalog, vector_store, query: str, mentioned_file_ids: list, on_event,
    result_collector: FinalResultCollector, thread_context: dict | None = None,
) -> OrchestratorResult | None:
    entries = _select_direct_route_files(catalog, mentioned_file_ids, "document")
    if entries is None:
        return None

    assigned_files = [FileRef(file_id=e.file_id) for e in entries]
    metadata_brief = build_document_metadata_brief(catalog, vector_store, [e.file_id for e in entries])
    agent = DocumentAgent(assigned_files, vector_store=vector_store, direct_route=True)
    findings = await agent.run(
        query, constraints={}, on_event=on_event, metadata_brief=metadata_brief,
        thread_context=thread_context,
    )
    result_collector.add_document_findings(
        findings, "invoke_document_agent", [e.file_id for e in entries],
    )
    return OrchestratorResult(
        final_answer=findings.summary, chart_paths=result_collector.chart_paths,
        artifacts=result_collector.artifacts, files_used=result_collector.files_used,
        open_questions=[], follow_up_questions=findings.follow_up_questions,
    )


async def run_investigation(
    ctx, investigation_id: str, chat_id: str, workspace_id: str, user_id: str, query: str,
    file_ids: list[str] | None = None, requested_at: str | None = None, route: str | None = None,
    email: str | None = None,
) -> None:
    picked_up_at = log_job_picked_up(
        logger, ctx, "run_investigation", requested_at=requested_at,
        investigation_id=investigation_id, chat_id=chat_id,
    )

    # No per-investigation sandbox pre-warm needed anymore: the shared sandbox pool is warmed
    # to min_size once at worker startup (see worker.py on_startup), and run_python calls just
    # acquire whatever's idle in the pool - there's no per-chat container to create here.
    sandbox_manager = ctx["sandbox_manager"]

    mentioned_file_ids = file_ids or []
    if mentioned_file_ids:
        logger.info("investigation %s: received %d @-mentioned file id(s): %s",
                    investigation_id, len(mentioned_file_ids), mentioned_file_ids)
    db = get_db()

    result_collector = FinalResultCollector()

    async def chart_capacity_checker() -> bool:
        return await usage.has_chart_capacity(user_id, email)

    async def on_event(event: dict) -> None:
        await _append_event(db, investigation_id, event["type"], event["message"], event.get("data"))

    async def cancel_check() -> bool:
        return await _is_cancelled(db, investigation_id)

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

    direct_route = route if route in ("tabular", "document") else None

    if skipped_files:
        await on_event({
            "type": "status",
            "message": (
                f"{len(skipped_files)} file(s) need to be re-uploaded (missing from local "
                f"storage) and were excluded from this investigation: {', '.join(skipped_files)}"
            ),
            "data": {"skipped_files": skipped_files},
        })

    thread_context = await _thread_context(db, chat_id)

    try:
        try:
            result = None

            if direct_route == "tabular":
                await on_event({
                    "type": "status",
                    "message": "Picked up your request",
                })
                result = await _run_tabular_direct(
                    catalog, storage, chat_id, sandbox_manager, workspace_id, query,
                    mentioned_file_ids, on_event, result_collector, thread_context,
                    chart_capacity_checker,
                )
                if result is None:
                    logger.info(
                        "investigation %s: tabular direct-route unsafe (no unambiguous file "
                        "selection) - falling back to the Orchestrator", investigation_id,
                    )
            elif direct_route == "document":
                await on_event({
                    "type": "status",
                    "message": "Picked up your request",
                })
                result = await _run_document_direct(
                    catalog, vector_store, query, mentioned_file_ids, on_event, result_collector,
                    thread_context,
                )
                if result is None:
                    logger.info(
                        "investigation %s: document direct-route unsafe (no unambiguous file "
                        "selection) - falling back to the Orchestrator", investigation_id,
                    )

            if result is not None:
                logger.info(
                    "investigation %s: handled directly by the %s agent (Orchestrator skipped)",
                    investigation_id, direct_route,
                )
            else:
                await on_event({
                    "type": "status",
                    "message": "Picked up your request",
                })
                orchestrator = OrchestratorAgent(
                    catalog, vector_store=vector_store, memory=memory, storage=storage,
                    reports_dir=engine_bootstrap.REPORTS_ROOT, chat_id=chat_id,
                    sandbox_manager=sandbox_manager, result_collector=result_collector,
                    chart_capacity_checker=chart_capacity_checker,
                )
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
            logger.exception("investigation %s failed", investigation_id)
            error_info = classify_llm_error(exc)
            user_facing = (
                error_info.user_message
                if error_info.kind != "unknown"
                else f"Something went wrong while investigating: {exc}"
            )

            message = Message(
                chat_id=chat_id, role="assistant",
                content=user_facing,
                investigation_id=investigation_id,
            )
            chart_ids: list = []
            report_id = None
            if result_collector.chart_paths or result_collector.artifacts:
                try:
                    chart_ids, report_id = await _persist_artifacts(
                        db, workspace_id, investigation_id, message.id, user_id,
                        result_collector.chart_paths, result_collector.artifacts, email,
                    )
                except Exception:
                    logger.exception(
                        "investigation %s: failed to persist partial results after error",
                        investigation_id,
                    )
                if chart_ids or report_id:
                    message.content += (
                        "\n\n_Some results were produced before this error - see below._"
                    )
            message.chart_ids = chart_ids
            message.report_id = report_id
            message.files_used = result_collector.files_used

            await _append_event(db, investigation_id, "error", user_facing)
            await db[INVESTIGATIONS].update_one(
                {"_id": investigation_id}, {"$set": {"status": "failed", "completed_at": _now()}},
            )
            await db[MESSAGES].insert_one(message.to_mongo())
            return

        message = Message(
            chat_id=chat_id, role="assistant", content=result.final_answer, investigation_id=investigation_id,
            follow_up_questions=result.follow_up_questions,
        )
        chart_ids, report_id = await _persist_artifacts(
            db, workspace_id, investigation_id, message.id, user_id,
            result.chart_paths, result.artifacts, email,
        )
        message.chart_ids = chart_ids
        message.report_id = report_id
        message.files_used = result.files_used
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

        await ctx["redis"].enqueue_job(
            "update_chat_memory",
            chat_id=chat_id, user_id=user_id, query=query, response=result.final_answer,
            files_used=result.files_used, files_created=result.artifact_refs,
            requested_at=now_iso(),
        )
    finally:
        log_job_finished(logger, "run_investigation", picked_up_at, investigation_id=investigation_id, chat_id=chat_id)
