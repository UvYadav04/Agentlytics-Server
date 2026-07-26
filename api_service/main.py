import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from api_service.routers import auth, charts, chats, dashboards, feedback, files, reports, usage, workspaces
from shared import intent_router
from shared.config import get_settings
from shared.db import close_client, ensure_indexes
from shared.logging_config import configure_logging
from shared.redis_client import close_redis

configure_logging("api_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    # Embeds every intent_examples.json phrase once, up front, so the hybrid router's embedding
    # tier never regenerates them per request (see shared/intent_router.py's module docstring).
    # Run off the event loop - it makes blocking network/HTTP calls (DeepInfra embeddings API).
    # A False return (e.g. DEEPINFRA_API_KEY not set yet) is intentionally non-fatal: routing
    # just falls back to the LLM classifier for every request until it's configured - never a
    # reason to fail startup.
    await asyncio.to_thread(intent_router.init)
    yield
    await close_redis()
    await close_client()


app = FastAPI(title="Data Analyzer API", lifespan=lifespan)

# Exposes GET /metrics (request count/latency/status codes, broken down by
# path+method) for Prometheus to scrape - see ../../observability/prometheus/prometheus.yml.
Instrumentator().instrument(app).expose(app)

	

# "/api" prefix matches the client's NEXT_PUBLIC_API_URL in production (e.g.
# https://agentlytics.duckdns.org/api - see Client/.env), which is forwarded to this service
# path-and-all by the reverse proxy rather than stripped. /health below is deliberately NOT under
# this prefix - it's for container/EC2-level liveness checks hitting the service directly on its
# own port, not traffic coming through the public reverse proxy.
API_PREFIX = "/api"

app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(workspaces.router, prefix=API_PREFIX)
app.include_router(files.router, prefix=API_PREFIX)
app.include_router(chats.router, prefix=API_PREFIX)
app.include_router(charts.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)
app.include_router(dashboards.router, prefix=API_PREFIX)
app.include_router(usage.router, prefix=API_PREFIX)
app.include_router(feedback.router, prefix=API_PREFIX)


@app.get("/health")
async def health():
    return {"status": "ok"}
