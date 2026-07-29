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
    # Run off the event loop - it loads the local ONNX model (shared/onnx_intent) and runs a
    # CPU inference pass. A False return (e.g. model files not downloaded yet - see
    # shared/onnx_intent/download_model.py) is intentionally non-fatal: route_query_intent_fast
    # just returns method="none" (full Orchestrator) for every request until it's configured -
    # never a reason to fail startup.
    await asyncio.to_thread(intent_router.init)
    yield
    await close_redis()
    await close_client()


app = FastAPI(title="Data Analyzer API", lifespan=lifespan)

# Client is on a different origin (localhost:3000 in dev, the duckdns domain in prod) and auth
# relies on an httpOnly cookie (see shared/auth.py's ACCESS_TOKEN_COOKIE_NAME), so the browser
# needs an explicit allow-list + allow_credentials=True - "*" isn't usable together with
# credentialed requests. CORS_ORIGINS is comma-separated in .env for anyone adding more origins
# later (e.g. a staging domain) without a code change; see shared/.env.example.
_default_cors_origins = "http://localhost:3000,https://agentlytics.duckdns.org"
_cors_origins = [
    origin.strip().rstrip("/")
    for origin in get_settings().get("CORS_ORIGINS", _default_cors_origins).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],  
)

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
