import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_service.routers import auth, charts, chats, dashboards, feedback, files, reports, usage, workspaces
from shared import intent_router
from shared.config import get_settings
from shared.db import close_client, ensure_indexes
from shared.logging_config import configure_logging
from shared.observability import init_observability, instrument_fastapi
from shared.redis_client import close_redis

configure_logging("api_service")
init_observability("api_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    await asyncio.to_thread(intent_router.init)
    yield
    await close_redis()
    await close_client()


app = FastAPI(title="Data Analyzer API", lifespan=lifespan)

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

# Auto-instruments every request with OTel spans + request count/latency/error/active-request
# metrics, exported via the OTLP pipeline set up in shared/observability.py. No-ops if
# OTEL_EXPORTER_OTLP_ENDPOINT isn't set (init_observability skipped setup above).
instrument_fastapi(app)

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
