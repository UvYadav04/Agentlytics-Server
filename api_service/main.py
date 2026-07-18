from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api_service.routers import auth, charts, chats, dashboards, feedback, files, reports, usage, workspaces
from shared.config import get_settings
from shared.db import close_client, ensure_indexes
from shared.redis_client import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    await ensure_indexes()
    yield
    await close_redis()
    await close_client()


app = FastAPI(title="Data Analyzer API", lifespan=lifespan)

frontend_origin = get_settings().get("FRONTEND_ORIGIN", "http://localhost:3000")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(workspaces.router)
app.include_router(files.router)
app.include_router(chats.router)
app.include_router(charts.router)
app.include_router(reports.router)
app.include_router(dashboards.router)
app.include_router(usage.router)
app.include_router(feedback.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
