# DataAnalyzer — Server

The backend for DataAnalyzer: a multi-agent data analysis platform. Users upload CSV/Excel/PDF/text files, chat with them in natural language, and get back analysis, charts, dashboards, CSV exports, and markdown reports.

This is a practical getting-started guide. For how the system is designed and why (agent architecture, sandbox pooling, latency work, etc.), see [`../ARCHITECTURE.md`](../ARCHITECTURE.md).

## Layout

Three Python services sharing one codebase root, plus a `shared` package both import:

| Directory | What it is |
|---|---|
| `api_service/` | FastAPI + Uvicorn. The only HTTP-facing service — auth, CRUD routes, enqueues work. Never runs an agent itself. |
| `worker_service/` | [arq](https://arq-docs.helpmanual.io/) (Redis-backed job queue) worker. Picks up ingestion/investigation/dashboard-refresh jobs and runs the agent pipelines off the request path. |
| `analyzerEngine/` | The agent "engine" — agents, tools, sandbox, vector store, ingestion, LLM provider abstraction. Not a standalone service; imported by `worker_service` only. |
| `shared/` | Cross-service code: Mongo models, Redis/S3/auth clients, usage limits, observability, email. Imported by both `api_service` and `worker_service`. |

Each of `api_service/`, `worker_service/`, `analyzerEngine/`, and `shared/` has its own `requirements.txt`; the root `requirements.txt` just includes all four for convenience (e.g. one IDE virtualenv).

## Prerequisites

- Python 3.12
- Docker (worker_service talks to the **host** Docker daemon to spin up sandbox containers for code execution — see `ARCHITECTURE.md` §5.10)
- A MongoDB database (Atlas or self-hosted)
- Redis (local via Docker is easiest — see below)
- A Chroma Cloud account (vector store for document chunks)
- An AWS S3 bucket (or S3-compatible storage) for uploads/artifacts
- At least one LLM provider API key (OpenAI, Anthropic, Gemini, Groq, or DeepInfra) — DeepInfra is the default for most agents in `.env.example`

## Environment setup

Two env files, both git-ignored, both loaded from example templates checked into the repo:

```bash
cp shared/.env.example shared/.env
cp analyzerEngine/.env.example analyzerEngine/.env
```

- **`shared/.env`** — Mongo, Redis, S3, JWT/auth, CORS, SMTP, free-tier usage limits, intent routing, and observability (Grafana Alloy/Langfuse) settings. Loaded by both `api_service` and `worker_service`.
- **`analyzerEngine/.env`** — LLM provider API keys, per-agent provider/model overrides, Chroma, LlamaParse, reranker, sandbox tuning, Langfuse tracing keys. Loaded by `worker_service` only (it's the only service that imports `analyzerEngine`).

Every setting in both files has an inline comment explaining what it does and what reads it — skim them before deploying anywhere real. A few things worth calling out:

- `OTEL_EXPORTER_OTLP_ENDPOINT` left blank disables OpenTelemetry entirely (no traces/metrics/logs export, everything else works fine) — useful for local dev without a Grafana Alloy collector running.
- `LANGFUSE_PUBLIC_KEY`/`LANGFUSE_SECRET_KEY` left blank disables LLM call tracing to Langfuse Cloud, same graceful no-op.
- `ONNX_INTENT_*` — the fast local intent classifier needs its ONNX weights downloaded once: `python shared/onnx_intent/download_model.py`.
- Free-tier limits (`FREE_TIER_MESSAGES`, `FREE_TIER_CHATS`, etc.) and `ADMIN_EMAILS` (which bypasses all of them) live in `shared/.env`.

## Running locally

### Option A — Docker Compose (recommended)

```bash
cd Server
docker compose up --build
```

This brings up Redis, `api_service` (port 8000), and `worker_service`, wired together with named volumes for the parquet data lake, sandbox Unix sockets, and agent memory. `worker_service` bind-mounts the host's `docker.sock` so it can create sandbox containers as siblings, not children.

`docker-compose.prod.yml` is the production overlay (same shape, prod-tuned env/image references). `docker-compose.observability.yml` is an optional self-hosted Langfuse/Prometheus/Loki/Grafana stack — not needed if you're pointed at Grafana Cloud + Langfuse Cloud via the env vars above.

### Option B — Run services directly

Each service needs its own virtualenv (or one shared one, using the root `requirements.txt`):

```bash
# api_service
cd Server
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate on macOS/Linux
pip install -r api_service/requirements.txt
uvicorn api_service.main:app --reload --port 8000

# worker_service (separate terminal, same or another venv)
pip install -r worker_service/requirements.txt
python shared/onnx_intent/download_model.py   # first run only
arq worker_service.worker.WorkerSettings
```

`worker_service` still needs a working Docker daemon reachable at the default socket for the sandbox pool to start — `worker.py`'s `on_startup` warms a couple of sandbox containers before the worker picks up its first job.

You'll also need Redis running somewhere (`docker run -p 6379:6379 redis:7-alpine` is the quickest local option) and `REDIS_URL` pointed at it.

## Key things to know before changing code

- **Never import `analyzerEngine` from `api_service`.** The whole point of the two-service split is that a slow/stuck agent run can't block the HTTP layer. If you need agent logic reachable from an API route, enqueue an arq job instead.
- **`analyzerEngine`'s internal imports are bare** (`from agents...`, `from tools...`, not `from analyzerEngine.agents...`) — `worker_service/engine_bootstrap.py` inserts `analyzerEngine/` directly into `sys.path` to make that resolve. Importing the same module two different ways creates two separate singletons (see `ARCHITECTURE.md` §5.2) — always thread shared objects like `SandboxManager` through constructors rather than re-resolving a module-level singleton in a new code path.
- **Every LLM call goes through `analyzerEngine/llm_provider/`**, never a raw SDK call — that's what gives every agent retry, fallback, and Langfuse tracing uniformly.
- **The sandbox pool is shared and stateless per call** — `execution_engine.py` reloads every table fresh from the parquet volume on every `/execute`, so nothing about a chat's prior turns lives inside a sandbox container between calls.

## Observability

`shared/observability.py` is the single module both services call into. It wires OpenTelemetry (traces/metrics/logs) to a Grafana Alloy OTLP endpoint and Langfuse Cloud for LLM call tracing — both fully optional and no-op if their env vars are unset. See the comments in `shared/.env.example` for the exact vars.

## Deeper reading

- [`../ARCHITECTURE.md`](../ARCHITECTURE.md) — full system design, the problems that shaped it, and the latency work done along the way.
- `analyzerEngine/agents/README.md` — agent-specific notes.
- `analyzerEngine/ingestion/README.md` — file ingestion pipeline notes.
