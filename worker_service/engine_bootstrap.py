"""Makes Server/analyzerEngine importable as top-level modules (`ingestion`,
`vectordb`, `agents`, `config`, ...) from inside worker_service.

The engine's own modules use bare top-level imports (e.g.
`from ingestion.manager import IngestionManager`, `from config import
get_settings`) because analyzerEngine/ is meant to be run with itself as the
import root - not as a subpackage of Server/. Rather than rewrite those
imports, we just add analyzerEngine's absolute path to sys.path once, here,
before worker_service imports anything from it.

Import this module FIRST, before any `from ingestion...` / `from agents...`
/ `from vectordb...` import, in every worker_service module that touches the
engine (see tasks/ingestion.py).
"""
import os
import sys
from pathlib import Path

_ENGINE_DIR = (Path(__file__).resolve().parent.parent / "analyzerEngine").resolve()


def _env_or_default(env_var: str, default: Path) -> str:
    """PARQUET_ROOT in particular may need to be a HOST path, not a path
    inside this process's own container - see the docker-outside-of-docker
    note below. Everything else just wants a stable default."""
    return os.environ.get(env_var) or str(default.resolve())

if str(_ENGINE_DIR) not in sys.path:
    # Insert at position 0 so a same-named module inside worker_service/
    # shared/ never wins over the intended engine module (there shouldn't be
    # a collision today, but this keeps resolution order predictable).
    sys.path.insert(0, str(_ENGINE_DIR))

ENGINE_DIR = _ENGINE_DIR
_DATA_DIR = Path(__file__).resolve().parent / "data"

# Where processed Parquet lives. LocalParquetStore, not R2: the Tabular
# Agent's DuckDB view registration (duckdb_utils.register_view) and its
# Docker sandbox (tools/tabular/sandbox_executor.py, bind-mounts
# `storage.root_dir`) both require real local filesystem paths - see the
# warning docstring in analyzerEngine/ingestion/storage/r2_store.py.
#
# IMPORTANT if you containerize worker_service and give it access to the
# HOST's Docker daemon via a mounted socket (docker-outside-of-docker, the
# only way a container can spin up sibling containers): PythonSandbox asks
# that daemon to bind-mount `root_dir` into a new sandbox container. Bind
# mounts are resolved by the DAEMON against ITS OWN filesystem - a path
# that only exists inside the worker's own container (e.g. /app/data/...)
# will not resolve on the host and the mount will silently be empty/wrong.
# Set PARQUET_ROOT to a path that is valid on the HOST (e.g. a docker-compose
# bind mount present at the identical path on both the host and the worker
# container) via this env var. Running worker_service as a bare process on
# a VM with local Docker (not containerized itself) sidesteps this
# entirely and is the simpler option - see README.md's deployment section.
PARQUET_ROOT = _env_or_default("PARQUET_ROOT", _DATA_DIR / "parquet")

# LongTermMemory (store_user_info/recall_user_info) is one JSON file per
# scope - we scope it per-user (see tasks/investigation.py) so preferences
# don't leak across users, since the engine's default is a single global file.
MEMORY_ROOT = _env_or_default("MEMORY_ROOT", _DATA_DIR / "memory")

# Scratch space for report/dashboard/csv generation (ReportingTools writes
# real files here before worker_service uploads them to R2 and deletes the
# local copy - see tasks/investigation.py::_persist_artifacts). No DooD
# concern here - only read/written by this process itself, never bind-mounted
# into a sandbox container.
REPORTS_ROOT = _env_or_default("REPORTS_ROOT", _DATA_DIR / "reports")

