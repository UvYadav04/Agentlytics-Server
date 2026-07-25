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
    """PARQUET_ROOT in particular is set explicitly in docker-compose.yml (see the
    docker-outside-of-docker note below - it no longer needs to be a HOST path, just this
    container's own mount point). Everything else just wants a stable default."""
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
# that daemon to bind a *named Docker volume* (not a host path - see
# docker-compose.yml's `parquet_data` volume and its top-of-file note) into
# every new sandbox container it creates. That works regardless of which
# path this worker container itself mounts that volume at, because the
# daemon resolves the volume by name, not by path - PARQUET_ROOT below only
# needs to be valid for THIS process's own reads/writes (and match the
# `:/data/parquet` target docker-compose.yml mounts the volume at), it does
# NOT need to equal any host filesystem path anymore. See
# sandbox_executor.py's PARQUET_VOLUME_NAME for the setting that actually
# ties the two containers to the same volume. Running worker_service as a
# bare process on a VM with local Docker (not containerized itself)
# sidesteps all of this and is the simpler option - see README.md's
# deployment section.
PARQUET_ROOT = _env_or_default("PARQUET_ROOT", _DATA_DIR / "parquet")

# LongTermMemory (store_user_info/recall_user_info) is one JSON file per
# scope - we scope it per-user (see tasks/investigation.py) so preferences
# don't leak across users, since the engine's default is a single global file.
#
# IMPORTANT if you containerize worker_service (docker-compose.yml): the
# `_DATA_DIR / "memory"` default below lives on the container's own writable
# layer, which is thrown away on every rebuild/redeploy/recreation - unlike
# PARQUET_ROOT above, nothing backs it with persistent storage unless you set
# MEMORY_ROOT explicitly to a named-volume mount point (see docker-compose.yml's
# `memory_data` volume) the same way PARQUET_ROOT is set. Deliberately NOT
# defaulted to a subfolder of PARQUET_ROOT: PythonSandbox bind-mounts the
# entirety of `root_dir` (== PARQUET_ROOT) read-write into every sandboxed
# code execution (see tools/tabular/sandbox_executor.py) - nesting per-user
# memory files under there would hand model-generated code read/write access
# to every user's stored preferences, not just the parquet tables it was
# assigned. Keeping MEMORY_ROOT a sibling path (its own bind mount) avoids
# that exposure entirely. Running worker_service as a bare process (not
# containerized) sidesteps the whole issue, same as PARQUET_ROOT's note above
# - the default below is already on real, persistent local disk in that case.
MEMORY_ROOT = _env_or_default("MEMORY_ROOT", _DATA_DIR / "memory")

# Scratch space for report/dashboard/csv generation (ReportingTools writes
# real files here before worker_service uploads them to R2 and deletes the
# local copy - see tasks/investigation.py::_persist_artifacts). No DooD
# concern here - only read/written by this process itself, never bind-mounted
# into a sandbox container.
REPORTS_ROOT = _env_or_default("REPORTS_ROOT", _DATA_DIR / "reports")

# This container's OWN mount point for the `sandbox_sockets` named Docker volume (see
# docker-compose.yml) - the .sock file each persistent sandbox container's Uvicorn server binds
# to (see sandbox/sandbox_server.py) is only visible to THIS process because both sides mount
# the same named volume, exactly like PARQUET_ROOT/`parquet_data` above. Handed to
# sandbox.sandbox_manager.get_manager() once, in worker.py's on_startup - see that file and
# sandbox/sandbox_manager.py's SANDBOX_SOCKET_VOLUME_NAME for the setting that ties this
# process's mount to the same volume every sandbox container mounts.
SANDBOX_SOCKET_ROOT = _env_or_default("SANDBOX_SOCKET_ROOT", _DATA_DIR / "sandbox_sockets")

