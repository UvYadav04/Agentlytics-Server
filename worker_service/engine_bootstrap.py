
import os
import sys
from pathlib import Path

_ENGINE_DIR = (Path(__file__).resolve().parent.parent / "analyzerEngine").resolve()

def _env_or_default(env_var: str, default: Path) -> str:
    return os.environ.get(env_var) or str(default.resolve())

if str(_ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(_ENGINE_DIR))

ENGINE_DIR = _ENGINE_DIR
_DATA_DIR = Path(__file__).resolve().parent / "data"

PARQUET_ROOT = _env_or_default("PARQUET_ROOT", _DATA_DIR / "parquet")

MEMORY_ROOT = _env_or_default("MEMORY_ROOT", _DATA_DIR / "memory")

REPORTS_ROOT = _env_or_default("REPORTS_ROOT", _DATA_DIR / "reports")

SANDBOX_SOCKET_ROOT = _env_or_default("SANDBOX_SOCKET_ROOT", _DATA_DIR / "sandbox_sockets")