"""Safe serialization and truncation utilities for trace payloads.

Ensures that monitoring payloads never cause memory leaks, gigantic disk writes,
or JSON serialization exceptions.
"""

from __future__ import annotations

import json
from typing import Any

MAX_STRING_LENGTH = 16384  # 16 KB per string field
MAX_LIST_ITEMS = 50
MAX_DICT_KEYS = 50


def safe_serialize(obj: Any, depth: int = 4) -> Any:
    """Recursively converts objects to JSON-serializable structures with size safety."""
    if depth <= 0:
        return "<max_depth_reached>"

    if obj is None or isinstance(obj, (bool, int, float)):
        return obj

    if isinstance(obj, str):
        if len(obj) > MAX_STRING_LENGTH:
            return obj[:MAX_STRING_LENGTH] + f"... [truncated {len(obj) - MAX_STRING_LENGTH} chars]"
        return obj

    if isinstance(obj, bytes):
        return f"<bytes len={len(obj)}>"

    if isinstance(obj, (list, tuple, set)):
        items = list(obj)
        serialized = [safe_serialize(x, depth - 1) for x in items[:MAX_LIST_ITEMS]]
        if len(items) > MAX_LIST_ITEMS:
            serialized.append(f"... [{len(items) - MAX_LIST_ITEMS} more items]")
        return serialized

    if isinstance(obj, dict):
        result = {}
        for idx, (k, v) in enumerate(obj.items()):
            if idx >= MAX_DICT_KEYS:
                result["__truncated__"] = f"{len(obj) - MAX_DICT_KEYS} more keys"
                break
            result[str(k)] = safe_serialize(v, depth - 1)
        return result

    # Check for pandas/polars DataFrame
    if hasattr(obj, "head") and hasattr(obj, "shape"):
        try:
            return {
                "__type__": "DataFrame",
                "shape": list(obj.shape),
                "columns": list(getattr(obj, "columns", [])),
                "head": safe_serialize(obj.head(5).to_dict(orient="records"), depth - 1),
            }
        except Exception:
            return f"<DataFrame shape={getattr(obj, 'shape', 'unknown')}>"

    # Check for Pydantic model
    if hasattr(obj, "model_dump"):
        try:
            return safe_serialize(obj.model_dump(), depth - 1)
        except Exception:
            pass
    elif hasattr(obj, "dict"):
        try:
            return safe_serialize(obj.dict(), depth - 1)
        except Exception:
            pass

    # Generic object with __dict__
    if hasattr(obj, "__dict__"):
        try:
            return {
                "__class__": obj.__class__.__name__,
                **safe_serialize(
                    {k: v for k, v in obj.__dict__.items() if not k.startswith("_")},
                    depth - 1,
                ),
            }
        except Exception:
            pass

    # Fallback to string representation
    try:
        val_str = str(obj)
        if len(val_str) > MAX_STRING_LENGTH:
            return val_str[:MAX_STRING_LENGTH] + "... [truncated]"
        return val_str
    except Exception:
        return f"<unserializable {type(obj).__name__}>"


def safe_json_dumps(obj: Any) -> str:
    """Safe json.dumps wrapper that always succeeds."""
    try:
        return json.dumps(safe_serialize(obj), default=str)
    except Exception as exc:
        return json.dumps({"__error__": f"json serialization failed: {exc}"})

