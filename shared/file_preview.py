import codecs
import csv
import io
import os
import threading
import time
from collections import OrderedDict
from datetime import date, datetime, time as dt_time
from decimal import Decimal
from pathlib import Path

from shared.dummy_files import DUMMY_STORAGE_PREFIX
from shared.storage import get_bucket_name, get_s3_client

MAX_PAGE_ROWS = 500
DEFAULT_PAGE_ROWS = 100
MAX_COLUMNS = 200
MAX_CELL_CHARS = 1000
TEXT_CHUNK_BYTES = 256 * 1024
MAX_CACHED_FILE_BYTES = 50 * 1024 * 1024
CACHE_BUDGET_BYTES = 150 * 1024 * 1024
CACHE_TTL_SECONDS = 600
CSV_SNIFF_BYTES = 64 * 1024

SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent / "analyzerEngine" / "sample_data"


class PreviewUnavailable(Exception):
    pass


class _ByteCache:
    def __init__(self, budget: int, ttl: float):
        self.budget = budget
        self.ttl = ttl
        self._items: "OrderedDict[str, tuple[float, bytes]]" = OrderedDict()
        self._size = 0
        self._lock = threading.Lock()

    def get(self, key: str) -> bytes | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires, data = item
            if expires < time.monotonic():
                self._items.pop(key)
                self._size -= len(data)
                return None
            self._items.move_to_end(key)
            return data

    def put(self, key: str, data: bytes) -> None:
        if len(data) > self.budget:
            return
        with self._lock:
            old = self._items.pop(key, None)
            if old is not None:
                self._size -= len(old[1])
            self._items[key] = (time.monotonic() + self.ttl, data)
            self._size += len(data)
            while self._size > self.budget and self._items:
                _, (_, evicted) = self._items.popitem(last=False)
                self._size -= len(evicted)


_cache = _ByteCache(CACHE_BUDGET_BYTES, CACHE_TTL_SECONDS)


def is_sample_key(storage_key: str | None) -> bool:
    return bool(storage_key) and storage_key.startswith(f"{DUMMY_STORAGE_PREFIX}/")


def _sample_path(storage_key: str) -> Path:
    name = os.path.basename(storage_key)
    path = SAMPLE_DATA_DIR / name
    if not path.is_file():
        raise PreviewUnavailable("Preview isn't available for this sample file.")
    return path


def object_size(storage_key: str) -> int:
    if is_sample_key(storage_key):
        return _sample_path(storage_key).stat().st_size
    try:
        head = get_s3_client().head_object(Bucket=get_bucket_name(), Key=storage_key)
    except Exception as exc:
        raise PreviewUnavailable("The original file couldn't be found in storage.") from exc
    return int(head.get("ContentLength") or 0)


def load_bytes(storage_key: str, max_bytes: int = MAX_CACHED_FILE_BYTES) -> bytes:
    cached = _cache.get(storage_key)
    if cached is not None:
        return cached
    size = object_size(storage_key)
    if size > max_bytes:
        raise PreviewUnavailable(
            f"This file is too large to preview ({size / (1024 * 1024):.0f} MB). Download it to view it."
        )
    if is_sample_key(storage_key):
        data = _sample_path(storage_key).read_bytes()
    else:
        try:
            body = get_s3_client().get_object(Bucket=get_bucket_name(), Key=storage_key)["Body"]
            data = body.read()
        except Exception as exc:
            raise PreviewUnavailable("The original file couldn't be read from storage.") from exc
    _cache.put(storage_key, data)
    return data


def read_range(storage_key: str, start: int, length: int) -> bytes:
    if length <= 0:
        return b""
    if is_sample_key(storage_key):
        with open(_sample_path(storage_key), "rb") as handle:
            handle.seek(start)
            return handle.read(length)
    cached = _cache.get(storage_key)
    if cached is not None:
        return cached[start:start + length]
    try:
        response = get_s3_client().get_object(
            Bucket=get_bucket_name(), Key=storage_key, Range=f"bytes={start}-{start + length - 1}",
        )
        return response["Body"].read()
    except Exception as exc:
        raise PreviewUnavailable("The original file couldn't be read from storage.") from exc


def format_cell(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        if value.hour == 0 and value.minute == 0 and value.second == 0 and value.microsecond == 0:
            return value.date().isoformat()
        return value.isoformat(sep=" ", timespec="seconds")
    if isinstance(value, (date, dt_time)):
        return value.isoformat()
    if isinstance(value, float):
        if value.is_integer() and abs(value) < 1e15:
            return str(int(value))
        return f"{value:.10g}"
    if isinstance(value, Decimal):
        return format(value.normalize(), "f")
    text = str(value)
    return text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "..."


def column_label(index: int) -> str:
    label = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        label = chr(65 + remainder) + label
    return label


def _trim_row(row: list) -> tuple[list[str], bool]:
    truncated = len(row) > MAX_COLUMNS
    return [format_cell(v) for v in row[:MAX_COLUMNS]], truncated


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1")


def _sniff_dialect(sample: str):
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        return csv.excel


def _table_page(rows_iter, offset: int, limit: int, count_all: bool) -> dict:
    rows: list[list[str]] = []
    width = 0
    truncated_columns = False
    total = 0
    has_more = False
    for index, raw in enumerate(rows_iter):
        total = index + 1
        if index < offset:
            continue
        if index >= offset + limit:
            has_more = True
            if not count_all:
                break
            continue
        cells, cut = _trim_row(list(raw))
        truncated_columns = truncated_columns or cut
        while cells and cells[-1] == "":
            cells.pop()
        width = max(width, len(cells))
        rows.append(cells)
    for row in rows:
        row.extend([""] * (width - len(row)))
    return {
        "columns": [column_label(i) for i in range(width)],
        "rows": rows,
        "offset": offset,
        "limit": limit,
        "total_rows": total if count_all else None,
        "has_more": has_more,
        "truncated_columns": truncated_columns,
    }


def csv_page(storage_key: str, offset: int, limit: int) -> dict:
    size = object_size(storage_key)
    if size <= MAX_CACHED_FILE_BYTES:
        text = _decode(load_bytes(storage_key))
        dialect = _sniff_dialect(text[:CSV_SNIFF_BYTES])
        page = _table_page(csv.reader(io.StringIO(text), dialect), offset, limit, count_all=True)
    else:
        page = _csv_stream_page(storage_key, offset, limit)
    page["sheet"] = None
    return page


def _csv_stream_page(storage_key: str, offset: int, limit: int) -> dict:
    try:
        body = get_s3_client().get_object(Bucket=get_bucket_name(), Key=storage_key)["Body"]
    except Exception as exc:
        raise PreviewUnavailable("The original file couldn't be read from storage.") from exc
    try:
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

        def lines():
            pending = ""
            for chunk in body.iter_chunks(chunk_size=256 * 1024):
                pending += decoder.decode(chunk)
                parts = pending.splitlines(keepends=True)
                pending = parts.pop() if parts and not parts[-1].endswith(("\n", "\r")) else ""
                yield from parts
            pending += decoder.decode(b"", final=True)
            if pending:
                yield pending

        line_iter = lines()
        first = next(line_iter, "")
        dialect = _sniff_dialect(first.lstrip("﻿"))

        def chained():
            yield first.lstrip("﻿")
            yield from line_iter

        return _table_page(csv.reader(chained(), dialect), offset, limit, count_all=False)
    finally:
        body.close()


def xlsx_sheets(storage_key: str) -> list[dict]:
    workbook = _open_workbook(storage_key)
    try:
        return [
            {"name": sheet.title, "rows": sheet.max_row, "columns": sheet.max_column}
            for sheet in workbook.worksheets
        ]
    finally:
        workbook.close()


def xlsx_page(storage_key: str, sheet_name: str | None, offset: int, limit: int) -> dict:
    workbook = _open_workbook(storage_key)
    try:
        if not workbook.worksheets:
            raise PreviewUnavailable("This workbook has no sheets.")
        if sheet_name is None:
            sheet = workbook.worksheets[0]
        elif sheet_name in workbook.sheetnames:
            sheet = workbook[sheet_name]
        else:
            raise PreviewUnavailable(f"Sheet '{sheet_name}' doesn't exist in this workbook.")
        max_col = min(sheet.max_column or 1, MAX_COLUMNS + 1)
        rows_iter = sheet.iter_rows(min_row=1, max_col=max_col, values_only=True)
        page = _table_page(rows_iter, offset, limit, count_all=True)
        if (sheet.max_column or 0) > MAX_COLUMNS:
            page["truncated_columns"] = True
        page["sheet"] = sheet.title
        return page
    finally:
        workbook.close()


def _open_workbook(storage_key: str):
    from openpyxl import load_workbook

    data = load_bytes(storage_key)
    try:
        return load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    except Exception as exc:
        raise PreviewUnavailable("This workbook couldn't be opened for preview.") from exc


def text_chunk(storage_key: str, offset: int) -> dict:
    total = object_size(storage_key)
    offset = max(0, min(offset, total))
    raw = read_range(storage_key, offset, min(TEXT_CHUNK_BYTES, total - offset))
    end = offset + len(raw)
    if end < total:
        cut = raw.rfind(b"\n")
        if cut > 0:
            raw = raw[:cut + 1]
            end = offset + len(raw)
        else:
            while raw and (raw[-1] & 0xC0) == 0x80:
                raw = raw[:-1]
            if raw and raw[-1] >= 0xC0:
                raw = raw[:-1]
            end = offset + len(raw)
    if offset == 0 and raw.startswith(codecs.BOM_UTF8):
        raw = raw[len(codecs.BOM_UTF8):]
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    return {
        "text": text,
        "offset": offset,
        "next_offset": end if end < total else None,
        "total_bytes": total,
    }
