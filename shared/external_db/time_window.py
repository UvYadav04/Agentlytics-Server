import calendar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

UNITS = ("day", "week", "month", "quarter", "year")
MAX_LAST = {"day": 3660, "week": 520, "month": 120, "quarter": 40, "year": 10}


@dataclass
class ResolvedWindow:
    start: datetime
    end: datetime


def _minus_months(value: datetime, months: int) -> datetime:
    month_index = value.month - 1 - months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def validate_window(window: Optional[dict]) -> Optional[dict]:
    if not window:
        return None
    column = str(window.get("column") or "").strip()
    unit = str(window.get("unit") or "").strip().lower().rstrip("s")
    try:
        last = int(window.get("last"))
    except (TypeError, ValueError):
        raise ValueError("time window 'last' must be a whole number")
    if not column:
        raise ValueError("time window needs the date column it filters on")
    if unit not in UNITS:
        raise ValueError(f"time window unit must be one of: {', '.join(UNITS)}")
    if not 1 <= last <= MAX_LAST[unit]:
        raise ValueError(f"time window 'last' must be between 1 and {MAX_LAST[unit]} for unit '{unit}'")
    return {"column": column, "last": last, "unit": unit}


def resolve_window(window: Optional[dict], now: Optional[datetime] = None) -> Optional[ResolvedWindow]:
    if not window:
        return None
    end = now or datetime.now(timezone.utc)
    last, unit = int(window["last"]), window["unit"]
    if unit == "day":
        start = end - timedelta(days=last)
    elif unit == "week":
        start = end - timedelta(weeks=last)
    elif unit == "month":
        start = _minus_months(end, last)
    elif unit == "quarter":
        start = _minus_months(end, last * 3)
    else:
        start = _minus_months(end, last * 12)
    return ResolvedWindow(start=start, end=end)
