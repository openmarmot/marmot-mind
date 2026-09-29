"""Room clock.

Stored instants are UTC. The think prompt and the page show America/Phoenix.
"""

import re
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

ROOM_TZ = ZoneInfo("America/Phoenix")

_TIME_ONLY_RE = re.compile(r"^(\d{1,2}):(\d{2})(?::(\d{2}))?$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def utcnow_iso() -> str:
    return utcnow().isoformat()


def parse_instant(value) -> datetime | None:
    """Parse a stored timestamp.

    Aware values keep their offset. Naive values are room-local, which is
    what datetime.now() wrote before storage moved to UTC.
    """
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str) and value.strip():
        try:
            dt = datetime.fromisoformat(value.strip())
        except ValueError:
            return None
    else:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=ROOM_TZ)
    return dt


def to_room(value) -> datetime | None:
    dt = parse_instant(value)
    if dt is None:
        return None
    return dt.astimezone(ROOM_TZ)


def format_room(value, *, with_date: bool = True) -> str:
    local = to_room(value)
    if local is None:
        return ""
    stamp = local.strftime("%Y-%m-%d %H:%M:%S" if with_date else "%H:%M")
    return f"{stamp} {local.tzname() or 'MST'}"


def format_room_day(value) -> str:
    local = to_room(value)
    if local is None:
        return ""
    return local.strftime("%Y-%m-%d")


def offset_label(value=None) -> str:
    local = to_room(value) if value is not None else utcnow().astimezone(ROOM_TZ)
    if local is None:
        local = utcnow().astimezone(ROOM_TZ)
    z = local.strftime("%z") or "-0700"
    hours = int(z[:3])
    minutes = int(z[3:5] or "0")
    if minutes:
        return f"UTC{hours:+d}:{minutes:02d}"
    return f"UTC{hours:+d}"


def prompt_clock_line(now=None) -> str:
    now = now or utcnow()
    local = now.astimezone(ROOM_TZ)
    stamp = local.strftime("%Y-%m-%d %H:%M:%S")
    name = local.tzname() or "MST"
    return (
        f"Current time: {stamp} {name} ({offset_label(local)}). "
        "Times in this prompt use this clock. "
        "When you tell the room a time, use it. "
        "Convert a source that is in UTC or another zone first."
    )


def parse_room_at(value, now=None) -> datetime | None:
    """Parse a wake time.

    Naive datetimes and bare clock times are America/Phoenix. A time with an
    offset is kept. A time-only value (HH:MM) is today, or tomorrow if that
    clock time has already passed.
    """
    raw = (value or "").strip() if isinstance(value, str) else ""
    if not raw:
        return None
    now_utc = now or utcnow()
    if now_utc.tzinfo is None:
        now_utc = now_utc.replace(tzinfo=timezone.utc)
    local_now = now_utc.astimezone(ROOM_TZ)

    m = _TIME_ONLY_RE.match(raw)
    if m:
        hour, minute, second = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if hour > 23 or minute > 59 or second > 59:
            return None
        target = local_now.replace(hour=hour, minute=minute, second=second, microsecond=0)
        if target <= local_now:
            target += timedelta(days=1)
        return target

    normalized = raw.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=ROOM_TZ)
    return dt


def after_seconds_iso(seconds: float, now=None) -> str:
    now = now or utcnow()
    return (now + timedelta(seconds=seconds)).isoformat()


def seconds_until(value, now=None) -> float | None:
    target = parse_instant(value)
    if target is None:
        return None
    now = now or utcnow()
    return (target - now).total_seconds()
