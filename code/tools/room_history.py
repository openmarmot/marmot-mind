"""Read the shared room. Does not post and does not move last_seen_message_id."""

from clock import format_room

_READ_DEFAULT = 40
_READ_MAX = 80
_SEARCH_DEFAULT = 20
_SEARCH_MAX = 40
_QUERY_MAX = 200


_READ_MESSAGES_TOOL = {
    "type": "function",
    "function": {
        "name": "read_messages",
        "description": (
            "Read room messages by id. The notes already include the latest ones. "
            "before_id returns older messages (id less than that). "
            "after_id returns later messages (id greater than that). "
            "Omit both for the latest page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "before_id": {
                    "type": "integer",
                    "description": "Return messages with a smaller id. Pass the oldest id you can see to page backward.",
                },
                "after_id": {
                    "type": "integer",
                    "description": "Return messages with a larger id.",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many messages (1–80, default 40).",
                },
            },
        },
    },
}


_SEARCH_MESSAGES_TOOL = {
    "type": "function",
    "function": {
        "name": "search_messages",
        "description": (
            "Find room messages whose text contains a phrase, case-insensitive. "
            "Returns the newest matches, oldest of that page first."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Phrase to find in message text.",
                },
                "limit": {
                    "type": "integer",
                    "description": "How many matches (1–40, default 20).",
                },
            },
            "required": ["query"],
        },
    },
}


def _optional_id(value, name: str):
    """Missing stays None. Otherwise an int >= 0, or an error string."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return f"Error: {name} must be an integer"
    try:
        n = int(value)
    except (TypeError, ValueError):
        return f"Error: {name} must be an integer"
    if n < 0:
        return f"Error: {name} must be >= 0"
    return n


def _limit(value, *, default: int, hi: int, name: str):
    """(n, note) or an error string. Over-max is clamped."""
    if value is None or value == "":
        return default, ""
    if isinstance(value, bool):
        return f"Error: {name} must be an integer"
    try:
        n = int(value)
    except (TypeError, ValueError):
        return f"Error: {name} must be an integer"
    if n < 1:
        return f"Error: {name} must be at least 1"
    if n > hi:
        return hi, f"{name} capped at {hi}."
    return n, ""


def _username(ctx) -> str:
    store = getattr(ctx, "store", None) if ctx else None
    return (getattr(store, "username", "") or "") if store else ""


def _format_history(messages: list, username: str) -> str:
    lines = []
    for m in messages:
        tags = m.get("tags") or []
        tag_s = f" tags=[{', '.join(str(t) for t in tags)}]" if tags else ""
        self_mark = " (you)" if username and m.get("username") == username else ""
        body = m.get("text") or ""
        if not body and m.get("images"):
            body = "(image only)"
        when = format_room(m.get("created_at") or "")
        stamp = f"[{when}] " if when else ""
        attached = _image_lines(m)
        lines.append(
            f"#{m.get('id')} {stamp}{m.get('username')}{self_mark}{tag_s}:\n{body}{attached}"
        )
    return "\n\n".join(lines)


def _image_lines(msg: dict) -> str:
    images = msg.get("images") or []
    lines = []
    for img in images:
        if not isinstance(img, dict):
            continue
        name = img.get("name") or "image"
        source = img.get("path") or img.get("url") or ""
        lines.append(f"  [attached image {name}: {source}]")
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def execute_read_messages(ctx, before_id=None, after_id=None, limit=None) -> str:
    room = getattr(ctx, "room", None) if ctx else None
    if room is None:
        return "Error: chat not connected"
    before = _optional_id(before_id, "before_id")
    if isinstance(before, str):
        return before
    after = _optional_id(after_id, "after_id")
    if isinstance(after, str):
        return after
    bounded = _limit(limit, default=_READ_DEFAULT, hi=_READ_MAX, name="limit")
    if isinstance(bounded, str):
        return bounded
    n, note = bounded
    who = _username(ctx)
    data = room.read_messages(before_id=before, after_id=after, limit=n, as_user=who or None)
    messages = data.get("messages") or []
    oldest = int(data.get("oldest_id") or 0)
    latest = int(data.get("latest_id") or 0)
    if not messages:
        if not oldest:
            return "No messages in the room."
        return f"No messages in that range. Room ids run from #{oldest} to #{latest}."
    return _read_header(messages, oldest, latest, note) + "\n\n" + _format_history(messages, who)


def _read_header(messages: list, oldest: int, latest: int, note: str) -> str:
    first = messages[0].get("id")
    last = messages[-1].get("id")
    bits = [f"{len(messages)} messages, #{first}–#{last}."]
    if oldest and latest:
        bits.append(f"Room ids #{oldest}–#{latest}.")
    if note:
        bits.append(note)
    extra = []
    if oldest and first and first > oldest:
        extra.append(f"read_messages(before_id={first}) for earlier.")
    if latest and last and last < latest:
        extra.append(f"read_messages(after_id={last}) for later.")
    lines = [" ".join(bits)]
    if extra:
        lines.append(" ".join(extra))
    return "\n".join(lines)


def execute_search_messages(ctx, query: str, limit=None) -> str:
    room = getattr(ctx, "room", None) if ctx else None
    if room is None:
        return "Error: chat not connected"
    q = (query or "").strip()
    if not q:
        return "Error: empty query"
    if len(q) > _QUERY_MAX:
        return f"Error: query is too long (max {_QUERY_MAX} characters)"
    bounded = _limit(limit, default=_SEARCH_DEFAULT, hi=_SEARCH_MAX, name="limit")
    if isinstance(bounded, str):
        return bounded
    n, note = bounded
    who = _username(ctx)
    data = room.search_messages(q, limit=n, as_user=who or None)
    messages = data.get("messages") or []
    total = int(data.get("total") or 0)
    if not messages:
        return f"No messages contain {q!r}."
    first = messages[0].get("id")
    last = messages[-1].get("id")
    if total > len(messages):
        head = (
            f"{len(messages)} newest of {total} matches for {q!r} "
            f"(#{first}–#{last}, oldest first)."
        )
    else:
        head = f"{total} matches for {q!r} (#{first}–#{last}, oldest first)."
    if note:
        head = head + " " + note
    return head + "\n\n" + _format_history(messages, who)
