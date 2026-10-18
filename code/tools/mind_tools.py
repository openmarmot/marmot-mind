import json

from clock import after_seconds_iso, format_room, parse_room_at, seconds_until

# Side effects (persist to store) are applied by the agent loop in mind.py.
# These executors return confirmation payloads for the LLM.

_MIN_WAKE_SECONDS = 20
_MAX_WAKE_SECONDS = 24 * 3600
_DEFAULT_WAKE_SECONDS = 300


_SET_FOCUS_TOOL = {
    "type": "function",
    "function": {
        "name": "set_focus",
        "description": "Replace your current focus. Empty text clears it. Shown on later loops.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "New focus, or empty to clear."}
            },
            "required": ["text"],
        },
    },
}


def execute_set_focus(text: str) -> str:
    txt = (text or "").strip()
    if not txt:
        return json.dumps({"status": "focus cleared"})
    return json.dumps({"status": "focus updated", "focus": txt}, ensure_ascii=False)


_LOG_OBSERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "log_observation",
        "description": (
            "Save a private scratch note. Not posted. The next loop sees the newest 10."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "This-loop scratch note."}
            },
            "required": ["note"],
        },
    },
}


def execute_log_observation(note: str) -> str:
    note = (note or "").strip()
    if not note:
        return json.dumps({"status": "error", "message": "empty note"})
    return json.dumps({"status": "observation recorded", "stored": note}, ensure_ascii=False)


_PLAN_WAKE_TOOL = {
    "type": "function",
    "function": {
        "name": "plan_next_wake",
        "description": (
            "Schedule the next time you think. Pass delay_seconds, or at as an "
            "America/Phoenix time (YYYY-MM-DD HH:MM, or HH:MM). If both are set, at wins. "
            "About 20 seconds to 24 hours. A new room message can wake you sooner. "
            "Skipping this waits 5 minutes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "integer",
                    "description": "Seconds until next think loop (min ~20). Ignored when at is set.",
                },
                "at": {
                    "type": "string",
                    "description": (
                        "Wake time in America/Phoenix, e.g. '2026-10-05 21:00' or '21:00'. "
                        "An offset in the string is honored."
                    ),
                },
                "reason": {"type": "string", "description": "Why this interval."},
            },
        },
    },
}


def resolve_next_wake(delay_seconds=None, at=None, now=None) -> dict | None:
    """Clamp a wake to 20s–24h. `at` wins over delay_seconds. None if `at` is junk."""
    at_raw = at.strip() if isinstance(at, str) else ""
    if at_raw:
        target = parse_room_at(at_raw, now=now)
        if target is None:
            return None
        delta = seconds_until(target, now=now)
        if delta is None:
            return None
        secs = max(_MIN_WAKE_SECONDS, min(_MAX_WAKE_SECONDS, int(delta)))
        return {
            "delay_seconds": secs,
            "next_wake_after": after_seconds_iso(secs, now=now),
            "at": format_room(target),
        }
    try:
        if delay_seconds is None or delay_seconds == "":
            ds = _DEFAULT_WAKE_SECONDS
        else:
            ds = int(delay_seconds)
        ds = max(_MIN_WAKE_SECONDS, min(_MAX_WAKE_SECONDS, ds))
    except Exception:
        ds = _DEFAULT_WAKE_SECONDS
    return {
        "delay_seconds": ds,
        "next_wake_after": after_seconds_iso(ds, now=now),
        "at": None,
    }


def execute_plan_next_wake(delay_seconds=None, reason: str = "", at: str = "") -> str:
    resolved = resolve_next_wake(delay_seconds, at)
    if not resolved:
        return json.dumps({
            "status": "error",
            "message": "could not parse at; use America/Phoenix YYYY-MM-DD HH:MM or HH:MM",
        })
    payload = {
        "status": "next wake scheduled",
        "delay_seconds": resolved["delay_seconds"],
        "reason": (reason or "")[:120],
    }
    if resolved["at"]:
        payload["at"] = resolved["at"]
    return json.dumps(payload, ensure_ascii=False)


_WRITE_NEXT_STEPS_TOOL = {
    "type": "function",
    "function": {
        "name": "write_next_steps",
        "description": "Replace the next-steps note shown on the next loop.",
        "parameters": {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "string",
                    "description": "Bullet list or short paragraph of next steps.",
                }
            },
            "required": ["steps"],
        },
    },
}


def execute_write_next_steps(steps: str) -> str:
    steps = (steps or "").strip()
    if not steps:
        return json.dumps({"status": "error", "message": "empty steps"})
    return json.dumps({"status": "next steps saved", "steps": steps}, ensure_ascii=False)


_UPDATE_GOALS_TOOL = {
    "type": "function",
    "function": {
        "name": "update_goals",
        "description": "Replace your goals. Empty text clears them. Shown on later loops.",
        "parameters": {
            "type": "object",
            "properties": {
                "goals": {"type": "string", "description": "Full updated goals text."}
            },
            "required": ["goals"],
        },
    },
}


def execute_update_goals(goals: str) -> str:
    goals = (goals or "").strip()
    return json.dumps({"status": "goals updated", "goals": goals}, ensure_ascii=False)


_UPDATE_PERSONALITY_TOOL = {
    "type": "function",
    "function": {
        "name": "update_personality",
        "description": "Replace your private note about yourself. Empty text clears it.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The whole note, or empty to clear.",
                }
            },
            "required": ["text"],
        },
    },
}


def execute_update_personality(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return json.dumps({"status": "personality cleared"})
    return json.dumps({"status": "personality noted", "text": text[:500]}, ensure_ascii=False)


_REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Create or replace a durable memory page. The same title overwrites that page."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Page name. Same title replaces the page.",
                },
                "body": {
                    "type": "string",
                    "description": "Full page text (markdown).",
                },
            },
            "required": ["title", "body"],
        },
    },
}


def execute_remember(ctx, title: str, body: str) -> str:
    store = getattr(ctx, "store", None)
    if store is None:
        return json.dumps({"status": "error", "message": "memory store unavailable"})
    written = store.write_topic(title, body)
    if not written:
        return json.dumps({"status": "error", "message": "need a title and a real body"})
    return json.dumps(
        {"status": "remembered", "title": written["title"], "slug": written["slug"]},
        ensure_ascii=False,
    )


_READ_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "read_memory",
        "description": "Open one memory page by title. The prompt lists titles, not the page body.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Page title from the memory index."}
            },
            "required": ["title"],
        },
    },
}


def execute_read_memory(ctx, title: str) -> str:
    store = getattr(ctx, "store", None)
    if store is None:
        return json.dumps({"status": "error", "message": "memory store unavailable"})
    found = store.read_topic(title)
    if not found:
        titles = [t["title"] for t in store.list_topics()]
        return json.dumps(
            {
                "status": "error",
                "message": "no page with that title",
                "titles": titles,
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {"status": "ok", "title": found["title"], "body": found["body"]},
        ensure_ascii=False,
    )


_FORGET_TOOL = {
    "type": "function",
    "function": {
        "name": "forget",
        "description": "Delete a durable memory page by title. The index updates immediately.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Page title to remove."}
            },
            "required": ["title"],
        },
    },
}


def execute_forget(ctx, title: str) -> str:
    store = getattr(ctx, "store", None)
    if store is None:
        return json.dumps({"status": "error", "message": "memory store unavailable"})
    removed = store.delete_topic(title)
    if not removed:
        titles = [t["title"] for t in store.list_topics()]
        return json.dumps(
            {
                "status": "error",
                "message": "no page with that title",
                "titles": titles,
            },
            ensure_ascii=False,
        )
    return json.dumps(
        {"status": "forgotten", "title": removed["title"]},
        ensure_ascii=False,
    )
