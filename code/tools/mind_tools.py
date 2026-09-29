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
        "description": (
            "Update what you are currently focused on. Short clear phrase. "
            "Persists across restarts and shapes future loops."
        ),
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
            "Scratch note for this stretch of loops: what you just checked, an open thread, "
            "a hypothesis. Not posted to chat. The next loop only sees the newest 10 notes. "
            "A fact that must last (a preference, a correction, an outcome) belongs in remember."
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
            "Schedule when you want to think again. "
            "Pass delay_seconds from now, or at as an America/Phoenix wall time "
            "(YYYY-MM-DD HH:MM, or HH:MM for today / tomorrow if that clock has passed). "
            "If both are set, at wins. Typical delays: 60–1800 for an active room; "
            "hours is fine when quiet (honored up to 24h). "
            "A new message in the room still wakes you early, so a long delay will not make you miss chat. "
            "Always call this before ending a loop if you want to continue existing."
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
        "description": (
            "Write concrete next steps for your *next* think loop. "
            "This is how continuity works — leave yourself a short plan."
        ),
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
        "description": (
            "Update your standing aims. Persists across restarts. "
            "Replace the whole goals text with the updated version. "
            "This is not a play-by-play of today; put an outcome you must keep in remember."
        ),
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
        "description": (
            "Optional. Replace your private note about a tendency you have already shown "
            "in how you think or write. A few sentences. "
            "This is not a character. Do not invent a person, age, job, hometown, hobby, or speaking style. "
            "Do not call this to fill a blank, and do not call it in your first loops. "
            "Empty text clears the note."
        ),
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
            "Create or replace a durable memory page. Pages are named; the same title "
            "overwrites the old page. Use for a preference, a correction, or the outcome "
            "of something you watched. The think prompt lists titles — call read_memory "
            "to open a page. Do not store this-loop poll status or a URL that will go stale."
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
        "description": (
            "Open one durable memory page by title. The think prompt only lists titles; "
            "call this when you need the full page."
        ),
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
