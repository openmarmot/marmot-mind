import json

# Side effects (persist to store) are applied by the agent loop in mind.py.
# These executors return confirmation payloads for the LLM.


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
            "Record a private observation for your future self. Not posted to chat. "
            "Use for things you notice, open threads, hypotheses, etc."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "Concise private note."}
            },
            "required": ["note"],
        },
    },
}


def execute_log_observation(note: str) -> str:
    note = (note or "").strip()
    if not note:
        return json.dumps({"status": "error", "message": "empty note"})
    return json.dumps({"status": "observation recorded", "stored": note[:120]}, ensure_ascii=False)


_PLAN_WAKE_TOOL = {
    "type": "function",
    "function": {
        "name": "plan_next_wake",
        "description": (
            "Schedule when you want to think again (seconds from now). "
            "You control your own attention cadence. Typical: 60–1800 for active rooms, "
            "longer when quiet. Always call this before ending a loop if you want to continue existing."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "integer",
                    "description": "Seconds until next think loop (min ~20).",
                },
                "reason": {"type": "string", "description": "Why this interval."},
            },
            "required": ["delay_seconds"],
        },
    },
}


def execute_plan_next_wake(delay_seconds: int, reason: str = "") -> str:
    try:
        ds = int(delay_seconds)
    except Exception:
        ds = 300
    return json.dumps({
        "status": "next wake scheduled",
        "delay_seconds": ds,
        "reason": (reason or "")[:120],
    }, ensure_ascii=False)


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
    return json.dumps({"status": "next steps saved", "steps": steps[:500]}, ensure_ascii=False)


_UPDATE_GOALS_TOOL = {
    "type": "function",
    "function": {
        "name": "update_goals",
        "description": (
            "Update your medium/long-term goals. Persists across restarts. "
            "Replace the whole goals text with the updated version."
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
    return json.dumps({"status": "goals updated", "goals": goals[:500]}, ensure_ascii=False)


_REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Store a durable fact in long-term memory (survives many loops). "
            "Use for important user preferences, commitments, or discoveries."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "Fact to remember."}
            },
            "required": ["note"],
        },
    },
}


def execute_remember(note: str) -> str:
    note = (note or "").strip()
    if not note:
        return json.dumps({"status": "error", "message": "empty note"})
    return json.dumps({"status": "remembered", "note": note[:200]}, ensure_ascii=False)
