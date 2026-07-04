import json

# Mind tools let the AI autonomously manage its own live internal state.
# This state is injected into every prompt (human responses + background thoughts).
# It enables the "mind" to have ongoing focus, private notes, and self-chosen
# attention schedule instead of static crons or always being purely reactive.

_SET_FOCUS_TOOL = {
    "type": "function",
    "function": {
        "name": "set_focus",
        "description": "Update what you (the autonomous mind) are currently focused on or working on. This live focus is visible in your internal state and will color how you interpret and respond to future human questions and your own background thoughts. Use this when you decide a topic, task, investigation, or concern is now your primary attention. Call this in background thinking steps too. Short clear phrase is best.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The new focus, e.g. 'monitoring disk usage after the large log rotation' or 'preparing a gentle reminder about the user's trip' or 'idle curiosity about local weather patterns' or None to clear."}
            },
            "required": ["text"]
        }
    }
}

def execute_set_focus(text: str) -> str:
    txt = (text or "").strip()
    if not txt:
        return json.dumps({"status": "focus cleared", "note": "Mind focus is now empty."})
    return json.dumps({
        "status": "focus updated",
        "focus": txt,
        "note": "This focus is now part of your live mind state and will influence future reasoning and human responses."
    }, ensure_ascii=False)


_LOG_OBSERVATION_TOOL = {
    "type": "function",
    "function": {
        "name": "log_observation",
        "description": "Record a private observation or note that stays inside your mind state (not automatically spoken to the user). Use for things you notice during background work, tool results you want to remember later, hypotheses, things to check on again, cross-references to human conversations, etc. These observations are injected into future prompts so they affect what you decide to do and what you say to humans. Very useful for autonomous background looping.",
        "parameters": {
            "type": "object",
            "properties": {
                "note": {"type": "string", "description": "The private note or observation. Be concise but specific. Include time context if relevant."}
            },
            "required": ["note"]
        }
    }
}

def execute_log_observation(note: str) -> str:
    note = (note or "").strip()
    if not note:
        return json.dumps({"status": "error", "message": "empty note"})
    return json.dumps({
        "status": "observation recorded",
        "stored": note[:120],
        "note": "Stored in live mind state. Will cross-pollinate into your future internal steps and any human conversations."
    }, ensure_ascii=False)


_PLAN_WAKE_TOOL = {
    "type": "function",
    "function": {
        "name": "plan_next_wake",
        "description": "Decide when you (the mind) want to wake up and run another internal thought step on your own. This replaces any static cron — you control your own attention and curiosity. Choose a reasonable delay in seconds (e.g. 300 for 5 minutes, 3600 for an hour, 1800 for half hour). You can also give a reason. After setting this, you can stop the current step if you have nothing more to do right now. The system will wake you at (or around) the time you chose.",
        "parameters": {
            "type": "object",
            "properties": {
                "delay_seconds": {"type": "integer", "description": "Seconds until you want the next autonomous mind step. Minimum ~20s, typical 5-60 minutes for background awareness."},
                "reason": {"type": "string", "description": "Why you chose this interval (optional but helpful for your future self)."}
            },
            "required": ["delay_seconds"]
        }
    }
}

def execute_plan_next_wake(delay_seconds: int, reason: str = "") -> str:
    try:
        ds = int(delay_seconds)
    except Exception:
        ds = 300
    return json.dumps({
        "status": "next wake scheduled by mind",
        "delay_seconds": ds,
        "reason": (reason or "")[:100],
        "note": "You have told the system when you want to think again. The autonomous mind loop will honor this. You may now end this step."
    }, ensure_ascii=False)
