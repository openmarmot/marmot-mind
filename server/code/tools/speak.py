import json

_SPEAK_TOOL = {
    "type": "function",
    "function": {
        "name": "speak",
        "description": "MANDATORY - ONLY way to talk to the user: Call this (and ONLY this) to deliver ANY text the user should hear, including direct answers, pronunciations, definitions, and explanations with no other tools needed. Plain content is NEVER delivered. Even for simple replies with no other tools required, you must call speak() instead of emitting plain text. Call speak() sparingly (1-2 times per query is usually enough). Use natural spoken English. Do additional non-speak tool work if needed before speaking again. NEVER repeat yourself with 'sorry', 'I'll stop', 'I'm done' or similar fillers. After giving the answer/greeting, stop calling speak and end. If nothing to say, stop without tool calls.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The exact natural spoken text to deliver to the user via TTS."}
            },
            "required": ["text"]
        }
    }
}


def execute_speak(text: str) -> str:
    """Pure function: returns the standardized result payload for the LLM.
    The actual queuing + TTS is performed by the server when it sees a speak tool call.
    This keeps the tool decoupled from server internals.
    """
    txt = (text or "").strip()
    if not txt:
        return json.dumps({"status": "error", "message": "empty text"})

    return json.dumps({
        "status": "audio queued for delivery to user",
        "text_spoken": txt,
        "note": "The user will hear the text above. If you are done communicating this to the user, stop now (do not emit plain content or repeat the text). You may call speak again or other tools if needed."
    }, ensure_ascii=False)
