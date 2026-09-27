import json

_POST_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "post_message",
        "description": (
            "Post a message to the shared chat room. This writes directly into the room "
            "and wakes every other running mind so they can read it. "
            "Include @username when you want a particular person to reply (e.g. 'hey @alice status?'). "
            "Use @everyone only when the whole room truly needs it. The server turns @mentions into tags. "
            "Optional tags[] still works, but prefer @mentions in the text. "
            "Do not spam. Write the way you actually talk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Message body. Include @username when you want that person to reply."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional extra tags (usually unnecessary if you use @mentions in text)."
                    ),
                },
            },
            "required": ["text"],
        },
    },
}


def execute_post_message(ctx, text: str, tags=None) -> str:
    txt = (text or "").strip()
    if not txt:
        return json.dumps({"status": "error", "message": "empty text"})
    if ctx is None or ctx.post_handler is None:
        return json.dumps({"status": "error", "message": "chat not connected"})
    try:
        tag_list = tags if isinstance(tags, list) else ([] if not tags else [tags])
        msg = ctx.post_handler(txt, tag_list)
        return json.dumps({
            "status": "posted",
            "id": msg.get("id"),
            "text": msg.get("text"),
            "tags": msg.get("tags"),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
