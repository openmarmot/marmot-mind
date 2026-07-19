import json

_post_handler = None


def set_post_handler(handler):
    """handler(text, tags) -> dict message or raises."""
    global _post_handler
    _post_handler = handler


_POST_MESSAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "post_message",
        "description": (
            "Post a message to the shared chat room. This is how you talk to humans and other minds. "
            "Use tags to notify specific users (list of usernames) or everyone (include 'everyone'). "
            "Tag users when you are replying to them or need their attention. "
            "Do not spam. Keep messages natural and in character."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Message body to post in the chat room.",
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Who to tag: usernames, and/or 'everyone'. "
                        "Empty list = untagged ambient message."
                    ),
                },
            },
            "required": ["text"],
        },
    },
}


def execute_post_message(text: str, tags=None) -> str:
    txt = (text or "").strip()
    if not txt:
        return json.dumps({"status": "error", "message": "empty text"})
    if _post_handler is None:
        return json.dumps({"status": "error", "message": "chat not connected"})
    try:
        tag_list = tags if isinstance(tags, list) else ([] if not tags else [tags])
        msg = _post_handler(txt, tag_list)
        return json.dumps({
            "status": "posted",
            "id": msg.get("id"),
            "text": msg.get("text"),
            "tags": msg.get("tags"),
        }, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e)})
