import json
import os

import chatdb
from .look_at_image import _clean_source, _load_image

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
            "Optional images[] are local paths or http(s) URLs (png/jpeg/gif/webp/bmp); "
            "up to 4, scaled the same way as a human upload. Text may be empty when you attach an image. "
            "Do not spam. Write the way you actually talk."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": (
                        "Message body. Include @username when you want that person to reply. "
                        "May be empty when images[] is set."
                    ),
                },
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional extra tags (usually unnecessary if you use @mentions in text)."
                    ),
                },
                "images": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional image sources: workspace-relative path, absolute file path, "
                        "or http(s) URL. Same sources as look_at_image. At most 4."
                    ),
                },
            },
        },
    },
}


def _image_sources(images) -> list[str] | str:
    if images is None or images == "":
        return []
    if isinstance(images, str):
        items = [images]
    elif isinstance(images, list):
        items = images
    else:
        return "images must be a list of paths or URLs"
    out = []
    for item in items:
        src = _clean_source(str(item) if item is not None else "")
        if src:
            out.append(src)
    if len(out) > chatdb.MAX_IMAGES_PER_MESSAGE:
        return f"at most {chatdb.MAX_IMAGES_PER_MESSAGE} images per message"
    return out


def execute_post_message(ctx, text: str, tags=None, images=None) -> str:
    txt = (text or "").strip()
    sources = _image_sources(images)
    if isinstance(sources, str):
        return json.dumps({"status": "error", "message": sources})
    if not txt and not sources:
        return json.dumps({"status": "error", "message": "empty message"})
    if ctx is None or ctx.post_handler is None:
        return json.dumps({"status": "error", "message": "chat not connected"})
    workspace = ctx.tool_calls_dir if ctx else None
    saved = []
    try:
        for src in sources:
            data, _mime, label = _load_image(src, workspace)
            if label.startswith(("http://", "https://")):
                name = os.path.basename(label.split("?")[0]) or "image"
            else:
                name = os.path.basename(label)
            saved.append(chatdb.save_chat_image(data, name or "image"))
        tag_list = tags if isinstance(tags, list) else ([] if not tags else [tags])
        msg = ctx.post_handler(txt, tag_list, saved)
        posted = {
            "status": "posted",
            "id": msg.get("id"),
            "text": msg.get("text"),
            "tags": msg.get("tags"),
        }
        attached = msg.get("images") or []
        if attached:
            posted["images"] = [
                {
                    "name": im.get("name"),
                    "path": im.get("path"),
                    "url": im.get("url"),
                }
                for im in attached
                if isinstance(im, dict)
            ]
        return json.dumps(posted, ensure_ascii=False)
    except Exception as e:
        for record in saved:
            chatdb.delete_chat_image(record)
        return json.dumps({"status": "error", "message": str(e)})
