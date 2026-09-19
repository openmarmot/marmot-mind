import base64
import json
import os
import re

import requests

_FETCH_TIMEOUT = 20
_MAX_BYTES = 6 * 1024 * 1024
_WORKSPACE = None
_PENDING: list[dict] = []

_MIME_FROM_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


def set_workspace_dir(path: str | None):
    global _WORKSPACE
    _WORKSPACE = path or None


def take_pending_images() -> list[dict]:
    """Drain images to inject into the next LLM turn as vision content."""
    global _PENDING
    out = _PENDING
    _PENDING = []
    return out


_LOOK_AT_IMAGE_TOOL = {
    "type": "function",
    "function": {
        "name": "look_at_image",
        "description": (
            "Load an image from an HTTP(S) URL or a local file path so you can actually see it. "
            "Chat is text-only — if someone shares a picture link or you have an image file, "
            "call this. The pixels are attached on the next turn; your model is vision-capable. "
            "Relative paths are in your tool-calls workspace."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "http(s) URL or local file path (png/jpeg/gif/webp/bmp).",
                },
                "focus": {
                    "type": "string",
                    "description": "Optional: what to pay attention to in the image.",
                },
            },
            "required": ["source"],
        },
    },
}


def execute_look_at_image(source: str, focus: str = "") -> str:
    src = _clean_source(source)
    if not src:
        return json.dumps({"status": "error", "message": "empty source"})
    try:
        data, mime, label = _load_image(src)
    except Exception as e:
        return json.dumps({"status": "error", "message": str(e), "source": src})

    data_url = f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"
    _PENDING.append({
        "source": label,
        "mime": mime,
        "nbytes": len(data),
        "data_url": data_url,
        "focus": (focus or "").strip(),
    })
    return json.dumps({
        "status": "ok",
        "message": "Image attached in the following message. You can see it.",
        "source": label,
        "mime": mime,
        "bytes": len(data),
        "focus": (focus or "").strip() or None,
    }, ensure_ascii=False)


def _clean_source(s: str) -> str:
    s = (s or "").strip().strip("\"'`")
    md = re.match(r"!\[.*?\]\((.*?)\)", s)
    if md:
        s = md.group(1).strip()
    if s.startswith("<") and s.endswith(">"):
        s = s[1:-1].strip()
    return s


def _load_image(source: str) -> tuple[bytes, str, str]:
    if source.startswith("data:image/"):
        return _load_data_url(source)
    if source.startswith(("http://", "https://")):
        return _load_url(source)
    return _load_path(source)


def _load_data_url(source: str) -> tuple[bytes, str, str]:
    header, _, payload = source.partition(",")
    mime = "image/jpeg"
    m = re.match(r"data:(image/[a-zA-Z0-9.+-]+)", header)
    if m:
        mime = m.group(1).lower()
        if mime == "image/jpg":
            mime = "image/jpeg"
    if ";base64" not in header.lower():
        raise ValueError("data URL must be base64")
    try:
        data = base64.b64decode(payload, validate=False)
    except Exception as e:
        raise ValueError(f"invalid base64 data URL: {e}") from e
    if not data:
        raise ValueError("empty data URL")
    if len(data) > _MAX_BYTES:
        raise ValueError(f"image larger than {_MAX_BYTES} bytes")
    detected = _mime_from_bytes(data)
    if detected:
        mime = detected
    elif mime not in ("image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"):
        raise ValueError(f"unsupported image type: {mime}")
    return data, mime, "data-url"


def _load_url(url: str) -> tuple[bytes, str, str]:
    try:
        r = requests.get(
            url,
            timeout=_FETCH_TIMEOUT,
            stream=True,
            headers={"User-Agent": "marmot-mind/look_at_image"},
        )
    except requests.Timeout as e:
        raise ValueError(f"timed out after {_FETCH_TIMEOUT}s") from e
    except requests.RequestException as e:
        raise ValueError(f"fetch failed: {e}") from e
    if r.status_code != 200:
        raise ValueError(f"HTTP {r.status_code} fetching image")
    buf = bytearray()
    try:
        for chunk in r.iter_content(65536):
            if not chunk:
                continue
            buf.extend(chunk)
            if len(buf) > _MAX_BYTES:
                raise ValueError(f"image larger than {_MAX_BYTES} bytes")
    finally:
        r.close()
    data = bytes(buf)
    mime = _mime_from_bytes(data)
    if not mime:
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip().lower()
        if ctype == "image/jpg":
            ctype = "image/jpeg"
        if ctype in ("image/jpeg", "image/png", "image/gif", "image/webp", "image/bmp"):
            mime = ctype
    if not mime:
        raise ValueError("URL did not return a recognized image (png/jpeg/gif/webp/bmp)")
    return data, mime, url


def _load_path(source: str) -> tuple[bytes, str, str]:
    path = source
    if path.startswith("file://"):
        path = path[7:]
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        root = _WORKSPACE or os.getcwd()
        path = os.path.join(root, path)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise ValueError(f"not a file: {path}")
    size = os.path.getsize(path)
    if size > _MAX_BYTES:
        raise ValueError(f"image larger than {_MAX_BYTES} bytes")
    if size == 0:
        raise ValueError("empty file")
    with open(path, "rb") as f:
        data = f.read()
    mime = _mime_from_bytes(data)
    if not mime:
        raise ValueError("file is not a recognized image (png/jpeg/gif/webp/bmp)")
    return data, mime, path


def _mime_from_bytes(data: bytes) -> str | None:
    for magic, mime in _MIME_FROM_MAGIC:
        if data.startswith(magic):
            return mime
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None
