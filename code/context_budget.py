"""Fit a think-loop prompt into a configured context window.

Token counts are a conservative estimate (no model tokenizer). Images count as
vision tokens, not as the length of their base64. When the prompt is over the
window, older tool results are shortened and older chat is dropped so a reply
still fits.
"""

import json

# Leave at least this much of the window for the reply, when the window is large.
_REPLY_FLOOR = 1024
# A vision part is expensive, but nothing like its base64 text.
_IMAGE_TOKENS = 1200


def estimate_tokens(text: str) -> int:
    """Overestimate tokens so we shrink before the server rejects the prompt."""
    if not text:
        return 0
    other = 0
    ascii_n = 0
    for ch in text:
        if ord(ch) < 128:
            ascii_n += 1
        else:
            other += 1
    return (ascii_n + 2) // 3 + other


def _content_tokens(content) -> int:
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_tokens(content)
    if isinstance(content, list):
        total = 0
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image_url":
                total += _IMAGE_TOKENS
            elif isinstance(part, dict):
                total += estimate_tokens(part.get("text") or "")
            else:
                total += estimate_tokens(str(part))
        return total
    return estimate_tokens(json.dumps(content, ensure_ascii=False))


def estimate_message(msg: dict) -> int:
    total = 4 + _content_tokens(msg.get("content"))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        total += 8
        total += estimate_tokens(fn.get("name") or "")
        total += estimate_tokens(fn.get("arguments") or "")
    return total


def estimate_messages(messages: list, tools=None) -> int:
    total = sum(estimate_message(m) for m in messages)
    if tools:
        total += estimate_tokens(json.dumps(tools, ensure_ascii=False))
    return total


def reply_room(max_context: int) -> int:
    """Tokens held back so the reply is not squeezed to nothing."""
    if max_context <= 2:
        return 1
    return min(_REPLY_FLOOR, max_context // 4)


def _oneline(content, limit: int = 160) -> str:
    if isinstance(content, list):
        texts = []
        images = 0
        for part in content:
            if not isinstance(part, dict):
                texts.append(str(part))
            elif part.get("type") == "image_url":
                images += 1
            elif part.get("text"):
                texts.append(part["text"])
        text = " ".join(texts)
        if images:
            text += f" [{images} image(s)]"
    else:
        text = "" if content is None else str(content)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: limit - 1] + "…"
    return text or "(empty)"


def _has_image(msg: dict) -> bool:
    content = msg.get("content")
    return isinstance(content, list) and any(
        isinstance(p, dict) and p.get("type") == "image_url" for p in content
    )


def _strip_images(msg: dict) -> bool:
    content = msg.get("content")
    if not isinstance(content, list) or not _has_image(msg):
        return False
    kept = [p for p in content if not (isinstance(p, dict) and p.get("type") == "image_url")]
    kept.append({"type": "text", "text": "[image cleared to fit context]"})
    msg["content"] = kept
    return True


def _groups(messages: list) -> tuple[list, list]:
    """Split off the system/context head from later assistant/tool rounds."""
    if messages and messages[0].get("role") == "system":
        head_end = 1
        if len(messages) > 1 and messages[1].get("role") == "user":
            head_end = 2
    else:
        head_end = 0
    head = messages[:head_end]
    rest = messages[head_end:]
    groups = []
    i = 0
    while i < len(rest):
        msg = rest[i]
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            ids = {tc.get("id") for tc in msg["tool_calls"]}
            j = i + 1
            while j < len(rest) and rest[j].get("role") == "tool" and rest[j].get("tool_call_id") in ids:
                j += 1
            groups.append(rest[i:j])
            i = j
        else:
            groups.append([msg])
            i += 1
    return head, groups


def _summarize_group(group: list) -> str:
    lines = []
    for msg in group:
        role = msg.get("role")
        if role == "assistant":
            for tc in msg.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name = fn.get("name") or "tool"
                args = " ".join((fn.get("arguments") or "").split())[:80]
                lines.append(f"- {name}({args})" if args else f"- {name}")
            if msg.get("content") and not msg.get("tool_calls"):
                lines.append("- assistant: " + _oneline(msg.get("content")))
        elif role == "tool":
            lines.append("  result: " + _oneline(msg.get("content")))
        else:
            lines.append(f"- {role or 'message'}: " + _oneline(msg.get("content")))
    text = "\n".join(lines)
    if len(text) > 600:
        text = text[:599] + "…"
    return text


def _clip(text: str, limit: int) -> str:
    note = "\n…[truncated to fit context]"
    if estimate_tokens(text) <= limit:
        return text
    if limit <= estimate_tokens(note):
        return note.strip()
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(text[:mid] + note) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + note


def _shrink_context_text(text: str, limit: int) -> tuple[str, str | None]:
    """Drop oldest chat blocks, then clip, until the text fits."""
    if estimate_tokens(text) <= limit:
        return text, None
    marker = "--- Recent chat room messages ---"
    if marker not in text:
        return _clip(text, limit), "prompt clipped to fit context"
    head, rest = text.split(marker, 1)
    # rest is "\n" + chat blocks + optional newer sections
    suffix = ""
    new_marker = "--- NEW messages since last loop ---"
    if new_marker in rest:
        rest, suffix = rest.split(new_marker, 1)
        suffix = new_marker + suffix
    chunks = []
    trailing = []
    for part in rest.split("\n\n"):
        stripped = part.lstrip("\n")
        if stripped.startswith("#") and not trailing:
            chunks.append(stripped)
        elif stripped.strip():
            trailing.append(part.strip("\n"))
    if trailing:
        extra = "\n\n".join(trailing)
        suffix = (extra + "\n\n" + suffix) if suffix else extra

    omitted = 0
    def assembled(note: str) -> str:
        body = head + marker + "\n"
        if note:
            body += note + "\n\n"
        if chunks:
            body += "\n\n".join(chunks)
        if suffix:
            body += "\n\n" + suffix
        return body

    while chunks and estimate_tokens(assembled("")) > limit:
        chunks.pop(0)
        omitted += 1
    note = f"Older chat omitted ({omitted} messages) to fit the context window." if omitted else ""
    text_out = assembled(note)
    if estimate_tokens(text_out) <= limit:
        return text_out, (f"omitted {omitted} older chat messages" if omitted else None)
    clipped = _clip(text_out, limit)
    return clipped, "prompt clipped to fit context"


def _assemble(head: list, groups: list, summary: str | None) -> list:
    out = list(head)
    if summary:
        out.append({
            "role": "user",
            "content": (
                "Earlier rounds in this loop were summarized to fit the context window:\n"
                + summary
            ),
        })
    for group in groups:
        out.extend(group)
    return out


def fit_messages(messages: list, max_context: int, tools=None) -> tuple[list, list[str]]:
    """Return messages that fit in the window, plus notes describing what changed.

    The list and message dicts may be updated in place. Tool-call groups stay
    paired so the chat request stays valid.
    """
    max_context = int(max_context)
    if max_context <= 0:
        return messages, []
    floor = reply_room(max_context)
    tool_tokens = estimate_tokens(json.dumps(tools, ensure_ascii=False)) if tools else 0
    limit = max(1, max_context - floor - tool_tokens)
    notes = []

    def over() -> bool:
        return estimate_messages(messages) > limit

    if not over():
        return messages, notes

    head, groups = _groups(messages)
    # Keep the newest image; clear older ones.
    image_msgs = [m for g in groups for m in g if _has_image(m)]
    for msg in image_msgs[:-1]:
        if _strip_images(msg):
            notes.append("cleared older images")
    messages = _assemble(head, groups, None)
    if not over():
        return messages, _dedupe(notes)

    # Shorten tool results from the oldest rounds, leaving the latest round intact.
    shortened = 0
    for group in groups[:-1] if len(groups) > 1 else []:
        for msg in group:
            if msg.get("role") != "tool":
                continue
            brief = "[shortened to fit context] " + _oneline(msg.get("content"))
            if estimate_tokens(brief) >= _content_tokens(msg.get("content")):
                continue
            msg["content"] = brief
            shortened += 1
        messages = _assemble(head, groups, None)
        if not over():
            break
    if shortened:
        notes.append(f"shortened {shortened} old tool results")
    if not over():
        return messages, _dedupe(notes)

    # Drop whole older rounds into one summary, stopping as soon as it fits.
    dropped = []
    while len(groups) > 1 and over():
        dropped.append(_summarize_group(groups.pop(0)))
        messages = _assemble(head, groups, "\n".join(dropped))
    if dropped:
        notes.append(f"summarized {len(dropped)} earlier tool rounds")
    if not over():
        return messages, _dedupe(notes)

    # The context block itself is too big: drop oldest chat, then clip.
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if not isinstance(content, str) or "--- Recent chat room messages ---" not in content:
            continue
        room = limit - (estimate_messages(messages) - _content_tokens(content))
        shrunk, note = _shrink_context_text(content, max(1, room))
        msg["content"] = shrunk
        if note:
            notes.append(note)
        break
    if not over():
        return messages, _dedupe(notes)

    # Last resort: clip the longest text until the estimate fits.
    guard = 0
    while over() and guard < 20:
        guard += 1
        target = None
        target_len = 0
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str) and len(content) > target_len:
                target = msg
                target_len = len(content)
        if target is None or target_len < 40:
            break
        room = limit - (estimate_messages(messages) - _content_tokens(target["content"]))
        target["content"] = _clip(target["content"], max(1, room))
        notes.append("clipped prompt text to fit context")
    return messages, _dedupe(notes)


def _dedupe(notes: list[str]) -> list[str]:
    out = []
    for note in notes:
        if note and note not in out:
            out.append(note)
    return out
