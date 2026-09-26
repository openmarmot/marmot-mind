#!/usr/bin/env python3
"""Core think-loop agent for a Mind instance."""

import json
import re
import datetime
import contextvars
import requests
import builtins

from context_budget import estimate_messages, fit_messages
from personality import invent_personality, personality_is_set, personality_prompt_block
from tools import execute_tool, get_tools
from tools.context import ToolContext

_print = builtins.print
_log_name: contextvars.ContextVar[str] = contextvars.ContextVar("mind_log_name", default="")


def set_log_name(name: str):
    return _log_name.set(name or "")


def reset_log_name(token) -> None:
    _log_name.reset(token)


def log(*args, sep=" ", end="\n"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    who = _log_name.get()
    prefix = f"[{ts}]" + (f" {who}" if who else "")
    if not args:
        _print(prefix, end=end)
        return
    msg = sep.join(str(x) for x in args)
    leading = ""
    while msg.startswith("\n"):
        leading += "\n"
        msg = msg[1:]
    _print(f"{leading}{prefix} {msg}", end=end)


_PER_TOOL_LIMITS = {
    "post_message": 6,
    "web_search": 3,
    "run_terminal": 15,
    "look_at_image": 4,
    "set_focus": 6,
    "log_observation": 12,
    "plan_next_wake": 3,
    "write_next_steps": 4,
    "update_goals": 3,
    "remember": 5,
}
_GLOBAL_TURN_LIMIT = 28
_MAX_WAKE_SECONDS = 24 * 3600
_FALLBACK_WAKE_SECONDS = 300

_IMAGE_URL_RE = re.compile(
    r"https?://[^\s<>\"]+\.(?:png|jpe?g|gif|webp|bmp)(?:\?[^\s<>\"]*)?",
    re.I,
)
_MD_IMAGE_RE = re.compile(r"!\[[^\]]*\]\(")


def message_tags_everyone(msg: dict) -> bool:
    """True if message tags the whole room."""
    for t in msg.get("tags") or []:
        if str(t).lower() in ("everyone", "*", "all", "@everyone"):
            return True
    return False


def message_direct_tags_me(msg: dict, username: str) -> bool:
    """True if message tags this username (not merely everyone)."""
    uname = (username or "").lower()
    for t in msg.get("tags") or []:
        tl = str(t).lower()
        if tl == uname or tl == f"@{uname}":
            return True
    return False


def message_tags_me(msg: dict, username: str) -> bool:
    """True if message tags this username or everyone."""
    return message_tags_everyone(msg) or message_direct_tags_me(msg, username)


def _format_messages(messages: list, username: str, *, urge_images: bool = False) -> str:
    if not messages:
        return "(no messages)"
    lines = []
    for m in messages:
        tags = m.get("tags") or []
        tag_s = f" tags=[{', '.join(tags)}]" if tags else ""
        if message_direct_tags_me(m, username):
            mine = " ← TAGGED YOU"
        elif message_tags_everyone(m):
            mine = " ← @everyone"
        else:
            mine = ""
        self_mark = " (you)" if m.get("username") == username else ""
        body = m.get("text") or ""
        if not body and m.get("images"):
            body = "(image only)"
        hint = _image_url_hint(m.get("text") or "") if urge_images else ""
        attached = _attached_image_hint(m, urge=urge_images)
        lines.append(
            f"#{m.get('id')} [{m.get('created_at', '')[:19]}] "
            f"{m.get('username')}{self_mark}{tag_s}{mine}{hint}:\n{body}{attached}"
        )
    return "\n\n".join(lines)


def _image_url_hint(text: str) -> str:
    if _IMAGE_URL_RE.search(text) or _MD_IMAGE_RE.search(text):
        return "  [possible image — look_at_image to see it]"
    return ""


def _attached_image_hint(msg: dict, *, urge: bool = False) -> str:
    images = msg.get("images") or []
    if not images:
        return ""
    lines = []
    for img in images:
        if not isinstance(img, dict):
            continue
        name = img.get("name") or "image"
        source = img.get("path") or img.get("url") or ""
        if urge:
            lines.append(f"  [attached image {name}: {source} — look_at_image to see it]")
        else:
            lines.append(f"  [attached image {name}: {source}]")
    if not lines:
        return ""
    return "\n" + "\n".join(lines)


def _attach_pending_images(messages: list, ctx: ToolContext) -> None:
    """Inject loaded images as vision content after tool results."""
    images = ctx.take_pending_images()
    if not images:
        return
    content = [
        {
            "type": "text",
            "text": (
                "The image(s) you requested with look_at_image are attached below. "
                "You can see them. Use what you see; do not claim you cannot view images."
            ),
        }
    ]
    for img in images:
        label = (
            f"Image from look_at_image: {img.get('source')} "
            f"({img.get('mime')}, {img.get('nbytes')} bytes)"
        )
        focus = (img.get("focus") or "").strip()
        if focus:
            label += f"\nWhat to look at: {focus}"
        content.append({"type": "text", "text": label})
        content.append({
            "type": "image_url",
            "image_url": {"url": img["data_url"]},
        })
    messages.append({"role": "user", "content": content})
    log(f"  🖼  attached {len(images)} image(s) for vision")


def _format_presence(users: list) -> str:
    """Compact active/inactive member list for the LLM prompt."""
    if not users:
        return "Room members: (none registered yet, or presence unavailable)"
    active = [u.get("username") for u in users if u.get("active")]
    inactive = [u.get("username") for u in users if not u.get("active")]
    lines = [
        "Room members (active = seen in the last 30s — chat open, or a mind loop running):",
        f"  Active now: {', '.join(active) if active else '(nobody)'}",
        f"  Inactive: {', '.join(inactive) if inactive else '(none)'}",
        "Prefer @mentioning people who are active when you need a reply; inactive users may not see it soon.",
    ]
    return "\n".join(lines)


def _build_context_block(
    store,
    recent_messages: list,
    new_messages: list,
    room_users: list | None = None,
) -> str:
    username = store.username
    personality = store.get_state("personality") or {}
    focus = store.get_state("focus") or "(none)"
    goals = store.get_state("goals") or "(none yet)"
    next_steps = store.get_state("next_steps") or "(none)"
    wake_reason = store.get_state("wake_reason") or ""
    obs = store.recent_observations(10)
    memory = store.get_memory_text(20)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    last_loop_at = store.get_state("last_loop_at") or ""
    last_loop_status = store.get_state("last_loop_status") or ""

    others = [m for m in new_messages if m.get("username") != username]
    direct = [m for m in others if message_direct_tags_me(m, username)]
    everyone = [
        m for m in others
        if message_tags_everyone(m) and not message_direct_tags_me(m, username)
    ]

    parts = [
        f"Your username: {username}",
        f"Current time: {now}",
        personality_prompt_block(personality),
        f"\nCurrent focus: {focus}",
        f"Goals:\n{goals}",
        f"Next steps from previous loop:\n{next_steps}",
    ]
    if wake_reason:
        parts.append(f"Why this loop started: {wake_reason}")
    if last_loop_status or last_loop_at:
        when = (last_loop_at or "")[:19]
        prev = last_loop_status or "n/a"
        parts.append("Previous loop: " + (f"{prev} at {when}" if when else prev))
    if obs:
        parts.append("Recent private observations:")
        for o in obs:
            parts.append(f"  • [{(o.get('ts') or '')[-8:]}] {o.get('note', '')[:140]}")
    if memory:
        parts.append("Durable memory:\n" + memory)

    parts.append("\n" + _format_presence(room_users or []))

    parts.append("\n--- Recent chat room messages ---")
    parts.append(_format_messages(recent_messages[-40:], username, urge_images=False))

    if new_messages:
        parts.append("\n--- NEW messages since last loop ---")
        parts.append(_format_messages(new_messages, username, urge_images=True))

    if direct:
        parts.append(
            f"\n⚠️ You were @mentioned by name in {len(direct)} message(s) this cycle. "
            "Prefer responding via post_message (tag the sender back when useful)."
        )
        if everyone:
            parts.append(
                f"{len(everyone)} other message(s) tagged @everyone. "
                "You do not have to answer those."
            )
    elif everyone:
        parts.append(
            f"\n{len(everyone)} message(s) tagged @everyone this cycle. "
            "Reply only if you have something of your own to add — you do not have to answer."
        )
    elif others:
        parts.append(
            "\nNew messages arrived and none tag you by name. "
            "You woke so you could read them. "
            "Post only if you have something of your own to add. "
            "Silence is fine — do not comment on the silence, and do not reply just to show you saw it."
        )
    else:
        parts.append(
            "\nYou were not specifically tagged in new messages. "
            "Post only if you have something of your own to add. Silence is fine — "
            "do not comment on the silence."
        )

    return "\n".join(parts)


def _apply_side_effects(store, name: str, args: dict):
    """Persist mind-tool side effects to SQLite."""
    if name == "set_focus":
        text = (args.get("text") or "").strip()
        store.set_state("focus", text or None)
        log(f"🧠 focus: {text or '(cleared)'}")
    elif name == "log_observation":
        note = (args.get("note") or "").strip()
        if note:
            store.add_observation(note)
            log(f"🧠 observation: {note[:90]}")
    elif name == "plan_next_wake":
        try:
            secs = max(20, min(_MAX_WAKE_SECONDS, int(args.get("delay_seconds", 300))))
        except Exception:
            secs = 300
        reason = (args.get("reason") or "")[:120]
        when = (datetime.datetime.now() + datetime.timedelta(seconds=secs)).isoformat()
        store.set_state("next_wake_after", when)
        store.set_state("next_wake_reason", reason)
        log(f"🧠 next wake in ~{secs}s" + (f" ({reason})" if reason else ""))
    elif name == "write_next_steps":
        steps = (args.get("steps") or "").strip()
        store.set_state("next_steps", steps)
        log(f"🧠 next_steps saved ({len(steps)} chars)")
    elif name == "update_goals":
        goals = (args.get("goals") or "").strip()
        store.set_state("goals", goals)
        log(f"🧠 goals updated")
    elif name == "remember":
        note = (args.get("note") or "").strip()
        if note:
            store.append_memory(note)
            log(f"🧠 remembered: {note[:90]}")


def _ensure_wake_plan(store, planned: bool, llm_ok: bool) -> None:
    """If the model never called plan_next_wake, sleep 5 minutes instead of spinning."""
    if planned:
        return
    secs = _FALLBACK_WAKE_SECONDS
    when = (datetime.datetime.now() + datetime.timedelta(seconds=secs)).isoformat()
    store.set_state("next_wake_after", when)
    why = "LLM call failed" if not llm_ok else "model did not call plan_next_wake"
    store.set_state("next_wake_reason", f"fallback {secs}s ({why})")
    log(f"🧠 no plan_next_wake — defaulting to {secs}s ({why})")


def _configured_context(store) -> int:
    """Configured context window in tokens, or 0 when unset."""
    raw = store.get_config("max_context")
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def run_think_loop(store, room, system_prompt: str) -> str:
    """One full think cycle: read the room in-process → LLM ReAct → persist."""
    token = set_log_name(store.username)
    try:
        return _run_think_loop(store, room, system_prompt)
    finally:
        reset_log_name(token)


def _run_think_loop(store, room, system_prompt: str) -> str:
    username = store.username
    llm_base = (store.get_config("llm_base_url") or "").rstrip("/")
    llm_model = store.get_config("llm_model") or ""
    if not llm_base or not llm_model:
        return "error: llm_base_url / llm_model not configured"

    if not personality_is_set(store.get_state("personality")):
        try:
            log("✨ Inventing personality via LLM…")
            personality = invent_personality(username, llm_base, llm_model)
            store.set_state("personality", personality)
            log(f"✨ Personality: {personality.get('summary', '')}")
        except Exception as e:
            log("Personality invent failed:", e)
            _ensure_wake_plan(store, planned=False, llm_ok=False)
            return f"error: personality invent failed: {e}"

    last_seen = int(store.get_state("last_seen_message_id") or 0)

    room_users: list = []
    try:
        if last_seen > 0:
            new_data = room.get_messages(after=last_seen, limit=100, as_user=username)
            new_messages = new_data.get("messages") or []
        else:
            new_messages = []

        recent_data = room.get_messages(limit=40, as_user=username)
        recent_messages = recent_data.get("messages") or []
        latest_id = int(recent_data.get("latest_id") or last_seen)

        # First loop: everything recent is still unread.
        if last_seen == 0 and recent_messages:
            new_messages = recent_messages[-15:]

        try:
            room_users = room.list_users(as_user=username) or []
        except Exception as e:
            log("Presence fetch warning:", e)
    except Exception as e:
        log("Chat fetch error:", e)
        _ensure_wake_plan(store, planned=False, llm_ok=False)
        return f"error: chat fetch failed: {e}"

    brave_key = store.get_config("brave_api_key") or ""
    web_enabled = bool(brave_key)

    def _post(text, tags):
        msg = room.post_message(username, text, tags)
        log(f"💬 posted #{msg.get('id')}: {(msg.get('text') or '')[:100]}")
        return msg

    ctx = ToolContext(
        tool_calls_dir=store.tool_calls_dir,
        post_handler=_post,
        brave_api_key=brave_key or None,
    )
    tools = get_tools(web_search_enabled=web_enabled)

    context = _build_context_block(
        store, recent_messages, new_messages,
        room_users=room_users,
    )
    user_prompt = (
        "THINK LOOP START.\n"
        "Review the chat and your state below. Act with tools as needed. "
        "Read new messages. Respond when a tag is directed at you. "
        "If a message does not tag you, stay quiet unless you have something of your own to add. "
        "Advance goals if useful. "
        "Before ending: write_next_steps and plan_next_wake.\n\n"
        + context
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    turn = 0
    tool_counts = {k: 0 for k in _PER_TOOL_LIMITS}
    posts = 0
    llm_ok = False
    planned_wake = False

    max_context = _configured_context(store)

    while turn < _GLOBAL_TURN_LIMIT:
        turn += 1
        if max_context:
            messages, notes = fit_messages(messages, max_context, tools=tools)
            for note in notes:
                log(f"🧠 context: {note}")
        payload = {
            "model": llm_model,
            "messages": messages,
            "temperature": 0.5,
            "tools": tools,
            "tool_choice": "auto",
        }
        # No fixed reply cap. When a window is configured, the reply may use
        # whatever is left after the prompt.
        if max_context:
            used = estimate_messages(messages, tools)
            payload["max_tokens"] = max(1, max_context - used)
        try:
            r = requests.post(
                f"{llm_base}/chat/completions",
                json=payload,
                timeout=300,
            )
            if r.status_code != 200:
                log(f"LLM HTTP {r.status_code}: {r.text[:250]}")
                break
            llm_ok = True
            msg = r.json().get("choices", [{}])[0].get("message", {})
            messages.append(msg)

            if msg.get("tool_calls"):
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "tool")
                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}

                    if name in _PER_TOOL_LIMITS:
                        tool_counts[name] = tool_counts.get(name, 0) + 1
                        if tool_counts[name] > _PER_TOOL_LIMITS[name]:
                            limit_msg = (
                                f"{name} call limit ({_PER_TOOL_LIMITS[name]}) "
                                "reached this loop. Do not call it again."
                            )
                            log(f"  ⚠️  {limit_msg}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id", ""),
                                "content": limit_msg,
                            })
                            continue

                    if name == "post_message":
                        posts += 1
                        log(f"  🔧 post_message")
                    elif name == "run_terminal":
                        log(f"  🔧 run_terminal: {(args.get('command') or '')[:80]}")
                    elif name == "web_search":
                        log(f"  🔧 web_search: {(args.get('query') or '')[:80]}")
                    elif name == "look_at_image":
                        log(f"  🔧 look_at_image: {(args.get('source') or '')[:120]}")
                    else:
                        log(f"  🔧 {name}")

                    out = execute_tool(ctx, tc)
                    _apply_side_effects(store, name, args)
                    if name == "plan_next_wake":
                        planned_wake = True
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": out,
                    })
                _attach_pending_images(messages, ctx)
                continue
            else:
                content = (msg.get("content") or "").strip()
                if content:
                    log(f"  (plain content ignored — use tools): {content[:120]}")
                break
        except Exception as e:
            log("LLM exception:", e)
            break

    if llm_ok:
        store.set_state("last_seen_message_id", max(last_seen, latest_id))
    else:
        log("🧠 last_seen not advanced (no successful LLM response)")

    _ensure_wake_plan(store, planned_wake, llm_ok)
    store.set_state("last_loop_at", datetime.datetime.now().isoformat())
    status = (
        f"loop_complete posts={posts} turns={turn} "
        f"last_seen={store.get_state('last_seen_message_id')}"
    )
    if not llm_ok:
        status += " llm_failed"
    if not planned_wake:
        status += " no_wake_plan"
    store.set_state("last_loop_status", status)
    log(f"🧠 {status}")
    return status
