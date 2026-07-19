#!/usr/bin/env python3
"""Core think-loop agent for a Mind instance."""

import json
import datetime
import requests
import builtins

from personality import personality_prompt_block
from tools import execute_tool, get_tools, configure_tools

_print = builtins.print


def log(*args, sep=" ", end="\n"):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not args:
        _print(f"[{ts}]", end=end)
        return
    msg = sep.join(str(x) for x in args)
    leading = ""
    while msg.startswith("\n"):
        leading += "\n"
        msg = msg[1:]
    _print(f"{leading}[{ts}] {msg}", end=end)


_PER_TOOL_LIMITS = {
    "post_message": 6,
    "web_search": 3,
    "run_terminal": 15,
    "set_focus": 6,
    "log_observation": 12,
    "plan_next_wake": 3,
    "write_next_steps": 4,
    "update_goals": 3,
    "remember": 5,
}
_GLOBAL_TURN_LIMIT = 28


def message_tags_me(msg: dict, username: str) -> bool:
    """True if message tags this username or everyone."""
    tags = msg.get("tags") or []
    uname = (username or "").lower()
    for t in tags:
        tl = str(t).lower()
        if tl in ("everyone", "*", "all", "@everyone"):
            return True
        if tl == uname or tl == f"@{uname}":
            return True
    return False


def _format_messages(messages: list, username: str) -> str:
    if not messages:
        return "(no messages)"
    lines = []
    for m in messages:
        tags = m.get("tags") or []
        tag_s = f" tags=[{', '.join(tags)}]" if tags else ""
        mine = " ← TAGGED YOU" if message_tags_me(m, username) else ""
        self_mark = " (you)" if m.get("username") == username else ""
        lines.append(
            f"#{m.get('id')} [{m.get('created_at', '')[:19]}] "
            f"{m.get('username')}{self_mark}{tag_s}{mine}:\n{m.get('text', '')}"
        )
    return "\n\n".join(lines)


def _format_presence(users: list) -> str:
    """Compact active/inactive member list for the LLM prompt."""
    if not users:
        return "Room members: (none registered yet, or presence unavailable)"
    active = [u.get("username") for u in users if u.get("active")]
    inactive = [u.get("username") for u in users if not u.get("active")]
    lines = [
        "Room members (active = hit the chat server recently, e.g. browser open or mind polling):",
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
    last_wake_reason = store.get_state("last_wake_reason") or ""
    obs = store.recent_observations(10)
    memory = store.get_memory_text(20)

    tagged = [m for m in new_messages if message_tags_me(m, username)]
    # also check recent for tags we might have missed if new_messages is empty on first run
    if not new_messages:
        tagged = [m for m in recent_messages if message_tags_me(m, username)
                  and m.get("username") != username]

    parts = [
        f"Your username: {username}",
        personality_prompt_block(personality),
        f"\nCurrent focus: {focus}",
        f"Goals:\n{goals}",
        f"Next steps from previous loop:\n{next_steps}",
    ]
    if last_wake_reason:
        parts.append(f"Last wake reason: {last_wake_reason}")
    if obs:
        parts.append("Recent private observations:")
        for o in obs:
            parts.append(f"  • [{(o.get('ts') or '')[-8:]}] {o.get('note', '')[:140]}")
    if memory:
        parts.append("Durable memory:\n" + memory)

    parts.append("\n" + _format_presence(room_users or []))

    parts.append("\n--- Recent chat room messages ---")
    parts.append(_format_messages(recent_messages[-40:], username))

    if new_messages:
        parts.append("\n--- NEW messages since last loop ---")
        parts.append(_format_messages(new_messages, username))

    if tagged:
        parts.append(
            f"\n⚠️ You were tagged in {len(tagged)} message(s) this cycle. "
            "Prefer responding via post_message (tag the sender back when useful)."
        )
    else:
        parts.append(
            "\nYou were not specifically tagged in new messages. "
            "Only post if you have something useful, goal-related, or social to add. Silence is fine."
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
            secs = max(20, int(args.get("delay_seconds", 300)))
        except Exception:
            secs = 300
        reason = (args.get("reason") or "")[:120]
        when = (datetime.datetime.now() + datetime.timedelta(seconds=secs)).isoformat()
        store.set_state("next_wake_after", when)
        if reason:
            store.set_state("last_wake_reason", reason)
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


def run_think_loop(store, chat_client, system_prompt: str) -> str:
    """One full think cycle: fetch messages → LLM ReAct → persist."""
    username = store.username
    llm_base = (store.get_config("llm_base_url") or "").rstrip("/")
    llm_model = store.get_config("llm_model") or ""
    if not llm_base or not llm_model:
        return "error: llm_base_url / llm_model not configured"
    if not chat_client or not chat_client.token:
        return "error: not connected to chat server"

    last_seen = int(store.get_state("last_seen_message_id") or 0)

    # Fetch new + recent context + room presence
    room_users: list = []
    try:
        if last_seen > 0:
            new_data = chat_client.get_messages(after=last_seen, limit=100)
            new_messages = new_data.get("messages") or []
        else:
            new_messages = []

        recent_data = chat_client.get_messages(limit=40)
        recent_messages = recent_data.get("messages") or []
        latest_id = int(recent_data.get("latest_id") or last_seen)

        # If first run, treat recent tagged messages as "new" for awareness
        if last_seen == 0 and recent_messages:
            new_messages = recent_messages[-15:]

        try:
            room_users = chat_client.list_users() or []
        except Exception as e:
            log("Presence fetch warning:", e)
    except Exception as e:
        log("Chat fetch error:", e)
        return f"error: chat fetch failed: {e}"

    # Wire tools
    brave_key = store.get_config("brave_api_key") or ""
    web_enabled = bool(brave_key)

    def _post(text, tags):
        msg = chat_client.post_message(text, tags)
        log(f"💬 posted #{msg.get('id')}: {(msg.get('text') or '')[:100]}")
        return msg

    configure_tools(
        tool_calls_dir=store.tool_calls_dir,
        brave_api_key=brave_key or None,
        post_handler=_post,
    )
    tools = get_tools(web_search_enabled=web_enabled)

    context = _build_context_block(store, recent_messages, new_messages, room_users=room_users)
    user_prompt = (
        "THINK LOOP START.\n"
        "Review the chat and your state below. Act with tools as needed. "
        "Respond to tags directed at you. Advance goals if useful. "
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

    while turn < _GLOBAL_TURN_LIMIT:
        turn += 1
        payload = {
            "model": llm_model,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.5,
            "tools": tools,
            "tool_choice": "auto",
        }
        try:
            r = requests.post(
                f"{llm_base}/chat/completions",
                json=payload,
                timeout=300,
            )
            if r.status_code != 200:
                log(f"LLM HTTP {r.status_code}: {r.text[:250]}")
                break
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
                    else:
                        log(f"  🔧 {name}")

                    out = execute_tool(tc)
                    _apply_side_effects(store, name, args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": out,
                    })
                continue
            else:
                content = (msg.get("content") or "").strip()
                if content:
                    log(f"  (plain content ignored — use tools): {content[:120]}")
                break
        except Exception as e:
            log("LLM exception:", e)
            break

    # Advance cursor
    store.set_state("last_seen_message_id", max(last_seen, latest_id))
    store.set_state("last_loop_at", datetime.datetime.now().isoformat())
    status = f"loop_complete posts={posts} turns={turn} last_seen={store.get_state('last_seen_message_id')}"
    store.set_state("last_loop_status", status)
    log(f"🧠 {status}")
    return status
