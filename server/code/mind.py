#!/usr/bin/env python3
"""
Marmot Mind

The actual AI "mind" — autonomous reasoning, state, background loop, and context management.
This is separate from the HTTP server layer (server.py) that interfaces with the outside world
(human via client polling, /inject, health, etc.).

The mind owns:
- conversation_history (the public spoken record)
- persistent_memory
- live mind_state (focus, private observations, self-scheduled wakes)
- process_with_llm (ReAct engine)
- autonomous background loop

The server wires in a speak handler so the mind can produce output without knowing about
queues, TTS, or Flask.
"""

import json
import datetime
import threading
import time
import os
import requests

# ====================== CONFIG (populated by server at startup) ======================
LLM_BASE_URL = None
LLM_MODEL = None
SYSTEM_PROMPT = None
TOOLS = []
TOOLS_ENABLED = True
MAX_TOOL_TURNS = 8
MAX_CONTEXT_TOKENS = 150000

def configure(*, llm_base_url=None, llm_model=None, system_prompt=None,
              tools=None, tools_enabled=None, max_tool_turns=None,
              max_context_tokens=None, memory_path=None):
    """Called by the server layer after loading config. Keeps mind decoupled."""
    global LLM_BASE_URL, LLM_MODEL, SYSTEM_PROMPT, TOOLS, TOOLS_ENABLED, MAX_TOOL_TURNS, MAX_CONTEXT_TOKENS, MEMORY_PATH
    if llm_base_url is not None:
        LLM_BASE_URL = llm_base_url
    if llm_model is not None:
        LLM_MODEL = llm_model
    if system_prompt is not None:
        SYSTEM_PROMPT = system_prompt
    if tools is not None:
        TOOLS = tools
    if tools_enabled is not None:
        TOOLS_ENABLED = tools_enabled
    if max_tool_turns is not None:
        MAX_TOOL_TURNS = max_tool_turns
    if max_context_tokens is not None:
        MAX_CONTEXT_TOKENS = max_context_tokens
    if memory_path is not None:
        MEMORY_PATH = memory_path


# ====================== LIVE MIND STATE + CONTEXT ======================

last_message_time = None  # Used for auto-clearing context after long inactivity (shared with server)

# conversation_history holds only the current session's user + final assistant turns.
# It is managed by trim_conversation_history which *prefers* LLM-generated compaction
# summaries over raw deletion when we approach the token limit.
conversation_history = []  # user + assistant messages (tool internals ephemeral per turn)

history_lock = threading.Lock()  # protects conversation_history appends from concurrent agent runs + poll delivery

persistent_memory = ""  # durable notes persisted across conversation clears (bounded ~100 lines)

# Live internal context (separate from the human conversation transcript).
# Injected into *every* LLM prompt.
mind_state = {
    "current_focus": None,
    "recent_observations": [],
    "last_wake_reason": None,
    "next_wake_after": None,
    "last_mind_activity": None,
    "recent_human_summary": "",   # LLM-generated compact summary of recent user interactions (for background mind context)
    "pending_direct_user_question": None,  # if set, a recent user question is being handled by the primary agent; background should not answer it
}
mind_lock = threading.Lock()
MIND_OBS_MAX = 25
mind_wake_event = threading.Event()


def _get_mind_state_block(for_background: bool = False) -> str:
    """Compact text block describing current live mind state for injection into LLM prompts."""
    with mind_lock:
        focus = mind_state.get("current_focus") or "(no active focus)"
        obs = list(mind_state.get("recent_observations", []))[-5:]
        next_w = mind_state.get("next_wake_after")
        last_act = mind_state.get("last_mind_activity")
        lines = [
            "Your current internal mind state (this affects how you respond and what you choose to think about next):",
            f"Current focus: {focus}"
        ]
        if for_background:
            human_sum = mind_state.get("recent_human_summary", "")
            pending_q = mind_state.get("pending_direct_user_question")
            if human_sum:
                lines.append("Recent human interactions (concise summary for your background awareness):")
                lines.append(human_sum)
            if pending_q:
                lines.append(f"Pending direct question being handled by primary agent (do NOT answer this in background): {pending_q}")
        if obs:
            lines.append("Recent private observations (internal only):")
            for o in obs:
                ts = o.get("ts", "?")[-8:]
                lines.append(f"  • [{ts}] {o.get('note', '')[:110]}")
        if next_w:
            lines.append(f"Next planned self-reflection: {next_w}")
        if last_act:
            lines.append(f"Last internal activity: {last_act}")
        return "\n".join(lines)


def _update_mind_focus(text: str):
    txt = (text or "").strip()
    with mind_lock:
        mind_state["current_focus"] = txt or None
        mind_state["last_mind_activity"] = datetime.datetime.now().isoformat()
    print(f"🧠 Mind focus updated: {mind_state['current_focus'] or '(cleared)'}")


def _log_mind_observation(note: str):
    note = (note or "").strip()
    if not note:
        return
    entry = {"ts": datetime.datetime.now().isoformat(), "note": note}
    with mind_lock:
        mind_state["recent_observations"].append(entry)
        if len(mind_state["recent_observations"]) > MIND_OBS_MAX:
            mind_state["recent_observations"].pop(0)
        mind_state["last_mind_activity"] = datetime.datetime.now().isoformat()
    print(f"🧠 Mind observation: {note[:90]}{'...' if len(note) > 90 else ''}")


def _plan_next_mind_wake(delay_seconds: int, reason: str = ""):
    try:
        secs = max(20, int(delay_seconds))
    except Exception:
        secs = 120
    when = (datetime.datetime.now() + datetime.timedelta(seconds=secs)).isoformat()
    with mind_lock:
        mind_state["next_wake_after"] = when
        if reason:
            mind_state["last_wake_reason"] = reason[:120]
        mind_state["last_mind_activity"] = datetime.datetime.now().isoformat()
    print(f"🧠 Mind scheduled next internal wake in ~{secs}s" + (f" ({reason[:60]})" if reason else ""))
    mind_wake_event.set()


def _get_mind_status_for_health() -> dict:
    with mind_lock:
        return {
            "current_focus": mind_state.get("current_focus"),
            "recent_observations_count": len(mind_state.get("recent_observations", [])),
            "last_mind_activity": mind_state.get("last_mind_activity"),
            "next_wake_after": mind_state.get("next_wake_after"),
            "last_wake_reason": mind_state.get("last_wake_reason"),
            "recent_human_summary": mind_state.get("recent_human_summary", ""),
            "pending_direct_user_question": mind_state.get("pending_direct_user_question"),
        }


def _get_memory_messages() -> list:
    mem = (persistent_memory or "").strip()
    if not mem:
        return []
    return [{
        "role": "system",
        "content": "Key facts and context remembered from previous conversations (carry these forward):\n" + mem
    }]


def _count_memory_lines() -> int:
    return len([ln for ln in (persistent_memory or "").splitlines() if ln.strip()])


def estimate_tokens(x) -> int:
    if x is None:
        return 0
    try:
        s = json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
    except Exception:
        s = str(x)
    return max(1, len(s) // 3)


def trim_conversation_history():
    """Ensure conversation_history (+ protected memory messages) stays under MAX_CONTEXT_TOKENS.

    Preferred path: LLM compaction of oldest turns...
    """
    global conversation_history
    if not conversation_history:
        return

    prefix = [{"role": "system", "content": SYSTEM_PROMPT}] + _get_memory_messages()
    pfx = len(prefix)
    max_compactions = 2
    compactions = 0

    while True:
        cur = prefix + conversation_history
        if len(cur) <= pfx or estimate_tokens(cur) <= MAX_CONTEXT_TOKENS:
            break

        if compactions < max_compactions:
            total = len(conversation_history)
            chunk = min(10, max(3, total // 3))
            if total >= 3:
                to_compact = conversation_history[:chunk]
                summary = summarize_for_compaction(to_compact)
                conversation_history = conversation_history[chunk:]
                low = (summary or "").lower()
                if summary and "no significant earlier context" not in low:
                    compacted_msg = {
                        "role": "assistant",
                        "content": "[Compacted summary of earlier turns in this conversation]\n" + summary.strip()
                    }
                    conversation_history.insert(0, compacted_msg)
                    print(f"🗜️  Compacted {chunk} older turns into a summary note")
                    compactions += 1
                    continue

        if conversation_history:
            if "Compacted summary" in conversation_history[0].get("content", "") and len(conversation_history) > 1:
                del conversation_history[1]
            else:
                conversation_history.pop(0)


# ====================== PERSISTENT MEMORY ======================

MEMORY_PATH = None  # set via configure if needed, but server manages the path usually

def _load_persistent_memory():
    global persistent_memory
    if not MEMORY_PATH or not os.path.exists(MEMORY_PATH):
        persistent_memory = ""
        return
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            persistent_memory = f.read()
    except Exception as e:
        print("Warning: could not load memory:", e)
        persistent_memory = ""


def set_persistent_memory(text: str):
    global persistent_memory
    persistent_memory = text or ""


def _save_persistent_memory():
    if not MEMORY_PATH:
        return
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            f.write(persistent_memory)
    except Exception as e:
        print("Warning: could not save memory:", e)


def _append_memory(new_text: str):
    global persistent_memory
    txt = (new_text or "").strip()
    if not txt:
        return
    low = txt.lower()
    if "nothing significant" in low or "nothing to remember" in low or low in ("", "none", "n/a"):
        return
    ts = datetime.datetime.now().strftime("%Y-%m-%d")
    entry = f"[{ts}] {txt}"
    combined = (persistent_memory + "\n\n" + entry).strip() if persistent_memory else entry
    lines = combined.splitlines()
    if len(lines) > 100:
        lines = lines[-100:]
    persistent_memory = "\n".join(lines)
    _save_persistent_memory()


def _call_llm_simple(messages: list, max_tokens: int = 512, temperature: float = 0.2) -> str:
    if not LLM_BASE_URL or not LLM_MODEL:
        return ""
    try:
        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        r = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=120)
        if r.status_code == 200:
            return (r.json().get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        print(f"LLM (simple) HTTP {r.status_code}")
    except Exception as e:
        print("LLM (simple) error:", e)
    return ""


def summarize_for_compaction(older_turns: list) -> str:
    if not older_turns:
        return ""
    instruction = {
        "role": "user",
        "content": (
            "The turns above are older parts of the *current ongoing conversation* and need to be compacted.\n"
            "Create an extremely concise summary (bullets or 1-3 short paragraphs) of the user goals, key facts, decisions, important discoveries or tool outcomes, and context that the assistant must remember to remain coherent and effective for the rest of *this* session.\n"
            "Ignore transient one-off details. If there is little still relevant, reply exactly with: No significant earlier context."
        )
    }
    msgs = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + older_turns
        + [instruction]
    )
    return _call_llm_simple(msgs, max_tokens=400, temperature=0.1)


def refresh_recent_human_summary(max_turns: int = 6) -> str:
    """Ask the LLM for a compact, background-mind-oriented summary of the most recent human interactions.
    This is injected into autonomous mind steps so the background process stays aware of what the
    human has said recently without re-reading the entire raw conversation_history every time.
    """
    global conversation_history, mind_state

    if not conversation_history:
        with mind_lock:
            mind_state["recent_human_summary"] = ""
        return ""

    # Take the most recent turns (user + assistant spoken turns)
    recent = conversation_history[-max_turns:]

    instruction = {
        "role": "user",
        "content": (
            "The turns above are the most recent interactions between the human and you.\n"
            "Create an extremely concise summary (1-4 short paragraphs or bullets) focused on:\n"
            "- What the human has said, asked, or expressed recently (key statements, questions, requests, tone/mood)\n"
            "- Any open topics, concerns, plans, or things the human seems to care about right now\n"
            "- Context that your *background autonomous thinking* should remember so you can be intelligently aware without repeating yourself.\n"
            "Ignore very old or transient details. If there is little new relevant human context, reply exactly: No significant recent human interactions.\n"
            "This summary will be used by your background mind loop (separate from direct replies)."
        )
    }

    msgs = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + recent
        + [instruction]
    )

    summary = _call_llm_simple(msgs, max_tokens=350, temperature=0.15)

    with mind_lock:
        if summary and "no significant recent human interactions" not in summary.lower():
            mind_state["recent_human_summary"] = summary.strip()
        else:
            mind_state["recent_human_summary"] = ""

    print(f"🧠 Refreshed recent human summary ({len(recent)} turns)")
    return mind_state["recent_human_summary"]


def extract_memory_from_history() -> str:
    global conversation_history
    if not conversation_history:
        return ""
    instruction = {
        "role": "user",
        "content": (
            "The conversation above is about to be cleared (inactivity or explicit reset).\n"
            "Before it is cleared, tell your future self the most important durable things to remember:\n"
            "- User preferences, name, style, or recurring requests\n"
            "- Key projects, tasks, files, or goals in progress\n"
            "- Important facts, decisions, or context that will help in future conversations\n\n"
            "Be extremely concise (a few bullets or short paragraphs at most).\n"
            "If there is truly nothing worth carrying forward, reply with exactly: Nothing significant to remember."
        )
    }
    msgs = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + conversation_history
        + [instruction]
    )
    return _call_llm_simple(msgs, max_tokens=450, temperature=0.15)


def commit_memory_before_clear():
    try:
        mem = extract_memory_from_history()
        if mem:
            _append_memory(mem)
            lines = [l for l in mem.splitlines() if l.strip()]
            print(f"🧠 Extracted memory ({len(lines)} lines) before clearing context")
    except Exception as e:
        print("Memory extraction failed (continuing):", e)


# ====================== LLM + MULTI-TURN TOOLS ======================

_PER_TOOL_LIMITS = {
    "web_search": 3,
    "run_terminal": 20,
    "speak": 5,
    "set_focus": 8,
    "log_observation": 12,
    "plan_next_wake": 5,
}
_GLOBAL_TURN_LIMIT = 30


# Speak handler injected by the server layer
_speak_handler = None

def set_speak_handler(handler):
    """Register the function to call when the mind wants to speak to the human.
    The server layer registers queue_proactive_message here.
    """
    global _speak_handler
    _speak_handler = handler


def process_with_llm(user_text: str = None, internal: bool = False) -> str:
    """Core ReAct-style agent loop (used for both human turns and autonomous mind steps).

    Context is now *dynamic*: live mind_state (focus, observations, self-scheduled wake)
    is always injected. This means human answers are influenced by whatever the
    mind is currently "up to".
    """
    global conversation_history

    trim_conversation_history()

    # Dynamic context assembly
    base = [{"role": "system", "content": SYSTEM_PROMPT}] + _get_memory_messages()
    mind_block = {
        "role": "system",
        "content": _get_mind_state_block(for_background=internal)
    }

    if internal:
        # For background / autonomous mind steps we use the compact mind state
        # (including the LLM-generated recent_human_summary) instead of dumping
        # the entire raw conversation_history. This prevents the background mind
        # from constantly re-seeing and re-responding to old human messages.
        messages = base + [mind_block]
    else:
        # Direct human interactions still get the full (trimmed) public transcript
        # so replies feel naturally continuous with the raw conversation.
        messages = base + [mind_block] + conversation_history

    if user_text:
        messages = messages + [{"role": "user", "content": user_text}]

    if not internal:
        # Explicit reminder for direct human replies so the model uses speak even
        # if the main system prompt + mind_block is diluted.
        messages.append({
            "role": "user",
            "content": "This is a direct response to the human. You MUST call the speak tool with natural spoken text for anything you want the user to hear. Do not emit plain content at the end."
        })

    turn = 0
    tool_counts = {k: 0 for k in _PER_TOOL_LIMITS}
    speaks_this_run = 0
    forced_speak_correction = 0

    while turn < _GLOBAL_TURN_LIMIT:
        turn += 1
        if turn > MAX_TOOL_TURNS:
            pass

        payload = {
            "model": LLM_MODEL,
            "messages": messages,
            "max_tokens": 4096,
            "temperature": 0.4,
        }
        if TOOLS_ENABLED and TOOLS:
            payload["tools"] = TOOLS
            payload["tool_choice"] = "auto"

        try:
            r = requests.post(f"{LLM_BASE_URL}/chat/completions", json=payload, timeout=300)
            if r.status_code != 200:
                print(f"LLM HTTP {r.status_code}: {r.text[:250]}")
                break
            data = r.json()
            msg = data.get("choices", [{}])[0].get("message", {})
            messages.append(msg)

            if msg.get("tool_calls"):
                non_speak_in_batch = False
                for tc in msg.get("tool_calls", []):
                    fn = tc.get("function", {})
                    name = fn.get("name", "tool")

                    if name in _PER_TOOL_LIMITS:
                        tool_counts[name] = tool_counts.get(name, 0) + 1
                        if tool_counts[name] > _PER_TOOL_LIMITS[name]:
                            if name == "web_search":
                                limit_msg = f"web_search call limit ({_PER_TOOL_LIMITS[name]}) reached this run. No more web searches allowed. You can continue using other tools (e.g. run_terminal to curl URLs or run other commands). Call speak() with a natural spoken summary only when you have something to tell the user."
                            elif name == "speak":
                                limit_msg = f"speak call limit ({_PER_TOOL_LIMITS[name]}) reached this run. You have spoken enough. Stop calling speak. Either use other tools if you need to, or stop the loop now and wait for the user. Do not produce more spoken messages."
                            else:
                                limit_msg = f"{name} call limit ({_PER_TOOL_LIMITS[name]}) reached this run. Respect the limit. You may use other tools or call speak() if you want to communicate with the user."
                            print(f"  ⚠️  {limit_msg}")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.get("id", ""),
                                "content": limit_msg
                            })
                            if name != "speak":
                                non_speak_in_batch = True
                            continue

                    try:
                        args = json.loads(fn.get("arguments", "{}"))
                    except Exception:
                        args = {}

                    speak_failed = False

                    if name == "web_search":
                        q = args.get("query", "")
                        print(f"  🔧 web_search: {q}" if q else "  🔧 web_search")
                    elif name == "run_terminal":
                        cmd = args.get("command", "")
                        print(f"  🔧 run_terminal: {cmd}" if cmd else "  🔧 run_terminal")
                    elif name == "speak":
                        speaks_this_run += 1
                        print(f"  🗣️  speak")
                        speak_text = (args.get("text") or "").strip()
                        speak_failed = False
                        if speak_text:
                            try:
                                if _speak_handler:
                                    _speak_handler(speak_text)
                                    print(f"🗣️  Speak queued: {speak_text[:120]}{'...' if len(speak_text) > 120 else ''}")
                                else:
                                    print(f"🗣️  (no speak handler registered): {speak_text[:80]}")
                            except Exception as e:
                                print("Speak handler error:", e)
                                speak_failed = True
                    elif name == "set_focus":
                        focus = args.get("text", "")
                        print(f"  🧠 set_focus: {focus[:60]}{'...' if len(focus) > 60 else ''}")
                        _update_mind_focus(focus)
                    elif name == "log_observation":
                        note = args.get("note", "")
                        print(f"  🧠 log_observation")
                        _log_mind_observation(note)
                    elif name == "plan_next_wake":
                        delay = args.get("delay_seconds", 0)
                        reason = args.get("reason", "")
                        print(f"  🧠 plan_next_wake: {delay}s")
                        _plan_next_mind_wake(delay, reason)
                    else:
                        print(f"  🔧 {name}")

                    from tools import execute_tool
                    out = execute_tool(tc)
                    if name == "speak" and speak_failed:
                        out = json.dumps({"status": "error", "message": "failed to queue audio for user"})
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": out
                    })
                    if name != "speak":
                        non_speak_in_batch = True

                if non_speak_in_batch:
                    messages.append({
                        "role": "user",
                        "content": "Reminder: If you now have information, a summary, or an update the user should hear, call the speak tool with natural spoken text. You may continue using other tools (such as run_terminal) for more work. Do not later repeat the same content as plain assistant messages (they are ignored). If you have nothing to communicate to the user right now, you can stop or keep working."
                    })
                continue
            else:
                content = (msg.get("content") or "").strip()
                if content:
                    if speaks_this_run > 0:
                        print(f"  (model emitted additional plain content after speaking; ignored. Content was: {content[:100]})")
                    else:
                        if not internal:
                            # For direct user input, force a speak so the client always gets a response.
                            # This ensures the user hears something even if the model emitted plain content.
                            if _speak_handler:
                                _speak_handler(content)
                            print(f"  (forced speak for direct user turn because model emitted plain content: {content[:80]})")
                            speaks_this_run += 1
                        else:
                            print(f"  (model emitted plain content and stopped; not spoken because no speak() was used. Model tried to say: {content[:150]})")
                            if forced_speak_correction < 1:
                                forced_speak_correction += 1
                                messages.append({
                                    "role": "user",
                                    "content": f"You emitted plain content instead of a speak() call. That is invalid — plain content is never delivered to the user. You MUST call the speak tool (following all CRITICAL rules: natural spoken English, no markdown, no lists, conversational). Call speak() now with a clean spoken version of the answer you wanted to give: {content[:300]}. If you have nothing to say, just stop without tool calls."
                                })
                                continue
                            else:
                                print("  (model still emitted plain content after correction; stopping with no user audio)")
                break
        except Exception as e:
            print("LLM exception:", e)
            break

    trim_conversation_history()
    status = f"agent_run_complete (speaks={speaks_this_run}, turns={turn})"
    if speaks_this_run == 0:
        print("  (agent completed with no speak() calls — user will hear nothing from this run)")
    return status


def _run_agent_for_user_turn(user_text: str):
    try:
        status = process_with_llm(user_text=None, internal=False)
        print(f"🐹 Background agent complete: {status}")
        # Refresh the recent-human summary now that the user turn (and our reply)
        # are complete, so background mind steps see an up-to-date compact view.
        try:
            refresh_recent_human_summary()
        except Exception:
            pass
    except Exception as ex:
        print("Background agent error:", ex)
    finally:
        clear_pending_direct_user_question()


# ====================== AUTONOMOUS MIND LOOP ======================

def _run_mind_step():
    try:
        # Make sure the recent-human summary is up to date for this background step
        # (cheap LLM call; only includes the most recent turns).
        try:
            refresh_recent_human_summary()
        except Exception:
            pass

        prompt = (
            "AUTONOMOUS INTERNAL MIND STEP. Time is passing on this machine. "
            "You have a live internal state (focus + private observations). "
            "Use set_focus, log_observation and plan_next_wake to steer your own attention. "
            "You may use run_terminal or web_search for investigation if useful. "
            "CRITICAL: You do NOT have to speak. Speak ONLY if you have genuinely urgent or important new information the human must hear *immediately* (e.g. critical alert, direct answer to an open question that the main agent turn did not cover). "
            "Routine status reports ('all quiet', 'system healthy', 'I'll check later', time of day, holiday notes) must stay internal — log them with observations and plan your next wake instead. "
            "Use the 'Recent human interactions' summary (provided in your internal mind state) to stay aware of what the human has said or asked recently. If there is a 'pending_direct_user_question', the primary agent is handling the direct reply — do not speak an answer to it. Do not re-answer old user messages that were already handled. "
            "When done, use plan_next_wake to schedule your next internal reflection. End without speak() unless truly necessary."
        )
        print("\n🧠 Autonomous mind step starting...")
        status = process_with_llm(prompt, internal=True)
        print(f"🧠 Autonomous mind step complete: {status}")
    except Exception as ex:
        print("Mind step error:", ex)


def _start_autonomous_mind_loop():
    def _mind_loop():
        time.sleep(8)
        while True:
            try:
                sleep_secs = 240
                with mind_lock:
                    nw = mind_state.get("next_wake_after")
                    if nw:
                        try:
                            target = datetime.datetime.fromisoformat(nw)
                            delta = (target - datetime.datetime.now()).total_seconds()
                            if delta > 15:
                                sleep_secs = min(7200, max(20, delta))
                        except Exception:
                            pass

                print(f"🧠 Mind quiet — will wake in ~{int(sleep_secs)}s unless woken")
                woke_early = mind_wake_event.wait(timeout=sleep_secs)
                mind_wake_event.clear()

                if woke_early:
                    print("🧠 Mind woken early (human interaction or explicit plan)")

                _run_mind_step()
            except Exception as e:
                print("Mind loop outer error (retrying):", e)
                time.sleep(45)

    t = threading.Thread(target=_mind_loop, daemon=True, name="marmot-mind")
    t.start()
    print("🧠 Autonomous self-scheduling mind loop started")
    mind_wake_event.set()


# Public helpers the server can use
def get_conversation_history():
    return conversation_history

def append_assistant_turn(text: str):
    """Used by server when a proactive message is delivered to the client."""
    with history_lock:
        conversation_history.append({"role": "assistant", "content": text})

def append_user_turn(text: str):
    with history_lock:
        conversation_history.append({"role": "user", "content": text})

def clear_conversation_history():
    global conversation_history, last_message_time
    conversation_history = []
    last_message_time = None
    with mind_lock:
        mind_state["recent_human_summary"] = ""
        mind_state["pending_direct_user_question"] = None

def get_last_message_time():
    return last_message_time

def set_last_message_time(ts):
    global last_message_time
    last_message_time = ts


def set_pending_direct_user_question(text: str):
    with mind_lock:
        mind_state["pending_direct_user_question"] = (text or "").strip() or None


def clear_pending_direct_user_question():
    with mind_lock:
        mind_state["pending_direct_user_question"] = None

def trim_history():
    trim_conversation_history()
