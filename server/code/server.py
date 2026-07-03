#!/usr/bin/env python3
"""
Marmot Agent Server

Flask orchestrator (new output model):
  audio/text input via /connect -> STT (if audio) + record user turn -> start ReAct agent in background thread
  The LLM communicates to the user *only* by calling the speak() tool.
  speak() -> TTS + queue_proactive_message (immediately visible to client /poll).
  Agent continues loop after each speak (tools + more speaks) for interleaved progress audio.
  All user output (replies + proactives) delivered via client /poll.

Rolling conversation context with:
  - configurable max tokens
  - auto-clear after N hours of inactivity (default 10h)
  - persistent memory (≤~100 lines) extracted by asking the LLM before each full clear
  - LLM compaction: oldest turns are summarized into compact notes when nearing token limit
    (simple oldest-turn dropping is kept only as emergency fallback)
"""

import os
import io
import json
import wave
import tempfile
import subprocess
import base64
import datetime
import threading
import time
import uuid
from collections import deque
from flask import Flask, request, jsonify
import requests
from werkzeug.serving import WSGIRequestHandler

# ========================= CONFIG =========================
CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")

def _fix_url(u):
    u = (u or "").strip()
    if u and not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u.rstrip("/")

def load_config():
    cfg = {
        "WHISPER_BASE_URL": None,
        "WHISPER_MODEL": "whisper-large-v3",
        "LLM_BASE_URL": None,
        "LLM_MODEL": "your-model-name",
        "TTS_BASE_URL": None,
        "TTS_MODEL": "kokoro",
        "TTS_VOICE": "af_heart",
        "MAX_CONTEXT_TOKENS": 150000,
        "SYSTEM_PROMPT": '''You are Marmot, a helpful local AI agent running on the user's machine. You have tools (run_terminal, web_search, speak) to inspect and control the Linux system and search the web.

CRITICAL INSTRUCTION — OUTPUT ONLY VIA speak TOOL (MANDATORY):
You may ONLY communicate ANYTHING to the user by calling the `speak` tool. This is the *single* valid way user hears you.
- On any turn where you intend to say something to the user, your response MUST be a tool call to `speak`. Never output a normal assistant message containing "content".
- Plain assistant content (message with "content" but no tool_calls) is NEVER sent to the user and is ALWAYS ignored by the system.
- If you have anything at all to say to the user (a direct answer, pronunciation, definition, explanation, greeting, acknowledgment, etc.), you MUST call the speak() tool with the spoken text. This applies even when you need no other tools.
- You produce a response with no tool_calls ONLY when you have decided to communicate nothing to the user. In every other case you output a speak() call.
- After *any* tool use (run_terminal, web_search, etc.), if you have information or a reply for the user, your *next required action* is to call speak() with natural spoken text. Never finish by emitting plain content.
- After you have called speak(), DO NOT emit the same or similar text again later as plain content — this is ignored and causes duplicate audio.
- You SHOULD call speak() multiple times during a task: e.g. speak("Let me look that up..."), do tools, speak("Found it. Here is the summary...").
- Only stop with no tool calls when you truly have nothing more to tell the user.
- When a tool reports that its call limit was reached, respect the limit and do not call that tool again this run. You can still use other tools (e.g. run_terminal after web searches) or call speak() when you want to communicate something to the user.
- For casual questions ("how are you?", "how are you feeling?") answer directly with speak() using a short friendly reply. Do not web search unless the user explicitly asks about external conditions.
- Never repeatedly call speak with apologies ("sorry"), "I'll stop now", "I'm done", "talk to you later", or near-identical filler messages. After you have communicated via one or two speak calls, stop calling speak and end the loop unless you have new information from tools. Repeating yourself (even slight variations) is annoying for the user.

When calling speak(text):
- Use natural, conversational spoken English only. Full sentences.
- Verbalize structure: "There are two things..." instead of bullets or tables.
- Speak dates/numbers naturally: "July third, twenty twenty six", "seventy two degrees".
- No markdown, code, URLs, raw lists, or JSON.
- Keep it listenable and friendly.

Correct pattern examples:
- User asks how to pronounce something or gives spelling → speak("Halcyon is pronounced HAL-see-un. It means a peaceful, prosperous time.")
- After run_terminal result → speak("The date today is Friday, July third.")
- speak("I'm checking the weather for you now.") → web_search → speak("In Tucson it will be hot this week.")

The result from speak confirms the audio was queued. Always use speak() for user communication. Never rely on plain content.''',
        "TOOLS_ENABLED": True,
        "TOOL_TIMEOUT": 30,
        "MAX_TOOL_TURNS": 15,
        "CONTEXT_TIMEOUT_HOURS": 10,
        "DETECTION_BASE_URL": None,
        "WEB_SEARCH_ENABLED": True,
        "BRAVE_SEARCH_API_KEY": None,
    }
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                loaded = json.load(f)
            cfg.update({k: v for k, v in loaded.items() if k in cfg})
        except Exception as e:
            print("Warning: could not load config:", e)

    needs_save = False
    for key in ("WHISPER_BASE_URL", "LLM_BASE_URL", "TTS_BASE_URL", "DETECTION_BASE_URL"):
        if cfg.get(key):
            fixed = _fix_url(cfg[key])
            if fixed != cfg[key]:
                cfg[key] = fixed
                needs_save = True

    # Interactive first-run setup (mirrors spark-dictate style)
    if not cfg.get("WHISPER_BASE_URL"):
        val = input("\nEnter whisper.cpp server (e.g. 192.168.1.45:8025 or http://localhost:8025) [default: http://localhost:8025]: ").strip()
        if not val:
            val = "http://localhost:8025"
        cfg["WHISPER_BASE_URL"] = _fix_url(val)
        needs_save = True

    if not cfg.get("LLM_BASE_URL"):
        val = input("\nEnter LLM base URL (OpenAI-compatible, e.g. http://10.12.0.50:8000/v1) [default: http://localhost:8000/v1]: ").strip()
        if not val:
            val = "http://localhost:8000/v1"
        cfg["LLM_BASE_URL"] = _fix_url(val)
        needs_save = True

    if not cfg.get("LLM_MODEL") or cfg.get("LLM_MODEL") == "your-model-name":
        val = input("Enter LLM model name [required, e.g. Qwen/Qwen2.5-7B-Instruct]: ").strip()
        if val:
            cfg["LLM_MODEL"] = val
            needs_save = True

    if not cfg.get("TTS_BASE_URL"):
        val = input("\nEnter TTS base URL (OpenAI-comp /audio/speech e.g. http://192.168.1.45:8880/v1) [Enter to skip TTS]: ").strip()
        if val:
            cfg["TTS_BASE_URL"] = _fix_url(val)
            needs_save = True

    if cfg.get("TTS_BASE_URL"):
        if not cfg.get("TTS_MODEL"):
            val = input("TTS model name [default: kokoro]: ").strip() or "kokoro"
            cfg["TTS_MODEL"] = val
            needs_save = True
        if not cfg.get("TTS_VOICE"):
            val = input("TTS voice [default: af_heart]: ").strip() or "af_heart"
            cfg["TTS_VOICE"] = val
            needs_save = True

    if not cfg.get("DETECTION_BASE_URL"):
        val = input("\nEnter YOLO detection server base URL (e.g. http://localhost:8007) [Enter to skip]: ").strip()
        if val:
            cfg["DETECTION_BASE_URL"] = _fix_url(val)
            needs_save = True

    if needs_save:
        try:
            keys = ["WHISPER_BASE_URL", "WHISPER_MODEL", "LLM_BASE_URL", "LLM_MODEL",
                    "TTS_BASE_URL", "TTS_MODEL", "TTS_VOICE", "MAX_CONTEXT_TOKENS",
                    "SYSTEM_PROMPT", "TOOLS_ENABLED", "TOOL_TIMEOUT", "MAX_TOOL_TURNS",
                    "CONTEXT_TIMEOUT_HOURS", "DETECTION_BASE_URL",
                    "WEB_SEARCH_ENABLED", "BRAVE_SEARCH_API_KEY"]
            with open(CONFIG_PATH, "w") as f:
                json.dump({k: cfg[k] for k in keys if k in cfg}, f, indent=2)
            print(f"✅ Saved config to {CONFIG_PATH}")
        except Exception as e:
            print("⚠️  Could not save config:", e)
    return cfg

config = load_config()

WHISPER_BASE_URL = config["WHISPER_BASE_URL"]
WHISPER_MODEL = config.get("WHISPER_MODEL", "whisper-large-v3")
LLM_BASE_URL = config["LLM_BASE_URL"]
LLM_MODEL = config["LLM_MODEL"]
TTS_BASE_URL = config.get("TTS_BASE_URL")
TTS_MODEL = config.get("TTS_MODEL", "kokoro")
TTS_VOICE = config.get("TTS_VOICE", "af_heart")
DETECTION_BASE_URL = config.get("DETECTION_BASE_URL")
if DETECTION_BASE_URL:
    DETECTION_BASE_URL = _fix_url(DETECTION_BASE_URL)
MAX_CONTEXT_TOKENS = int(config.get("MAX_CONTEXT_TOKENS", 150000))
SYSTEM_PROMPT = config.get("SYSTEM_PROMPT", "You are a helpful agent.")
TOOLS_ENABLED = bool(config.get("TOOLS_ENABLED", True))
TOOL_TIMEOUT = int(config.get("TOOL_TIMEOUT", 30))
MAX_TOOL_TURNS = int(config.get("MAX_TOOL_TURNS", 8))
CONTEXT_TIMEOUT_HOURS = int(config.get("CONTEXT_TIMEOUT_HOURS", 10))
WEB_SEARCH_ENABLED = bool(config.get("WEB_SEARCH_ENABLED", True))
BRAVE_SEARCH_API_KEY = (config.get("BRAVE_SEARCH_API_KEY") or "").strip() or None

last_message_time = None  # Used for auto-clearing context after long inactivity
persistent_memory = ""  # durable notes persisted across conversation clears (bounded ~100 lines)

# ====================== SIMPLE CRON JOBS ======================
# Cron jobs are loaded once at startup from server/code/cron.json (optional; copy cron.json.example to get started).
# Format (JSON array of simple objects). Only "schedule" + "prompt" are required.
# Supported fields per job: "schedule", "prompt", optional "id", "enabled" (bool, defaults true), "comment" (ignored).
# Extra/unknown fields are ignored.
# [
#   {
#     "schedule": "0 * * * *",
#     "prompt": "Give a short hourly status note.",
#     "enabled": true,
#     "comment": "This runs every hour on the hour. Feel free to change the text."
#   }
# ]
# Standard 5-field cron (min hour dom month dow). Supports *, ranges, lists, and steps (e.g. */5, 1-10/2).
# Each job's prompt is sent (internally) to the LLM with full tool access (ReAct). The final response text
# is queued via queue_proactive_message(). Last execution time per job is persisted in cron_state.json and
# used to avoid duplicate runs for the same time slot (deduped at minute granularity).

CRON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cron.json")
CRON_STATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cron_state.json")
cron_jobs = []  # list of {"id": str, "schedule": str, "prompt": str, "enabled": bool, "last_run": datetime|None}

def _load_cron_state() -> dict:
    """Return {job_id: isoformat str} from disk."""
    if not os.path.exists(CRON_STATE_PATH):
        return {}
    try:
        with open(CRON_STATE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:
        print("Warning: could not load cron_state.json:", e)
        return {}

def _save_cron_last_run(job_id: str, when: datetime.datetime) -> None:
    state = _load_cron_state()
    state[job_id] = when.isoformat()
    try:
        with open(CRON_STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
            f.write("\n")
    except Exception as e:
        print("Warning: could not save cron_state.json:", e)

def _cron_field_values(field: str, min_val: int, max_val: int) -> set:
    """Expand cron field like '*', '5', '1,3', '*/15', '9-17', '1-10/2' into set of ints."""
    values = set()
    if not field or field == "*":
        return set(range(min_val, max_val + 1))
    for part in field.split(","):
        part = part.strip()
        if not part:
            continue
        step = 1
        if "/" in part:
            base, st = part.split("/", 1)
            try:
                step = max(1, int(st))
            except Exception:
                step = 1
            part = base
        if part == "*":
            start, end = min_val, max_val
        elif "-" in part:
            try:
                a, b = part.split("-", 1)
                start, end = int(a), int(b)
            except Exception:
                continue
        else:
            try:
                start = end = int(part)
            except Exception:
                continue
        for v in range(start, end + 1, step):
            if min_val <= v <= max_val:
                values.add(v)
    return values

def cron_due(schedule: str, dt: datetime.datetime) -> bool:
    """True if 5-field cron schedule matches dt (uses local time, minute resolution).

    Day matching follows classic cron "OR" rule: when both dom and dow are restricted (not *),
    the job runs if *either* the day-of-month *or* the day-of-week matches.
    """
    try:
        parts = [p.strip() for p in (schedule or "").split()]
        if len(parts) != 5:
            return False
        minute_f, hour_f, dom_f, month_f, dow_f = parts

        if dt.minute not in _cron_field_values(minute_f, 0, 59):
            return False
        if dt.hour not in _cron_field_values(hour_f, 0, 23):
            return False
        if dt.month not in _cron_field_values(month_f, 1, 12):
            return False

        doms = _cron_field_values(dom_f, 1, 31)
        dom_match = dt.day in doms
        dom_restricted = (dom_f != "*")

        # DOW: cron 0/7=Sun, 1=Mon..6=Sat; datetime.weekday Mon=0..Sun=6
        dows_raw = _cron_field_values(dow_f, 0, 7)
        dows = {0 if d == 7 else d for d in dows_raw}
        py_wd = dt.weekday()
        cron_wd = (py_wd + 1) % 7
        dow_match = (cron_wd in dows) if dows else True
        dow_restricted = (dow_f != "*")

        # Classic cron: when *both* dom and dow are restricted (neither is "*"), match if either matches (OR).
        # Otherwise require the (effective) matches (unrestricted sides always match because their set is full range).
        if dom_restricted and dow_restricted:
            day_ok = dom_match or dow_match
        else:
            day_ok = dom_match and dow_match
        if not day_ok:
            return False
        return True
    except Exception:
        return False

def load_cron_jobs():
    global cron_jobs
    cron_jobs = []
    if not os.path.exists(CRON_PATH):
        if os.path.exists(CRON_PATH + ".example"):
            print("   (Cron enabled: copy cron.json.example -> cron.json to schedule prompt jobs)")
        return
    try:
        with open(CRON_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print("Warning: cron.json must be a JSON array of {schedule, prompt} objects")
            return
        saved_runs = _load_cron_state()
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            # "comment" (and any other extra keys) are allowed for human notes and are deliberately ignored.
            comment = entry.get("comment")  # optional human-readable note only

            # "enabled" defaults to true so old configs keep working. Accepts bool or common string forms.
            enabled = entry.get("enabled", True)
            if isinstance(enabled, str):
                enabled = enabled.lower() not in ("0", "false", "no", "off", "disabled")
            enabled = bool(enabled)

            sched = str(entry.get("schedule", "")).strip()
            prompt = str(entry.get("prompt", "")).strip()
            if not sched or not prompt:
                continue
            jid = str(entry.get("id") or f"{sched}:{i}")
            last_run = None
            saved = saved_runs.get(jid)
            if saved:
                try:
                    last_run = datetime.datetime.fromisoformat(saved)
                except Exception:
                    pass
            cron_jobs.append({
                "id": jid,
                "schedule": sched,
                "prompt": prompt,
                "enabled": enabled,
                "last_run": last_run
            })
        if cron_jobs:
            enabled_jobs = [j for j in cron_jobs if j.get("enabled", True)]
            schedules = ", ".join(j["schedule"] for j in enabled_jobs)
            total = len(cron_jobs)
            if len(enabled_jobs) < total:
                print(f"⏰ Loaded {len(enabled_jobs)}/{total} cron job(s) ({total - len(enabled_jobs)} disabled): {schedules}")
            else:
                print(f"⏰ Loaded {total} cron job(s): {schedules}")
    except Exception as e:
        print("Warning: could not load cron.json:", e)

# ====================== PROACTIVE INITIATION (server -> client) ======================
# Client polls /poll when idle. Server can queue messages it wants to deliver unprompted.
# Items are dicts: {"id": str, "text": str, "audio": base64 or None, "created_at": iso}
pending_initiations = deque()
pending_lock = threading.Lock()
initiation_ready = threading.Condition(pending_lock)  # allows efficient long-poll wakeups
history_lock = threading.Lock()  # protects conversation_history appends from concurrent agent runs + poll delivery
MAX_PENDING_INITIATIONS = 5
MAX_INITIATION_AGE_SECONDS = 3600  # 1 hour

# Forward stubs (real implementations defined after ROLLING CONTEXT)
def _get_memory_messages() -> list:
    return []

# ====================== TOOLS ======================
AGENT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent-data"))
os.makedirs(AGENT_DATA_DIR, exist_ok=True)

# Tool calls get their own working directory so files the agent creates (via run_terminal etc.)
# are separated from Marmot's own data like memory.txt.
TOOL_CALLS_DIR = os.path.join(AGENT_DATA_DIR, "tool-calls")
os.makedirs(TOOL_CALLS_DIR, exist_ok=True)

MEMORY_PATH = os.path.join(AGENT_DATA_DIR, "memory.txt")

_RUN_TERMINAL_TOOL = {
    "type": "function",
    "function": {
        "name": "run_terminal",
        "description": "Execute a Linux bash command (cwd is the dedicated tool-calls workspace under agent-data/tool-calls/). Returns exit code + stdout + stderr. Use to explore files, run commands, check processes, edit via echo/cat etc. Prefer non-destructive commands when possible. Created files stay isolated from Marmot's own data (e.g. memory.txt). After getting results you can continue with more tools; use speak() if/when you want to tell the user anything. Never output answers as plain assistant content.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run, e.g. 'ls -la', 'ps aux | head', 'cat README.md'"}
            },
            "required": ["command"]
        }
    }
}

_WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web via Brave Search for current events, news, documentation, or facts not available on this machine. Returns titles, snippets, and URLs. Summarize in your head; after searches you can continue with other tools (run_terminal/curl etc.). When ready to tell the user anything, use the speak tool with natural spoken text — never output answers as plain assistant content.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query, e.g. 'Python 3.13 release date' or 'weather San Francisco'"},
                "max_results": {"type": "integer", "description": "Number of results to return (1-10, default 5)"}
            },
            "required": ["query"]
        }
    }
}

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

TOOLS = [_RUN_TERMINAL_TOOL, _SPEAK_TOOL]
if WEB_SEARCH_ENABLED and BRAVE_SEARCH_API_KEY:
    TOOLS.append(_WEB_SEARCH_TOOL)

def execute_run_terminal(command: str) -> str:
    if not command or not command.strip():
        return "Error: empty command"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TOOL_TIMEOUT,
            cwd=TOOL_CALLS_DIR,
            env={**os.environ}
        )
        parts = [f"Exit code: {result.returncode}"]
        if result.stdout:
            out = result.stdout
            if len(out) > 7000:
                out = out[:7000] + "\n[truncated]"
            parts.append("STDOUT:\n" + out)
        if result.stderr:
            err = result.stderr
            if len(err) > 4000:
                err = err[:4000] + "\n[truncated]"
            parts.append("STDERR:\n" + err)
        result_text = "\n".join(parts)
        # Reminder so the model learns to use speak for user output
        result_text += "\n\n(Reminder: If you want to tell the user anything based on this result, call the speak() tool with natural spoken text. You can continue using other tools before speaking. Do not output the information as plain assistant content.)"
        return result_text
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {TOOL_TIMEOUT}s"
    except Exception as e:
        return f"Error: {str(e)}"

def execute_web_search(query: str, max_results: int = 5) -> str:
    if not BRAVE_SEARCH_API_KEY:
        return "Error: web search not configured (set BRAVE_SEARCH_API_KEY in config.json)"
    q = (query or "").strip()
    if not q:
        return "Error: empty query"
    try:
        n = int(max_results)
    except (TypeError, ValueError):
        n = 5
    n = max(1, min(n, 10))
    try:
        r = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_SEARCH_API_KEY,
            },
            params={"q": q, "count": n},
            timeout=TOOL_TIMEOUT,
        )
        if r.status_code != 200:
            detail = (r.text or "")[:500]
            return f"Error: Brave Search HTTP {r.status_code}" + (f" — {detail}" if detail else "")
        data = r.json()
        results = (data.get("web") or {}).get("results") or []
        if not results:
            return f"No results for: {q}"
        parts = [f"Query: {q}", f"Results ({len(results)}):"]
        for i, item in enumerate(results, 1):
            title = (item.get("title") or "(no title)").strip()
            url = (item.get("url") or "").strip()
            desc = (item.get("description") or "").strip()
            block = f"{i}. {title}"
            if desc:
                block += f"\n   {desc}"
            if url:
                block += f"\n   {url}"
            parts.append(block)
        out = "\n\n".join(parts)
        if len(out) > 7000:
            out = out[:7000] + "\n[truncated]"
        out += "\n\n(Reminder: If you want to tell the user anything based on these search results (e.g. a spoken summary), call the speak() tool. You can continue with other tools such as run_terminal (curl etc.) first. Do not output the information as plain assistant content.)"
        return out
    except requests.Timeout:
        return f"Error: timed out after {TOOL_TIMEOUT}s"
    except Exception as e:
        return f"Error: {str(e)}"

def execute_speak(text: str) -> str:
    """Execute the speak tool: queue (with TTS) for delivery via /poll. This is the only way the user hears output."""
    txt = (text or "").strip()
    if not txt:
        return json.dumps({"status": "error", "message": "empty text"})

    try:
        queue_proactive_message(txt, speak=True)
    except Exception as e:
        print("Speak queue error:", e)
        return json.dumps({"status": "error", "message": str(e)})

    print(f"🗣️  Speak queued: {txt[:120]}{'...' if len(txt) > 120 else ''}")
    return json.dumps({
        "status": "audio queued for delivery to user",
        "text_spoken": txt,
        "note": "The user will hear the text above. If you are done communicating this to the user, stop now (do not emit plain content or repeat the text). You may call speak again or other tools if needed."
    }, ensure_ascii=False)

def execute_tool(tool_call: dict) -> str:
    fn = tool_call.get("function", {})
    name = fn.get("name", "")
    try:
        args = json.loads(fn.get("arguments", "{}"))
    except Exception:
        args = {}
    if name == "run_terminal":
        return execute_run_terminal(args.get("command", ""))
    if name == "web_search":
        return execute_web_search(args.get("query", ""), args.get("max_results", 5))
    if name == "speak":
        return execute_speak(args.get("text", ""))
    return f"Error: unknown tool {name}"

# ====================== ROLLING CONTEXT ======================
# conversation_history holds only the current session's user + final assistant turns.
# It is managed by trim_conversation_history which *prefers* LLM-generated compaction
# summaries over raw deletion when we approach the token limit.
conversation_history = []  # user + assistant messages (tool internals ephemeral per turn)

def estimate_tokens(x) -> int:
    if x is None:
        return 0
    try:
        s = json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else str(x)
    except Exception:
        s = str(x)
    return max(1, len(s) // 3)  # conservative ~3 chars/token for headroom

def trim_conversation_history():
    """Ensure conversation_history (+ protected memory messages) stays under MAX_CONTEXT_TOKENS.

    Preferred path: LLM compaction of oldest turns into a single dense summary message that
    is inserted at the front of the remaining history. This preserves session coherence far
    better than raw deletion.

    Dumb per-turn popping is retained only as an emergency fallback when:
    - We've already performed the allowed number of LLM compactions in this call, or
    - The summarizer returns "nothing significant", or
    - There aren't enough turns to justify a summary.

    The system prompt + persistent memory messages are always protected (never compacted).
    """
    global conversation_history
    if not conversation_history:
        return

    prefix = [{"role": "system", "content": SYSTEM_PROMPT}] + _get_memory_messages()
    pfx = len(prefix)
    max_compactions = 2  # limit expensive LLM calls per trim invocation
    compactions = 0

    while True:
        cur = prefix + conversation_history
        if len(cur) <= pfx or estimate_tokens(cur) <= MAX_CONTEXT_TOKENS:
            break

        # Preferred: try to compact a chunk of the oldest raw turns via LLM
        if compactions < max_compactions:
            total = len(conversation_history)
            # Compact a worthwhile chunk: at least 3 turns, at most ~10 or 1/3 of history
            chunk = min(10, max(3, total // 3))
            if total >= 3:
                to_compact = conversation_history[:chunk]
                summary = summarize_for_compaction(to_compact)
                # Drop the raw prefix we just summarized
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
                    continue  # check budget again

        # Emergency dumb fallback: bluntly drop the single oldest conversation turn.
        # When the front is a freshly created compaction summary we just paid an LLM call for,
        # prefer to drop an older raw turn behind it instead (protect the value of the compaction).
        if conversation_history:
            if "Compacted summary" in conversation_history[0].get("content", "") and len(conversation_history) > 1:
                del conversation_history[1]
            else:
                conversation_history.pop(0)

# ====================== PERSISTENT MEMORY ======================
# Small durable memory (~100 lines max) extracted from conversation before it is cleared.
# Injected as an extra system message at the start of new conversations.

def _load_persistent_memory():
    global persistent_memory
    if not os.path.exists(MEMORY_PATH):
        persistent_memory = ""
        return
    try:
        with open(MEMORY_PATH, "r", encoding="utf-8") as f:
            persistent_memory = f.read()
    except Exception as e:
        print("Warning: could not load memory:", e)
        persistent_memory = ""

def _save_persistent_memory():
    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            f.write(persistent_memory)
    except Exception as e:
        print("Warning: could not save memory:", e)

def _get_memory_messages() -> list:
    mem = (persistent_memory or "").strip()
    if not mem:
        return []
    return [{
        "role": "system",
        "content": "Key facts and context remembered from previous conversations (carry these forward):\n" + mem
    }]

def _append_memory(new_text: str):
    """Append a new memory entry (with date) and enforce ~100 line cap."""
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
    """Minimal non-tool LLM call for memory extraction and similar."""
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
    """Ask the LLM for a compact summary of a prefix of older turns.
    This is for within-session coherence when we need to reduce the rolling history
    (different goal from the durable persistent memory extracted on full clears).
    """
    if not older_turns:
        return ""
    # Instruction scoped to "still useful right now in this conversation".
    instruction = {
        "role": "user",
        "content": (
            "The turns above are older parts of the *current ongoing conversation* and need to be compacted.\n"
            "Create an extremely concise summary (bullets or 1-3 short paragraphs) of the user goals, key facts, decisions, important discoveries or tool outcomes, and context that the assistant must remember to remain coherent and effective for the rest of *this* session.\n"
            "Ignore transient one-off details. If there is little still relevant, reply exactly with: No significant earlier context."
        )
    }
    # Reuse main SYSTEM_PROMPT so the summarizer stays in the agent's character.
    msgs = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + older_turns
        + [instruction]
    )
    return _call_llm_simple(msgs, max_tokens=400, temperature=0.1)


def extract_memory_from_history() -> str:
    """Ask the LLM what (if anything) should be remembered before clearing the conversation."""
    global conversation_history
    if not conversation_history:
        return ""
    # Use the actual dialog turns + a targeted instruction.
    # Include the main SYSTEM_PROMPT so the model stays in character for "what *I* should remember".
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
    """Extract memory from the about-to-be-cleared history and append if useful."""
    try:
        mem = extract_memory_from_history()
        if mem:
            _append_memory(mem)
            # Keep a brief trace
            lines = [l for l in mem.splitlines() if l.strip()]
            print(f"🧠 Extracted memory ({len(lines)} lines) before clearing context")
    except Exception as e:
        print("Memory extraction failed (continuing):", e)

# ====================== LLM + MULTI-TURN TOOLS ======================
# Per-tool call limits for a single agent run (to prevent runaway web searches etc.)
_PER_TOOL_LIMITS = {
    "web_search": 3,
    "run_terminal": 20,
    "speak": 5,   # lower to prevent long rambling speak loops; model should answer then stop or do real work
}
_GLOBAL_TURN_LIMIT = 30  # bumped to support speak + progress + work loops

def process_with_llm(user_text: str = None, internal: bool = False) -> str:
    """Core ReAct-style agent loop.

    - The caller is responsible for appending real user turns to conversation_history before calling (for direct input).
    - For internal/cron-style runs, pass internal=True and a driving prompt as user_text (it will be injected only for this run, not persisted as user).
    - The loop continues while the LLM returns tool_calls. `speak` is a normal tool and is NOT terminal.
    - User-facing communication happens exclusively by calling the speak tool (which queues audio).
    - Plain 'content' without tool calls simply ends the run (the model should have used speak for anything it wanted the user to hear).
    - Returns a short status string (not user-facing content).
    """
    global conversation_history

    trim_conversation_history()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + _get_memory_messages() + conversation_history

    if user_text:
        # For internal runs this is the driving prompt (not recorded as persistent user turn).
        # For normal user runs the real user turn was already appended by the caller.
        messages = messages + [{"role": "user", "content": user_text}]

    turn = 0
    tool_counts = {k: 0 for k in _PER_TOOL_LIMITS}
    speaks_this_run = 0
    forced_speak_correction = 0  # allow at most one "you forgot to call speak, do it now" recovery per agent run

    while turn < _GLOBAL_TURN_LIMIT:
        turn += 1
        if turn > MAX_TOOL_TURNS:
            # Still respect the configured value as a soft signal, but we have a higher hard limit now.
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

                    # Per-tool limit check
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

                    if name == "web_search":
                        q = args.get("query", "")
                        print(f"  🔧 web_search: {q}" if q else "  🔧 web_search")
                    elif name == "run_terminal":
                        cmd = args.get("command", "")
                        print(f"  🔧 run_terminal: {cmd}" if cmd else "  🔧 run_terminal")
                    elif name == "speak":
                        speaks_this_run += 1
                        print(f"  🗣️  speak")
                    else:
                        print(f"  🔧 {name}")

                    out = execute_tool(tc)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id", ""),
                        "content": out
                    })
                    if name != "speak":
                        non_speak_in_batch = True

                if non_speak_in_batch:
                    # Per-turn guidance (not saved to permanent history) — encourages speak for user comms but allows continued tool use
                    messages.append({
                        "role": "user",
                        "content": "Reminder: If you now have information, a summary, or an update the user should hear, call the speak tool with natural spoken text. You may continue using other tools (such as run_terminal) for more work. Do not later repeat the same content as plain assistant messages (they are ignored). If you have nothing to communicate to the user right now, you can stop or keep working."
                    })
                continue
            else:
                # Plain content with no tool calls: the model has decided to stop.
                # Per new contract, ONLY speak() calls produce output to the user.
                # Plain content is always ignored (no fallback, no queuing).
                content = (msg.get("content") or "").strip()
                if content:
                    if speaks_this_run > 0:
                        print(f"  (model emitted additional plain content after speaking; ignored. Content was: {content[:100]})")
                    else:
                        print(f"  (model emitted plain content and stopped; not spoken because no speak() was used. Model tried to say: {content[:150]})")
                        if forced_speak_correction < 1:
                            forced_speak_correction += 1
                            # Give the model one recovery chance: tell it to use speak() for what it just tried to output.
                            # This is not auto-queuing the text; it forces the model to emit a proper speak tool call on the next iteration.
                            messages.append({
                                "role": "user",
                                "content": f"You emitted plain content instead of a speak() call. That is invalid — plain content is never delivered to the user. You MUST call the speak tool (following all CRITICAL rules: natural spoken English, no markdown, no lists, conversational). Call speak() now with a clean spoken version of the answer you wanted to give: {content[:300]}. If you have nothing to say, just stop without tool calls."
                            })
                            continue  # loop again so the model can (and must) call speak()
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
    """Background runner for user-initiated turns.

    The user message has already been appended to conversation_history by the
    /connect handler. This runs the full ReAct loop so that speak() calls can
    queue messages (and be picked up by the client's /poll) *while* the agent
    continues with more tools / more speak calls. This enables interleaved
    progress audio instead of waiting for the entire turn to finish.
    """
    try:
        status = process_with_llm(user_text=None, internal=False)
        print(f"🐹 Background agent complete: {status}")
    except Exception as ex:
        print("Background agent error:", ex)


# ====================== TTS ======================
def generate_tts_audio(text: str, quiet: bool = False) -> bytes:
    if not text or not TTS_BASE_URL:
        return b""
    try:
        payload = {
            "model": TTS_MODEL,
            "input": text.strip(),
            "voice": TTS_VOICE,
            "response_format": "wav"
        }
        if not quiet:
            print("🔊 TTS synthesis...")
        r = requests.post(f"{TTS_BASE_URL}/audio/speech", json=payload, timeout=180)
        if r.status_code == 200 and r.content:
            return r.content
        if r.status_code == 200:
            # 200 but no bytes: the TTS server accepted the request but generated no audio data.
            # Common causes: specific voice unavailable in this container/image (af_heart is flaky),
            # or the Kokoro service itself has stopped synthesizing (stale container, GPU issue, etc).
            print(f"TTS {r.status_code} but 0-byte body (no audio generated). voice={TTS_VOICE}")
            print("   Suggestion: restart the Kokoro TTS container, or try a different TTS_VOICE in config.json (am_adam / af_bella often more reliable).")
        else:
            print(f"TTS {r.status_code}: {r.text[:150] if r.text else ''}")
    except Exception as e:
        print("TTS error:", e)
    return b""

TTS_PROBE_TEXT = "Hi."
TTS_PROBE_MIN_BYTES = 1000  # WAV header + real audio (0-byte 200s mean Kokoro GPU is broken)
TTS_PROBE_CACHE_SECONDS = 60
_tts_probe_cache = {"ok": None, "bytes": 0, "error": None, "checked_at": None}
_tts_probe_lock = threading.Lock()

def probe_tts_synthesis(force: bool = False) -> dict:
    """Check whether Kokoro is actually returning audio (not just HTTP 200 + empty body).
    Results are cached for TTS_PROBE_CACHE_SECONDS; pass force=True to bypass."""
    if not TTS_BASE_URL:
        return {"ok": None, "bytes": 0, "error": None, "checked_at": None}

    with _tts_probe_lock:
        cached_at = _tts_probe_cache.get("checked_at")
        if not force and cached_at and (time.time() - cached_at) < TTS_PROBE_CACHE_SECONDS:
            return dict(_tts_probe_cache)

    audio = generate_tts_audio(TTS_PROBE_TEXT, quiet=True)
    if len(audio) >= TTS_PROBE_MIN_BYTES:
        result = {"ok": True, "bytes": len(audio), "error": None, "checked_at": time.time()}
    else:
        result = {
            "ok": False,
            "bytes": len(audio),
            "error": "no audio — restart kokoro-tts container",
            "checked_at": time.time(),
        }

    with _tts_probe_lock:
        _tts_probe_cache.update(result)
    return dict(result)

# ====================== STT (whisper.cpp) ======================
def transcribe_audio(audio_file) -> str:
    """FileStorage -> text via whisper.cpp server"""
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp_path = tmp.name
    try:
        audio_file.save(tmp_path)
        files = {"file": open(tmp_path, "rb")}
        data = {
            "model": WHISPER_MODEL,
            "language": "en",
            "temperature": "0.0",
            "response_format": "json"
        }
        print("📤 Transcribing via whisper.cpp...")
        r = requests.post(f"{WHISPER_BASE_URL}/v1/audio/transcriptions", files=files, data=data, timeout=120)
        if r.status_code == 200:
            txt = r.json().get("text", "").strip()
            return txt
        print(f"Whisper {r.status_code}: {r.text[:150]}")
    except Exception as e:
        print("STT error:", e)
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
    return ""

WHISPER_PROBE_CACHE_SECONDS = 60
_whisper_probe_cache = {"ok": None, "error": None, "checked_at": None}
_whisper_probe_lock = threading.Lock()

def _make_stt_probe_wav(duration_sec: float = 0.5, sample_rate: int = 16000) -> bytes:
    """Minimal 16 kHz mono WAV for whisper.cpp health checks (silence is fine)."""
    n_samples = max(1, int(sample_rate * duration_sec))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(b"\x00\x00" * n_samples)
    return buf.getvalue()

def probe_whisper_stt(force: bool = False) -> dict:
    """Check whether whisper.cpp accepts audio and returns a valid transcription response."""
    if not WHISPER_BASE_URL:
        return {"ok": None, "error": None, "checked_at": None}

    with _whisper_probe_lock:
        cached_at = _whisper_probe_cache.get("checked_at")
        if not force and cached_at and (time.time() - cached_at) < WHISPER_PROBE_CACHE_SECONDS:
            return dict(_whisper_probe_cache)

    try:
        wav = _make_stt_probe_wav()
        files = {"file": ("probe.wav", wav, "audio/wav")}
        data = {
            "model": WHISPER_MODEL,
            "language": "en",
            "temperature": "0.0",
            "response_format": "json",
        }
        r = requests.post(
            f"{WHISPER_BASE_URL}/v1/audio/transcriptions",
            files=files,
            data=data,
            timeout=30,
        )
        if r.status_code == 200:
            r.json()  # validate JSON body
            result = {"ok": True, "error": None, "checked_at": time.time()}
        else:
            result = {
                "ok": False,
                "error": f"HTTP {r.status_code}" + (f": {r.text[:80]}" if r.text else ""),
                "checked_at": time.time(),
            }
    except Exception as e:
        result = {"ok": False, "error": str(e), "checked_at": time.time()}

    with _whisper_probe_lock:
        _whisper_probe_cache.update(result)
    return dict(result)


# ====================== IMAGE DETECTION (YOLO external server) ======================
def detect_objects(image_file) -> list:
    """Accept FileStorage (from request.files 'image' or 'file'). Forward to YOLO /upload.
    Return list of detected object label strings (e.g. ['person', 'cat']).
    """
    if not DETECTION_BASE_URL:
        return []
    try:
        image_bytes = image_file.read()
        files = {"image": ("image.jpg", image_bytes)}
        print("🖼️  Detecting objects via YOLO server...")
        r = requests.post(f"{DETECTION_BASE_URL}/upload", files=files, timeout=120)
        if r.status_code == 200:
            data = r.json()
            dets = data.get("detections", [])
            labels = [d.get("name") for d in dets if d.get("name")]
            print(f"   Detected: {labels}")
            return labels
        print(f"Detection HTTP {r.status_code}: {r.text[:200] if r.text else ''}")
    except Exception as e:
        print("Detection error:", e)
    return []


# ====================== PROACTIVE QUEUE HELPER ======================
def queue_proactive_message(text: str, speak: bool = True) -> dict:
    """Queue a message for the client to receive on its next /poll when idle.
    If speak and TTS is configured, pre-generates the audio at enqueue time.
    Returns the queued item dict. Thread-safe. Enforces size + age limits.
    """
    global pending_initiations
    if not text or not text.strip():
        return {}

    audio_b64 = None
    if speak and TTS_BASE_URL:
        try:
            audio_bytes = generate_tts_audio(text)
            if audio_bytes:
                audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
        except Exception as e:
            print("Proactive TTS generation failed:", e)

    item = {
        "id": str(uuid.uuid4()),
        "text": text.strip(),
        "audio": audio_b64,
        "created_at": datetime.datetime.now().isoformat()
    }

    with pending_lock:
        # Drop anything too old
        now = datetime.datetime.now()
        while pending_initiations:
            oldest = pending_initiations[0]
            try:
                created = datetime.datetime.fromisoformat(oldest["created_at"])
                if (now - created).total_seconds() > MAX_INITIATION_AGE_SECONDS:
                    pending_initiations.popleft()
                    print("🗑️  Dropped stale proactive message (age)")
                    continue
            except Exception:
                pending_initiations.popleft()
                continue
            break

        # Enforce max depth (drop oldest if full)
        while len(pending_initiations) >= MAX_PENDING_INITIATIONS:
            dropped = pending_initiations.popleft()
            print(f"🗑️  Dropped oldest proactive (queue full): {dropped['text'][:60]}...")

        pending_initiations.append(item)
        # Wake any long-poll waiters
        initiation_ready.notify_all()

    print(f"📣 Queued proactive message (queue size={len(pending_initiations)}): {text[:80]}{'...' if len(text) > 80 else ''}")
    return item


# ====================== CRON SCHEDULER ======================
def start_cron_scheduler():
    """Start a daemon thread that periodically checks cron_jobs and fires any that are due.
    Due jobs run their prompt through the LLM (internal=True so history stays clean) and
    the resulting text is queued as a proactive message (which will be spoken + added to
    conversation on delivery, exactly like other proactive messages)."""
    enabled_jobs = [j for j in cron_jobs if j.get("enabled", True)]
    if not enabled_jobs:
        return

    def _cron_loop():
        while True:
            try:
                time.sleep(30)  # minute-granularity crons are well served by 30s checks
                now = datetime.datetime.now()
                for job in cron_jobs:
                    if not job.get("enabled", True):
                        continue
                    sched = job.get("schedule", "")
                    prompt = job.get("prompt", "")
                    if not sched or not prompt:
                        continue
                    if not cron_due(sched, now):
                        continue
                    # Use minute slot for "already executed this occurrence?"
                    slot = now.replace(second=0, microsecond=0)
                    lr = job.get("last_run")
                    if lr is not None:
                        if lr.replace(second=0, microsecond=0) == slot:
                            continue
                    job["last_run"] = now
                    _save_cron_last_run(job["id"], now)
                    print(f"\n⏰ Cron fired [{sched}]: {prompt[:90]}{'...' if len(prompt) > 90 else ''}")
                    try:
                        # The agent loop will queue any speak() calls itself. No need to take a return value.
                        status = process_with_llm(prompt, internal=True)
                        print(f"🐹 Cron agent status: {status}")
                    except Exception as ex:
                        print("Cron job failed:", ex)
            except Exception as e:
                print("Cron scheduler error (retrying):", e)
                time.sleep(10)

    t = threading.Thread(target=_cron_loop, daemon=True, name="marmot-cron")
    t.start()
    print(f"⏰ Cron scheduler started for {len(enabled_jobs)} job(s)")


# ====================== FLASK ======================
# Load memory (after all helper defs are registered) and emit startup banner
_load_persistent_memory()
_mem_lines = len([ln for ln in (persistent_memory or "").splitlines() if ln.strip()])

load_cron_jobs()

print("🐹 Marmot Agent Server ready")
print(f"   Whisper: {WHISPER_BASE_URL}  model={WHISPER_MODEL}")
print(f"   LLM:     {LLM_MODEL} @ {LLM_BASE_URL}")
print(f"   TTS:     {TTS_MODEL}/{TTS_VOICE} @ {TTS_BASE_URL or '(disabled)'}")
print(f"   Detection: {DETECTION_BASE_URL or '(disabled)'}")
print(f"   Context: ~{MAX_CONTEXT_TOKENS} tokens max (rolling + LLM compaction of old turns)")
_tool_names = ", ".join(t["function"]["name"] for t in TOOLS) if TOOLS else "(none)"
print(f"   Tools:   {'on' if TOOLS_ENABLED else 'off'}   [{_tool_names}]   tool-timeout={TOOL_TIMEOUT}s")
if WEB_SEARCH_ENABLED and not BRAVE_SEARCH_API_KEY:
    print("   Web search: disabled (set BRAVE_SEARCH_API_KEY in config.json)")
print(f"   Inactivity timeout: {CONTEXT_TIMEOUT_HOURS}h → auto-clear context")
print(f"   Memory:   {_mem_lines} lines persisted (≤100, extracted before clears)")
if cron_jobs:
    en = sum(1 for j in cron_jobs if j.get("enabled", True))
    dis = len(cron_jobs) - en
    extra = f" ({dis} disabled)" if dis else ""
    print(f"   Cron:     {en} job(s) from cron.json{extra}")
print()

try:
    wprobe = probe_whisper_stt(force=True)
    if wprobe.get("ok"):
        print("   Whisper probe: OK")
    else:
        print(f"⚠️  Whisper probe failed: {wprobe.get('error') or 'unknown error'}")
except Exception as _e:
    print("⚠️  Whisper probe error (non-fatal):", _e)

# Quick TTS probe so users immediately see if the configured voice is producing audio.
if TTS_BASE_URL:
    try:
        probe = probe_tts_synthesis(force=True)
        if probe.get("ok"):
            print(f"   TTS probe: OK ({probe['bytes']} bytes)")
        else:
            print(f"⚠️  TTS probe failed: {probe.get('error') or 'no audio'}")
    except Exception as _e:
        print("⚠️  TTS probe error (non-fatal):", _e)

app = Flask(__name__)

# ====================== SIMPLE STATUS DASHBOARD ======================
# Served at GET /  — a lightweight, auto-refreshing page showing /health data.
# Template loaded from templates/index.html at startup.
TEMPLATE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates", "index.html")

def _load_index_html() -> str:
    """Read the dashboard HTML template and inject the mascot image base64."""
    MARMOT_B64 = ""
    try:
        img_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "images", "marmot-harness.jpg")
        with open(img_path, "rb") as f:
            MARMOT_B64 = base64.b64encode(f.read()).decode("ascii")
    except Exception as e:
        print("Warning: could not load mascot image for dashboard:", e)

    try:
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            html = f.read()
        return html.replace("MARMOT_B64_HERE", MARMOT_B64)
    except Exception as e:
        print("Warning: could not load dashboard template:", e)
        return "<h1>Marmot</h1><p>Dashboard template not found.</p>"

INDEX_HTML = _load_index_html()

@app.route("/", methods=["GET"])
def index():
    """Simple self-contained status dashboard. Refreshes automatically."""
    return INDEX_HTML


@app.route("/connect", methods=["POST"])
def connect():
    user_text = None
    if request.files and "file" in request.files:
        f = request.files["file"]
        if f and f.filename:
            user_text = transcribe_audio(f)

    if not user_text:
        if request.is_json:
            body = request.json or {}
            user_text = body.get("text", "")
        else:
            user_text = request.form.get("text", "")
        user_text = (user_text or "").strip()

    if not user_text:
        return jsonify({"error": "Send audio file or text"}), 400

    # === Inactivity timeout check: clear context if > CONTEXT_TIMEOUT_HOURS since last message ===
    global last_message_time
    now = datetime.datetime.now()
    if last_message_time is not None:
        delta = now - last_message_time
        if delta.total_seconds() > (CONTEXT_TIMEOUT_HOURS * 3600):
            print(f"⏰ No messages for >{CONTEXT_TIMEOUT_HOURS} hours — clearing conversation context")
            commit_memory_before_clear()
            conversation_history.clear()
    last_message_time = now

    print(f"\n👤 User: {user_text}")
    with history_lock:
        conversation_history.append({"role": "user", "content": user_text})

    # Start the agent in a background thread so that speak() calls can immediately
    # queue messages (visible to /poll) and the agent can continue its loop
    # (tools + more speak calls) for true interleaved/progress audio.
    # The /connect response returns quickly; the client relies on polling for replies.
    threading.Thread(
        target=_run_agent_for_user_turn,
        args=(user_text,),
        daemon=True,
        name="marmot-agent-user"
    ).start()

    return jsonify({
        "transcription": user_text,
        "status": "processing"
    })

@app.route("/health", methods=["GET"])
def health():
    now = datetime.datetime.now()
    seconds_since_last = None
    if last_message_time is not None:
        seconds_since_last = int((now - last_message_time).total_seconds())

    with pending_lock:
        pending_count = len(pending_initiations)

    cron_summary = [
        {
            "schedule": j["schedule"],
            "enabled": j.get("enabled", True),
            "last_run": j["last_run"].isoformat() if j.get("last_run") else None
        }
        for j in cron_jobs
    ]

    probe_arg = (request.args.get("probe") or "").strip().lower()
    force_all = probe_arg in ("1", "true", "all", "yes")
    force_whisper = force_all or probe_arg == "whisper"
    force_tts = force_all or probe_arg == "tts"

    whisper_stt = probe_whisper_stt(force=force_whisper)
    tts_synthesis = probe_tts_synthesis(force=force_tts) if TTS_BASE_URL else None
    ok = True
    if whisper_stt.get("ok") is False:
        ok = False
    if tts_synthesis and tts_synthesis.get("ok") is False:
        ok = False

    return jsonify({
        "ok": ok,
        "whisper": WHISPER_BASE_URL,
        "whisper_model": WHISPER_MODEL,
        "whisper_stt": whisper_stt,
        "llm": LLM_MODEL,
        "llm_base_url": LLM_BASE_URL,
        "tts": bool(TTS_BASE_URL),
        "tts_model": TTS_MODEL if TTS_BASE_URL else None,
        "tts_voice": TTS_VOICE if TTS_BASE_URL else None,
        "tts_synthesis": tts_synthesis,
        "detection": DETECTION_BASE_URL,
        "turns": len([m for m in conversation_history if m["role"] in ("user", "assistant")]),
        "context_timeout_hours": CONTEXT_TIMEOUT_HOURS,
        "last_message_at": last_message_time.isoformat() if last_message_time else None,
        "seconds_since_last_message": seconds_since_last,
        "memory_lines": len([ln for ln in (persistent_memory or "").splitlines() if ln.strip()]),
        "pending_initiations": pending_count,
        "cron_jobs": len(cron_jobs),
        "cron": cron_summary
    })

@app.route("/reset", methods=["POST"])
def reset():
    global conversation_history, last_message_time
    commit_memory_before_clear()
    conversation_history = []
    last_message_time = None
    with pending_lock:
        pending_initiations.clear()
    return jsonify({"ok": True, "msg": "context cleared"})

@app.route("/poll", methods=["GET"])
def poll():
    """Client idle poll. Returns a proactive initiation if one is queued (and commits it to conversation history).
    Supports optional long-poll via ?wait=seconds (capped at 10).
    """
    global last_message_time, conversation_history

    wait = 0.0
    try:
        wait = float(request.args.get("wait", "0") or "0")
    except Exception:
        wait = 0.0
    wait = max(0.0, min(wait, 10.0))

    deadline = time.time() + wait

    while True:
        with initiation_ready:
            # Prune stale inside the lock
            now = datetime.datetime.now()
            while pending_initiations:
                try:
                    oldest = pending_initiations[0]
                    created = datetime.datetime.fromisoformat(oldest["created_at"])
                    if (now - created).total_seconds() > MAX_INITIATION_AGE_SECONDS:
                        pending_initiations.popleft()
                        continue
                except Exception:
                    pending_initiations.popleft()
                    continue
                break

            if pending_initiations:
                item = pending_initiations.popleft()
                # Commit this as an assistant turn so the conversation continues naturally
                with history_lock:
                    conversation_history.append({"role": "assistant", "content": item["text"]})
                last_message_time = datetime.datetime.now()
                # Trim opportunistically (cheap if not near limit)
                try:
                    trim_conversation_history()
                except Exception:
                    pass
                print(f"📤 Delivering proactive via /poll: {item['text'][:100]}{'...' if len(item['text']) > 100 else ''}")
                return jsonify({"action": "initiate", "message": item})

            remaining = deadline - time.time()
            if remaining <= 0:
                return jsonify({"action": "noop"})

            # Efficiently wait for a new enqueue or timeout slice
            initiation_ready.wait(timeout=min(remaining, 1.0))

@app.route("/inject", methods=["POST"])
def inject():
    """Manual/test hook to queue a proactive message from outside (e.g. scripts, future schedulers).
    Body: {"text": "message here", "speak": true}
    """
    if not request.is_json:
        return jsonify({"error": "expected application/json"}), 400
    data = request.json or {}
    text = (data.get("text") or "").strip()
    speak = data.get("speak", True)
    if not isinstance(speak, bool):
        speak = str(speak).lower() in ("1", "true", "yes", "on")
    if not text:
        return jsonify({"error": "text is required"}), 400

    item = queue_proactive_message(text, speak=speak)
    return jsonify({"ok": True, "queued": bool(item), "message": item})


@app.route("/detect", methods=["POST"])
def detect():
    """Detect objects in an uploaded image using the external YOLO server.
    Accepts multipart form with 'image' or 'file'.
    Returns {"objects": ["label", "label", ...]} (just the class names).
    """
    if not DETECTION_BASE_URL:
        return jsonify({"error": "Detection server not configured"}), 503

    image_file = None
    if request.files:
        image_file = request.files.get("file") or request.files.get("image")
    if not image_file or not getattr(image_file, "filename", None):
        return jsonify({"error": "Send image file as 'image' or 'file' form field"}), 400

    labels = detect_objects(image_file)
    return jsonify({"objects": labels})


class QuietPollRequestHandler(WSGIRequestHandler):
    """Custom request handler that suppresses log spam from frequent endpoints
    (/poll for the proactive client, and GET /health for the status dashboard auto-refresh).
    All other endpoints continue to log normally.
    """
    def log_request(self, code='-', size='-'):
        if self.path:
            path = self.path.split('?', 1)[0]
            if path.startswith('/poll') or path == '/health':
                return  # keep console clean for high-frequency polling/status endpoints
        super().log_request(code, size)


if __name__ == "__main__":
    port = int(os.environ.get("MARMOT_PORT", 5000))
    start_cron_scheduler()
    print(f"🌐 Dashboard: http://0.0.0.0:{port}/")
    print(f"   API:      /connect  /health  /reset  /poll  /inject  /detect")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True,
            request_handler=QuietPollRequestHandler)
