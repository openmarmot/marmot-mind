#!/usr/bin/env python3
"""
Marmot Agent Server

Flask orchestrator (new output model):
  audio/text input via /connect -> STT (if audio) + record user turn -> start ReAct agent in background thread
  The LLM communicates to the user *only* by calling the speak() tool.
  speak() -> TTS + queue_proactive_message (immediately visible to client /poll).
  Agent continues loop after each speak (tools + more speaks) for interleaved progress audio.
  All user output (replies + proactives) delivered via client /poll.

Rolling conversation context (public spoken record) + live dynamic mind state.

The mind has:
  - Separate live internal state (focus, observations) injected into all LLM prompts
  - Self-determined wake schedule (via plan_next_wake tool — no more static human cron for thinking)
  - Background autonomous loop for its own thoughts/work (cross-pollinates human responses)
  - conversation_history is only the human-visible transcript
"""

import os
import io
import json
import wave
import tempfile
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
        "TOOLS_ENABLED": True,
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
                    "TOOLS_ENABLED", "MAX_TOOL_TURNS",
                    "CONTEXT_TIMEOUT_HOURS", "DETECTION_BASE_URL",
                    "WEB_SEARCH_ENABLED", "BRAVE_SEARCH_API_KEY"]
            with open(CONFIG_PATH, "w") as f:
                json.dump({k: cfg[k] for k in keys if k in cfg}, f, indent=2)
            print(f"✅ Saved config to {CONFIG_PATH}")
        except Exception as e:
            print("⚠️  Could not save config:", e)
    return cfg


def load_system_prompt() -> str:
    """Load the system prompt from the prompts/ directory.

    This is intentionally *not* stored in config.json. Config is for
    user-provided connection details (URLs, keys, etc.). The prompt
    is part of the agent's core behavior and lives alongside the code.
    """
    prompt_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "prompts",
        "system_prompt.txt"
    )
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if not content:
            raise ValueError("prompt file is empty")
        return content
    except Exception as e:
        print("⚠️  Could not load system_prompt.txt:", e)
        print("    Falling back to minimal prompt.")
        return "You are a helpful agent."


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
SYSTEM_PROMPT = load_system_prompt()
TOOLS_ENABLED = bool(config.get("TOOLS_ENABLED", True))
MAX_TOOL_TURNS = int(config.get("MAX_TOOL_TURNS", 8))
CONTEXT_TIMEOUT_HOURS = int(config.get("CONTEXT_TIMEOUT_HOURS", 10))
WEB_SEARCH_ENABLED = bool(config.get("WEB_SEARCH_ENABLED", True))
BRAVE_SEARCH_API_KEY = (config.get("BRAVE_SEARCH_API_KEY") or "").strip() or None

# ====================== PROACTIVE INITIATION (server -> client) ======================
# Client polls /poll when idle. Server can queue messages it wants to deliver unprompted.
# Items are dicts: {"id": str, "text": str, "audio": base64 or None, "created_at": iso}
pending_initiations = deque()
pending_lock = threading.Lock()
initiation_ready = threading.Condition(pending_lock)  # allows efficient long-poll wakeups
MAX_PENDING_INITIATIONS = 5
MAX_INITIATION_AGE_SECONDS = 3600  # 1 hour


def _prune_stale_initiations(log_drops: bool = False):
    """Drop proactive messages that are too old from the front of the queue.

    Must be called while the pending_lock (or initiation_ready Condition) is held.
    If log_drops, print a message for each age-based drop (matches historical queue path).
    """
    now = datetime.datetime.now()
    while pending_initiations:
        oldest = pending_initiations[0]
        try:
            created = datetime.datetime.fromisoformat(oldest["created_at"])
            if (now - created).total_seconds() > MAX_INITIATION_AGE_SECONDS:
                pending_initiations.popleft()
                if log_drops:
                    print("🗑️  Dropped stale proactive message (age)")
                continue
        except Exception:
            pending_initiations.popleft()
            continue
        break





# ====================== TOOLS ======================
AGENT_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "agent-data"))
os.makedirs(AGENT_DATA_DIR, exist_ok=True)

# Tool calls get their own working directory so files the agent creates (via run_terminal etc.)
# are separated from Marmot's own data like memory.txt.
TOOL_CALLS_DIR = os.path.join(AGENT_DATA_DIR, "tool-calls")
os.makedirs(TOOL_CALLS_DIR, exist_ok=True)

MEMORY_PATH = os.path.join(AGENT_DATA_DIR, "memory.txt")

from tools import BASE_TOOLS, WEB_SEARCH_TOOL, execute_tool
from tools import configure_tools

# Inject server-owned state into the (now decoupled) tools.
# This replaces the previous "from server import ..." inside tool modules.
configure_tools(tool_calls_dir=TOOL_CALLS_DIR, brave_api_key=BRAVE_SEARCH_API_KEY)

TOOLS = list(BASE_TOOLS)
if WEB_SEARCH_ENABLED and BRAVE_SEARCH_API_KEY:
    TOOLS.append(WEB_SEARCH_TOOL)

# ====================== MIND WIRING ======================
import mind

mind.configure(
    llm_base_url=LLM_BASE_URL,
    llm_model=LLM_MODEL,
    system_prompt=SYSTEM_PROMPT,
    tools=TOOLS,
    tools_enabled=TOOLS_ENABLED,
    max_tool_turns=MAX_TOOL_TURNS,
    max_context_tokens=MAX_CONTEXT_TOKENS,
    memory_path=MEMORY_PATH,
)

# ====================== TTS ======================


















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
        # Drop anything too old (log_drops=True reproduces the original per-item prints in queue path)
        _prune_stale_initiations(log_drops=True)


        # Enforce max depth (drop oldest if full)
        while len(pending_initiations) >= MAX_PENDING_INITIATIONS:
            dropped = pending_initiations.popleft()
            print(f"🗑️  Dropped oldest proactive (queue full): {dropped['text'][:60]}...")

        pending_initiations.append(item)
        # Wake any long-poll waiters
        initiation_ready.notify_all()

    print(f"📣 Queued proactive message (queue size={len(pending_initiations)}): {text[:80]}{'...' if len(text) > 80 else ''}")
    return item


# Register speak handler *after* the function is defined (Python executes top-to-bottom at import time).
mind.set_speak_handler(queue_proactive_message)


# ====================== FLASK ======================
# Data loads are intentionally at import time (they populate globals used by routes/agents).
# The human-facing banner + probes are emitted only when run as a script (see __main__).
mind._load_persistent_memory()

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
    now = datetime.datetime.now()
    if mind.get_last_message_time() is not None:
        delta = now - mind.get_last_message_time()
        if delta.total_seconds() > (CONTEXT_TIMEOUT_HOURS * 3600):
            print(f"⏰ No messages for >{CONTEXT_TIMEOUT_HOURS} hours — clearing conversation context")
            mind.commit_memory_before_clear()
            mind.conversation_history.clear()
    mind.set_last_message_time(now)

    print(f"\n👤 User: {user_text}")

    # Cross-pollination: record the human arrival in live mind state.
    # Note: the direct reply to this human turn is handled by the main agent; background steps should rarely speak.
    mind._log_mind_observation(f"Human just spoke (primary reply handled separately): {user_text[:140]}")
    mind.mind_wake_event.set()

    with mind.history_lock:
        mind.conversation_history.append({"role": "user", "content": user_text})

    mind.set_pending_direct_user_question(user_text)

    # Generate a fresh LLM summary of recent human interactions so the
    # background mind sees a compact view instead of the raw full transcript.
    try:
        mind.refresh_recent_human_summary()
    except Exception:
        pass

    # Start the agent in a background thread. The runner lives in mind now.
    threading.Thread(
        target=mind._run_agent_for_user_turn,
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
    if mind.get_last_message_time() is not None:
        seconds_since_last = int((now - mind.get_last_message_time()).total_seconds())

    with pending_lock:
        pending_count = len(pending_initiations)

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
        "turns": len([m for m in mind.conversation_history if m["role"] in ("user", "assistant")]),
        "context_timeout_hours": CONTEXT_TIMEOUT_HOURS,
        "last_message_at": mind.get_last_message_time().isoformat() if mind.get_last_message_time() else None,
        "seconds_since_last_message": seconds_since_last,
        "memory_lines": mind._count_memory_lines(),
        "pending_initiations": pending_count,
        "mind": mind._get_mind_status_for_health()
    })

@app.route("/reset", methods=["POST"])
def reset():
    mind.commit_memory_before_clear()
    mind.clear_conversation_history()
    mind.set_last_message_time(None)
    with pending_lock:
        pending_initiations.clear()
    # Clear focus on reset (mind keeps other observations)
    with mind.mind_lock:
        mind.mind_state["current_focus"] = "context was reset by human; re-evaluating"
    mind.mind_wake_event.set()
    print("🧠 Mind acknowledged context reset")
    return jsonify({"ok": True, "msg": "context cleared"})

@app.route("/poll", methods=["GET"])
def poll():
    """Client idle poll. Returns a proactive initiation if one is queued (and commits it to conversation history).
    Supports optional long-poll via ?wait=seconds (capped at 10).
    """
    wait = 0.0
    try:
        wait = float(request.args.get("wait", "0") or "0")
    except Exception:
        wait = 0.0
    wait = max(0.0, min(wait, 10.0))

    deadline = time.time() + wait

    while True:
        with initiation_ready:
            # Prune stale inside the lock (no logging here, unlike queue path)
            _prune_stale_initiations(log_drops=False)

            if pending_initiations:
                item = pending_initiations.popleft()
                # Commit this as an assistant turn so the conversation continues naturally
                with mind.history_lock:
                    mind.conversation_history.append({"role": "assistant", "content": item["text"]})
                mind.set_last_message_time(datetime.datetime.now())
                try:
                    mind.trim_history()
                except Exception:
                    pass
                # Refresh the recent-human summary so background mind steps see that
                # a spoken message (proactive or otherwise) was just part of the exchange.
                try:
                    mind.refresh_recent_human_summary()
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


def _emit_startup_banner():
    """Human-facing startup information + quick connectivity probes.

    Called only from the `if __name__ == "__main__"` path so that plain
    `import server` (tests, gunicorn workers, etc.) stays quiet.
    """
    mem_lines = mind._count_memory_lines()

    print("🐹 Marmot Agent Server ready")
    print(f"   Whisper: {WHISPER_BASE_URL}  model={WHISPER_MODEL}")
    print(f"   LLM:     {LLM_MODEL} @ {LLM_BASE_URL}")
    print(f"   TTS:     {TTS_MODEL}/{TTS_VOICE} @ {TTS_BASE_URL or '(disabled)'}")
    print(f"   Detection: {DETECTION_BASE_URL or '(disabled)'}")
    print(f"   Context: ~{MAX_CONTEXT_TOKENS} tokens max (rolling + LLM compaction of old turns)")
    _tool_names = ", ".join(t["function"]["name"] for t in TOOLS) if TOOLS else "(none)"
    print(f"   Tools:   {'on' if TOOLS_ENABLED else 'off'}   [{_tool_names}]")
    if WEB_SEARCH_ENABLED and not BRAVE_SEARCH_API_KEY:
        print("   Web search: disabled (set BRAVE_SEARCH_API_KEY in config.json)")
    print(f"   Inactivity timeout: {CONTEXT_TIMEOUT_HOURS}h → auto-clear context")
    print(f"   Memory:   {mem_lines} lines persisted (≤100, extracted before clears)")
    print(f"   Mind:     live state + autonomous self-wake loop (dynamic, LLM-driven)")
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


if __name__ == "__main__":
    port = int(os.environ.get("MARMOT_PORT", 5000))
    _emit_startup_banner()
    mind._start_autonomous_mind_loop()
    print(f"🌐 Dashboard: http://0.0.0.0:{port}/")
    print(f"   API:      /connect  /health  /reset  /poll  /inject  /detect")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True,
            request_handler=QuietPollRequestHandler)
