#!/usr/bin/env python3
"""
Marmot Mind — independent AI chat participant.

Each process is one mind (one username). Multiple concurrent mind.py processes
can join the same chat server and interact.

- Status / config UI on a random free port (Flask)
- Single think loop: fetch chat → act (tools) → schedule next wake
- All state per username in SQLite under mind/data/{username}/
"""

import os
import sys
import socket
import argparse
import threading
import time
import random
import datetime
import builtins
from flask import Flask, request, jsonify, render_template

from storage import MindStore, list_usernames
from chat_client import ChatClient
from personality import generate_personality
from agent import run_think_loop, log, message_tags_me

# ========================= PATHS =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.join(os.path.dirname(BASE_DIR), "data")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "system_prompt.txt")

os.makedirs(DATA_ROOT, exist_ok=True)

# ========================= RUNTIME =========================
app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

_store: MindStore | None = None
_chat: ChatClient | None = None
_system_prompt: str = ""
_port: int = 0
_loop_thread: threading.Thread | None = None
_mention_thread: threading.Thread | None = None
_loop_stop = threading.Event()
_loop_wake = threading.Event()
_mention_stop = threading.Event()
_loop_running = False
_loop_lock = threading.Lock()
_tick_lock = threading.Lock()
_last_mention_wake_at: str | None = None
_last_mention_info: str | None = None


def _load_system_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content
    except Exception as e:
        log("Warning: could not load system prompt:", e)
    return "You are an autonomous mind in a chat room. Use post_message to talk."


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("0.0.0.0", 0))
        return s.getsockname()[1]


def _fix_url(u: str) -> str:
    u = (u or "").strip()
    if u and not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u.rstrip("/")


def _require_store():
    if _store is None:
        return None, (jsonify({"error": "no mind identity loaded — create or resume first"}), 400)
    return _store, None


def _build_chat_from_store(store: MindStore) -> ChatClient | None:
    url = _fix_url(store.get_config("chat_server_url") or "")
    token = store.get_config("chat_token")
    if not url:
        return None
    return ChatClient(url, token=token, username=store.username)


def _connect_chat(store: MindStore) -> ChatClient:
    url = _fix_url(store.get_config("chat_server_url") or "")
    if not url:
        raise RuntimeError("chat_server_url not configured")
    client = ChatClient(url, username=store.username)
    # Prefer stored token; fall back to login/signup
    token = store.get_config("chat_token")
    if token:
        client.token = token
        try:
            client.me()
            return client
        except Exception:
            pass
    data = client.ensure_registered(store.username)
    store.set_config("chat_token", data["token"])
    store.set_config("chat_username", data["username"])
    return client


# ========================= THINK LOOP =========================
def _seconds_until_wake(store: MindStore) -> float:
    nw = store.get_state("next_wake_after")
    if not nw:
        return 120.0
    try:
        target = datetime.datetime.fromisoformat(nw)
        # handle timezone-naive
        if target.tzinfo is not None:
            now = datetime.datetime.now(target.tzinfo)
        else:
            now = datetime.datetime.now()
        delta = (target - now).total_seconds()
        return max(20.0, min(7200.0, delta))
    except Exception:
        return 120.0


def _loop_body():
    global _loop_running, _chat
    log("🧠 Think loop thread started")
    # Short delay so config can settle after start
    time.sleep(2)
    while not _loop_stop.is_set():
        try:
            store = _store
            if store is None:
                time.sleep(2)
                continue
            if not store.get_config("loop_enabled"):
                # Wait until re-enabled or stop
                _loop_wake.wait(timeout=5)
                _loop_wake.clear()
                continue

            with _tick_lock:
                try:
                    if _chat is None or not _chat.token:
                        _chat = _connect_chat(store)
                    status = run_think_loop(store, _chat, _system_prompt)
                    log(f"🧠 tick: {status}")
                except Exception as e:
                    log("Think loop error:", e)
                    store.set_state("last_loop_status", f"error: {e}")
                    store.set_state("last_loop_at", datetime.datetime.now().isoformat())

            sleep_secs = _seconds_until_wake(store)
            log(f"🧠 sleeping ~{int(sleep_secs)}s until next wake (mentions still watched)")
            woke = _loop_wake.wait(timeout=sleep_secs)
            _loop_wake.clear()
            if woke:
                reason = (store.get_state("last_wake_reason") or "").strip()
                if reason.startswith("mention"):
                    log(f"🧠 woken early by {reason}")
                else:
                    log("🧠 woken early")
        except Exception as e:
            log("Loop outer error:", e)
            time.sleep(30)

    _loop_running = False
    log("🧠 Think loop thread stopped")


# ========================= MENTION WATCHER =========================
def _mention_watcher_body():
    """Fast poll for new @mentions / @everyone; wakes the think loop when found.

    Interval is randomized 1–5s so multiple concurrent minds don't stampede the
    chat server on the same beat. Uses last_seen_message_id so only unprocessed
    messages are considered.
    """
    global _chat, _last_mention_wake_at, _last_mention_info
    log("👀 Mention watcher started (poll every 1–5s jitter)")
    while not _mention_stop.is_set() and not _loop_stop.is_set():
        try:
            delay = random.uniform(1.0, 5.0)
            if _mention_stop.wait(timeout=delay):
                break
            if _loop_stop.is_set():
                break

            store = _store
            if store is None or not store.get_config("loop_enabled"):
                continue

            # Need a chat connection; retry quietly until available
            if _chat is None or not _chat.token:
                try:
                    _chat = _connect_chat(store)
                except Exception:
                    continue

            last_seen = int(store.get_state("last_seen_message_id") or 0)
            try:
                data = _chat.get_messages(after=last_seen, limit=50)
            except Exception as e:
                log(f"👀 mention poll error: {e}")
                continue

            msgs = data.get("messages") or []
            username = store.username
            mentions = [
                m for m in msgs
                if m.get("username") != username and message_tags_me(m, username)
            ]
            if not mentions:
                continue

            first = mentions[0]
            info = f"#{first.get('id')} from {first.get('username')}"
            if len(mentions) > 1:
                info += f" (+{len(mentions) - 1} more)"

            _last_mention_wake_at = datetime.datetime.now().isoformat()
            _last_mention_info = info
            store.set_state("last_wake_reason", f"mention {info}")

            # Avoid log spam if we already signalled and mind hasn't slept yet
            if not _loop_wake.is_set():
                log(f"👀 Mention detected ({info}) — waking mind")
            _loop_wake.set()
        except Exception as e:
            log("Mention watcher error:", e)
            time.sleep(5)

    log("👀 Mention watcher stopped")


def start_loop():
    global _loop_thread, _mention_thread, _loop_running
    with _loop_lock:
        if _store is None:
            raise RuntimeError("no identity")
        _store.set_config("loop_enabled", True)
        if _loop_thread and _loop_thread.is_alive():
            _loop_wake.set()
            _loop_running = True
            # Ensure mention watcher is up if loop was already running
            if not (_mention_thread and _mention_thread.is_alive()):
                _mention_stop.clear()
                _mention_thread = threading.Thread(
                    target=_mention_watcher_body, daemon=True, name="mind-mentions"
                )
                _mention_thread.start()
            return
        _loop_stop.clear()
        _mention_stop.clear()
        _loop_wake.clear()
        _loop_running = True
        _loop_thread = threading.Thread(target=_loop_body, daemon=True, name="mind-loop")
        _loop_thread.start()
        _mention_thread = threading.Thread(
            target=_mention_watcher_body, daemon=True, name="mind-mentions"
        )
        _mention_thread.start()


def stop_loop():
    global _loop_running
    with _loop_lock:
        if _store:
            _store.set_config("loop_enabled", False)
        _loop_stop.set()
        _mention_stop.set()
        _loop_wake.set()
        _loop_running = False


# ========================= FLASK ROUTES =========================
@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    loop_on = bool(
        _loop_running and _loop_thread and _loop_thread.is_alive()
        and _store and _store.get_config("loop_enabled")
    )
    base = {
        "port": _port,
        "loop_running": loop_on,
        "mention_watcher_running": bool(
            loop_on and _mention_thread and _mention_thread.is_alive()
        ),
        "last_mention_wake_at": _last_mention_wake_at,
        "last_mention_info": _last_mention_info,
        "username": _store.username if _store else None,
        "data_root": DATA_ROOT,
    }
    if _store is None:
        return jsonify(base)
    snap = _store.status_snapshot()
    snap.update(base)
    # include brave presence without exposing full key in UI needlessly
    snap.setdefault("config", {})
    if _store.get_config("brave_api_key"):
        snap["config"]["brave_api_key_set"] = True
    return jsonify(snap)


@app.get("/api/minds")
def api_minds():
    return jsonify({"usernames": list_usernames(DATA_ROOT)})


@app.post("/api/identity/create")
def api_identity_create():
    global _store, _chat
    if _store is not None:
        return jsonify({"error": "identity already loaded in this process — restart mind.py to switch"}), 400
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    if len(username) < 2 or len(username) > 32:
        return jsonify({"error": "username must be 2–32 characters"}), 400
    if not all(c.isalnum() or c in "_-" for c in username):
        return jsonify({"error": "username may only contain letters, numbers, _ and -"}), 400
    if username in list_usernames(DATA_ROOT):
        return jsonify({"error": "local mind data already exists for this username — resume instead"}), 409

    store = MindStore(DATA_ROOT, username)
    personality = generate_personality(username)
    store.set_state("personality", personality)
    store.set_state("focus", None)
    store.set_state("goals", "Get oriented in the chat room and be a helpful participant.")
    store.set_state("next_steps", "Connect to chat, introduce yourself briefly if the room is quiet, set a gentle wake schedule.")
    store.set_state("last_seen_message_id", 0)
    store.set_config("loop_enabled", False)
    # sensible defaults
    if not store.get_config("chat_server_url"):
        store.set_config("chat_server_url", "http://127.0.0.1:5000")
    _store = store
    _chat = None
    log(f"✨ Created mind identity: {username}")
    log(f"   Personality: {personality.get('summary', '')}")
    return jsonify({"username": username, "personality": personality}), 201


@app.post("/api/identity/resume")
def api_identity_resume():
    global _store, _chat
    if _store is not None:
        return jsonify({"error": "identity already loaded in this process — restart mind.py to switch"}), 400
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    if username not in list_usernames(DATA_ROOT):
        return jsonify({"error": "unknown local mind username"}), 404
    store = MindStore(DATA_ROOT, username)
    _store = store
    _chat = _build_chat_from_store(store)
    log(f"📂 Resumed mind identity: {username}")
    # Auto-start loop if it was enabled before restart
    if store.get_config("loop_enabled"):
        try:
            start_loop()
            log("🧠 Auto-resumed think loop (was enabled)")
        except Exception as e:
            log("Could not auto-start loop:", e)
    return jsonify({"username": username, "personality": store.get_state("personality")})


@app.post("/api/config")
def api_config():
    store, err = _require_store()
    if err:
        return err
    body = request.get_json(silent=True) or {}
    if "chat_server_url" in body:
        store.set_config("chat_server_url", _fix_url(body.get("chat_server_url") or ""))
    if "llm_base_url" in body:
        store.set_config("llm_base_url", _fix_url(body.get("llm_base_url") or ""))
    if "llm_model" in body:
        store.set_config("llm_model", (body.get("llm_model") or "").strip())
    if "brave_api_key" in body:
        key = (body.get("brave_api_key") or "").strip()
        store.set_config("brave_api_key", key if key else None)
    return jsonify({"ok": True, "config": store.get_all_config()})


@app.post("/api/connect")
def api_connect():
    global _chat
    store, err = _require_store()
    if err:
        return err
    try:
        _chat = _connect_chat(store)
        me = _chat.me()
        return jsonify({
            "message": f"connected as {me.get('username')}",
            "username": me.get("username"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/loop/start")
def api_loop_start():
    store, err = _require_store()
    if err:
        return err
    if not store.get_config("llm_base_url") or not store.get_config("llm_model"):
        return jsonify({"error": "configure llm_base_url and llm_model first"}), 400
    if not store.get_config("chat_server_url"):
        return jsonify({"error": "configure chat_server_url first"}), 400
    try:
        start_loop()
        return jsonify({"ok": True, "message": "loop started"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.post("/api/loop/stop")
def api_loop_stop():
    stop_loop()
    return jsonify({"ok": True, "message": "loop stopped"})


@app.post("/api/loop/tick")
def api_loop_tick():
    global _chat
    store, err = _require_store()
    if err:
        return err
    if not store.get_config("llm_base_url") or not store.get_config("llm_model"):
        return jsonify({"error": "configure llm first"}), 400
    try:
        with _tick_lock:
            if _chat is None or not _chat.token:
                _chat = _connect_chat(store)
            status = run_think_loop(store, _chat, _system_prompt)
        _loop_wake.set()
        return jsonify({"status": status})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ========================= MAIN =========================
def main():
    global _store, _chat, _system_prompt, _port

    parser = argparse.ArgumentParser(description="Marmot Mind — autonomous chat participant")
    parser.add_argument("--create", metavar="USERNAME", help="Create a new mind with this username")
    parser.add_argument("--resume", metavar="USERNAME", help="Resume an existing mind username")
    parser.add_argument("--port", type=int, default=0, help="Status UI port (0 = random free port)")
    parser.add_argument("--chat-server", default=None, help="Chat server base URL")
    parser.add_argument("--llm-url", default=None, help="LLM OpenAI-compatible base URL")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--start-loop", action="store_true", help="Start think loop immediately after identity load")
    args = parser.parse_args()

    _system_prompt = _load_system_prompt()
    _port = args.port or _find_free_port()

    if args.create and args.resume:
        print("Use only one of --create or --resume")
        sys.exit(2)

    if args.create:
        username = args.create.strip()
        if username in list_usernames(DATA_ROOT):
            print(f"Local data already exists for '{username}'. Use --resume {username}")
            sys.exit(1)
        # Reuse API logic
        with app.test_request_context(json={"username": username}):
            # manual create
            store = MindStore(DATA_ROOT, username)
            personality = generate_personality(username)
            store.set_state("personality", personality)
            store.set_state("goals", "Get oriented in the chat room and be a helpful participant.")
            store.set_state("next_steps", "Connect to chat, introduce yourself briefly if the room is quiet.")
            store.set_state("last_seen_message_id", 0)
            store.set_config("loop_enabled", False)
            store.set_config("chat_server_url", "http://127.0.0.1:5000")
            _store = store
            log(f"✨ Created mind: {username}")
            log(f"   {personality.get('summary')}")

    if args.resume:
        username = args.resume.strip()
        if username not in list_usernames(DATA_ROOT):
            print(f"No local mind data for '{username}'. Use --create {username}")
            sys.exit(1)
        _store = MindStore(DATA_ROOT, username)
        _chat = _build_chat_from_store(_store)
        log(f"📂 Resumed mind: {username}")

    if _store:
        if args.chat_server:
            _store.set_config("chat_server_url", _fix_url(args.chat_server))
        if args.llm_url:
            _store.set_config("llm_base_url", _fix_url(args.llm_url))
        if args.llm_model:
            _store.set_config("llm_model", args.llm_model.strip())
        if args.start_loop:
            try:
                start_loop()
            except Exception as e:
                log("Could not start loop:", e)
        elif _store.get_config("loop_enabled"):
            try:
                start_loop()
                log("🧠 Auto-resumed think loop")
            except Exception as e:
                log("Could not auto-start loop:", e)

    print()
    print("🧠 Marmot Mind")
    print(f"   Status UI : http://127.0.0.1:{_port}/")
    print(f"   Data dir  : {DATA_ROOT}")
    if _store:
        print(f"   Username  : {_store.username}")
    else:
        print("   Username  : (open the status UI to create or resume)")
    print()

    # Disable Flask request logging spam a bit
    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    app.run(host="0.0.0.0", port=_port, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
