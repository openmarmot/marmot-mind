#!/usr/bin/env python3
"""
Marmot Chat Server

Single-room chat app (Discord/Slack style, kept simple).
Users (humans and minds) sign up with a username, post tagged messages,
and poll for new messages by sequential id.

Persistence: SQLite (survives restarts).
"""

import os
import functools
from flask import Flask, request, jsonify, render_template, g
from werkzeug.serving import WSGIRequestHandler

import db

# ========================= PATHS / CONFIG =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASE_DIR), "data")
DB_PATH = os.path.join(DATA_DIR, "chat.db")
DEFAULT_PORT = int(os.environ.get("MARMOT_PORT", "5000"))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
app.config["JSON_SORT_KEYS"] = False

db.init_db(DB_PATH)


# ========================= AUTH HELPERS =========================
def _token_from_request() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("X-Auth-Token")
        or request.args.get("token")
        or (request.get_json(silent=True) or {}).get("token")
    )


def require_auth(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        user = db.user_from_token(_token_from_request())
        if not user:
            return jsonify({"error": "unauthorized", "message": "valid token required"}), 401
        # Presence: any authenticated hit marks the user active (writes throttled in db).
        db.touch_last_seen(user["username"])
        g.user = user
        return fn(*args, **kwargs)
    return wrapper


# ========================= WEB UI =========================
@app.get("/")
def index():
    return render_template("index.html")


# ========================= API =========================
@app.get("/health")
def health():
    presence = db.list_users_by_presence()
    return jsonify({
        "status": "ok",
        "service": "marmot-chat",
        "users": len(presence["users"]),
        "active_users": len(presence["active"]),
        "active_within_seconds": presence["active_within_seconds"],
        "messages": db.message_count(),
        "latest_message_id": db.latest_message_id(),
    })


@app.post("/api/signup")
def api_signup():
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required"}), 400
    if len(username) < 2:
        return jsonify({"error": "username must be at least 2 characters"}), 400
    if len(username) > 32:
        return jsonify({"error": "username must be at most 32 characters"}), 400
    if not all(c.isalnum() or c in "_-" for c in username):
        return jsonify({"error": "username may only contain letters, numbers, _ and -"}), 400

    result = db.create_user(username)
    if not result:
        return jsonify({"error": "username already taken"}), 409
    return jsonify({
        "username": result["username"],
        "token": result["token"],
        "message": "signed up successfully",
    }), 201


@app.post("/api/login")
def api_login():
    body = request.get_json(silent=True) or {}
    username = (body.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required"}), 400
    result = db.login_user(username)
    if not result:
        return jsonify({"error": "unknown username — sign up first"}), 404
    return jsonify({
        "username": result["username"],
        "token": result["token"],
        "message": "logged in",
    })


@app.get("/api/me")
@require_auth
def api_me():
    return jsonify(g.user)


@app.get("/api/users")
@require_auth
def api_users():
    """Registered users with presence.

    Presence is derived from last authenticated request (polls, posts, /me, …).
    Active = last_seen within active_within_seconds (default 30).
    """
    return jsonify(db.list_users_by_presence())


@app.get("/api/messages")
@require_auth
def api_get_messages():
    """Retrieve messages.

    Query params:
      after — return messages with id > after (incremental poll)
      limit — max messages (default 100, max 500)
      recent — if set (and after not used as incremental), return last N messages
    """
    after = request.args.get("after")
    limit = request.args.get("limit", 100)

    if after is not None and after != "":
        try:
            after_id = int(after)
        except ValueError:
            return jsonify({"error": "after must be an integer"}), 400
        messages = db.get_messages_after(after_id, limit)
    else:
        # Initial load: recent history
        try:
            lim = int(limit)
        except ValueError:
            lim = 50
        messages = db.get_recent_messages(lim)

    return jsonify({
        "messages": messages,
        "latest_id": db.latest_message_id(),
    })


@app.post("/api/messages")
@require_auth
def api_post_message():
    body = request.get_json(silent=True) or {}
    text = body.get("text")
    tags = body.get("tags")
    try:
        msg = db.post_message(g.user["username"], text, tags)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(msg), 201


# ========================= QUIET LOGGING FOR POLLS =========================
class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        path = self.path.split("?", 1)[0]
        # Suppress high-frequency poll noise
        if path in ("/api/messages", "/health") and str(code).startswith("2"):
            return
        super().log_request(code, size)


def main():
    port = DEFAULT_PORT
    print(f"🐹 Marmot Chat Server")
    print(f"   Web UI : http://127.0.0.1:{port}/")
    print(f"   DB     : {DB_PATH}")
    print(f"   API    : /api/signup, /api/login, /api/messages, /api/users")
    app.run(host="0.0.0.0", port=port, threaded=True, request_handler=QuietRequestHandler)


if __name__ == "__main__":
    main()
