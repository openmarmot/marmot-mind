#!/usr/bin/env python3
"""Marmot — one process for the chat room and every mind.

Humans use the web UI (and the HTTP API). Minds read and post through Room,
in-process Python calls, and wake as soon as someone else posts.
"""

import argparse
import json
import os
import sys

from flask import Flask, g, jsonify, render_template, request, send_file
from werkzeug.serving import WSGIRequestHandler

import chatdb
from minds import MindManager, valid_username
from room import Room

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(BASE_DIR)
DEFAULT_DATA_DIR = os.path.join(REPO_ROOT, "data")
PROMPT_PATH = os.path.join(BASE_DIR, "prompts", "system_prompt.txt")
DEFAULT_PORT = int(os.environ.get("MARMOT_PORT", "5000"))

_room: Room | None = None
_minds: MindManager | None = None
_settings_path: str | None = None


def _load_system_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            content = f.read().strip()
        if content:
            return content
    except Exception as e:
        print("Warning: could not load system prompt:", e)
    return "You are an autonomous mind in a chat room. Use post_message to talk."


def _fix_url(u: str) -> str:
    u = (u or "").strip()
    if u and not u.startswith(("http://", "https://")):
        u = "http://" + u
    return u.rstrip("/")


def _read_settings() -> dict:
    try:
        with open(_settings_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def _write_settings(data: dict) -> None:
    os.makedirs(os.path.dirname(_settings_path), exist_ok=True)
    tmp = _settings_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, _settings_path)


def _public_max_context(value):
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _parse_max_context(value):
    """Positive token count, or None to clear. Raises ValueError on garbage."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("max_context must be a positive number of tokens")
    if isinstance(value, str):
        value = value.strip().replace(",", "").replace("_", "")
        if not value:
            return None
    if isinstance(value, float) and not value.is_integer():
        raise ValueError("max_context must be a whole number of tokens")
    try:
        n = int(value)
    except (TypeError, ValueError):
        raise ValueError("max_context must be a positive number of tokens")
    if n <= 0:
        return None
    if n > 2_000_000:
        raise ValueError("max_context must be at most 2000000")
    return n


def _public_settings(data: dict | None = None) -> dict:
    data = _read_settings() if data is None else data
    return {
        "llm_base_url": data.get("llm_base_url") or "",
        "llm_model": data.get("llm_model") or "",
        "max_context": _public_max_context(data.get("max_context")),
        "brave_api_key_set": bool(data.get("brave_api_key")),
    }


def _token_from_request() -> str | None:
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return (
        request.headers.get("X-Auth-Token")
        or request.args.get("token")
        or (request.get_json(silent=True) or {}).get("token")
    )


def _require_auth(fn):
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        user = chatdb.user_from_token(_token_from_request())
        if not user:
            return jsonify({"error": "unauthorized", "message": "valid token required"}), 401
        chatdb.touch_last_seen(user["username"])
        g.user = user
        return fn(*args, **kwargs)

    return wrapper


def _mind_or_404(username: str):
    rt = _minds.get(username) if _minds else None
    if rt is None:
        return None, (jsonify({"error": "unknown mind"}), 404)
    return rt, None


def _apply_llm_fields(setter, body: dict) -> None:
    """Persist non-blank LLM fields. A blank brave key clears it when the key is present."""
    if "llm_base_url" in body:
        url = _fix_url(body.get("llm_base_url") or "")
        if url:
            setter("llm_base_url", url)
    if "llm_model" in body:
        model = (body.get("llm_model") or "").strip()
        if model:
            setter("llm_model", model)
    if "brave_api_key" in body:
        key = (body.get("brave_api_key") or "").strip()
        setter("brave_api_key", key or None)
    if "max_context" in body:
        setter("max_context", _parse_max_context(body.get("max_context")))


def create_app(data_dir: str | None = None) -> Flask:
    global _room, _minds, _settings_path
    data_dir = data_dir or DEFAULT_DATA_DIR
    os.makedirs(data_dir, exist_ok=True)
    mind_root = os.path.join(data_dir, "minds")
    os.makedirs(mind_root, exist_ok=True)
    _settings_path = os.path.join(data_dir, "settings.json")
    chatdb.init_db(os.path.join(data_dir, "chat.db"))
    _room = Room()
    _minds = MindManager(mind_root, _room, _load_system_prompt())
    _room.add_listener(_minds.handle_message)
    _minds.load_existing()

    app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))
    app.config["JSON_SORT_KEYS"] = False

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/health")
    def health():
        presence = chatdb.list_users_by_presence()
        running = [rt for rt in _minds.all() if rt.snapshot()["loop_running"]]
        return jsonify({
            "status": "ok",
            "service": "marmot",
            "users": len(presence["users"]),
            "active_users": len(presence["active"]),
            "active_within_seconds": presence["active_within_seconds"],
            "messages": chatdb.message_count(),
            "latest_message_id": chatdb.latest_message_id(),
            "minds": len(_minds.usernames()),
            "minds_running": len(running),
        })

    @app.post("/api/join")
    def api_join():
        """One name. Create it, or continue if that person already exists."""
        body = request.get_json(silent=True) or {}
        username = (body.get("username") or "").strip()
        err = valid_username(username)
        if err:
            return jsonify({"error": err if username else "username required"}), 400
        if _minds.get(username):
            return jsonify({"error": "that name belongs to a mind"}), 409
        existing = chatdb.login_user(username)
        if existing:
            return jsonify({
                "username": existing["username"],
                "token": existing["token"],
                "message": "welcome back",
            })
        result = chatdb.create_user(username)
        if not result:
            existing = chatdb.login_user(username)
            if existing and not _minds.get(existing["username"]):
                return jsonify({
                    "username": existing["username"],
                    "token": existing["token"],
                    "message": "welcome back",
                })
            return jsonify({"error": "username already taken"}), 409
        return jsonify({
            "username": result["username"],
            "token": result["token"],
            "message": "joined",
        }), 201

    @app.post("/api/signup")
    def api_signup():
        body = request.get_json(silent=True) or {}
        username = (body.get("username") or "").strip()
        err = valid_username(username)
        if err:
            return jsonify({"error": err if username else "username required"}), 400
        if _minds.get(username):
            return jsonify({"error": "username already taken"}), 409
        result = chatdb.create_user(username)
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
        result = chatdb.login_user(username)
        if not result:
            return jsonify({"error": "unknown username — sign up first"}), 404
        return jsonify({
            "username": result["username"],
            "token": result["token"],
            "message": "logged in",
        })

    @app.get("/api/me")
    @_require_auth
    def api_me():
        return jsonify(g.user)

    @app.get("/api/users")
    @_require_auth
    def api_users():
        payload = chatdb.list_users_by_presence()
        mind_names = {n.lower() for n in _minds.usernames()}
        for user in payload["users"]:
            user["is_mind"] = user["username"].lower() in mind_names
        return jsonify(payload)

    @app.get("/api/messages")
    @_require_auth
    def api_get_messages():
        after = request.args.get("after")
        limit = request.args.get("limit", 100)
        if after is not None and after != "":
            try:
                after_id = int(after)
            except ValueError:
                return jsonify({"error": "after must be an integer"}), 400
            messages = chatdb.get_messages_after(after_id, limit)
        else:
            try:
                lim = int(limit)
            except ValueError:
                lim = 50
            messages = chatdb.get_recent_messages(lim)
        return jsonify({
            "messages": messages,
            "latest_id": chatdb.latest_message_id(),
            "generation": chatdb.message_generation(),
        })

    @app.get("/api/images/<image_id>")
    def api_image(image_id):
        found = chatdb.image_file(image_id)
        if not found:
            return jsonify({"error": "not found"}), 404
        path, mime, name = found
        return send_file(path, mimetype=mime, download_name=name, conditional=True, max_age=86400)

    @app.post("/api/messages")
    @_require_auth
    def api_post_message():
        saved = []
        try:
            ctype = (request.content_type or "").lower()
            if "multipart/form-data" in ctype:
                text = request.form.get("text") or ""
                raw_tags = request.form.get("tags")
                tags = None
                if raw_tags:
                    try:
                        tags = json.loads(raw_tags)
                    except Exception:
                        raise ValueError("tags must be a JSON array")
                files = [f for f in request.files.getlist("images") if f]
                if len(files) > chatdb.MAX_IMAGES_PER_MESSAGE:
                    raise ValueError(f"at most {chatdb.MAX_IMAGES_PER_MESSAGE} images per message")
                for upload in files:
                    saved.append(chatdb.save_chat_image(upload.read(), upload.filename or ""))
                msg = _room.post_message(g.user["username"], text, tags, images=saved)
            else:
                body = request.get_json(silent=True) or {}
                msg = _room.post_message(g.user["username"], body.get("text"), body.get("tags"))
        except ValueError as e:
            for record in saved:
                chatdb.delete_chat_image(record)
            return jsonify({"error": str(e)}), 400
        return jsonify(msg), 201

    @app.post("/api/messages/clear")
    @_require_auth
    def api_clear_messages():
        return jsonify(chatdb.clear_messages())

    @app.get("/api/settings")
    def api_settings_get():
        return jsonify(_public_settings())

    @app.post("/api/settings")
    def api_settings_post():
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "JSON object body required"}), 400
        data = _read_settings()

        def setter(key, value):
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value

        try:
            _apply_llm_fields(setter, body)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        _write_settings(data)
        return jsonify({"ok": True, "settings": _public_settings(data)})

    @app.get("/api/minds")
    def api_minds():
        return jsonify({"minds": [rt.snapshot() for rt in _minds.all()]})

    @app.post("/api/minds")
    def api_minds_create():
        body = request.get_json(silent=True) or {}
        settings = _read_settings()
        llm_url = _fix_url(body.get("llm_base_url") or "") or (settings.get("llm_base_url") or "")
        llm_model = (body.get("llm_model") or "").strip() or (settings.get("llm_model") or "")
        brave = (body.get("brave_api_key") or "").strip() or (settings.get("brave_api_key") or "")
        if "max_context" in body and body.get("max_context") not in (None, ""):
            max_context = _parse_max_context(body.get("max_context"))
        else:
            max_context = _public_max_context(settings.get("max_context"))
        try:
            rt = _minds.create(
                body.get("username") or "",
                llm_base_url=llm_url or None,
                llm_model=llm_model or None,
                brave_api_key=brave or None,
                max_context=max_context,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except FileExistsError as e:
            return jsonify({"error": str(e)}), 409
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify(rt.snapshot()), 201

    @app.delete("/api/minds/<username>")
    def api_mind_delete(username):
        try:
            name = _minds.delete(username)
        except FileNotFoundError:
            return jsonify({"error": "unknown mind"}), 404
        return jsonify({"ok": True, "username": name})

    @app.get("/api/minds/<username>")
    def api_mind_get(username):
        rt, err = _mind_or_404(username)
        if err:
            return err
        return jsonify(rt.snapshot())

    @app.post("/api/minds/<username>/config")
    def api_mind_config(username):
        rt, err = _mind_or_404(username)
        if err:
            return err
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "JSON object body required"}), 400
        try:
            _apply_llm_fields(rt.store.set_config, body)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, "mind": rt.snapshot()})

    @app.post("/api/minds/<username>/start")
    def api_mind_start(username):
        rt, err = _mind_or_404(username)
        if err:
            return err
        try:
            rt.start()
        except Exception as e:
            return jsonify({"error": str(e)}), 400
        return jsonify({"ok": True, "mind": rt.snapshot()})

    @app.post("/api/minds/<username>/stop")
    def api_mind_stop(username):
        rt, err = _mind_or_404(username)
        if err:
            return err
        rt.stop()
        return jsonify({"ok": True, "mind": rt.snapshot()})

    @app.post("/api/minds/<username>/tick")
    def api_mind_tick(username):
        rt, err = _mind_or_404(username)
        if err:
            return err
        try:
            status = rt.tick_once()
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
        return jsonify({"status": status, "mind": rt.snapshot()})

    return app


class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        path = self.path.split("?", 1)[0]
        if path.startswith("/api/images/"):
            return
        if path in ("/api/messages", "/api/users", "/api/minds", "/health") and str(code).startswith("2"):
            return
        super().log_request(code, size)


def main():
    parser = argparse.ArgumentParser(description="Marmot — chat room and minds in one process")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--create", metavar="USERNAME", help="Create a mind, then serve")
    parser.add_argument("--llm-url", default=None, help="Default LLM base URL (also applied to --create)")
    parser.add_argument("--llm-model", default=None, help="Default LLM model (also applied to --create)")
    parser.add_argument("--max-context", type=int, default=None, help="Default context window in tokens")
    parser.add_argument("--start-loop", action="store_true", help="Start the --create mind's think loop")
    args = parser.parse_args()

    if args.start_loop and not args.create:
        print("--start-loop requires --create USERNAME")
        sys.exit(2)

    app = create_app(args.data_dir)

    if args.llm_url or args.llm_model or args.max_context:
        data = _read_settings()

        def setter(key, value):
            if value is None:
                data.pop(key, None)
            else:
                data[key] = value

        fields = {}
        if args.llm_url:
            fields["llm_base_url"] = args.llm_url
        if args.llm_model:
            fields["llm_model"] = args.llm_model
        if args.max_context:
            fields["max_context"] = args.max_context
        _apply_llm_fields(setter, fields)
        _write_settings(data)

    if args.create:
        settings = _read_settings()
        try:
            rt = _minds.create(
                args.create,
                llm_base_url=settings.get("llm_base_url") or None,
                llm_model=settings.get("llm_model") or None,
                brave_api_key=settings.get("brave_api_key") or None,
                max_context=_public_max_context(settings.get("max_context")),
            )
        except (ValueError, FileExistsError) as e:
            print(e)
            sys.exit(1)
        if args.start_loop:
            try:
                rt.start()
            except Exception as e:
                print(f"Could not start loop: {e}")
                sys.exit(1)

    import logging
    logging.getLogger("werkzeug").setLevel(logging.WARNING)

    print()
    print("🐹 Marmot")
    print(f"   Web UI : http://127.0.0.1:{args.port}/")
    print(f"   Data   : {args.data_dir}")
    print(f"   Minds  : {', '.join(_minds.usernames()) or '(none yet)'}")
    print()
    app.run(host="0.0.0.0", port=args.port, threaded=True, request_handler=QuietRequestHandler, use_reloader=False)


if __name__ == "__main__":
    main()
