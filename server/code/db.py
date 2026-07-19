#!/usr/bin/env python3
"""SQLite persistence for the chat server."""

import os
import sqlite3
import threading
import secrets
import json
from contextlib import contextmanager
from datetime import datetime, timezone

_DB_PATH = None
_lock = threading.RLock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(path: str):
    global _DB_PATH
    _DB_PATH = path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with _connect() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                token TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                text TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_messages_id ON messages(id);
            """
        )


@contextmanager
def _connect():
    if not _DB_PATH:
        raise RuntimeError("Database not initialized")
    with _lock:
        conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()


def create_user(username: str) -> dict | None:
    """Create a user. Returns {username, token} or None if taken."""
    username = (username or "").strip()
    if not username:
        return None
    if len(username) > 32:
        return None
    if not all(c.isalnum() or c in "_-" for c in username):
        return None
    token = secrets.token_hex(24)
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, token, created_at) VALUES (?, ?, ?)",
                (username, token, _utcnow()),
            )
        return {"username": username, "token": token}
    except sqlite3.IntegrityError:
        return None


def login_user(username: str) -> dict | None:
    """Return existing user credentials or None."""
    username = (username or "").strip()
    if not username:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT username, token FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
    if not row:
        return None
    return {"username": row["username"], "token": row["token"]}


def user_from_token(token: str) -> dict | None:
    token = (token or "").strip()
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, created_at FROM users WHERE token = ?",
            (token,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
    }


def list_users() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT username, created_at FROM users ORDER BY username COLLATE NOCASE"
        ).fetchall()
    return [{"username": r["username"], "created_at": r["created_at"]} for r in rows]


def post_message(username: str, text: str, tags: list[str] | None = None) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty message")
    if len(text) > 8000:
        raise ValueError("message too long (max 8000 chars)")

    clean_tags = _normalize_tags(tags)
    tags_json = json.dumps(clean_tags)
    created = _utcnow()
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO messages (username, text, tags, created_at) VALUES (?, ?, ?, ?)",
            (username, text, tags_json, created),
        )
        msg_id = cur.lastrowid
    return {
        "id": msg_id,
        "username": username,
        "text": text,
        "tags": clean_tags,
        "created_at": created,
    }


def get_messages_after(after_id: int = 0, limit: int = 100) -> list[dict]:
    after_id = max(0, int(after_id or 0))
    limit = max(1, min(int(limit or 100), 500))
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, username, text, tags, created_at
            FROM messages
            WHERE id > ?
            ORDER BY id ASC
            LIMIT ?
            """,
            (after_id, limit),
        ).fetchall()
    return [_row_to_message(r) for r in rows]


def get_recent_messages(limit: int = 50) -> list[dict]:
    limit = max(1, min(int(limit or 50), 500))
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT id, username, text, tags, created_at
            FROM messages
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    msgs = [_row_to_message(r) for r in rows]
    msgs.reverse()
    return msgs


def latest_message_id() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT MAX(id) AS m FROM messages").fetchone()
    return int(row["m"] or 0)


def message_count() -> int:
    with _connect() as conn:
        row = conn.execute("SELECT COUNT(*) AS c FROM messages").fetchone()
    return int(row["c"] or 0)


def _normalize_tags(tags) -> list[str]:
    if not tags:
        return []
    if isinstance(tags, str):
        tags = [tags]
    out = []
    seen = set()
    for t in tags:
        t = (t or "").strip()
        if not t:
            continue
        key = t.lower()
        if key in ("*", "all", "everyone", "@everyone"):
            t = "everyone"
            key = "everyone"
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
        if len(out) >= 20:
            break
    return out


def _row_to_message(row) -> dict:
    try:
        tags = json.loads(row["tags"] or "[]")
    except Exception:
        tags = []
    if not isinstance(tags, list):
        tags = []
    return {
        "id": row["id"],
        "username": row["username"],
        "text": row["text"],
        "tags": tags,
        "created_at": row["created_at"],
    }
