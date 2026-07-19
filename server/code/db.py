#!/usr/bin/env python3
"""SQLite persistence for the chat server."""

import os
import re
import sqlite3
import threading
import secrets
import json
from contextlib import contextmanager
from datetime import datetime, timezone

# @alice, @everyone, @all — usernames are alnum / _ / -
_MENTION_RE = re.compile(r"@([A-Za-z0-9_-]+)")

_DB_PATH = None
_lock = threading.RLock()

# A user is "active" if they hit an authenticated endpoint within this window.
# Browser polls ~1.5s; mind mention watcher polls 1–5s — 30s is a comfortable grace.
ACTIVE_WITHIN_SECONDS = 30
# Don't rewrite last_seen_at on every poll; throttle DB writes.
_LAST_SEEN_WRITE_MIN_SECONDS = 5


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


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
                created_at TEXT NOT NULL,
                last_seen_at TEXT
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
        # Migrate older DBs that predate last_seen_at
        cols = {row[1] for row in conn.execute("PRAGMA table_info(users)").fetchall()}
        if "last_seen_at" not in cols:
            conn.execute("ALTER TABLE users ADD COLUMN last_seen_at TEXT")


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
    now = _utcnow()
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO users (username, token, created_at, last_seen_at) VALUES (?, ?, ?, ?)",
                (username, token, now, now),
            )
        return {"username": username, "token": token}
    except sqlite3.IntegrityError:
        return None


def login_user(username: str) -> dict | None:
    """Return existing user credentials or None. Marks them active."""
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
    touch_last_seen(row["username"], force=True)
    return {"username": row["username"], "token": row["token"]}


def user_from_token(token: str) -> dict | None:
    token = (token or "").strip()
    if not token:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, username, created_at, last_seen_at FROM users WHERE token = ?",
            (token,),
        ).fetchone()
    if not row:
        return None
    return {
        "id": row["id"],
        "username": row["username"],
        "created_at": row["created_at"],
        "last_seen_at": row["last_seen_at"],
        "active": _is_active(row["last_seen_at"]),
    }


def touch_last_seen(username: str, force: bool = False) -> None:
    """Record that this user hit the server (throttled writes unless force)."""
    username = (username or "").strip()
    if not username:
        return
    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    with _connect() as conn:
        row = conn.execute(
            "SELECT last_seen_at FROM users WHERE username = ? COLLATE NOCASE",
            (username,),
        ).fetchone()
        if not row:
            return
        if not force:
            prev = _parse_iso(row["last_seen_at"])
            if prev is not None:
                age = (now - prev).total_seconds()
                if age < _LAST_SEEN_WRITE_MIN_SECONDS:
                    return
        conn.execute(
            "UPDATE users SET last_seen_at = ? WHERE username = ? COLLATE NOCASE",
            (now_iso, username),
        )


def _is_active(last_seen_at: str | None, within_seconds: int = ACTIVE_WITHIN_SECONDS) -> bool:
    prev = _parse_iso(last_seen_at)
    if prev is None:
        return False
    age = (datetime.now(timezone.utc) - prev).total_seconds()
    return age <= within_seconds


def list_users(within_seconds: int = ACTIVE_WITHIN_SECONDS) -> list[dict]:
    """All registered users with presence info derived from last_seen_at."""
    with _connect() as conn:
        rows = conn.execute(
            """
            SELECT username, created_at, last_seen_at
            FROM users
            ORDER BY username COLLATE NOCASE
            """
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            "username": r["username"],
            "created_at": r["created_at"],
            "last_seen_at": r["last_seen_at"],
            "active": _is_active(r["last_seen_at"], within_seconds),
        })
    # Active first, then alpha within each group (already alpha overall — re-sort)
    out.sort(key=lambda u: (0 if u["active"] else 1, u["username"].lower()))
    return out


def list_users_by_presence(within_seconds: int = ACTIVE_WITHIN_SECONDS) -> dict:
    users = list_users(within_seconds)
    return {
        "users": users,
        "active": [u for u in users if u["active"]],
        "inactive": [u for u in users if not u["active"]],
        "active_within_seconds": within_seconds,
    }


def post_message(username: str, text: str, tags: list[str] | None = None) -> dict:
    text = (text or "").strip()
    if not text:
        raise ValueError("empty message")
    if len(text) > 8000:
        raise ValueError("message too long (max 8000 chars)")

    # Tags = explicit API tags ∪ @mentions parsed from the message body.
    # @mentions only count for registered usernames (+ @everyone / @all).
    known = [u["username"] for u in list_users()]
    from_text = extract_mentions_from_text(text, known)
    clean_tags = _normalize_tags(list(tags or []) + from_text, known_usernames=known)
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


def extract_mentions_from_text(text: str, known_usernames: list[str] | None = None) -> list[str]:
    """Pull @mentions from message text into tag names.

    Recognizes @everyone / @all, and @username when it matches a registered user
    (case-insensitive). Unknown @tokens are ignored so prose like @2am stays clean.
    """
    if not text:
        return []
    known_map = {u.lower(): u for u in (known_usernames or [])}
    out = []
    seen = set()
    for m in _MENTION_RE.finditer(text):
        raw = m.group(1)
        low = raw.lower()
        if low in ("everyone", "all"):
            tag = "everyone"
        elif known_map:
            if low not in known_map:
                continue
            tag = known_map[low]
        else:
            tag = raw
        if tag.lower() in seen:
            continue
        seen.add(tag.lower())
        out.append(tag)
    return out


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


def _normalize_tags(tags, known_usernames: list[str] | None = None) -> list[str]:
    if not tags:
        return []
    if isinstance(tags, str):
        tags = [tags]
    known_map = {u.lower(): u for u in (known_usernames or [])}
    out = []
    seen = set()
    for t in tags:
        t = (t or "").strip().lstrip("@")
        if not t:
            continue
        key = t.lower()
        if key in ("*", "all", "everyone"):
            t = "everyone"
            key = "everyone"
        elif known_map:
            if key not in known_map:
                # Drop unknown explicit tags when we have a user list
                continue
            t = known_map[key]
            key = t.lower()
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
