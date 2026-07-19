#!/usr/bin/env python3
"""Per-username SQLite storage for a Mind instance.

All state for a mind lives under data/{username}/mind.db so multiple concurrent
mind processes can run without sharing files.
"""

import os
import json
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone

_lock = threading.RLock()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class MindStore:
    def __init__(self, data_root: str, username: str):
        self.username = username
        self.dir = os.path.join(data_root, username)
        os.makedirs(self.dir, exist_ok=True)
        self.db_path = os.path.join(self.dir, "mind.db")
        self.tool_calls_dir = os.path.join(self.dir, "tool-calls")
        os.makedirs(self.tool_calls_dir, exist_ok=True)
        self._init_schema()

    @contextmanager
    def _connect(self):
        with _lock:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS mind_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    note TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    note TEXT NOT NULL
                );
                """
            )

    # ----- config -----
    def get_config(self, key: str, default=None):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM config WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return row["value"]

    def set_config(self, key: str, value):
        raw = json.dumps(value)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO config (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, raw),
            )

    def get_all_config(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM config").fetchall()
        out = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                out[r["key"]] = r["value"]
        return out

    # ----- mind state (focus, next steps, goals, etc.) -----
    def get_state(self, key: str, default=None):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM mind_state WHERE key = ?", (key,)
            ).fetchone()
        if not row:
            return default
        try:
            return json.loads(row["value"])
        except Exception:
            return row["value"]

    def set_state(self, key: str, value):
        raw = json.dumps(value)
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO mind_state (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, raw),
            )

    def get_all_state(self) -> dict:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM mind_state").fetchall()
        out = {}
        for r in rows:
            try:
                out[r["key"]] = json.loads(r["value"])
            except Exception:
                out[r["key"]] = r["value"]
        return out

    # ----- observations -----
    def add_observation(self, note: str, max_keep: int = 50):
        note = (note or "").strip()
        if not note:
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO observations (ts, note) VALUES (?, ?)",
                (_utcnow(), note),
            )
            # prune old
            conn.execute(
                """
                DELETE FROM observations WHERE id NOT IN (
                    SELECT id FROM observations ORDER BY id DESC LIMIT ?
                )
                """,
                (max_keep,),
            )

    def recent_observations(self, limit: int = 15) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, note FROM observations ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"ts": r["ts"], "note": r["note"]} for r in reversed(rows)]

    # ----- durable memory -----
    def append_memory(self, note: str, max_keep: int = 100):
        note = (note or "").strip()
        if not note:
            return
        low = note.lower()
        if "nothing significant" in low or low in ("none", "n/a"):
            return
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory (ts, note) VALUES (?, ?)",
                (_utcnow(), note),
            )
            conn.execute(
                """
                DELETE FROM memory WHERE id NOT IN (
                    SELECT id FROM memory ORDER BY id DESC LIMIT ?
                )
                """,
                (max_keep,),
            )

    def get_memory_text(self, limit: int = 40) -> str:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, note FROM memory ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        if not rows:
            return ""
        lines = []
        for r in reversed(rows):
            day = (r["ts"] or "")[:10]
            lines.append(f"[{day}] {r['note']}")
        return "\n".join(lines)

    # ----- convenience -----
    def status_snapshot(self) -> dict:
        return {
            "username": self.username,
            "config": {
                k: self.get_config(k)
                for k in ("chat_server_url", "llm_base_url", "llm_model", "loop_enabled")
            },
            "has_token": bool(self.get_config("chat_token")),
            "personality": self.get_state("personality"),
            "focus": self.get_state("focus"),
            "goals": self.get_state("goals"),
            "next_steps": self.get_state("next_steps"),
            "next_wake_after": self.get_state("next_wake_after"),
            "last_wake_reason": self.get_state("last_wake_reason"),
            "last_loop_at": self.get_state("last_loop_at"),
            "last_loop_status": self.get_state("last_loop_status"),
            "last_seen_message_id": self.get_state("last_seen_message_id") or 0,
            "recent_observations": self.recent_observations(8),
            "memory_preview": self.get_memory_text(8),
        }


def list_usernames(data_root: str) -> list[str]:
    if not os.path.isdir(data_root):
        return []
    names = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "mind.db")):
            names.append(name)
    return names
