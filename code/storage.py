#!/usr/bin/env python3
"""Per-username SQLite storage for a mind.

All state for one mind lives under data/minds/{username}/mind.db.
Each store has its own lock so minds do not block each other.
"""

import os
import re
import json
import sqlite3
import threading
from contextlib import contextmanager
from clock import utcnow_iso


class MindStore:
    def __init__(self, data_root: str, username: str):
        self.username = username
        self.dir = os.path.join(data_root, username)
        os.makedirs(self.dir, exist_ok=True)
        self.db_path = os.path.join(self.dir, "mind.db")
        self.tool_calls_dir = os.path.join(self.dir, "tool-calls")
        os.makedirs(self.tool_calls_dir, exist_ok=True)
        self.memory_dir = os.path.join(self.dir, "memory")
        self.topics_dir = os.path.join(self.memory_dir, "topics")
        os.makedirs(self.topics_dir, exist_ok=True)
        self._lock = threading.RLock()
        self._init_schema()

    @contextmanager
    def _connect(self):
        with self._lock:
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
                (utcnow_iso(), note),
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

    # ----- durable memory (markdown topics) -----
    def write_topic(self, title: str, body: str) -> dict | None:
        title = (title or "").strip()
        body = (body or "").strip()
        slug = _topic_slug(title)
        if not slug or not title:
            return None
        low = body.lower()
        if not body or low in ("none", "n/a", "nothing significant"):
            return None
        text = _topic_file_text(title, body)
        path = os.path.join(self.topics_dir, slug + ".md")
        with self._lock:
            os.makedirs(self.topics_dir, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text if text.endswith("\n") else text + "\n")
            self._write_memory_index()
        return {"title": title, "slug": slug}

    def read_topic(self, title: str) -> dict | None:
        slug = _topic_slug(title)
        if not slug:
            return None
        with self._lock:
            hit = self._topic_from_slug(slug)
            if hit:
                return hit
            want = (title or "").strip().lower()
            for item in self._list_topics_unlocked():
                if item["title"].lower() == want:
                    return item
        return None

    def delete_topic(self, title: str) -> dict | None:
        found = self.read_topic(title)
        if not found:
            return None
        path = os.path.join(self.topics_dir, found["slug"] + ".md")
        with self._lock:
            if os.path.isfile(path):
                os.remove(path)
            self._write_memory_index()
        return {"title": found["title"], "slug": found["slug"]}

    def list_topics(self) -> list[dict]:
        with self._lock:
            return self._list_topics_unlocked()

    def get_memory_index(self) -> str:
        topics = self.list_topics()
        if not topics:
            return ""
        lines = ["Topics:"]
        for t in topics:
            summary = t["summary"]
            line = f"- **{t['title']}**"
            if summary:
                line += f" — {summary}"
            lines.append(line)
        lines.append("Call read_memory with a title to open a page. Same title on remember replaces that page.")
        return "\n".join(lines)

    def _topic_from_slug(self, slug: str) -> dict | None:
        path = os.path.join(self.topics_dir, slug + ".md")
        if not os.path.isfile(path):
            return None
        with open(path, encoding="utf-8") as f:
            text = f.read()
        title, summary = _parse_topic(text, slug)
        return {"title": title, "slug": slug, "summary": summary, "body": text}

    def _list_topics_unlocked(self) -> list[dict]:
        if not os.path.isdir(self.topics_dir):
            return []
        out = []
        for name in os.listdir(self.topics_dir):
            if not name.endswith(".md") or name.startswith("."):
                continue
            slug = name[:-3]
            item = self._topic_from_slug(slug)
            if item:
                out.append(item)
        out.sort(key=lambda t: t["title"].lower())
        return out

    def _write_memory_index(self) -> None:
        os.makedirs(self.memory_dir, exist_ok=True)
        topics = self._list_topics_unlocked()
        lines = [
            "# Memory",
            "",
            "> Generated. Use remember / forget; do not edit this file.",
            "",
        ]
        if topics:
            lines.append("## Topics")
            lines.append("")
            for t in topics:
                summary = t["summary"]
                line = f"- **{t['title']}**"
                if summary:
                    line += f" — {summary}"
                line += f" (`topics/{t['slug']}.md`)"
                lines.append(line)
            lines.append("")
        else:
            lines.append("No topics yet.")
            lines.append("")
        path = os.path.join(self.memory_dir, "MEMORY.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    # ----- convenience -----
    def status_snapshot(self) -> dict:
        return {
            "username": self.username,
            "config": {
                "llm_base_url": self.get_config("llm_base_url") or "",
                "llm_model": self.get_config("llm_model") or "",
                "max_context": self.get_config("max_context") or None,
                "loop_enabled": bool(self.get_config("loop_enabled")),
                "brave_api_key_set": bool(self.get_config("brave_api_key")),
            },
            "personality": self.get_state("personality"),
            "focus": self.get_state("focus"),
            "goals": self.get_state("goals"),
            "next_steps": self.get_state("next_steps"),
            "next_wake_after": self.get_state("next_wake_after"),
            "wake_reason": self.get_state("wake_reason"),
            "next_wake_reason": self.get_state("next_wake_reason"),
            "last_loop_at": self.get_state("last_loop_at"),
            "last_loop_status": self.get_state("last_loop_status"),
            "last_seen_message_id": self.get_state("last_seen_message_id") or 0,
            "recent_observations": self.recent_observations(8),
            "memory_topics": [
                {"title": t["title"], "summary": t["summary"], "body": t["body"]}
                for t in self.list_topics()
            ],
        }


def _topic_slug(title: str) -> str:
    s = (title or "").strip().lower()
    s = re.sub(r"[^\w]+", "-", s, flags=re.UNICODE).strip("-")
    s = s.replace("_", "-")
    return s[:80]


def _topic_file_text(title: str, body: str) -> str:
    body = (body or "").strip()
    heading = f"# {title}"
    if body.startswith("# "):
        first = body.split("\n", 1)[0][2:].strip()
        if first.lower() == title.lower():
            return body if body.endswith("\n") else body + "\n"
    if body:
        return f"{heading}\n\n{body}\n"
    return heading + "\n"


def _parse_topic(text: str, slug: str) -> tuple[str, str]:
    lines = (text or "").splitlines()
    title = slug.replace("-", " ")
    i = 0
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip() or title
        i = 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    summary = ""
    if i < len(lines):
        summary = lines[i].strip()
        if summary.startswith("#"):
            summary = ""
    if len(summary) > 160:
        summary = summary[:157].rstrip() + "…"
    return title, summary


def list_usernames(data_root: str) -> list[str]:
    if not os.path.isdir(data_root):
        return []
    names = []
    for name in sorted(os.listdir(data_root)):
        path = os.path.join(data_root, name)
        if ".deleting-" in name:
            continue
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, "mind.db")):
            names.append(name)
    return names
