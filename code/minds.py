"""Several minds in one process. Each mind has its own store, think loop, and wake event.

Chat access is a Room method call. Any post from someone else sets a running
mind's wake event immediately so it can read. A post that arrives during a tick
is picked up when that tick finishes and re-reads the room.
"""

import os
import secrets
import shutil
import threading
import time

import chatdb
from agent import log, message_tags_me, reset_log_name, run_think_loop, set_log_name
from clock import seconds_until, utcnow_iso
from storage import MindStore, list_usernames

_MAX_SLEEP_SECONDS = 24 * 3600
_DEFAULT_SLEEP_SECONDS = 300.0


def _seconds_until_wake(store: MindStore) -> float:
    """Seconds until next_wake_after. Overdue → 0. Missing → 5 min. Cap 24h."""
    nw = store.get_state("next_wake_after")
    if not nw:
        return _DEFAULT_SLEEP_SECONDS
    delta = seconds_until(nw)
    if delta is None:
        return _DEFAULT_SLEEP_SECONDS
    if delta <= 0:
        return 0.0
    return min(float(_MAX_SLEEP_SECONDS), delta)


def valid_username(username: str) -> str | None:
    username = (username or "").strip()
    if len(username) < 2 or len(username) > 32:
        return "username must be 2–32 characters"
    if not all(c.isalnum() or c in "_-" for c in username):
        return "username may only contain letters, numbers, _ and -"
    return None


class MindRuntime:
    def __init__(self, store: MindStore, room, system_prompt: str):
        self.store = store
        self.room = room
        self.system_prompt = system_prompt
        self.loop_stop = threading.Event()
        self.loop_wake = threading.Event()
        self.tick_lock = threading.Lock()
        self.loop_lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.loop_thread: threading.Thread | None = None
        self.loop_running = False
        # Message ids already attempted when a tick failed to advance last_seen.
        self.suppress_until_id = 0

    @property
    def username(self) -> str:
        return self.store.username

    def llm_ready(self) -> bool:
        return bool(self.store.get_config("llm_base_url") and self.store.get_config("llm_model"))

    def _describe_wake(self, msg: dict) -> str:
        mid = int(msg.get("id") or 0)
        kind = "mention" if message_tags_me(msg, self.username) else "message"
        return f"{kind} #{mid} from {msg.get('username')}"

    def on_message(self, msg: dict) -> None:
        """Wake so this mind can read a post from someone else.

        Safe to call from the posting thread. The mind's own posts do not wake it.
        A post that arrives while a tick is running is left for that tick's
        unread check. A direct tag or @everyone is labeled a mention in the log.
        """
        if not self.store.get_config("loop_enabled"):
            return
        if msg.get("username") == self.username:
            return
        mid = int(msg.get("id") or 0)
        if mid <= int(self.store.get_state("last_seen_message_id") or 0):
            return
        if mid <= self.suppress_until_id:
            return
        if self.tick_lock.locked():
            return
        info = self._describe_wake(msg)
        with self.state_lock:
            if self.tick_lock.locked():
                return
            first = not self.loop_wake.is_set()
            self.store.set_state("wake_reason", info)
            self.loop_wake.set()
        if first:
            token = set_log_name(self.username)
            try:
                log(f"👀 {info} — waking")
            finally:
                reset_log_name(token)

    def _unseen_messages(self) -> str | None:
        """Oldest unread post from someone else, or None if the room is caught up."""
        last_seen = int(self.store.get_state("last_seen_message_id") or 0)
        floor = max(last_seen, self.suppress_until_id)
        data = self.room.get_messages(after=floor, limit=50, as_user=self.username)
        hits = [
            m for m in (data.get("messages") or [])
            if m.get("username") != self.username
        ]
        if not hits:
            return None
        info = self._describe_wake(hits[0])
        if len(hits) > 1:
            info += f" (+{len(hits) - 1} more)"
        return info

    def run_tick(self, reason: str | None = None) -> tuple[str, bool]:
        """One think cycle. Returns (status, should_think_again)."""
        if reason:
            self.store.set_state("wake_reason", reason)
        with self.tick_lock:
            status = run_think_loop(self.store, self.room, self.system_prompt)
        with self.state_lock:
            self.loop_wake.clear()
        failed = "llm_failed" in (status or "")
        if not failed:
            self.suppress_until_id = 0
        unseen = self._unseen_messages()
        if unseen and not failed:
            self.store.set_state("wake_reason", unseen)
            return status, True
        if unseen and failed:
            # The model call failed and that post is still waiting.
            # Hold it so the watcher does not spin, then retry shortly.
            self.suppress_until_id = max(self.suppress_until_id, self.room.latest_id())
            return status, False
        self.suppress_until_id = 0
        return status, False

    def _loop_body(self):
        token = set_log_name(self.username)
        try:
            self._loop_body_inner()
        finally:
            reset_log_name(token)

    def _loop_body_inner(self):
        log("🧠 Think loop started")
        if self.loop_stop.wait(2):
            self.loop_running = False
            log("🧠 Think loop stopped")
            return
        if not self.store.get_state("wake_reason"):
            self.store.set_state("wake_reason", "loop start")
        while not self.loop_stop.is_set():
            try:
                if not self.store.get_config("loop_enabled"):
                    self.loop_wake.wait(timeout=5)
                    self.loop_wake.clear()
                    continue
                try:
                    status, again = self.run_tick()
                    log(f"🧠 tick: {status}")
                except Exception as e:
                    log("Think loop error:", e)
                    self.store.set_state("last_loop_status", f"error: {e}")
                    self.store.set_state("last_loop_at", utcnow_iso())
                    again = False
                if again and not self.loop_stop.is_set():
                    continue
                # A failed tick leaves posts unread. Hold them briefly so we
                # don't tight-loop, then try again. Newer posts still wake early.
                last_seen_now = int(self.store.get_state("last_seen_message_id") or 0)
                if self.suppress_until_id > last_seen_now and not self.loop_stop.is_set():
                    log("🧠 unread messages after a failed tick — retrying in 30s")
                    self.store.set_state("next_wake_reason", "retry after failed tick")
                    self.loop_wake.wait(timeout=30)
                    self.loop_wake.clear()
                    self.suppress_until_id = 0
                    if not self.loop_stop.is_set():
                        self.store.set_state("wake_reason", "retry unread messages")
                    continue
                sleep_secs = _seconds_until_wake(self.store)
                log(f"🧠 sleeping ~{int(sleep_secs)}s (new messages wake immediately)")
                woke = self.loop_wake.wait(timeout=sleep_secs)
                self.loop_wake.clear()
                if self.loop_stop.is_set():
                    break
                if woke:
                    reason = (self.store.get_state("wake_reason") or "").strip()
                    if reason:
                        log(f"🧠 woken early by {reason}")
                    else:
                        log("🧠 woken early")
                        self.store.set_state("wake_reason", "interrupted")
                else:
                    nxt = (self.store.get_state("next_wake_reason") or "").strip()
                    self.store.set_state(
                        "wake_reason",
                        f"scheduled: {nxt}" if nxt else "scheduled",
                    )
            except Exception as e:
                log("Loop outer error:", e)
                time.sleep(30)
        self.loop_running = False
        log("🧠 Think loop stopped")

    def start(self):
        with self.loop_lock:
            if not self.llm_ready():
                raise RuntimeError("configure llm_base_url and llm_model first")
            self.room.ensure_user(self.username)
            self.store.set_config("loop_enabled", True)
            self.loop_running = True
            if self.loop_thread and self.loop_thread.is_alive():
                self.store.set_state("wake_reason", "loop start")
                self.loop_wake.set()
                return
            self.loop_stop.clear()
            self.loop_wake.clear()
            self.suppress_until_id = 0
            self.loop_thread = threading.Thread(
                target=self._loop_body, daemon=True, name=f"mind-{self.username}"
            )
            self.loop_thread.start()

    def stop(self):
        with self.loop_lock:
            self.store.set_config("loop_enabled", False)
            self.loop_stop.set()
            self.loop_wake.set()
            self.loop_running = False

    def tick_once(self) -> str:
        if not self.llm_ready():
            raise RuntimeError("configure llm_base_url and llm_model first")
        self.room.ensure_user(self.username)
        status, again = self.run_tick("manual tick")
        if again and self.loop_running:
            self.loop_wake.set()
        return status

    def snapshot(self) -> dict:
        loop_on = bool(
            self.loop_running
            and self.loop_thread
            and self.loop_thread.is_alive()
            and self.store.get_config("loop_enabled")
        )
        snap = self.store.status_snapshot()
        snap.update({
            "loop_running": loop_on,
        })
        return snap


class MindManager:
    def __init__(self, mind_root: str, room, system_prompt: str):
        self.mind_root = mind_root
        self.room = room
        self.system_prompt = system_prompt
        self._minds: dict[str, MindRuntime] = {}
        self._lock = threading.Lock()

    def handle_message(self, msg: dict) -> None:
        for rt in self.all():
            try:
                rt.on_message(msg)
            except Exception as e:
                log(f"message dispatch error for {rt.username}: {e}")

    def all(self) -> list[MindRuntime]:
        with self._lock:
            return list(self._minds.values())

    def usernames(self) -> list[str]:
        return [rt.username for rt in self.all()]

    def get(self, username: str) -> MindRuntime | None:
        key = (username or "").strip().lower()
        with self._lock:
            for name, rt in self._minds.items():
                if name.lower() == key:
                    return rt
        return None

    def load_existing(self) -> None:
        if os.path.isdir(self.mind_root):
            for name in os.listdir(self.mind_root):
                if ".deleting-" not in name:
                    continue
                shutil.rmtree(os.path.join(self.mind_root, name), ignore_errors=True)
        for name in list_usernames(self.mind_root):
            if self.get(name):
                continue
            store = MindStore(self.mind_root, name)
            rt = MindRuntime(store, self.room, self.system_prompt)
            with self._lock:
                self._minds[store.username] = rt
            try:
                self.room.ensure_user(store.username)
            except Exception as e:
                log(f"Could not register chat user {store.username}: {e}")
            if store.get_config("loop_enabled") and rt.llm_ready():
                try:
                    rt.start()
                    log(f"🧠 Auto-resumed {store.username}")
                except Exception as e:
                    log(f"Could not auto-start {store.username}: {e}")

    def create(
        self,
        username: str,
        *,
        llm_base_url: str | None = None,
        llm_model: str | None = None,
        brave_api_key: str | None = None,
        max_context: int | None = None,
    ) -> MindRuntime:
        username = (username or "").strip()
        err = valid_username(username)
        if err:
            raise ValueError(err)
        with self._lock:
            known = {n.lower() for n in list_usernames(self.mind_root)}
            if username.lower() in known or any(n.lower() == username.lower() for n in self._minds):
                raise FileExistsError("a mind with that username already exists")
            if self.room.user_exists(username):
                raise FileExistsError("that username is already in the chat")
            store = MindStore(self.mind_root, username)
            store.set_state("last_seen_message_id", 0)
            store.set_config("loop_enabled", False)
            if llm_base_url:
                store.set_config("llm_base_url", llm_base_url)
            if llm_model:
                store.set_config("llm_model", llm_model)
            if brave_api_key:
                store.set_config("brave_api_key", brave_api_key)
            if max_context:
                store.set_config("max_context", int(max_context))
            self.room.ensure_user(username)
            rt = MindRuntime(store, self.room, self.system_prompt)
            self._minds[store.username] = rt
            log(f"✨ Created mind {username}")
            return rt

    def delete(self, username: str) -> str:
        """Stop the mind, free its chat name, and remove its files.

        An in-flight tick is allowed to finish. The directory is renamed
        immediately so a restart will not bring the mind back.
        """
        rt = self.get(username)
        if rt is None:
            raise FileNotFoundError(username)
        name = rt.username
        rt.stop()
        with self._lock:
            self._minds.pop(name, None)
        try:
            chatdb.delete_user(name)
        except Exception as e:
            log(f"Could not remove chat user {name}: {e}")
        src = rt.store.dir
        trash = src + ".deleting-" + secrets.token_hex(4)
        try:
            os.rename(src, trash)
        except OSError as e:
            log(f"Could not move {name} aside: {e}")
            trash = src
        thread = rt.loop_thread

        def _finish():
            if thread and thread.is_alive():
                thread.join(timeout=360)
            shutil.rmtree(trash, ignore_errors=True)
            log(f"🗑 Deleted mind {name}")

        threading.Thread(target=_finish, daemon=True, name=f"delete-{name}").start()
        return name
