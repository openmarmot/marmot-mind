"""In-process chat room. Minds call this directly; HTTP routes use it too."""

import chatdb


class Room:
    def __init__(self):
        self._listeners = []

    def add_listener(self, fn):
        """fn(message_dict) is called after every successful post."""
        self._listeners.append(fn)

    def user_exists(self, username: str) -> bool:
        return chatdb.find_user(username) is not None

    def ensure_user(self, username: str) -> dict:
        """Return the chat user, creating them if needed. Does not mark presence."""
        found = chatdb.find_user(username)
        if found:
            return found
        created = chatdb.create_user(username)
        if not created:
            raise RuntimeError(f"could not register chat user {username}")
        return created

    def touch(self, username: str, force: bool = False) -> None:
        chatdb.touch_last_seen(username, force=force)

    def list_users(self, *, as_user: str | None = None) -> list[dict]:
        if as_user:
            chatdb.touch_last_seen(as_user)
        return chatdb.list_users()

    def get_messages(self, *, after: int | None = None, limit: int = 100, as_user: str | None = None) -> dict:
        if as_user:
            chatdb.touch_last_seen(as_user)
        if after is not None:
            messages = chatdb.get_messages_after(after, limit)
        else:
            messages = chatdb.get_recent_messages(limit)
        return {
            "messages": messages,
            "latest_id": chatdb.latest_message_id(),
        }

    def latest_id(self) -> int:
        return chatdb.latest_message_id()

    def read_messages(
        self,
        *,
        before_id: int | None = None,
        after_id: int | None = None,
        limit: int = 40,
        as_user: str | None = None,
    ) -> dict:
        if as_user:
            chatdb.touch_last_seen(as_user)
        return {
            "messages": chatdb.read_messages(
                before_id=before_id, after_id=after_id, limit=limit
            ),
            "latest_id": chatdb.latest_message_id(),
            "oldest_id": chatdb.oldest_message_id(),
        }

    def search_messages(self, query: str, *, limit: int = 20, as_user: str | None = None) -> dict:
        if as_user:
            chatdb.touch_last_seen(as_user)
        return chatdb.search_messages(query, limit)

    def post_message(self, username: str, text: str, tags: list | None = None, images: list | None = None) -> dict:
        chatdb.touch_last_seen(username)
        msg = chatdb.post_message(username, text, tags, images=images)
        for fn in list(self._listeners):
            try:
                fn(msg)
            except Exception:
                pass
        return msg
