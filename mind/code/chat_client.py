#!/usr/bin/env python3
"""HTTP client for the Marmot Chat Server."""

import requests


class ChatClient:
    def __init__(self, base_url: str, token: str | None = None, username: str | None = None):
        self.base_url = (base_url or "").rstrip("/")
        self.token = token
        self.username = username
        self._session = requests.Session()

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def health(self) -> dict:
        r = self._session.get(self._url("/health"), timeout=10)
        r.raise_for_status()
        return r.json()

    def signup(self, username: str) -> dict:
        r = self._session.post(
            self._url("/api/signup"),
            json={"username": username},
            timeout=15,
        )
        data = r.json() if r.content else {}
        if r.status_code == 409:
            # already exists — fall through to login
            return self.login(username)
        if r.status_code >= 400:
            raise RuntimeError(data.get("error") or f"signup failed HTTP {r.status_code}")
        self.token = data["token"]
        self.username = data["username"]
        return data

    def login(self, username: str) -> dict:
        r = self._session.post(
            self._url("/api/login"),
            json={"username": username},
            timeout=15,
        )
        data = r.json() if r.content else {}
        if r.status_code >= 400:
            raise RuntimeError(data.get("error") or f"login failed HTTP {r.status_code}")
        self.token = data["token"]
        self.username = data["username"]
        return data

    def ensure_registered(self, username: str) -> dict:
        """Login if exists, otherwise signup."""
        try:
            return self.login(username)
        except RuntimeError:
            return self.signup(username)

    def me(self) -> dict:
        r = self._session.get(self._url("/api/me"), headers=self._headers(), timeout=10)
        r.raise_for_status()
        return r.json()

    def list_users(self) -> list[dict]:
        r = self._session.get(self._url("/api/users"), headers=self._headers(), timeout=10)
        r.raise_for_status()
        return r.json().get("users") or []

    def get_messages(self, after: int | None = None, limit: int = 100) -> dict:
        params = {"limit": limit}
        if after is not None:
            params["after"] = after
        r = self._session.get(
            self._url("/api/messages"),
            headers=self._headers(),
            params=params,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()

    def post_message(self, text: str, tags: list[str] | None = None) -> dict:
        payload = {"text": text}
        if tags is not None:
            payload["tags"] = tags
        r = self._session.post(
            self._url("/api/messages"),
            headers=self._headers(),
            json=payload,
            timeout=15,
        )
        data = r.json() if r.content else {}
        if r.status_code >= 400:
            raise RuntimeError(data.get("error") or f"post failed HTTP {r.status_code}")
        return data
