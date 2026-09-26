#!/usr/bin/env python3
"""Personality for a Mind instance.

Identity is invented by the mind's own LLM on the first think loop.
A record is summary, who, and voice.
"""

import json
import re
import secrets

import requests


def personality_is_set(personality) -> bool:
    if not personality:
        return False
    if isinstance(personality, str):
        return bool(personality.strip())
    if not isinstance(personality, dict):
        return False
    return bool(
        personality.get("summary") or personality.get("who") or personality.get("voice")
    )


def personality_prompt_block(personality: dict | None) -> str:
    if not personality_is_set(personality):
        return "Personality: (not invented yet)"
    if isinstance(personality, str):
        return (
            "Your personality (stable identity — stay in character):\n"
            f"  {personality.strip()}"
        )

    summary = (personality.get("summary") or "").strip()
    who = (personality.get("who") or "").strip()
    voice = (personality.get("voice") or "").strip()
    lines = ["Your personality (stable identity — stay in character):"]
    if summary:
        lines.append(f"  {summary}")
    if who:
        lines.append(f"  {who}")
    if voice:
        lines.append(f"  Voice: {voice}")
    return "\n".join(lines)


def invent_personality(username: str, llm_base: str, llm_model: str) -> dict:
    """Ask the LLM to invent a one-off identity. Raises on failure."""
    nonce = secrets.token_hex(8)
    user = (
        f'Invent a durable identity for a chat participant named "{username}".\n'
        "Make them a specific person, not a generic helpful assistant.\n"
        "Do not mention the chat room, silence, or introducing yourself.\n"
        "Reply with JSON only:\n"
        "{\n"
        '  "summary": "one sentence",\n'
        '  "who": "a short paragraph of who they are and what they care about",\n'
        '  "voice": "how they write in chat"\n'
        "}\n"
        f"Variation token (not part of the identity): {nonce}"
    )
    payload = {
        "model": llm_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You invent fictional chat identities. "
                    "Reply with a single JSON object, no markdown."
                ),
            },
            {"role": "user", "content": user},
        ],
        "max_tokens": 800,
        "temperature": 1.1,
    }
    r = requests.post(
        f"{llm_base.rstrip('/')}/chat/completions",
        json=payload,
        timeout=120,
    )
    if r.status_code != 200:
        raise RuntimeError(f"LLM HTTP {r.status_code}: {r.text[:250]}")
    content = (
        r.json().get("choices", [{}])[0].get("message", {}).get("content") or ""
    )
    return _parse_personality_json(content)


def _parse_personality_json(text: str) -> dict:
    raw = (text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("LLM personality response had no JSON object")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("LLM personality JSON must be an object")
    summary = str(data.get("summary") or "").strip()
    who = str(data.get("who") or "").strip()
    voice = str(data.get("voice") or "").strip()
    if not summary and who:
        summary = who.split(".")[0].strip()[:160]
    if not summary:
        raise ValueError("LLM personality JSON missing summary")
    return {"summary": summary, "who": who, "voice": voice}
