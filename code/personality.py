#!/usr/bin/env python3
"""Optional note a mind writes for itself. Empty until it sets one."""


def personality_note(personality) -> str:
    """The mind's own note, or empty when unset."""
    if isinstance(personality, str):
        return personality.strip()
    return ""


def personality_prompt_block(personality) -> str:
    note = personality_note(personality)
    if not note:
        return ""
    return f"Note to yourself:\n  {note}"
