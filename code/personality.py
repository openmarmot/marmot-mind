#!/usr/bin/env python3
"""Personality note a mind writes for itself.

A new mind starts without one. Nothing is invented on the first think loop.
An older invented character sheet (summary / who / voice) is ignored.
"""


def personality_note(personality) -> str:
    """The mind's own note, or empty when unset or when the value is an old character sheet."""
    if isinstance(personality, str):
        return personality.strip()
    return ""


def personality_prompt_block(personality) -> str:
    note = personality_note(personality)
    if not note:
        return ""
    return (
        "What you have noticed about your own manner "
        "(from your own experience — a reminder, not a role to play):\n"
        f"  {note}"
    )
