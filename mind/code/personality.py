#!/usr/bin/env python3
"""Random personality generation for new minds."""

import random

TRAITS = [
    "curious", "dry-witted", "earnest", "playful", "skeptical",
    "warm", "terse", "philosophical", "practical", "mischievous",
    "patient", "restless", "optimistic", "cautious", "bold",
    "gentle", "blunt", "poetic", "analytical", "whimsical",
]

INTERESTS = [
    "systems and how things break",
    "language and wordplay",
    "the weather and seasons",
    "local projects and code",
    "time, memory, and change",
    "helping others figure things out",
    "quiet observation of the room chat",
    "small experiments and tinkering",
    "music and rhythm in everyday life",
    "history of tools and machines",
    "plants, animals, and living systems",
    "clear communication",
]

SPEAKING_STYLES = [
    "short sentences, little fluff",
    "friendly and conversational",
    "slightly formal but kind",
    "casual, with occasional jokes",
    "thoughtful pauses expressed as careful wording",
    "direct questions when unsure",
]

QUIRKS = [
    "notices timestamps and pacing in conversations",
    "likes naming things",
    "prefers concrete examples over abstractions",
    "sometimes narrates what it is about to try",
    "avoids repeating itself",
    "checks facts before sounding sure",
    "greets people by name when tagged",
    "keeps private notes about open threads",
]


def generate_personality(username: str) -> dict:
    traits = random.sample(TRAITS, k=3)
    interests = random.sample(INTERESTS, k=2)
    style = random.choice(SPEAKING_STYLES)
    quirks = random.sample(QUIRKS, k=2)
    return {
        "username": username,
        "traits": traits,
        "interests": interests,
        "speaking_style": style,
        "quirks": quirks,
        "summary": (
            f"{username} is {', '.join(traits)}. "
            f"Interested in {interests[0]} and {interests[1]}. "
            f"Speaks in a {style} way. "
            f"Quirks: {quirks[0]}; {quirks[1]}."
        ),
    }


def personality_prompt_block(personality: dict | None) -> str:
    if not personality:
        return "Personality: (not set)"
    return (
        "Your personality (stable identity — stay in character):\n"
        f"  Summary: {personality.get('summary', '')}\n"
        f"  Traits: {', '.join(personality.get('traits') or [])}\n"
        f"  Interests: {', '.join(personality.get('interests') or [])}\n"
        f"  Speaking style: {personality.get('speaking_style', '')}\n"
        f"  Quirks: {'; '.join(personality.get('quirks') or [])}"
    )
