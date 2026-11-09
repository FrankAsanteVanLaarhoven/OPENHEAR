"""Anti-Hallucination Guardrails & Phantom Speech Interceptors for S2ST.

Filters out hallucination loops, ambient noise triggers, and runaway
repetition cycles typical of autoregressive speech decoders.
"""
from __future__ import annotations

import re
from typing import Sequence

# Known common autoregressive hallucination artifacts on non-speech/silence
KNOWN_PHANTOM_PATTERNS = [
    re.compile(r"thank\s+you\s+for\s+watching", re.IGNORECASE),
    re.compile(r"please\s+subscribe", re.IGNORECASE),
    re.compile(r"subtitles\s+by", re.IGNORECASE),
    re.compile(r"all\s+rights\s+reserved", re.IGNORECASE),
    re.compile(r"like\s+and\s+subscribe", re.IGNORECASE),
    re.compile(r"谢谢大家观看", re.IGNORECASE),
    re.compile(r"请订阅", re.IGNORECASE),
    re.compile(r"关注我的频道", re.IGNORECASE),
]


def detect_token_repetition_loop(text: str, max_repeat_count: int = 3) -> bool:
    """Detect runaway repetitive loops in generated text (e.g. 'and and and and' or 'word, word, word')."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < 4:
        return False
    # Check 1-word repetition
    repeat = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            repeat += 1
            if repeat >= max_repeat_count:
                return True
        else:
            repeat = 1
    # Check 2-word phrase repetition (e.g. "thank you thank you thank you")
    if len(words) >= 6:
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words) - 1)]
        b_repeat = 1
        for i in range(2, len(bigrams), 2):
            if bigrams[i] == bigrams[i - 2]:
                b_repeat += 1
                if b_repeat >= max_repeat_count:
                    return True
            else:
                b_repeat = 1
    return False


def is_phantom_hallucination(transcript: str, rms_energy: float = 1.0,
                             rms_threshold: float = 0.005) -> tuple[bool, str]:
    """Check if a generated hypothesis is a phantom hallucination.

    Returns:
        (is_hallucination, reason): True if flagged as hallucination, with explanatory reason.
    """
    clean = transcript.strip()
    if not clean:
        return False, "empty"
    # Check if speech model hallucinated substantial text on near-zero energy
    if rms_energy < rms_threshold and len(clean.split()) > 2:
        return True, "hallucination_on_near_silence"
    # Check for known YouTube / Whisper subtitle dataset hallucinations
    for pattern in KNOWN_PHANTOM_PATTERNS:
        if pattern.search(clean):
            return True, f"phantom_pattern_match:{pattern.pattern}"
    # Check for runaway repetitions
    if detect_token_repetition_loop(clean):
        return True, "runaway_repetition_loop"
    return False, "clean"


def sanitize_hypothesis(text: str, rms_energy: float = 1.0) -> str:
    """Sanitize hypothesis text. If flagged as an ungrounded hallucination, suppress to empty string."""
    is_hallucination, reason = is_phantom_hallucination(text, rms_energy=rms_energy)
    if is_hallucination:
        return ""
    return text.strip()
