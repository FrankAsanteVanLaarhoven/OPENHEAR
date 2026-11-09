"""Conversational Dialogue Dynamics: Backchannels, Interruption Ducking & Speech Repair.

Provides human-centric conversational features:
- Backchannel acknowledgment filtering (e.g. 'yeah', 'uh-huh', '对', '好的')
- Instantaneous sub-30ms logarithmic audio ducking for conversational turn interruptions
- False-start and speech-repair normalization
"""
from __future__ import annotations

import re
import numpy as np

EN_BACKCHANNELS = frozenset({
    "yeah", "yea", "yes", "mhm", "uh-huh", "uh huh", "okay", "ok",
    "got it", "i see", "right", "sure", "cool", "yep", "yup"
})

ZH_BACKCHANNELS = frozenset({
    "嗯", "对", "好的", "行", "明白", "知道了", "是吗", "确实", "没错", "哦", "好", "对对对"
})


def is_backchannel(text: str, language: str = "auto") -> bool:
    """Classify if an utterance is an acknowledgment backchannel rather than a substantive turn."""
    cleaned = re.sub(r"[-_]", " ", text.strip().lower())
    cleaned = re.sub(r"[^\w\s\u4e00-\u9fff]", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return False
    if language in ("en", "auto") and cleaned in EN_BACKCHANNELS:
        return True
    if language in ("zh", "auto") and cleaned in ZH_BACKCHANNELS:
        return True
    return False


def generate_ducking_fade_out(duration_ms: float = 25.0, sample_rate: int = 16000) -> np.ndarray:
    """Generate a logarithmic fade-out gain curve for sub-30ms audio ducking on speech interruption."""
    n_samples = max(2, int((duration_ms / 1000.0) * sample_rate))
    # Logarithmic decay curve: starts at 1.0, smoothly decays to 0.0
    t = np.linspace(0.0, 1.0, n_samples)
    gain = 0.5 * (1.0 + np.cos(np.pi * t))
    return gain.astype(np.float32)


def apply_interruption_ducking(audio_buffer: np.ndarray, interruption_index: int,
                               fade_duration_ms: float = 25.0, sample_rate: int = 16000) -> np.ndarray:
    """Fade out ongoing audio playback smoothly from the interruption point, avoiding clicks/pops."""
    if interruption_index >= len(audio_buffer):
        return audio_buffer.copy()
    result = audio_buffer.copy()
    fade = generate_ducking_fade_out(duration_ms=fade_duration_ms, sample_rate=sample_rate)
    fade_len = min(len(fade), len(result) - interruption_index)
    result[interruption_index:interruption_index + fade_len] *= fade[:fade_len]
    result[interruption_index + fade_len:] = 0.0
    return result


def normalize_speech_repairs(text: str) -> str:
    """Detect and clean conversational false-starts (e.g. 'I will... wait no, I want to go')."""
    # Pattern: "... wait, no ...", "... I mean ..."
    cleaned = re.sub(r"\b(wait\s+no|i\s+mean|actually|or\s+rather),?\s*", " ", text, flags=re.IGNORECASE)
    # Pattern in Mandarin: "... 等等，不是，..." 或 "... 我是说 ..."
    cleaned = re.sub(r"(等等[，,]?不是|我是说|不对[，,]?应该是)\s*", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()
