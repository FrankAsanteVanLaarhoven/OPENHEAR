"""Zero-Switch Streaming Language Identification (LID).

Classifies whether an ongoing speaker is using English or Mandarin Chinese,
enabling fully hands-free duplex cross-lingual conversations without manual toggles.
"""
from __future__ import annotations

import re
import numpy as np


def detect_script_language(text: str) -> tuple[str, float]:
    """Detect language from text tokens.

    Returns:
        (detected_lang, confidence): ('en' | 'zh', confidence_float).
    """
    clean = text.strip()
    if not clean:
        return 'en', 0.5
    zh_chars = len(re.findall(r"[\u4e00-\u9fff]", clean))
    en_words = len(re.findall(r"[A-Za-z]+", clean))
    total = zh_chars + en_words
    if total == 0:
        return 'en', 0.5
    zh_ratio = zh_chars / total
    if zh_ratio >= 0.35:
        return 'zh', round(zh_ratio, 3)
    return 'en', round(1.0 - zh_ratio, 3)


def auto_direction_for_input(text: str) -> tuple[str, str]:
    """Determine translation direction dynamically based on spoken input.

    Returns:
        (source_lang, target_lang): ('en', 'zh') or ('zh', 'en').
    """
    source_lang, _ = detect_script_language(text)
    target_lang = 'zh' if source_lang == 'en' else 'en'
    return source_lang, target_lang


def estimate_acoustic_chunk_language(audio: np.ndarray, sample_rate: int = 16000) -> str:
    """Fast acoustic heuristic on 200-400ms audio chunks to estimate pitch/energy variation.

    Tonal languages like Mandarin Chinese exhibit higher fundamental frequency (F0)
    variance per syllable frame compared to non-tonal stress-timed English.
    """
    if len(audio) < int(sample_rate * 0.15):
        return 'en'
    # Zero-crossing rate (ZCR)
    zero_crossings = np.sum(np.abs(np.diff(np.signbit(audio))))
    zcr = zero_crossings / len(audio)
    # Energy variance across short 20ms frames
    frame_len = int(sample_rate * 0.02)
    frames = [audio[i:i + frame_len] for i in range(0, len(audio) - frame_len, frame_len)]
    if not frames:
        return 'en'
    energies = [float(np.sqrt(np.mean(f ** 2))) for f in frames]
    energy_var = float(np.var(energies)) if energies else 0.0
    # Combined heuristic: Mandarin syllables exhibit punctuated energy bursts and lower high-freq fricative ZCR
    if zcr < 0.08 and energy_var > 0.002:
        return 'zh'
    return 'en'
