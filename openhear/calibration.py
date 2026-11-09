"""Physical Acoustic Loopback Calibration for True Mouth-to-Ear Latency.

Measures hardware, OS buffer, CoreAudio and DAC/ADC delays by injecting a
known acoustic chirp signal and calculating sub-millisecond cross-correlation
time-of-flight against loopback microphone audio.
"""
from __future__ import annotations

import math
import numpy as np


def generate_calibration_chirp(sample_rate: int = 16000, duration_s: float = 0.1,
                               f_start: float = 500.0, f_end: float = 4000.0) -> np.ndarray:
    """Generate a logarithmic frequency chirp with Tukey/Hann windowing for latency calibration."""
    n_samples = int(sample_rate * duration_s)
    t = np.linspace(0.0, duration_s, n_samples, endpoint=False)
    # Logarithmic chirp instantaneous phase
    k = math.log(f_end / f_start) / duration_s
    phase = 2.0 * math.pi * f_start * (np.exp(k * t) - 1.0) / k
    signal = np.sin(phase)
    # Apply 10% cosine fade-in/fade-out to eliminate edge clicks
    fade_len = int(n_samples * 0.1)
    if fade_len > 0:
        window = np.ones(n_samples)
        fade_curve = 0.5 * (1.0 - np.cos(np.linspace(0, math.pi, fade_len)))
        window[:fade_len] = fade_curve
        window[-fade_len:] = fade_curve[::-1]
        signal *= window
    return signal.astype(np.float32)


def detect_loopback_latency_ms(reference_chirp: np.ndarray, recorded_stream: np.ndarray,
                               sample_rate: int = 16000) -> tuple[float, float]:
    """Calculate the physical time-of-flight delay in milliseconds via matched filter cross-correlation.

    Returns:
        (delay_ms, correlation_peak): Delay in ms, and peak normalized cross-correlation (0.0 to 1.0).
    """
    if len(recorded_stream) < len(reference_chirp):
        return 0.0, 0.0
    ref_norm = reference_chirp - np.mean(reference_chirp)
    rec_norm = recorded_stream - np.mean(recorded_stream)
    ref_energy = np.sum(ref_norm ** 2)
    if ref_energy <= 1e-12:
        return 0.0, 0.0
    # Cross-correlation via FFT
    corr = np.correlate(rec_norm, ref_norm, mode='valid')
    peak_idx = int(np.argmax(np.abs(corr)))
    peak_val = float(corr[peak_idx])
    # Compute local window energy to normalize peak
    window_energy = np.sum(rec_norm[peak_idx:peak_idx + len(reference_chirp)] ** 2)
    norm_denom = math.sqrt(ref_energy * max(window_energy, 1e-12))
    normalized_confidence = min(1.0, max(0.0, abs(peak_val) / norm_denom))
    delay_ms = (peak_idx / sample_rate) * 1000.0
    return round(delay_ms, 2), round(normalized_confidence, 4)


def evaluate_latency_budget(delay_ms: float) -> dict[str, any]:
    """Assess acoustic latency against psychoacoustic human conversation limits."""
    # Under 150ms: Imperceptible (ITU-T G.114 standard for natural conversation)
    # 150ms - 400ms: Acceptable for simultaneous interpretation
    # > 400ms: Cognitive collision and turn overlap threshold
    status = "EXCELLENT" if delay_ms < 150.0 else "ACCEPTABLE" if delay_ms < 400.0 else "EXCESSIVE"
    return {
        "delay_ms": delay_ms,
        "status": status,
        "itu_g114_compliant": delay_ms < 150.0,
        "human_conversational_pacing": delay_ms < 400.0,
    }
