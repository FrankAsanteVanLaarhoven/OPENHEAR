"""Bounded, offline, deterministic audio validation and preprocessing."""
from __future__ import annotations
import math
from pathlib import Path

MAX_AUDIO_SECONDS = 29.0   # Whisper short-form decoding; no false streaming claims
MAX_AUDIO_BYTES = 16 * 1024 * 1024


def load_speech(path: str, *, max_seconds: float = MAX_AUDIO_SECONDS):
    """Return normalized mono float32 PCM at 16 kHz, reject silent/oversized input.

    Files must be local, and audio resampling is explicit. Measured metrics should
    keep the source sample rate and preprocess timing separate from model timings.
    """
    import numpy as np
    import soundfile as sf

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(path)
    if p.stat().st_size > MAX_AUDIO_BYTES:
        raise ValueError(f"Audio exceeds {MAX_AUDIO_BYTES} byte limit")
    with sf.SoundFile(p) as f:
        frames, sr = f.frames, f.samplerate
        if sr < 8000 or sr > 96000:
            raise ValueError(f"Unsupported sample rate: {sr}")
        if frames == 0 or frames / sr > max_seconds:
            raise ValueError(f"Audio must last between 0 and {max_seconds} seconds")
        audio = f.read(dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if not np.all(np.isfinite(mono)):
        raise ValueError("Non-finite PCM samples")
    if float(np.sqrt(np.mean(mono.astype(np.float64)**2))) < 0.0005:
        raise ValueError("Audio is silent or too quiet; ask the speaker to repeat")
    if sr != 16000:
        from scipy.signal import resample_poly
        d = math.gcd(sr, 16000)
        mono = resample_poly(mono, 16000 // d, sr // d)
    return np.ascontiguousarray(mono, dtype=np.float32)


def validate_uploaded_wav(audio: bytes) -> None:
    if len(audio) < 44 or len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("WAV must be between 44 bytes and 16 MiB")
    if not (audio[:4] == b"RIFF" and audio[8:12] == b"WAVE"):
        raise ValueError("Only RIFF/WAVE PCM audio is accepted by the microphone interface")


def write_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Encode float32 PCM numpy array into standard 16-bit PCM RIFF/WAVE bytes."""
    import io
    import soundfile as sf
    buf = io.BytesIO()
    clipped = np.clip(audio, -1.0, 1.0)
    sf.write(buf, clipped, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()

