"""Explicit best-effort turn-based VAD policy; not causal token-level S2ST."""
from dataclasses import asdict, dataclass

@dataclass(frozen=True)
class VADPolicy:
    speech_on_rms: float = 0.018
    speech_off_rms: float = 0.012
    min_speech_ms: int = 420
    pre_roll_ms: int = 260
    long_utterance_ms: int = 3500
    pause_short_ms: int = 750
    pause_long_ms: int = 520
    max_segment_ms: int = 11000
    preview_after_ms: int = 1800
    preview_every_ms: int = 2500
    max_pending_segments: int = 2

    def __post_init__(self):
        if not 0 < self.speech_off_rms <= self.speech_on_rms < 1:
            raise ValueError('Invalid RMS thresholds')
        if not 0 < self.min_speech_ms < self.long_utterance_ms < self.max_segment_ms <= 28000:
            raise ValueError('Invalid segment durations')
        if not 0 < self.pause_long_ms <= self.pause_short_ms:
            raise ValueError('Invalid pause thresholds')
        if not 0 <= self.pre_roll_ms < 1000 or self.max_pending_segments < 1:
            raise ValueError('Invalid pre-roll or queue capacity')

    def pause_required_ms(self, utterance_ms: int) -> int:
        return self.pause_long_ms if utterance_ms >= self.long_utterance_ms else self.pause_short_ms

    def public(self):
        return asdict(self)


def common_stable_prefix(previous: str, current: str, language: str) -> str:
    """Only a display hint. Confirmed transcript comes from end-of-turn ASR."""
    if language == 'zh':
        a = list(previous.strip()); b = list(current.strip())
        common = []
        for l, r in zip(a, b):
            if l != r: break
            common.append(l)
        return ''.join(common).strip()
    a, b = previous.strip().split(), current.strip().split()
    stable = []
    for l, r in zip(a, b):
        if l.lower() != r.lower(): break
        stable.append(l)
    return ' '.join(stable)
