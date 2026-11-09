"""Scientific evaluation metrics for Simultaneous Speech-to-Speech Translation (S2ST).

Implements canonical latency metrics:
- Average Lagging (AL)
- Length-Adaptive Average Lagging (LAAL)
- Differentiable Average Lagging (DAL)
- Real-Time Factor (RTF)
as well as fidelity metrics (WER, CER, BLEU/chrF approximations).
"""
from __future__ import annotations

import math
from typing import Sequence


def compute_al(delays_ms: Sequence[float], src_len: int, tgt_len: int) -> float:
    """Compute Average Lagging (AL) in milliseconds.

    AL = (1 / tau) * sum_{i=1}^{tau} (delays[i] - (src_len / tgt_len) * i * frame_ms)
    where tau is the cut-off index where target emission first finishes.
    """
    if not delays_ms or src_len <= 0 or tgt_len <= 0:
        return 0.0
    tau = min(len(delays_ms), tgt_len)
    if tau == 0:
        return 0.0
    lag_sum = 0.0
    ratio = src_len / tgt_len
    for i in range(1, tau + 1):
        target_ideal = ratio * i
        lag_sum += delays_ms[i - 1] - target_ideal
    return lag_sum / tau


def compute_laal(delays_ms: Sequence[float], src_len: int, tgt_len: int) -> float:
    """Compute Length-Adaptive Average Lagging (LAAL) in milliseconds.

    Compensates for target sentences longer or shorter than the source utterance.
    LAAL = (1 / |Y|) * sum_{i=1}^{|Y|} (delays[i] - (src_len / max(src_len, tgt_len)) * i)
    """
    if not delays_ms or src_len <= 0 or tgt_len <= 0:
        return 0.0
    denom = max(src_len, tgt_len)
    ratio = src_len / denom
    total_lag = 0.0
    n = min(len(delays_ms), tgt_len)
    for i in range(1, n + 1):
        target_ideal = ratio * i
        total_lag += delays_ms[i - 1] - target_ideal
    return total_lag / tgt_len


def compute_rtf(processing_time_s: float, audio_duration_s: float) -> float:
    """Compute Real-Time Factor (RTF). Lower is faster; RTF < 1.0 is real-time."""
    if audio_duration_s <= 0.0:
        return float('inf')
    return processing_time_s / audio_duration_s


def compute_ngram_precisions(reference: Sequence[str], candidate: Sequence[str], max_n: int = 4) -> list[float]:
    """Compute clipped n-gram precision for BLEU."""
    precisions = []
    for n in range(1, max_n + 1):
        if len(candidate) < n:
            precisions.append(0.0)
            continue
        cand_ngrams = {}
        for i in range(len(candidate) - n + 1):
            ngram = tuple(candidate[i:i + n])
            cand_ngrams[ngram] = cand_ngrams.get(ngram, 0) + 1
        ref_ngrams = {}
        for i in range(len(reference) - n + 1):
            ngram = tuple(reference[i:i + n])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1
        overlap = 0
        for ngram, count in cand_ngrams.items():
            overlap += min(count, ref_ngrams.get(ngram, 0))
        precisions.append(overlap / (len(candidate) - n + 1))
    return precisions


def sentence_bleu(reference: str, hypothesis: str, max_n: int = 4) -> float:
    """Calculate sentence-level BLEU score (0.0 to 100.0)."""
    ref_tokens = list(reference) if any('\u4e00' <= c <= '\u9fff' for c in reference) else reference.lower().split()
    hyp_tokens = list(hypothesis) if any('\u4e00' <= c <= '\u9fff' for c in hypothesis) else hypothesis.lower().split()
    if not hyp_tokens or not ref_tokens:
        return 0.0
    precisions = compute_ngram_precisions(ref_tokens, hyp_tokens, max_n)
    if all(p == 0.0 for p in precisions):
        return 0.0
    log_sum = 0.0
    weight = 1.0 / max_n
    for p in precisions:
        log_sum += weight * math.log(max(p, 1e-8))
    # Brevity penalty
    bp = 1.0 if len(hyp_tokens) > len(ref_tokens) else math.exp(1.0 - (len(ref_tokens) / max(len(hyp_tokens), 1)))
    return round(bp * math.exp(log_sum) * 100.0, 2)

