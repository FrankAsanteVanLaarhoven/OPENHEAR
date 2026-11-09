"""Reproducible local feasibility evaluation (NOT an independent clinical study).

CSV entries describe real, locally controlled WAV files. No manufactured audio,
reference translations or performance values are included in generated templates.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

FIELDNAMES = (
    "id", "source", "target", "audio_path", "reference_transcript",
    "reference_translation", "speaker_id", "accent", "condition"
)


def init_manifest(destination: str, per_direction: int = 20) -> Path:
    if not 1 <= per_direction <= 10000:
        raise ValueError("per_direction must be between 1 and 10000")
    out = Path(destination).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Will not overwrite an existing benchmark: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        for source, target in (("en", "zh"), ("zh", "en")):
            for i in range(1, per_direction + 1):
                writer.writerow({**dict.fromkeys(FIELDNAMES, ""),
                                 "id": f"{source}-{target}-{i:03d}",
                                 "source": source, "target": target,
                                 "condition": "clean"})
    return out


def _norm(s: str) -> str:
    # Whitespace and punctuation are not meaningful ASR character errors.
    import unicodedata
    s = unicodedata.normalize("NFKC", s)
    return "".join(ch.casefold() for ch in s if not ch.isspace() and
                   not unicodedata.category(ch).startswith("P"))


def _edit_distance(a, b) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j-1] + (ca != cb)))
        previous = current
    return previous[-1]


def char_error_rate(reference: str, hypothesis: str) -> float | None:
    ref = _norm(reference)
    if not ref:
        return None
    return _edit_distance(ref, _norm(hypothesis)) / len(ref)


def word_error_rate(reference: str, hypothesis: str) -> float | None:
    """Only defined here for space-delimited language (English)."""
    import unicodedata
    def tokens(text):
        raw = unicodedata.normalize("NFKC", text).casefold().split()
        return ["".join(ch for ch in token if not unicodedata.category(ch).startswith("P"))
                for token in raw]
    ref = [x for x in tokens(reference) if x]
    if not ref:
        return None
    hyp = [x for x in tokens(hypothesis) if x]
    return _edit_distance(ref, hyp) / len(ref)


def percentile(values: list[float], pct: int) -> float | None:
    if not values:
        return None
    xs = sorted(values)
    k = (len(xs) - 1) * pct / 100
    lo, hi = math.floor(k), math.ceil(k)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (k - lo), 3)


def evaluate_manifest(dataset_path: str, bridge, *,
                      model_manifest_path: str | None = None) -> dict:
    """Evaluate valid rows; report *all* missing rows and failed attempts.

    The caller supplies a real local Bridge, or test fixtures. We disclose the
    runner type so fixture output cannot be mistaken for a production benchmark.
    """
    dataset = Path(dataset_path).expanduser().resolve()
    with dataset.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = set(FIELDNAMES) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Dataset missing columns: {sorted(missing)}")
        rows = list(reader)
    results = []
    for row in rows:
        identifier = (row.get("id") or "").strip()
        src, dst = row["source"].strip(), row["target"].strip()
        entry = {"id": identifier, "source": src, "target": dst,
                 "speaker_id": row["speaker_id"].strip(),
                 "accent": row["accent"].strip(),
                 "condition": row["condition"].strip()}
        if (src, dst) not in (("en", "zh"), ("zh", "en")):
            entry.update(status="invalid", error="unsupported direction")
            results.append(entry)
            continue
        necessary = ("audio_path", "reference_transcript", "reference_translation", "speaker_id")
        if any(not row[col].strip() for col in necessary):
            entry.update(status="pending", error="Missing audio path/reference/speaker ID")
            results.append(entry)
            continue
        audio = (dataset.parent / row["audio_path"].strip()).resolve()
        if not audio.is_file():
            entry.update(status="missing_audio", error="Audio file not found")
            results.append(entry)
            continue
        try:
            processed = bridge.process(str(audio), src, dst)
            entry.update(status="completed", transcript=processed.transcript,
                         translation=processed.translation,
                         timings_ms=processed.timings_ms,
                         reference_transcript=row["reference_transcript"],
                         reference_translation=row["reference_translation"],
                         asr_cer=char_error_rate(row["reference_transcript"], processed.transcript),
                         asr_wer=(word_error_rate(row["reference_transcript"], processed.transcript)
                                  if src == "en" else None))
        except Exception as error:
            entry.update(status="failed", error_type=type(error).__name__,
                         error=str(error)[:350])
        results.append(entry)

    summaries = {}
    for src, dst in (("en", "zh"), ("zh", "en")):
        group = [r for r in results if r["source"] == src and r["target"] == dst]
        attempted = [r for r in group if r["status"] in ("completed", "failed")]
        successful = [r for r in group if r["status"] == "completed"]
        deltas = {key: [r["timings_ms"][key] for r in successful]
                  for key in ("asr", "mt", "tts", "total")}
        chrf = None
        try:
            from sacrebleu.metrics import CHRF
        except ImportError:
            pass
        else:
            if successful:
                chrf = round(CHRF(word_order=2).corpus_score(
                    [r["translation"] for r in successful],
                    [[r["reference_translation"] for r in successful]]).score, 3)
        cer_values = [r["asr_cer"] for r in successful if r.get("asr_cer") is not None]
        wer_values = [r["asr_wer"] for r in successful if r.get("asr_wer") is not None]
        summaries[f"{src}-{dst}"] = {
            "planned": len(group), "attempted": len(attempted),
            "completed": len(successful), "failed": len(attempted) - len(successful),
            "pending_or_missing": len(group) - len(attempted),
            "completion_rate_attempted": (round(len(successful)/len(attempted), 4)
                                          if attempted else None),
            "asr_cer_mean": round(statistics.mean(cer_values), 4) if cer_values else None,
            "asr_wer_mean": round(statistics.mean(wer_values), 4) if src == "en" and wer_values else None,
            "translation_chrf_plus_plus": chrf,
            "time_ms": {key: {f"p{p}": percentile(values, p) for p in (50, 95, 99)}
                        for key, values in deltas.items()},
            "feasibility_min_20_completed": len(successful) >= 20,
        }
    model_hash = None
    if model_manifest_path:
        p = Path(model_manifest_path)
        if p.is_file():
            model_hash = hashlib.sha256(p.read_bytes()).hexdigest()
    return {"schema_version": "0.3", "research_stage": "feasibility_not_clinical",
            "method": "post-utterance processing wall time; not acoustic end-to-end latency",
            "captured_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(),
            "dataset": str(dataset), "model_manifest_sha256": model_hash,
            "engine_type": type(bridge.asr).__name__ + "/" + type(bridge.translator).__name__,
            "privacy_notice": "Contains reference, transcript and translated text. Store securely; do not publish participant-identifiable data.",
            "score_note": "chrF++ is reference-based text similarity, NOT a certified semantic or human intelligibility score. WER shown only for English source. Percentiles at N=20 remain unstable.",
            "summary": summaries, "utterances": results}


def save_report(report: dict, destination: str) -> Path:
    out = Path(destination).expanduser().resolve()
    if out.exists():
        raise FileExistsError(f"Will not overwrite existing evidence file: {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out
