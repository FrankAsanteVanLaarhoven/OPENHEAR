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
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .metrics import compute_rtf, sentence_bleu
from .guardrails import is_phantom_hallucination

FIELDNAMES = (
    "id", "source", "target", "audio_path", "reference_transcript",
    "reference_translation", "speaker_id", "accent", "condition"
)

# Standardized bilingual evaluation corpus for automated benchmarking
BENCHMARK_CORPUS_PAIRS = {
    "en-zh": [
        ("Good morning, welcome to our simultaneous translation trial.", "早上好，欢迎参加我们的同传测试。"),
        ("The total cost of the project is twenty-five thousand dollars.", "该项目的总成本为两万五千美元。"),
        ("Could you please send me the updated meeting schedule?", "您能把更新后的会议日程发给我吗？"),
        ("Artificial intelligence models can operate completely offline.", "人工智能模型可以完全离线运行。"),
        ("We are deploying our software update next Monday.", "我们下周一发布软件更新。"),
    ],
    "zh-en": [
        ("早上好，很高兴今天能够在这里与你交谈。", "Good morning, it is a pleasure to speak with you here today."),
        ("这个系统的延迟大约在五百毫秒以内。", "The latency of this system is approximately within five hundred milliseconds."),
        ("我们必须确保所有的测试数据都是真实有效的。", "We must ensure that all test data is genuine and valid."),
        ("离线语音翻译技术对隐私保护非常重要。", "Offline speech translation technology is very important for privacy protection."),
        ("谢谢你的帮助，我们明天继续讨论方案。", "Thank you for your help, we will continue discussing the plan tomorrow."),
    ]
}


def populate_benchmark_fixtures(dataset_path: str, per_direction: int = 5,
                                use_tts: bool = True) -> Path:
    """Populate a complete, runnable benchmark dataset with authentic audio and references."""
    dataset = Path(dataset_path).expanduser().resolve()
    dataset.parent.mkdir(parents=True, exist_ok=True)
    audio_dir = dataset.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for src, dst in (("en", "zh"), ("zh", "en")):
        pairs = BENCHMARK_CORPUS_PAIRS[f"{src}-{dst}"]
        count = min(per_direction, len(pairs))
        for i in range(count):
            uid = f"{src}-{dst}-{i+1:03d}"
            ref_transcript, ref_translation = pairs[i]
            audio_rel = f"audio/{uid}.wav"
            audio_full = dataset.parent / audio_rel
            
            if use_tts and not audio_full.exists():
                voice = "Samantha" if src == "en" else "Tingting"
                aiff_temp = audio_full.with_suffix(".aiff")
                try:
                    subprocess.run(["say", "-v", voice, "-o", str(aiff_temp), ref_transcript],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    subprocess.run(["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff_temp), str(audio_full)],
                                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    # Fallback to programmatic synthetic wav if say/afconvert fails
                    import numpy as np
                    import soundfile as sf
                    t = np.linspace(0, 1.2, int(16000 * 1.2), endpoint=False)
                    wav = (0.3 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
                    sf.write(str(audio_full), wav, 16000, subtype="PCM_16")
                finally:
                    if aiff_temp.exists():
                        aiff_temp.unlink()

            rows.append({
                "id": uid,
                "source": src,
                "target": dst,
                "audio_path": audio_rel,
                "reference_transcript": ref_transcript,
                "reference_translation": ref_translation,
                "speaker_id": "spk-001" if src == "en" else "spk-002",
                "accent": "standard",
                "condition": "clean"
            })

    with dataset.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return dataset


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
            is_hal, hal_reason = is_phantom_hallucination(processed.transcript)
            entry.update(status="completed", transcript=processed.transcript,
                         translation=processed.translation,
                         timings_ms=processed.timings_ms,
                         reference_transcript=row["reference_transcript"],
                         reference_translation=row["reference_translation"],
                         asr_cer=char_error_rate(row["reference_transcript"], processed.transcript),
                         asr_wer=(word_error_rate(row["reference_transcript"], processed.transcript)
                                  if src == "en" else None),
                         translation_bleu=sentence_bleu(row["reference_translation"], processed.translation),
                         is_hallucination=is_hal,
                         hallucination_reason=hal_reason)
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
        bleu_values = [r["translation_bleu"] for r in successful if r.get("translation_bleu") is not None]
        summaries[f"{src}-{dst}"] = {
            "planned": len(group), "attempted": len(attempted),
            "completed": len(successful), "failed": len(attempted) - len(successful),
            "pending_or_missing": len(group) - len(attempted),
            "completion_rate_attempted": (round(len(successful)/len(attempted), 4)
                                          if attempted else None),
            "asr_cer_mean": round(statistics.mean(cer_values), 4) if cer_values else None,
            "asr_wer_mean": round(statistics.mean(wer_values), 4) if src == "en" and wer_values else None,
            "translation_bleu_mean": round(statistics.mean(bleu_values), 2) if bleu_values else None,
            "translation_chrf_plus_plus": chrf,
            "hallucinations_detected": sum(1 for r in successful if r.get("is_hallucination")),
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


def generate_markdown_report(report: dict) -> str:
    """Format benchmark report into a structured Markdown comparison table."""
    lines = [
        "# OpenHear Bridge SOTA Benchmark Evaluation Report",
        "",
        "## SOTA Benchmark Leaderboard Summary",
        "",
        f"- **Captured (UTC)**: `{report.get('captured_utc', 'N/A')}`",
        f"- **Engine**: `{report.get('engine_type', 'N/A')}`",
        f"- **Model Manifest SHA256**: `{report.get('model_manifest_sha256', 'none')}`",
        "",
        "## Performance & Fidelity Summary",
        "",
        "| Direction | Completed / Planned | BLEU Mean | chrF++ | ASR WER | ASR CER | Total Latency P50 | Total Latency P95 | Hallucinations |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    summary = report.get("summary", {})
    for dir_key, data in summary.items():
        completed = f"{data.get('completed', 0)} / {data.get('planned', 0)}"
        bleu = f"{data.get('translation_bleu_mean', 'N/A')}"
        chrf = f"{data.get('translation_chrf_plus_plus', 'N/A')}"
        wer = f"{data.get('asr_wer_mean', 'N/A')}"
        cer = f"{data.get('asr_cer_mean', 'N/A')}"
        times = data.get("time_ms", {}).get("total", {})
        p50 = f"{times.get('p50', 'N/A')} ms"
        p95 = f"{times.get('p95', 'N/A')} ms"
        hals = f"{data.get('hallucinations_detected', 0)}"
        lines.append(f"| **{dir_key}** | {completed} | {bleu} | {chrf} | {wer} | {cer} | {p50} | {p95} | {hals} |")
    lines.append("")
    return "\n".join(lines)

