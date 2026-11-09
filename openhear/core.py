"""Deterministic, auditable translation orchestration. No network calls."""
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Protocol
from subprocess import SubprocessError
import hashlib, json, time, uuid

LANGUAGES = {"en": "English", "zh": "Mandarin Chinese"}

class ASR(Protocol):
    def transcribe(self, audio_path: str, language: str) -> str: ...
class Translator(Protocol):
    def translate(self, text: str, source: str, target: str) -> str: ...
class Synthesizer(Protocol):
    def synthesize(self, text: str, language: str, output_path: str) -> str: ...

@dataclass
class Result:
    request_id: str
    source_language: str
    target_language: str
    transcript: str
    translation: str
    audio_path: str | None
    timings_ms: dict
    mode: str
    status: str
    tts_status: str = "disabled"
    tts_error: str | None = None

class Bridge:
    def __init__(self, asr: ASR, translator: Translator, tts: Synthesizer | None = None, *, tts_policy: str = "strict"):
        if tts_policy not in ("strict", "best-effort"):
            raise ValueError("tts_policy must be strict or best-effort")
        self.asr, self.translator, self.tts = asr, translator, tts
        self.tts_policy = tts_policy

    def process(self, audio_path: str, source: str, target: str, output_path: str | None = None) -> Result:
        if source not in LANGUAGES or target not in LANGUAGES or source == target:
            raise ValueError("Supported directions: en->zh and zh->en")
        if not Path(audio_path).is_file():
            raise FileNotFoundError(audio_path)
        t0 = time.perf_counter()
        transcript = self.asr.transcribe(audio_path, source).strip()
        t1 = time.perf_counter()
        if not transcript:
            raise ValueError("Empty transcript; request repeat rather than inventing speech")
        translation = self.translator.translate(transcript, source, target).strip()
        t2 = time.perf_counter()
        if not translation:
            raise ValueError("Empty translation; request repeat")
        rendered = None
        tts_status = "disabled" if self.tts is None else "not_requested"
        tts_error = None
        if self.tts and output_path:
            try:
                rendered = self.tts.synthesize(translation, target, output_path)
                tts_status = "generated"
            except (RuntimeError, OSError, ValueError, SubprocessError) as exc:
                if self.tts_policy == "strict":
                    raise
                tts_status, tts_error = "unavailable", f"{type(exc).__name__}: {exc}"
        t3 = time.perf_counter()
        return Result(str(uuid.uuid4()), source, target, transcript, translation, rendered,
                      {"asr": round((t1-t0)*1000, 2), "mt": round((t2-t1)*1000, 2),
                       "tts": round((t3-t2)*1000, 2), "total": round((t3-t0)*1000, 2)},
                      "local", "completed" if tts_status != "unavailable" else "completed_text_only",
                      tts_status, tts_error)

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def evidence_record(result: Result, input_path: str) -> dict:
    return {"schema_version": "0.1", "input_sha256": sha256_file(input_path),
            "result": asdict(result), "latency_note": "software pipeline wall time, NOT acoustic mouth-to-ear"}
