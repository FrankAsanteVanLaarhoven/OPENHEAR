"""Local model adapters. No model downloads occur at inference time."""
from __future__ import annotations
import platform
from pathlib import Path
from .audio import load_speech

class FixtureASR:
    """Only for tests. No actual recognition."""
    def __init__(self, transcript: str): self.transcript = transcript
    def transcribe(self, audio_path: str, language: str) -> str: return self.transcript

class FixtureMT:
    """Only for tests. No actual translation."""
    def __init__(self, mapping: dict): self.mapping = mapping
    def translate(self, text: str, source: str, target: str) -> str:
        return self.mapping.get((source, target, text), "")


def choose_device(requested: str) -> str:
    if requested not in ("auto", "cpu", "mps"):
        raise ValueError("Device must be auto, cpu or mps")
    import torch
    has_mps = bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if requested == "mps" and not has_mps:
        raise RuntimeError("MPS unavailable; use --device cpu")
    return "mps" if requested == "auto" and has_mps else "cpu" if requested == "auto" else requested


def local_directory(path: str) -> str:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_dir() or not (resolved / "config.json").is_file():
        raise FileNotFoundError(f"Local model directory missing config.json: {resolved}. Run 'openhear prepare' while online.")
    return str(resolved)


class LocalWhisperASR:
    """Short-form multilingual Whisper ASR (not Whisper English-only checkpoint)."""
    def __init__(self, model_dir: str, device: str = "auto"):
        import torch
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
        self.device = choose_device(device)
        path = local_directory(model_dir)
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(path, local_files_only=True).eval().to(self.device)
        self.torch = torch

    def transcribe(self, audio_path: str, language: str) -> str:
        if language not in ("en", "zh"):
            raise ValueError("Only English and Mandarin supported")
        audio = load_speech(audio_path)
        features = self.processor.feature_extractor(audio, sampling_rate=16000, return_tensors="pt").input_features.to(self.device)
        with self.torch.inference_mode():
            tokens = self.model.generate(features, language={"en": "english", "zh": "chinese"}[language], task="transcribe", max_new_tokens=128)
        return self.processor.batch_decode(tokens, skip_special_tokens=True)[0].strip()


class LocalMarianMT:
    def __init__(self, en_zh_dir: str, zh_en_dir: str, device: str = "auto"):
        import torch
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        self.device = choose_device(device)
        self.torch = torch
        self.models = {}
        for direction, raw in (("en-zh", en_zh_dir), ("zh-en", zh_en_dir)):
            path = local_directory(raw)
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True).eval().to(self.device)
            self.models[direction] = (tokenizer, model)

    def translate(self, text: str, source: str, target: str) -> str:
        direction = f"{source}-{target}"
        if direction not in self.models:
            raise ValueError(f"Unsupported direction: {direction}")
        tok, model = self.models[direction]
        # opus-mt-en-zh is multi-target; the model card requires a prefix to
        # force Simplified Mandarin rather than an arbitrary Chinese variety.
        if direction == "en-zh":
            target_token = ">>cmn_Hans<<"
            if tok.convert_tokens_to_ids(target_token) == tok.unk_token_id:
                raise RuntimeError("The English-to-Chinese model lacks the documented cmn_Hans target token")
            text = target_token + " " + text
        payload = tok(text, return_tensors="pt", truncation=True, max_length=384)
        payload = {key: value.to(self.device) for key, value in payload.items()}
        with self.torch.inference_mode():
            prediction = model.generate(**payload, max_new_tokens=256, num_beams=2)
        return tok.decode(prediction[0], skip_special_tokens=True).strip()


class MacOSSayTTS:
    """Local macOS speech synthesis through `say`, writing a real PCM WAV.

    The voice must already be installed. Never downloads a voice automatically.
    Native `say` does not prove audiological quality or end-to-end latency.
    """
    def __init__(self, voice_en: str | None = None, voice_zh: str | None = None,
                 speech_rate: int = 175):
        import shutil
        if platform.system() != "Darwin" or not shutil.which("say"):
            raise RuntimeError("Native speech requires macOS `say` (available on Mac)")
        if not 100 <= speech_rate <= 300:
            raise ValueError("Speech rate must be 100..300 words/minute")
        self.available = self._list_voices()
        self.voice_overrides = {"en": voice_en, "zh": voice_zh}
        self.speech_rate = speech_rate
        for language, name in self.voice_overrides.items():
            if name:
                self.voice_for(language)  # reject wrong-language voice up front

    @staticmethod
    def _parse_voices(text: str):
        # macOS: 'Ting-Ting        zh_CN  # 你好' (names may contain spaces)
        voices = []
        for line in text.splitlines():
            if "#" not in line:
                continue
            prefix = line.split("#", 1)[0].strip()
            parts = prefix.rsplit(None, 1)
            if len(parts) == 2 and len(parts[1]) >= 2 and parts[1][:2].isalpha():
                voices.append((parts[0], parts[1]))
        return voices

    def _list_voices(self):
        import subprocess
        proc = subprocess.run(["say", "-v", "?"], capture_output=True,
                              text=True, check=True, timeout=12)
        return self._parse_voices(proc.stdout)

    def voice_for(self, language: str) -> str:
        if language not in ("en", "zh"):
            raise ValueError("English/Mandarin only")
        if language == "zh":
            # zh_HK may be Cantonese; never silently label it Mandarin.
            found = [(name, locale) for name, locale in self.available
                     if locale.lower().replace("-", "_").startswith(("zh_cn", "zh_tw", "zh_sg"))]
        else:
            found = [(name, locale) for name, locale in self.available
                     if locale.lower().startswith("en")]
        if not found:
            raise RuntimeError(f"No installed {language} system voice suitable for {'Mandarin' if language == 'zh' else 'English'}. Add it in macOS System Settings > Accessibility > Spoken Content > System Voice.")
        desired_name = self.voice_overrides.get(language)
        if desired_name:
            for name, _ in found:
                if name.casefold() == desired_name.casefold():
                    return name
            raise RuntimeError(f"Requested voice {desired_name!r} is not installed for {language}")
        priority = {"en": ("samantha", "alex"), "zh": ("ting-ting", "tingting", "meijia")}[language]
        for preference in priority:
            for name, _ in found:
                if name.casefold() == preference:
                    return name
        return found[0][0]

    def synthesize(self, text: str, language: str, output_path: str) -> str:
        import soundfile as sf
        import subprocess
        import tempfile
        import numpy as np
        if language not in ("en", "zh") or not text.strip():
            raise ValueError("TTS requires supported nonempty text")
        if len(text) > 2000:
            raise ValueError("Refusing an overly long TTS string")
        voice = self.voice_for(language)
        output = Path(output_path).expanduser().resolve()
        if output.suffix.lower() != ".wav":
            raise ValueError("Native speech output must be a .wav file")
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="openhear-say-") as tempdir:
            txt = Path(tempdir) / "speech.txt"
            aiff = Path(tempdir) / "speech.aiff"
            txt.write_text(text, encoding="utf-8")
            # File input avoids shell interpolation and text exposure in argv.
            subprocess.run(["say", "-v", voice, "-r", str(self.speech_rate),
                            "-f", str(txt), "-o", str(aiff)],
                           capture_output=True, check=True, timeout=90)
            samples, rate = sf.read(aiff, dtype="float32", always_2d=True)
            if samples.shape[0] == 0 or not np.isfinite(samples).all():
                raise RuntimeError("macOS speech produced empty/non-finite audio")
            if samples.shape[0] / rate > 60:
                raise RuntimeError("Generated audio exceeds the 60-second limit")
            sf.write(output, samples, rate, format="WAV", subtype="PCM_16")
        return str(output)


def play_wav_macos(path: str):
    """Explicit, user-invoked local playback; never used in the server."""
    import shutil
    import subprocess
    if platform.system() != "Darwin" or not shutil.which("afplay"):
        raise RuntimeError("`afplay` is available on macOS only")
    subprocess.run(["afplay", str(Path(path).resolve())], check=True, timeout=75)
