"""Read-only data/voice/model preflight; no implication of live human speech.

An audio file can be genuine PCM yet contain synthetic speech or a mislabeled
language. Only the operator and subsequent human transcription can establish
those conditions.
"""
from __future__ import annotations

import hashlib
import platform
from datetime import datetime, timezone
from pathlib import Path


def inspect_audio(path: str, label: str) -> dict:
    from .core import sha256_file
    from .audio import load_speech
    result = {'label': label, 'path': str(Path(path).expanduser().resolve()),
              'valid_audio': False, 'speaker_consent_verified': False,
              'human_source_verified': False}
    try:
        import soundfile as sf
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f'Audio file missing: {p}')
        with sf.SoundFile(p) as wav:
            result.update({'seconds': round(wav.frames / wav.samplerate, 3),
                           'sample_rate_hz': wav.samplerate, 'channels': wav.channels})
        load_speech(str(p))
        result.update({'valid_audio': True, 'sha256': sha256_file(str(p)),
                       'size_bytes': p.stat().st_size})
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as err:
        result['error'] = f'{type(err).__name__}: {err}'[:300]
    return result


def preflight(models_dir: str, en_audio: str, zh_audio: str,
              *, require_tts: bool = True) -> dict:
    from .models import verify_manifest
    issues = verify_manifest(models_dir)
    model_root = Path(models_dir).expanduser().resolve()
    manifest = model_root / 'models.lock.json'
    sources = [inspect_audio(en_audio, 'en'), inspect_audio(zh_audio, 'zh')]
    voices = {'en': None, 'zh': None}
    if platform.system() == 'Darwin':
        try:
            from .adapters import MacOSSayTTS
            voice_adapter = MacOSSayTTS()
            for language in voices:
                try:
                    voices[language] = voice_adapter.voice_for(language)
                except RuntimeError:
                    pass
        except (RuntimeError, OSError, ValueError):
            pass
    result = {
        'schema_version': '0.5-preflight',
        'utc': datetime.now(timezone.utc).isoformat(),
        'model_dir': str(model_root),
        'model_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest()
                                  if manifest.is_file() else None,
        'model_verification': 'pass' if not issues else 'fail',
        'model_issues': issues[:10],
        'inputs': sources,
        'macos_voices': voices,
        'require_tts': require_tts,
        'preflight_ready': not issues and all(x['valid_audio'] for x in sources) and
                           (not require_tts or all(voices.values())),
        'evidence_class': 'Local file and software checks only',
        'limitations': [
            'No inference executed; recording language, speaker identity and consent are not automatically verified',
            'Model SHA256 checks are not signed provenance',
            'Audio metadata does not prove acoustic speech quality or translation correctness',
            'Offline network isolation is not established by this preflight',
        ],
    }
    return result
