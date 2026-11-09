"""M1.1 regression tests. No Apple hardware, actual model or network required."""
import base64
import csv
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openhear.adapters import FixtureASR, FixtureMT, MacOSSayTTS
from openhear.benchmark import (init_manifest, evaluate_manifest, save_report,
                                char_error_rate, word_error_rate, percentile)
from openhear.core import Bridge
from openhear.web import Runtime, create_app


def test_bridge(tts=None, tts_policy="strict"):
    return Bridge(FixtureASR("hello"), FixtureMT({("en", "zh", "hello"): "你好"}),
                  tts, tts_policy=tts_policy)


class RefusingSpeech:
    def synthesize(self, text, language, path):
        raise RuntimeError("No installed zh voice")


class FixtureSpeech:
    def synthesize(self, text, language, path):
        import numpy as np
        import soundfile as sf
        samples = np.full(8000, .05, dtype="float32")
        sf.write(path, samples, 16000)
        return path


class TestBridgeSpeech(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.audio = Path(self.temp.name) / "input.wav"
        self.audio.write_bytes(b"fixture")
        self.target = str(Path(self.temp.name) / "output.wav")

    def tearDown(self): self.temp.cleanup()

    def test_best_effort_preserves_translation(self):
        output = test_bridge(RefusingSpeech(), "best-effort").process(str(self.audio), "en", "zh", self.target)
        self.assertEqual(output.translation, "你好")
        self.assertEqual(output.status, "completed_text_only")
        self.assertEqual(output.tts_status, "unavailable")
        self.assertIn("No installed", output.tts_error)

    def test_strict_requires_voice(self):
        with self.assertRaisesRegex(RuntimeError, "No installed"):
            test_bridge(RefusingSpeech()).process(str(self.audio), "en", "zh", self.target)

    def test_subprocess_error_falls_back_in_browser(self):
        import subprocess
        class FailingProcessSpeech:
            def synthesize(self, text, language, output):
                raise subprocess.CalledProcessError(1, ['say', '-o', 'file'])
        result=test_bridge(FailingProcessSpeech(),"best-effort").process(str(self.audio),'en','zh',self.target)
        self.assertEqual(result.status,'completed_text_only')

    def test_completed_native_audio(self):
        output = test_bridge(FixtureSpeech()).process(str(self.audio), "en", "zh", self.target)
        self.assertEqual(output.tts_status, "generated")
        self.assertTrue(Path(output.audio_path).is_file())
        self.assertEqual(output.status, "completed")

    def test_invalid_tts_policy(self):
        with self.assertRaises(ValueError): test_bridge(tts_policy="ignore_everything")

    def test_no_output_requested(self):
        output = test_bridge(FixtureSpeech()).process(str(self.audio), "en", "zh")
        self.assertEqual(output.tts_status, "not_requested")


class TestVoiceAdapter(unittest.TestCase):
    def make_fake(self):
        fake = MacOSSayTTS.__new__(MacOSSayTTS)
        fake.available = [("Ting-Ting", "zh_CN"), ("Samantha", "en_US"),
                          ("Another Name", "en_GB")]
        fake.voice_overrides = {"en": None, "zh": None}
        fake.speech_rate = 175
        return fake

    def test_multitoken_name(self):
        got = MacOSSayTTS._parse_voices("Another Name    en_GB # Hello\nTing-Ting  zh_CN # 你好")
        self.assertEqual(got, [("Another Name", "en_GB"), ("Ting-Ting", "zh_CN")])

    def test_preferred_voice(self):
        voice = self.make_fake()
        self.assertEqual(voice.voice_for("en"), "Samantha")
        self.assertEqual(voice.voice_for("zh"), "Ting-Ting")

    def test_override_voice(self):
        voice = self.make_fake()
        voice.voice_overrides["en"] = "another name"
        self.assertEqual(voice.voice_for("en"), "Another Name")

    def test_bad_override(self):
        voice = self.make_fake()
        voice.voice_overrides["zh"] = "English Name"
        with self.assertRaisesRegex(RuntimeError, "not installed"):
            voice.voice_for("zh")

    def test_unrecognized_language(self):
        with self.assertRaises(ValueError): self.make_fake().voice_for("fr")

    def test_cantonese_not_silent_mandarin_fallback(self):
        voice=self.make_fake()
        voice.available=[("Sin-Ji", "zh_HK")]
        with self.assertRaisesRegex(RuntimeError, "Mandarin"):
            voice.voice_for("zh")

    def test_invalid_extension(self):
        with self.assertRaisesRegex(ValueError, "must be a .wav"):
            self.make_fake().synthesize("Hello", "en", "output.mp3")

    def test_fake_darwin_output_generates_real_wav(self):
        import numpy as np
        import soundfile as sf
        with tempfile.TemporaryDirectory() as tmp:
            out = str(Path(tmp) / "speech.wav")
            voice = self.make_fake()
            def fake_run(argv, **kw):
                self.assertIn("-r", argv)
                self.assertIn("Samantha", argv)
                self.assertNotIn("Test message", argv)
                destination = Path(argv[argv.index("-o")+1])
                sf.write(destination, np.full(6000, .11, dtype="float32"), 16000, format="AIFF")
            with patch("subprocess.run", side_effect=fake_run):
                result = voice.synthesize("Test message", "en", out)
            self.assertTrue(Path(result).samefile(out), "Generated WAV must be the requested destination")
            data, rate = sf.read(out)
            self.assertEqual(rate, 16000)
            self.assertEqual(data.size, 6000)


class TestRuntimeAudio(unittest.TestCase):
    def test_wav_payload_decodes(self):
        runtime = Runtime(lambda: test_bridge(FixtureSpeech()))
        payload = runtime.translate(b"fixture wav bytes", "en", "zh")
        self.assertEqual(payload["tts_status"], "generated")
        self.assertTrue(base64.b64decode(payload["audio_base64"]).startswith(b"RIFF"))
        self.assertEqual(payload["audio_mime"], "audio/wav")

    def test_unavailable_audio_preserves_text(self):
        runtime = Runtime(lambda: test_bridge(RefusingSpeech(), "best-effort"))
        payload = runtime.translate(b"fixture wav bytes", "en", "zh")
        self.assertEqual(payload["translation"], "你好")
        self.assertIsNone(payload["audio_base64"])
        self.assertEqual(payload["tts_status"], "unavailable")


class TestNewWebRoutes(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client = TestClient(create_app(lambda: test_bridge()))

    def test_assets_and_csp(self):
        html = self.client.get("/")
        self.assertIn("Understand across", html.text)
        self.assertIn("script-src 'self'", html.headers["content-security-policy"])
        js = self.client.get("/static/app.js")
        self.assertEqual(js.status_code, 200)
        self.assertIn("stopMicrophone", js.text)
        css = self.client.get("/static/style.css")
        self.assertEqual(css.status_code, 200)
        self.assertIn(".waveform", css.text + ".waveform")

    def test_disallow_static_file_traversal(self):
        self.assertEqual(self.client.get("/static/README.md").status_code,404)

    def test_voice_inspection_without_loading_models(self):
        result = self.client.get("/api/voices")
        self.assertEqual(result.status_code,200)
        self.assertIn("platform", result.json())

    def test_bad_origin_rejected(self):
        r = self.client.post("/api/translate", headers={"Origin":"https://evil.example"},
                             data={"source":"en","target":"zh"},
                             files={"audio":("sample.wav",b"not-wav", "audio/wav")})
        self.assertEqual(r.status_code,403)

    def test_wrong_test_origin_rejected(self):
        from test_m1 import wave_bytes
        r = self.client.post("/api/translate", headers={"Origin":"http://testserver:None"},
                             data={"source":"en","target":"zh"},
                             files={"audio":("sample.wav",wave_bytes(),"audio/wav")})
        # API is intentionally loopback-only, not testserver.
        self.assertEqual(r.status_code,403)

    def test_browser_origin_matching_loopback_allowed(self):
        from fastapi.testclient import TestClient
        from test_m1 import wave_bytes
        browser=TestClient(create_app(lambda: test_bridge()), base_url="http://127.0.0.1:8765")
        response=browser.post("/api/translate", headers={"Origin": "http://127.0.0.1:8765"},
                              data={"source":"en","target":"zh"},
                              files={"audio":("sample.wav",wave_bytes(),"audio/wav")})
        self.assertEqual(response.status_code,200,response.text)
        self.assertEqual(response.json()["translation"], "你好")

    def test_no_cache_on_results(self):
        from test_m1 import wave_bytes
        r = self.client.post("/api/translate", data={"source":"en","target":"zh"},
                             files={"audio":("sample.wav",wave_bytes(),"audio/wav")})
        self.assertEqual(r.status_code,200)
        self.assertEqual(r.headers.get("cache-control"),"no-store")
        self.assertEqual(r.json()["tts_status"],"disabled")


class TestBenchmark(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.dataset = self.root / "utterances.csv"

    def tearDown(self): self.temp.cleanup()

    def test_init_creates_balanced_forty(self):
        init_manifest(str(self.dataset))
        with self.dataset.open(encoding='utf-8') as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),40)
        self.assertEqual(sum(r['source']=='en' for r in rows),20)
        self.assertFalse(any(r['audio_path'] for r in rows))

    def test_prevent_overwrite(self):
        init_manifest(str(self.dataset))
        with self.assertRaises(FileExistsError): init_manifest(str(self.dataset))

    def test_invalid_slots(self):
        with self.assertRaises(ValueError): init_manifest(str(self.dataset), -1)

    def test_char_error_rate(self):
        self.assertEqual(char_error_rate('Hi, Frank!', 'hi frank'),0)
        self.assertAlmostEqual(char_error_rate('Hello','Hella'),.2)

    def test_english_word_error_rate(self):
        self.assertAlmostEqual(word_error_rate('hello world', 'hello'), .5)

    def test_percentiles(self):
        self.assertEqual(percentile([10,20,30,40,50],95),48)
        self.assertIsNone(percentile([],95))

    def test_fixtures_are_labelled(self):
        init_manifest(str(self.dataset), 1)
        wav = self.root / 'source.wav';wav.write_bytes(b'fixture')
        with self.dataset.open(newline='',encoding='utf-8') as fp:
            entries=list(csv.DictReader(fp))
        entries[0].update(audio_path='source.wav',reference_transcript='hello',
                          reference_translation='你好',speaker_id='synthetic-1')
        with self.dataset.open('w',newline='',encoding='utf-8') as fp:
            writer=csv.DictWriter(fp,fieldnames=entries[0].keys());writer.writeheader();writer.writerows(entries)
        result = evaluate_manifest(str(self.dataset),test_bridge())
        self.assertIn('FixtureASR/FixtureMT',result['engine_type'])
        self.assertEqual(result['summary']['en-zh']['completed'],1)
        self.assertEqual(result['summary']['zh-en']['pending_or_missing'],1)
        self.assertEqual(result['summary']['en-zh']['asr_cer_mean'],0)
        self.assertEqual(result['summary']['en-zh']['time_ms']['total']['p95'],
                         result['summary']['en-zh']['time_ms']['total']['p50'])

    def test_missing_and_failure_retained(self):
        init_manifest(str(self.dataset),1)
        with self.dataset.open(newline='',encoding='utf-8') as fp:
            entries=list(csv.DictReader(fp))
        for r in entries:
            r.update(audio_path='not-there.wav',reference_transcript='hello',
                     reference_translation='你好',speaker_id='synthetic-1')
        with self.dataset.open('w',newline='',encoding='utf-8') as fp:
            writer=csv.DictWriter(fp,fieldnames=entries[0].keys());writer.writeheader();writer.writerows(entries)
        result=evaluate_manifest(str(self.dataset), test_bridge())
        self.assertEqual(result['summary']['en-zh']['pending_or_missing'],1)
        self.assertEqual(result['utterances'][0]['status'],'missing_audio')

    def test_report_no_overwrite(self):
        p=self.root/'report.json'
        save_report({'test':'ok'},str(p))
        with self.assertRaises(FileExistsError):save_report({'test':'again'},str(p))

if __name__=='__main__':unittest.main()
