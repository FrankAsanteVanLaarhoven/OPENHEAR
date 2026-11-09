"""Tests for SOTA M3 capabilities: calibration, metrics, guardrails, dialogue dynamics, and LID."""
from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
import numpy as np

from openhear.calibration import generate_calibration_chirp, detect_loopback_latency_ms, evaluate_latency_budget
from openhear.metrics import compute_al, compute_laal, compute_rtf, sentence_bleu
from openhear.guardrails import is_phantom_hallucination, detect_token_repetition_loop, sanitize_hypothesis
from openhear.dialogue import is_backchannel, apply_interruption_ducking, normalize_speech_repairs
from openhear.lid import detect_script_language, auto_direction_for_input, estimate_acoustic_chunk_language
from openhear.cli import main


class TestCalibration(unittest.TestCase):
    def test_chirp_generation(self):
        chirp = generate_calibration_chirp(sample_rate=16000, duration_s=0.1)
        self.assertEqual(len(chirp), 1600)
        self.assertTrue(np.max(chirp) > 0.5)

    def test_loopback_latency_detection(self):
        chirp = generate_calibration_chirp(sample_rate=16000, duration_s=0.05)
        # Pad with 800 samples = 50ms delay
        offset = 800
        stream = np.pad(chirp, (offset, 400), mode='constant')
        delay_ms, conf = detect_loopback_latency_ms(chirp, stream, sample_rate=16000)
        self.assertAlmostEqual(delay_ms, 50.0, delta=1.0)
        self.assertGreater(conf, 0.9)

    def test_latency_budget(self):
        good = evaluate_latency_budget(120.0)
        self.assertEqual(good["status"], "EXCELLENT")
        self.assertTrue(good["itu_g114_compliant"])

        excessive = evaluate_latency_budget(500.0)
        self.assertEqual(excessive["status"], "EXCELLENT" if 500 < 150 else "EXCESSIVE")
        self.assertFalse(excessive["human_conversational_pacing"])


class TestMetrics(unittest.TestCase):
    def test_al_and_laal(self):
        delays = [200.0, 400.0, 600.0]
        al = compute_al(delays, src_len=3, tgt_len=3)
        self.assertGreater(al, 0.0)
        laal = compute_laal(delays, src_len=3, tgt_len=3)
        self.assertGreater(laal, 0.0)

    def test_rtf(self):
        self.assertAlmostEqual(compute_rtf(0.5, 2.0), 0.25)
        self.assertEqual(compute_rtf(0.5, 0.0), float('inf'))

    def test_sentence_bleu(self):
        ref = "This is a test of simultaneous speech translation."
        hyp = "This is a test of simultaneous speech translation."
        score = sentence_bleu(ref, hyp)
        self.assertEqual(score, 100.0)

        diff_score = sentence_bleu(ref, "Something completely unrelated.")
        self.assertLess(diff_score, 30.0)


class TestGuardrails(unittest.TestCase):
    def test_phantom_pattern_flagged(self):
        flagged, reason = is_phantom_hallucination("Thank you for watching! Please subscribe.")
        self.assertTrue(flagged)
        self.assertIn("phantom", reason)
        self.assertEqual(sanitize_hypothesis("Thank you for watching!"), "")

    def test_repetition_loop(self):
        loop_text = "I think that that that that is fine."
        self.assertTrue(detect_token_repetition_loop(loop_text))

        clean_text = "Good morning, how are you today?"
        self.assertFalse(detect_token_repetition_loop(clean_text))
        flagged, _ = is_phantom_hallucination(clean_text)
        self.assertFalse(flagged)

    def test_hallucination_on_silence(self):
        flagged, reason = is_phantom_hallucination("A long sentence that should not appear.", rms_energy=0.001)
        self.assertTrue(flagged)
        self.assertEqual(reason, "hallucination_on_near_silence")


class TestDialogueDynamics(unittest.TestCase):
    def test_backchannel_classification(self):
        self.assertTrue(is_backchannel("yeah", "en"))
        self.assertTrue(is_backchannel("uh-huh", "en"))
        self.assertTrue(is_backchannel("对", "zh"))
        self.assertTrue(is_backchannel("好的", "zh"))
        self.assertFalse(is_backchannel("We need to discuss the project schedule today.", "en"))

    def test_interruption_ducking(self):
        buffer = np.ones(16000, dtype=np.float32)
        ducked = apply_interruption_ducking(buffer, interruption_index=8000, fade_duration_ms=25.0)
        self.assertEqual(ducked[0], 1.0)
        self.assertEqual(ducked[-1], 0.0)
        # Verify smooth decay
        self.assertLess(ducked[8200], ducked[8050])

    def test_speech_repairs(self):
        repaired = normalize_speech_repairs("I will leave at five wait no I mean six PM.")
        self.assertNotIn("wait no", repaired)
        self.assertNotIn("I mean", repaired)


class TestLanguageIdentification(unittest.TestCase):
    def test_script_detection(self):
        lang_en, conf_en = detect_script_language("Welcome to the bilingual speech trial.")
        self.assertEqual(lang_en, 'en')
        self.assertGreater(conf_en, 0.8)

        lang_zh, conf_zh = detect_script_language("欢迎参加实时双语同传测试。")
        self.assertEqual(lang_zh, 'zh')
        self.assertGreater(conf_zh, 0.8)

    def test_auto_direction(self):
        src, tgt = auto_direction_for_input("Hello there")
        self.assertEqual((src, tgt), ('en', 'zh'))

        src, tgt = auto_direction_for_input("你好朋友")
        self.assertEqual((src, tgt), ('zh', 'en'))

    def test_acoustic_chunk(self):
        audio = np.random.randn(4800).astype(np.float32) * 0.1
        lang = estimate_acoustic_chunk_language(audio)
        self.assertIn(lang, ('en', 'zh'))


class TestSotaCLI(unittest.TestCase):
    def test_calibrate_cli(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["calibrate", "--test-latency"])
        output = json.loads(buf.getvalue())
        self.assertIn("measured_delay_ms", output)
        self.assertIn("confidence", output)

    def test_metrics_cli(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["metrics", "--delays", "150", "300", "450", "--src-len", "3", "--tgt-len", "3"])
        output = json.loads(buf.getvalue())
        self.assertIn("al_ms", output)
        self.assertIn("laal_ms", output)

    def test_guardrails_cli(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(["guardrails", "--text", "Thank you for watching"])
        output = json.loads(buf.getvalue())
        self.assertTrue(output["is_hallucination"])

    def test_benchmark_populate_and_report(self):
        import tempfile
        from pathlib import Path
        from openhear.benchmark import generate_markdown_report
        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "test_manifest.csv"
            buf = io.StringIO()
            with redirect_stdout(buf):
                main(["benchmark", "populate", "--dataset", str(csv_path), "--per-direction", "2", "--no-tts"])
            self.assertTrue(csv_path.exists())
            # Test Markdown report generator
            mock_report = {
                "captured_utc": "2026-11-09T00:00:00Z",
                "engine_type": "Whisper/Marian",
                "summary": {
                    "en-zh": {
                        "completed": 2, "planned": 2,
                        "translation_bleu_mean": 28.5,
                        "translation_chrf_plus_plus": 45.2,
                        "asr_wer_mean": 0.05,
                        "asr_cer_mean": 0.02,
                        "time_ms": {"total": {"p50": 450.0, "p95": 600.0}},
                        "hallucinations_detected": 0
                    }
                }
            }
            md = generate_markdown_report(mock_report)
            self.assertIn("en-zh", md)
            self.assertIn("28.5", md)
            self.assertIn("Leaderboard", md)


class TestSotaWebAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from starlette.testclient import TestClient
        from openhear.web import create_app
        # Mock runtime factory
        class MockBridge:
            asr = None
            translator = None
            tts = None
        cls.client = TestClient(create_app(lambda: MockBridge()))

    def test_calibrate_chirp_route(self):
        resp = self.client.get("/api/calibrate/chirp")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/wav")
        self.assertGreater(len(resp.content), 1000)

    def test_lid_route(self):
        resp = self.client.post("/api/lid", data={"text": "Hello this is English"})
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["detected"], "en")
        self.assertEqual(data["direction"], "en:zh")

        resp_zh = self.client.post("/api/lid", data={"text": "你好世界"})
        self.assertEqual(resp_zh.status_code, 200)
        self.assertEqual(resp_zh.json()["detected"], "zh")
        self.assertEqual(resp_zh.json()["direction"], "zh:en")

    def test_benchmarks_summary_route(self):
        resp = self.client.get("/api/benchmarks/summary")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("available", data)


if __name__ == "__main__":
    unittest.main()

