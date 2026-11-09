import tempfile
from pathlib import Path
import unittest
from openhear.core import Bridge, evidence_record
from openhear.adapters import FixtureASR, FixtureMT

class TestBridge(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.audio = Path(self.tmp.name) / "audio.wav"
        self.audio.write_bytes(b"fixture audio")
        self.bridge = Bridge(FixtureASR("hello"), FixtureMT({("en", "zh", "hello"): "你好"}))
    def tearDown(self): self.tmp.cleanup()
    def test_direction(self):
        r = self.bridge.process(str(self.audio), "en", "zh")
        self.assertEqual(r.translation, "你好")
        self.assertEqual(r.status, "completed")
    def test_evidence(self):
        e = evidence_record(self.bridge.process(str(self.audio), "en", "zh"), str(self.audio))
        self.assertEqual(len(e["input_sha256"]), 64)
    def test_unsupported(self):
        with self.assertRaises(ValueError): self.bridge.process(str(self.audio), "fr", "en")
    def test_same_language(self):
        with self.assertRaises(ValueError): self.bridge.process(str(self.audio), "en", "en")
    def test_missing_audio(self):
        with self.assertRaises(FileNotFoundError): self.bridge.process("missing.wav", "en", "zh")
    def test_empty_transcript(self):
        with self.assertRaisesRegex(ValueError, "Empty transcript"):
            Bridge(FixtureASR(""), FixtureMT({})).process(str(self.audio), "en", "zh")
    def test_empty_translation(self):
        with self.assertRaisesRegex(ValueError, "Empty translation"):
            Bridge(FixtureASR("hello"), FixtureMT({})).process(str(self.audio), "en", "zh")

if __name__ == "__main__": unittest.main()
