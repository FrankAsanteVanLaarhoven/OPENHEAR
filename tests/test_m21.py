"""M2.1 operations: read-only model discovery and offline preflight.

These fixtures contain no neural models and no human speech; success does not
assert ASR, translation, voice playback, or WAN isolation on a Mac.
"""
from __future__ import annotations

import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from openhear.models import manifest_for
from openhear.model_locator import (
    ModelDiscoveryError, manifest_candidates, verified_candidates,
    select_model_root, resolve_model_root,
)
from openhear.preflight import inspect_audio, preflight
from test_m1 import wave_bytes


def fake_model_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for key in ('whisper', 'en-zh', 'zh-en'):
        (root / key).mkdir(exist_ok=True)
        (root / key / 'config.json').write_text('{"model_type": "test_only"}', encoding='utf-8')
    (root / 'models.lock.json').write_text(json.dumps(manifest_for(str(root))))
    return root


class TestModelLocator(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.downloads = self.root / 'Downloads'
        self.downloads.mkdir()
    def tearDown(self):
        self.tmp.cleanup()

    def test_deep_nested_verified_model_found(self):
        deep = fake_model_dir(self.downloads/'hotfix'/'nested folder with spaces'/'openhear'/'m1'/'models')
        self.assertEqual(select_model_root([self.downloads]), deep.resolve())
        self.assertEqual(verified_candidates([self.downloads])[0]['valid'], True)

    def test_corrupt_manifest_contents_not_trusted(self):
        root = fake_model_dir(self.downloads/'models')
        (root/'whisper'/'config.json').write_text('tampered')
        with self.assertRaisesRegex(ModelDiscoveryError, 'No complete'):
            select_model_root([self.downloads])

    def test_broken_symlink_ignored_and_complete_real_used(self):
        fake_model_dir(self.downloads/'nested'/'models')
        (self.downloads/'bad-link').symlink_to(self.downloads/'missing', target_is_directory=True)
        self.assertEqual(len(manifest_candidates([self.downloads])), 1)

    def test_multiple_distinct_manifests_rejected(self):
        fake_model_dir(self.downloads/'models1')
        second = fake_model_dir(self.downloads/'models2')
        manifest = json.loads((second/'models.lock.json').read_text())
        manifest['research_note'] = 'different revision'
        (second/'models.lock.json').write_text(json.dumps(manifest))
        with self.assertRaisesRegex(ModelDiscoveryError, 'Multiple different'):
            select_model_root([self.downloads])

    def test_identical_manifests_accept_shortest_copy(self):
        first = fake_model_dir(self.downloads/'models1')
        second = fake_model_dir(self.downloads/'subdir'/'models2')
        (second/'models.lock.json').write_bytes((first/'models.lock.json').read_bytes())
        self.assertEqual(select_model_root([self.downloads]), first.resolve())

    def test_explicit_dir_unchanged(self):
        path = str(self.downloads/'something')
        self.assertEqual(resolve_model_root(path), path)

    def test_empty_candidates_fail_without_download(self):
        with self.assertRaises(ModelDiscoveryError):
            select_model_root([self.downloads])

    def test_direct_manifest_file(self):
        model = fake_model_dir(self.downloads/'nested')
        self.assertEqual(manifest_candidates([model/'models.lock.json']), [model.resolve()])

    def test_locator_cli_print_path_only(self):
        from openhear.cli import main
        model = fake_model_dir(self.downloads/'with spaces'/'models')
        buf = io.StringIO()
        with redirect_stdout(buf):
            main(['models','locate','--search-root',str(self.downloads),'--print-path'])
        self.assertEqual(buf.getvalue().strip(), str(model.resolve()))


class TestOfflinePreflight(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.models = fake_model_dir(self.root/'models')
        self.en = self.root/'english.wav'
        self.zh = self.root/'mandarin.wav'
        self.en.write_bytes(wave_bytes(seconds=1.0))
        self.zh.write_bytes(wave_bytes(seconds=1.25))
    def tearDown(self):
        self.tmp.cleanup()

    def test_audio_inspection_no_transcript_claim(self):
        result=inspect_audio(str(self.en), 'en')
        self.assertTrue(result['valid_audio'])
        self.assertEqual(result['sample_rate_hz'],16000)
        self.assertFalse(result['human_source_verified'])
        self.assertEqual(len(result['sha256']),64)
        self.assertNotIn('transcript',result)

    def test_missing_file_retained(self):
        result=inspect_audio(str(self.root/'missing.wav'),'zh')
        self.assertFalse(result['valid_audio'])
        self.assertIn('missing',result['error'])

    def test_preflight_without_tts_checks_valid_inputs(self):
        with patch('openhear.preflight.platform.system', return_value='Linux'):
            result=preflight(str(self.models),str(self.en),str(self.zh),require_tts=False)
        self.assertTrue(result['preflight_ready'])
        self.assertEqual(result['model_verification'],'pass')
        self.assertIn('not established',result['limitations'][-1])

    def test_preflight_tts_on_nonmac_unavailable(self):
        with patch('openhear.preflight.platform.system', return_value='Linux'):
            result=preflight(str(self.models),str(self.en),str(self.zh),require_tts=True)
        self.assertFalse(result['preflight_ready'])

    def test_preflight_invalid_audio_false(self):
        self.zh.write_bytes(wave_bytes(seconds=1.0,silence=True))
        with patch('openhear.preflight.platform.system', return_value='Linux'):
            result=preflight(str(self.models),str(self.en),str(self.zh),require_tts=False)
        self.assertFalse(result['preflight_ready'])
        self.assertFalse(result['inputs'][1]['valid_audio'])

    def test_preflight_cli_create_only_evidence(self):
        from openhear.cli import main
        target = self.root/'preflight.json'
        with redirect_stdout(io.StringIO()):
            main(['offline','preflight','--models-dir',str(self.models),
                  '--en-audio',str(self.en),'--zh-audio',str(self.zh),
                  '--no-tts','--output',str(target)])
        result=json.loads(target.read_text())
        self.assertTrue(result['preflight_ready'])
        with redirect_stdout(io.StringIO()):
            with self.assertRaises(FileExistsError):
                main(['offline','preflight','--models-dir',str(self.models),
                      '--en-audio',str(self.en),'--zh-audio',str(self.zh),
                      '--no-tts','--output',str(target)])


if __name__ == '__main__':
    unittest.main()
