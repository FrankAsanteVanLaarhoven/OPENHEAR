import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openhear.adapters import FixtureASR, FixtureMT, MacOSSayTTS
from openhear.audio import load_speech, validate_uploaded_wav
from openhear.core import Bridge
from openhear.models import verify_manifest, manifest_for, model_dirs
from openhear.web import Runtime, BusyError, create_app


def wave_bytes(seconds=0.75, rate=16000, frequency=220.0, silence=False):
    import numpy as np
    import soundfile as sf
    samples = (0.0 if silence else 0.1) * np.sin(2*np.pi*frequency*np.arange(int(rate*seconds))/rate)
    out = io.BytesIO()
    sf.write(out, samples.astype('float32'), rate, format='WAV', subtype='PCM_16')
    return out.getvalue()


class TestAudio(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = Path(self.tmp.name)/'source.wav'
    def tearDown(self): self.tmp.cleanup()
    def test_valid_wav(self):
        blob = wave_bytes()
        validate_uploaded_wav(blob)
        self.wav.write_bytes(blob)
        arr=load_speech(str(self.wav))
        self.assertEqual(len(arr),12000)
        self.assertEqual(arr.dtype.name,'float32')
    def test_resample(self):
        self.wav.write_bytes(wave_bytes(rate=48000))
        self.assertEqual(len(load_speech(str(self.wav))),12000)
    def test_silence_rejected(self):
        self.wav.write_bytes(wave_bytes(silence=True))
        with self.assertRaisesRegex(ValueError,'silent'):
            load_speech(str(self.wav))
    def test_long_rejected(self):
        self.wav.write_bytes(wave_bytes(seconds=30.1))
        with self.assertRaisesRegex(ValueError, 'must last'):
            load_speech(str(self.wav))
    def test_upload_rejects_fake_wav(self):
        with self.assertRaises(ValueError):validate_uploaded_wav(b'a'*80)
    def test_upload_rejects_oversize(self):
        with self.assertRaises(ValueError):validate_uploaded_wav(b'RIFF'+b'x'*17000000)
    def test_small_rejected(self):
        with self.assertRaises(ValueError):validate_uploaded_wav(b'RIFF')

class TestTTS(unittest.TestCase):
    def test_parser(self):
        names=MacOSSayTTS._parse_voices('Samantha   en_US # Hi\nTing-Ting   zh_CN # 你好\nBroken line\n')
        self.assertEqual(names,[('Samantha','en_US'),('Ting-Ting','zh_CN')])
    def test_language_required(self):
        fake=MacOSSayTTS.__new__(MacOSSayTTS)
        fake.available=[('Samantha','en_US')]
        with self.assertRaisesRegex(RuntimeError,'No installed zh'):
            fake.voice_for('zh')

class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        root=Path(self.tmp.name)
        for x in ('whisper','en-zh','zh-en'):
            (root/x).mkdir()
            (root/x/'config.json').write_text('{}')
    def tearDown(self):self.tmp.cleanup()
    def test_hash_manifest(self):
        manifest=manifest_for(self.tmp.name)
        self.assertEqual(sorted(manifest['models']),['en-zh','whisper','zh-en'])
        (Path(self.tmp.name)/'models.lock.json').write_text(json.dumps(manifest))
        self.assertEqual(verify_manifest(self.tmp.name),[])
    def test_tamper_detection(self):
        manifest=manifest_for(self.tmp.name)
        (Path(self.tmp.name)/'models.lock.json').write_text(json.dumps(manifest))
        (Path(self.tmp.name)/'whisper'/'config.json').write_text('{"changed":true}')
        self.assertTrue(any('whisper/config.json' in x for x in verify_manifest(self.tmp.name)))
    def test_missing_lock(self):
        self.assertTrue(verify_manifest(self.tmp.name))
    def test_paths(self):
        paths=model_dirs(self.tmp.name)
        self.assertEqual(Path(paths['en-zh']).name,'en-zh')

class TestRuntime(unittest.TestCase):
    def setUp(self):
        self.wav=wave_bytes()
        self.asr=FixtureASR('hello')
        self.mt=FixtureMT({('en','zh','hello'):'你好'})
        self.runtime=Runtime(lambda:Bridge(self.asr,self.mt))
    def test_runtime_fixture_translation(self):
        answer=self.runtime.translate(self.wav,'en','zh')
        self.assertEqual(answer['translation'],'你好')
        self.assertIsNone(answer['audio_base64'])
        self.assertEqual(answer['mode'],'local-models')
    def test_busy(self):
        self.runtime.lock.acquire()
        try:
            with self.assertRaises(BusyError):self.runtime.translate(self.wav,'en','zh')
        finally:self.runtime.lock.release()
    def test_temp_files_discarded(self):
        # Integration path cleans temp directory on completion.
        self.runtime.translate(self.wav,'en','zh')
        self.assertIsNotNone(self.runtime.bridge)

class TestHTTP(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        self.client=TestClient(create_app(lambda:Bridge(FixtureASR('hello'),FixtureMT({('en','zh','hello'):'你好'}))))
        self.wav=wave_bytes()
    def test_home(self):
        r=self.client.get('/')
        self.assertEqual(r.status_code,200)
        self.assertIn('Understand across',r.text)
        self.assertIn('no-store',r.headers.get('cache-control',''))
    def test_health(self):
        self.assertEqual(self.client.get('/api/health').json()['status'],'not_loaded')
    def test_translation(self):
        r=self.client.post('/api/translate',data={'source':'en','target':'zh'},files={'audio':('test.wav',self.wav,'audio/wav')})
        self.assertEqual(r.status_code,200,r.text)
        self.assertEqual(r.json()['translation'],'你好')
    def test_invalid_direction(self):
        r=self.client.post('/api/translate',data={'source':'en','target':'en'},files={'audio':('test.wav',self.wav,'audio/wav')})
        self.assertEqual(r.status_code,422)
    def test_invalid_file(self):
        r=self.client.post('/api/translate',data={'source':'en','target':'zh'},files={'audio':('test.wav',b'x'*80,'audio/wav')})
        self.assertEqual(r.status_code,422)
    def test_bounded_size(self):
        r=self.client.post('/api/translate',data={'source':'en','target':'zh'},files={'audio':('test.wav',b'RIFF'+b'x'*16777220,'audio/wav')})
        self.assertEqual(r.status_code,413)

if __name__=='__main__':unittest.main()
