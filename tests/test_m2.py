"""M1.2/M2-alpha deterministic tests. No network, Apple device or ML weights."""
import json
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openhear.offline import (
    PythonEgressGuard, _allowed_host, _allowed_address, _write_exclusive,
    run_bilateral_trial, snapshot,
)
from openhear.stream import VADPolicy, common_stable_prefix
from openhear.adapters import FixtureASR, FixtureMT
from openhear.core import Bridge
from openhear.web import Runtime, BusyError, create_app
from test_m1 import wave_bytes


class TestNetworkGuard(unittest.TestCase):
    def test_numeric_loopback(self):
        self.assertTrue(_allowed_host('127.0.0.1'))
        self.assertTrue(_allowed_host('::1'))
        self.assertTrue(_allowed_host('localhost'))
        self.assertTrue(_allowed_host('LOCALHOST.'))
        self.assertFalse(_allowed_host('0.0.0.0'))
        self.assertFalse(_allowed_host('192.168.1.1'))
        self.assertFalse(_allowed_host('8.8.8.8'))
        self.assertFalse(_allowed_host('example.com'))
        self.assertFalse(_allowed_host('::'))

    def test_address_unix(self):
        self.assertTrue(_allowed_address('/tmp/foo.sock', socket.AF_UNIX))
        self.assertFalse(_allowed_address(('203.0.113.4', 443), socket.AF_INET))

    def test_guard_blocks_external_tcp_without_connecting(self):
        with PythonEgressGuard() as guard:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                with self.assertRaises(PermissionError):
                    s.connect(('203.0.113.99', 443))
        self.assertEqual(guard.attempts[0]['operation'], 'connect')
        self.assertTrue(guard.attempts[0]['blocked'])

    def test_guard_restores_methods(self):
        before = socket.socket.connect
        with PythonEgressGuard():
            self.assertIsNot(before, socket.socket.connect)
        self.assertIs(before, socket.socket.connect)

    def test_guard_blocks_external_udp(self):
        with PythonEgressGuard() as guard:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                with self.assertRaises(PermissionError):
                    s.sendto(b'abc', ('198.51.100.2', 53))
        self.assertEqual(guard.attempts[0]['operation'], 'sendto')

    def test_guard_blocks_dns_lookup(self):
        with PythonEgressGuard() as guard:
            with self.assertRaises(PermissionError):
                socket.getaddrinfo('example.com', 80)
        self.assertEqual(guard.attempts[0]['operation'], 'getaddrinfo')

    def test_guard_allows_loopback_dns(self):
        with PythonEgressGuard() as guard:
            entries = socket.getaddrinfo('127.0.0.1', 80, type=socket.SOCK_STREAM)
        self.assertTrue(entries)
        self.assertEqual(guard.attempts, [])

    def test_connect_ex_blocks_external(self):
        with PythonEgressGuard() as guard:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                with self.assertRaises(PermissionError):
                    s.connect_ex(('203.0.113.9', 443))
        self.assertEqual(guard.attempts[0]['operation'], 'connect_ex')


class TestOfflineReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.en = self.root/'en.wav'
        self.zh = self.root/'zh.wav'
        self.en.write_bytes(wave_bytes())
        self.zh.write_bytes(wave_bytes())

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def fixture_engine():
        return Bridge(FixtureASR('hello'), FixtureMT({
            ('en','zh','hello'):'你好',('zh','en','hello'):'Hello',
        }))

    def test_trial_both_directions(self):
        result = run_bilateral_trial(self.fixture_engine, {'en':str(self.en),'zh':str(self.zh)})
        self.assertTrue(result['functional_pass'])
        self.assertTrue(result['python_egress_gate_pass'])
        self.assertEqual(len(result['trials']),2)
        self.assertNotIn('translation', result['trials'][0])
        self.assertNotIn('transcript', result['trials'][0])
        self.assertEqual(len(result['trials'][0]['translation_sha256']),64)

    def test_trial_retains_one_failure(self):
        def bridge(): return Bridge(FixtureASR('hello'), FixtureMT({('en','zh','hello'):'你好'}))
        record=run_bilateral_trial(bridge, {'en':str(self.en),'zh':str(self.zh)})
        self.assertFalse(record['functional_pass'])
        self.assertEqual(record['trials'][0]['status'],'completed')
        self.assertEqual(record['trials'][1]['status'],'error')

    def test_model_init_failure_is_retained_for_both_directions(self):
        def broken(): raise RuntimeError('Local checkpoint could not load')
        record=run_bilateral_trial(broken, {'en':str(self.en),'zh':str(self.zh)})
        self.assertFalse(record['functional_pass'])
        self.assertEqual(len(record['trials']),2)
        self.assertEqual(record['trials'][0]['error_stage'],'model_initialization')

    def test_model_snapshot_missing_manifest(self):
        record=snapshot(str(self.root))
        self.assertEqual(record['model_verification'],'fail')
        self.assertIsNone(record['model_manifest_sha256'])
        self.assertIn('evidence_class',record)

    def test_no_overwrite(self):
        path=self.root/'reports'/'snapshot.json'
        _write_exclusive(str(path), {'a':1})
        with self.assertRaises(FileExistsError):_write_exclusive(str(path),{'a':2})
        self.assertEqual(json.loads(path.read_text()),{'a':1})

    def test_network_attempt_fails_gate(self):
        class UnsafeASR:
            def transcribe(self, *_):
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.connect(('203.0.113.1',443))
                return 'hello'
        engine = lambda:Bridge(UnsafeASR(),FixtureMT({('en','zh','hello'):'你好'}))
        report=run_bilateral_trial(engine, {'en':str(self.en),'zh':str(self.zh)})
        self.assertFalse(report['functional_pass'])
        self.assertFalse(report['python_egress_gate_pass'])
        self.assertEqual(len(report['python_non_loopback_attempts']),2)


class TestStreamPolicy(unittest.TestCase):
    def test_config_public(self):
        d=VADPolicy().public()
        self.assertEqual(d['max_pending_segments'],2)
        self.assertEqual(d['pause_long_ms'],520)

    def test_long_utterance_shortens_hangover(self):
        policy=VADPolicy()
        self.assertEqual(policy.pause_required_ms(2000),750)
        self.assertEqual(policy.pause_required_ms(5000),520)

    def test_reject_invalid_threshold(self):
        with self.assertRaises(ValueError):VADPolicy(speech_off_rms=.2)

    def test_reject_bad_max(self):
        with self.assertRaises(ValueError):VADPolicy(max_segment_ms=40000)

    def test_stable_prefix_english(self):
        self.assertEqual(common_stable_prefix('Hello my friend','Hello my name is','en'), 'Hello my')

    def test_stable_prefix_mandarin(self):
        self.assertEqual(common_stable_prefix('我叫弗兰克', '我叫小明', 'zh'),'我叫')


class TestPreviewAPI(unittest.TestCase):
    def setUp(self):
        from fastapi.testclient import TestClient
        engine=lambda:Bridge(FixtureASR('provisional hello'),FixtureMT({}))
        self.client=TestClient(create_app(engine),base_url='http://127.0.0.1:8765')
        self.wav=wave_bytes()

    def test_stream_config(self):
        response=self.client.get('/api/stream/config')
        self.assertEqual(response.status_code,200)
        self.assertIn('vad',response.json())
        self.assertIn('not simultaneous',response.json()['warning'])

    def test_preview_returns_uncommitted(self):
        r=self.client.post('/api/preview',data={'source':'en'},files={'audio':('t.wav',self.wav,'audio/wav')})
        self.assertEqual(r.status_code,200,r.text)
        self.assertTrue(r.json()['provisional'])
        self.assertEqual(r.json()['transcript'],'provisional hello')
        self.assertNotIn('translation',r.json())

    def test_preview_requires_source(self):
        r=self.client.post('/api/preview',files={'audio':('t.wav',self.wav,'audio/wav')})
        self.assertEqual(r.status_code,422)

    def test_preview_rejects_cross_origin(self):
        r=self.client.post('/api/preview',headers={'Origin':'https://evil.example'},data={'source':'en'},files={'audio':('t.wav',self.wav,'audio/wav')})
        self.assertEqual(r.status_code,403)

    def test_preview_rejects_bad_audio(self):
        r=self.client.post('/api/preview',data={'source':'en'},files={'audio':('t.wav',b'garbage','audio/wav')})
        self.assertEqual(r.status_code,422)

    def test_preview_rejects_bad_language(self):
        r=self.client.post('/api/preview',data={'source':'de'},files={'audio':('t.wav',self.wav,'audio/wav')})
        self.assertEqual(r.status_code,422)

    def test_live_js_and_worklet_available(self):
        for asset in ('live.js','capture-worklet.js'):
            r=self.client.get(f'/static/{asset}')
            self.assertEqual(r.status_code,200)
            self.assertIn('no-store',r.headers['cache-control'])
        self.assertIn('Continuous conversation',self.client.get('/').text)

    def test_live_unavailable_asset(self):
        self.assertEqual(self.client.get('/static/not_a_script.js').status_code,404)

    def test_preview_busy_raises(self):
        runtime=Runtime(lambda:Bridge(FixtureASR('hello'),FixtureMT({})))
        runtime.lock.acquire()
        try:
            with self.assertRaises(BusyError):runtime.preview(self.wav,'en')
        finally:runtime.lock.release()


if __name__=='__main__':unittest.main()
