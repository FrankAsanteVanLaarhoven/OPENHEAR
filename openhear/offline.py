"""Offline-assurance evidence. Explicitly *not* an OS-wide packet firewall.

A Python socket guard denies non-loopback targets for Python socket APIs.
It cannot intercept native C libraries, macOS services, subprocesses or browsers;
independent host-level monitoring remains a separate acceptance gate.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import platform
import socket
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


def _allowed_host(host: object) -> bool:
    if host is None or host == '':
        return True  # local binding / passive name resolution
    if isinstance(host, bytes):
        host = host.decode('ascii', errors='replace')
    if not isinstance(host, str):
        return False
    if host.lower().rstrip('.') == 'localhost':
        return True
    try:
        address = ipaddress.ip_address(host.split('%', 1)[0])
        return address.is_loopback or bool(getattr(address, 'ipv4_mapped', None) and address.ipv4_mapped.is_loopback)
    except ValueError:
        return False


def _allowed_address(address: object, family: int | None = None) -> bool:
    if family == socket.AF_UNIX:
        return True
    if isinstance(address, tuple) and len(address) >= 2:
        return _allowed_host(address[0])
    return False


class PythonEgressGuard:
    """Process-wide guard for common Python TCP/UDP socket operations.

    Activate before loading ML models in a dedicated process. Do not install
    alongside arbitrary concurrent tasks; changes to `socket` are global.
    """
    def __init__(self):
        self.attempts: list[dict] = []
        self._originals: dict[str, object] = {}
        self._lock = Lock()
        self.active = False

    def _deny(self, operation: str, destination: object):
        with self._lock:
            self.attempts.append({"operation": operation, "destination": str(destination)[:200], "blocked": True})
        raise PermissionError(f"OpenHear Python guard: non-loopback {operation} denied")

    def activate(self):
        if self.active:
            return self
        originals = {
            'connect': socket.socket.connect,
            'connect_ex': socket.socket.connect_ex,
            'sendto': socket.socket.sendto,
            'getaddrinfo': socket.getaddrinfo,
        }
        if hasattr(socket.socket, 'sendmsg'):
            originals['sendmsg'] = socket.socket.sendmsg
        self._originals = originals
        guard = self

        def connect(sock, address):
            if not _allowed_address(address, sock.family):
                guard._deny('connect', address)
            return originals['connect'](sock, address)

        def connect_ex(sock, address):
            if not _allowed_address(address, sock.family):
                guard._deny('connect_ex', address)
            return originals['connect_ex'](sock, address)

        def sendto(sock, data, *args):
            # socket.sendto(data, address) or socket.sendto(data, flags, address)
            if args and not _allowed_address(args[-1], sock.family):
                guard._deny('sendto', args[-1])
            return originals['sendto'](sock, data, *args)

        def getaddrinfo(host, *args, **kwargs):
            if not _allowed_host(host):
                guard._deny('getaddrinfo', host)
            return originals['getaddrinfo'](host, *args, **kwargs)

        socket.socket.connect = connect
        socket.socket.connect_ex = connect_ex
        socket.socket.sendto = sendto
        socket.getaddrinfo = getaddrinfo
        if 'sendmsg' in originals:
            def sendmsg(sock, buffers, ancdata=(), flags=0, address=None):
                if address is not None and not _allowed_address(address, sock.family):
                    guard._deny('sendmsg', address)
                if address is None:
                    return originals['sendmsg'](sock, buffers, ancdata, flags)
                return originals['sendmsg'](sock, buffers, ancdata, flags, address)
            socket.socket.sendmsg = sendmsg
        self.active = True
        return self

    def deactivate(self):
        if not self.active:
            return
        for key, original in self._originals.items():
            if key == 'getaddrinfo':
                socket.getaddrinfo = original
            else:
                setattr(socket.socket, key, original)
        self.active = False

    def __enter__(self):
        return self.activate()

    def __exit__(self, *_):
        self.deactivate()


def local_route_observation():
    """Advisory observation only: a route is not proof of connectivity or egress."""
    output = {"os": platform.system(), "default_route_observed": None,
              "network_interface_state": "not independently inspected"}
    if platform.system() != 'Darwin':
        return output
    try:
        r = subprocess.run(['/sbin/route', '-n', 'get', 'default'], capture_output=True,
                           text=True, timeout=5, check=False)
        output['default_route_observed'] = r.returncode == 0 and 'gateway:' in r.stdout
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        r = subprocess.run(['/usr/sbin/scutil', '--nwi'], capture_output=True,
                           text=True, timeout=5, check=False)
        output['network_interface_state'] = 'scutil reports network paths' if 'Network interfaces:' in r.stdout else 'scutil returned no recognized network path listing'
    except (OSError, subprocess.TimeoutExpired):
        pass
    return output


def snapshot(models_dir: str, *, include_verification=True):
    from .models import verify_manifest
    manifest = Path(models_dir).expanduser().resolve() / 'models.lock.json'
    return {
        'schema_version': '0.4-offline-snapshot',
        'utc': datetime.now(timezone.utc).isoformat(),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'network_observation': local_route_observation(),
        'model_manifest_sha256': hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.is_file() else None,
        'model_verification': 'pass' if not verify_manifest(models_dir) else 'fail',
        'evidence_class': 'host-self-report; NOT an independent network capture',
        'limitations': ['Python socket hooks are not a native OS firewall',
                        'Native libraries, subprocesses and browser network calls are outside this guard',
                        'Functional no-WAN evidence requires physical/OS link isolation'],
    }


def _write_exclusive(destination: str, record: dict):
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2)
        handle.write('\n')
    return str(path)


def run_bilateral_trial(bridge_factory, inputs: dict[str, str], guard: PythonEgressGuard | None = None):
    """Run two local directions; retain setup/errors without saving plaintext."""
    from .core import sha256_file
    import tempfile
    guard = guard if guard is not None else PythonEgressGuard()
    trials = []
    directions = [('en', 'zh'), ('zh', 'en')]
    with guard:
        init_error = None
        try:
            bridge = bridge_factory()
        except Exception as exc:
            init_error = exc
            bridge = None
        for source, target in directions:
            audio_path = inputs[source]
            record = {'direction': f'{source}-{target}', 'input_sha256': sha256_file(audio_path),
                      'status': 'error', 'timings_ms': None, 'tts_status': None}
            if init_error is not None:
                record['error_stage'] = 'model_initialization'
                record['error_class'] = type(init_error).__name__
                record['error_summary'] = str(init_error)[:300]
                trials.append(record)
                continue
            try:
                with tempfile.TemporaryDirectory(prefix='openhear-offline-') as tmp:
                    destination = str(Path(tmp) / 'translated.wav') if bridge.tts is not None else None
                    result = bridge.process(audio_path, source, target, destination)
                    record.update({'status': result.status, 'timings_ms': result.timings_ms,
                                   'tts_status': result.tts_status,
                                   'translation_sha256': hashlib.sha256(result.translation.encode()).hexdigest(),
                                   'transcript_sha256': hashlib.sha256(result.transcript.encode()).hexdigest()})
                    if destination and Path(destination).is_file() and result.tts_status == 'generated':
                        record['tts_wav_sha256'] = sha256_file(destination)
            except Exception as error:
                record['error_stage'] = 'inference'
                record['error_class'] = type(error).__name__
                record['error_summary'] = str(error)[:300]
            trials.append(record)
    return {'schema_version': '0.4-offline-trial', 'utc': datetime.now(timezone.utc).isoformat(),
            'trials': trials, 'python_non_loopback_attempts': guard.attempts,
            'functional_pass': all(x['status'] in ('completed', 'completed_text_only') for x in trials),
            'python_egress_gate_pass': not guard.attempts,
            'evidence_class': 'Python socket guard plus local-model trial, NOT native network capture',
            'limitations': ['Run with all host WAN interfaces disconnected for functional no-WAN gate',
                            'No claim about native-library, subprocess, browser or OS-service egress',
                            'SHA256 manifests are not independently signed']}
