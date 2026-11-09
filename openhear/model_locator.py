"""Discover *existing*, verified model caches without moving or downloading weights.

Discovery is deliberately read-only. A discovered manifest is not trusted merely
because it exists: select_model_root verifies every listed file before selecting.
Conflicting model manifests are never silently resolved to an arbitrary revision.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

_SKIP = frozenset({'.git', '.venv', 'node_modules', '__pycache__',
                   '.cache', 'site-packages', 'dist', 'build'})


class ModelDiscoveryError(RuntimeError):
    pass


def search_roots() -> list[Path]:
    """Priority: explicit OPENHEAR_MODEL_DIR, cwd/models, user Downloads.

    Only direct paths and the user's Downloads tree are searched. The user's
    entire home directory is *not* traversed.
    """
    roots = []
    supplied = os.environ.get('OPENHEAR_MODEL_DIR', '').strip()
    if supplied:
        roots.append(Path(supplied).expanduser())
    roots.extend((Path.cwd() / 'models', Path.home() / 'Downloads'))
    return list(dict.fromkeys(x.absolute() for x in roots))


def manifest_candidates(roots: list[Path] | None = None, *, max_depth: int = 14,
                        max_candidates: int = 80) -> list[Path]:
    roots = roots if roots is not None else search_roots()
    candidates = set()
    for entry in roots:
        entry = Path(entry).expanduser()
        if entry.is_file() and entry.name == 'models.lock.json':
            candidates.add(entry.resolve().parent)
            continue
        if not entry.is_dir():
            continue
        if (entry / 'models.lock.json').is_file():
            candidates.add(entry.resolve())
        if entry.name == 'models':
            # A direct models root should not cause a scan of the entire model cache.
            continue
        base_depth = len(entry.parts)
        for current, dirs, files in os.walk(entry, followlinks=False):
            depth = len(Path(current).parts) - base_depth
            dirs[:] = sorted(d for d in dirs if d not in _SKIP and
                             not (Path(current) / d).is_symlink()) if depth < max_depth else []
            if 'models.lock.json' in files:
                candidates.add(Path(current).resolve())
                if len(candidates) >= max_candidates:
                    raise ModelDiscoveryError(
                        f'Exceeded {max_candidates} model manifests; specify --models-dir explicitly')
    return sorted(candidates, key=lambda x: (len(x.parts), str(x)))


def verified_candidates(roots: list[Path] | None = None):
    from .models import verify_manifest
    results = []
    for root in manifest_candidates(roots):
        manifest = root / 'models.lock.json'
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
        try:
            issues = verify_manifest(str(root))
        except (OSError, ValueError, KeyError) as exc:
            issues = [f'{type(exc).__name__}: {exc}']
        results.append({'path': str(root), 'manifest_sha256': digest,
                        'valid': not issues, 'problems': issues[:5]})
    return results


def select_model_root(roots: list[Path] | None = None) -> Path:
    results = verified_candidates(roots)
    valid = [item for item in results if item['valid']]
    if not valid:
        raise ModelDiscoveryError(
            'No complete verified model cache found. Run `models locate` to inspect existing '
            'paths; do not redownload unless the files are genuinely absent.')
    distinct_manifests = {item['manifest_sha256'] for item in valid}
    if len(distinct_manifests) > 1:
        options = '\n'.join(f"  {item['path']} (SHA256 {item['manifest_sha256'][:12]}...)"
                            for item in valid)
        raise ModelDiscoveryError('Multiple different VERIFIED model revisions found. '
                                  'Specify --models-dir explicitly:\n' + options)
    # Different copies of the same manifest can safely use the shortest accessible path.
    return Path(min(valid, key=lambda item: (len(Path(item['path']).parts),
                                             len(item['path']), item['path']))['path'])


def resolve_model_root(value: str) -> str:
    if value != 'auto':
        return str(Path(value).expanduser())
    return str(select_model_root())
