"""EXPLICIT ONLINE stage: download known model repositories, record provenance.

Inference never calls this module, and never calls remote APIs.
"""
from __future__ import annotations
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

MODEL_SOURCES = {
    "whisper": "openai/whisper-base",             # multilingual, not whisper-base.en
    "en-zh": "Helsinki-NLP/opus-mt-en-zh",       # model card license Apache-2.0
    "zh-en": "Helsinki-NLP/opus-mt-zh-en",       # model card license CC-BY-4.0
}
WHISPER_ALTERNATIVES = ("openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small")
LICENSE_CARD = {"whisper": "apache-2.0", "en-zh": "apache-2.0", "zh-en": "cc-by-4.0"}


def model_dirs(root: str) -> dict[str, str]:
    return {key: str(Path(root).expanduser().resolve() / key) for key in MODEL_SOURCES}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_for(root: str, revisions: dict | None = None) -> dict:
    resolved = Path(root).expanduser().resolve()
    items = {}
    for key, repo in MODEL_SOURCES.items():
        directory = resolved / key
        if not (directory / "config.json").is_file():
            raise FileNotFoundError(f"Missing {directory / 'config.json'}")
        files = []
        for p in sorted(directory.rglob("*")):
            if p.is_file() and ".cache" not in p.relative_to(directory).parts:
                files.append({"path": str(p.relative_to(directory)), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
        items[key] = {"repo": (revisions or {}).get(key, {}).get("repo", repo),
                      "revision": (revisions or {}).get(key, {}).get("revision"),
                      "model_card_license_unverified": LICENSE_CARD[key], "files": files}
    return {"schema_version": "0.2", "created_utc": datetime.now(timezone.utc).isoformat(),
            "platform": platform.platform(), "models": items,
            "notice": "Metadata for reproducibility; this is not a digital signature, licence clearance, or proof of offline integrity."}


def prepare(root: str, whisper_repo: str = "openai/whisper-base") -> Path:
    if whisper_repo not in WHISPER_ALTERNATIVES:
        raise ValueError(f"Whisper must be one of: {WHISPER_ALTERNATIVES}")
    from huggingface_hub import HfApi, snapshot_download
    rootpath = Path(root).expanduser().resolve()
    rootpath.mkdir(parents=True, exist_ok=True)
    revisions = {}
    for key, original_repo in MODEL_SOURCES.items():
        repo = whisper_repo if key == "whisper" else original_repo
        info = HfApi().model_info(repo)
        print(f"Downloading {repo} at revision {info.sha} into {rootpath / key}", flush=True)
        snapshot_download(
            repo_id=repo, revision=info.sha, local_dir=str(rootpath / key),
            allow_patterns=["*.json", "*.md", "*.txt", "*.spm", "*.model", "*.safetensors", "*.bin", "*.tiktoken"],
            ignore_patterns=["tf_model.*", "flax_model.*", "rust_model.*", "*.msgpack", "*.h5", "*.ot", "onnx/*"]
        )
        revisions[key] = {"repo": repo, "revision": info.sha}
    manifest = manifest_for(str(rootpath), revisions)
    result = rootpath / "models.lock.json"
    result.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def verify_manifest(root: str) -> list[str]:
    parent = Path(root).expanduser().resolve()
    lock = parent / "models.lock.json"
    if not lock.is_file():
        return ["models.lock.json is missing (run prepare while online)"]
    manifest = json.loads(lock.read_text(encoding="utf-8"))
    problems = []
    for required in MODEL_SOURCES:
        if required not in manifest.get("models", {}):
            problems.append(f"Model absent from lockfile: {required}")
    for name, data in manifest.get("models", {}).items():
        if name not in MODEL_SOURCES:
            problems.append(f"Unknown model key: {name}")
            continue
        for item in data.get("files", []):
            target = (parent / name / item["path"]).resolve()
            if not target.is_relative_to((parent / name).resolve()):
                problems.append(f"Unsafe relative path: {item['path']}")
            elif not target.is_file() or sha256_file(target) != item["sha256"]:
                problems.append(f"SHA256 mismatch or missing: {name}/{item['path']}")
    return problems
