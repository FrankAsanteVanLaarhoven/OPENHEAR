"""CLI for explicit model downloads, offline inference, verification and local UI."""
from __future__ import annotations
import argparse
import json
import os
import platform
from pathlib import Path


def offline_env():
    # Applied before importing Transformers/HF Hub in inference runtime.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["DO_NOT_TRACK"] = "1"


def create_runtime(models_dir: str, device: str, use_tts: bool, *,
                   voice_en: str | None = None, voice_zh: str | None = None,
                   speech_rate: int = 175, tts_policy: str = "strict"):
    offline_env()
    from .adapters import LocalWhisperASR, LocalMarianMT, MacOSSayTTS
    from .core import Bridge
    from .models import model_dirs
    paths = model_dirs(models_dir)
    asr = LocalWhisperASR(paths["whisper"], device=device)
    translator = LocalMarianMT(paths["en-zh"], paths["zh-en"], device=device)
    if not use_tts:
        tts = None
    else:
        try:
            tts = MacOSSayTTS(voice_en=voice_en, voice_zh=voice_zh,
                              speech_rate=speech_rate)
        except (RuntimeError, OSError, ValueError) as error:
            if tts_policy != "best-effort":
                raise
            reason = str(error)
            class UnavailableSpeech:
                def synthesize(self, text: str, language: str, output_path: str) -> str:
                    raise RuntimeError(f"Native voice initialization unavailable: {reason}")
            tts = UnavailableSpeech()
    return Bridge(asr, translator, tts, tts_policy=tts_policy)


def doctor(models_dir: str):
    from .models import model_dirs
    state = {"python": platform.python_version(), "platform": platform.platform(), "models_dir": str(Path(models_dir).expanduser().resolve()), "models": {}, "mps": None, "voices": []}
    for label, dirname in model_dirs(models_dir).items():
        state["models"][label] = (Path(dirname) / "config.json").exists()
    try:
        import torch
        state["torch"] = torch.__version__
        state["mps"] = bool(torch.backends.mps.is_available())
    except ImportError:
        state["torch"] = "NOT INSTALLED"
    try:
        from importlib.metadata import version
        state["transformers"] = version("transformers")
    except Exception:
        state["transformers"] = "NOT INSTALLED"
    if platform.system() == "Darwin":
        try:
            from .adapters import MacOSSayTTS
            state["voices"] = [{"name": name, "locale": locale} for name, locale in MacOSSayTTS().available if locale.lower().startswith(("en", "zh"))]
        except Exception as error:
            state["voices_error"] = str(error)
    print(json.dumps(state, indent=2))
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(prog="openhear", description="OpenHear Bridge M1.2/M2-alpha offline-audited conversation research prototype")
    commands = parser.add_subparsers(dest="action", required=True)
    models = commands.add_parser("models", help="Find existing model caches without downloading")
    model_commands = models.add_subparsers(dest="model_action", required=True)
    locate = model_commands.add_parser("locate", help="Inspect cached model manifests and select an intact cache")
    locate.add_argument("--search-root", action="append", default=None,
                        help="Restrict discovery to the supplied directory; may repeat")
    locate.add_argument("--print-path", action="store_true", help="Print only the selected model directory")
    prepare = commands.add_parser("prepare", help="ONLINE ONLY: download approved checkpoints and write hash manifest")
    prepare.add_argument("--models-dir", default="models")
    prepare.add_argument("--whisper", choices=["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small"], default="openai/whisper-base")
    check = commands.add_parser("doctor", help="Show local runtime/model and Mac speech voice status")
    check.add_argument("--models-dir", default="models")
    verify = commands.add_parser("verify", help="Check cached model files against downloaded hashes")
    verify.add_argument("--models-dir", default="models")
    translate = commands.add_parser("translate", help="OFFLINE: translate real local WAV and optionally synthesize speech")
    translate.add_argument("audio")
    translate.add_argument("--source", required=True, choices=["en", "zh"])
    translate.add_argument("--target", required=True, choices=["en", "zh"])
    translate.add_argument("--models-dir", default="models")
    translate.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    translate.add_argument("--no-tts", action="store_true")
    translate.add_argument("--output-audio", default="translated.wav")
    translate.add_argument("--evidence", default="evidence.json")
    serve = commands.add_parser("serve", help="OFFLINE: start loopback-only browser microphone console")
    serve.add_argument("--models-dir", default="models")
    serve.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    serve.add_argument("--no-tts", action="store_true")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--python-guard", action="store_true", help="Deny Python socket non-loopback egress; not an OS firewall")
    live_report = commands.add_parser("live-report", help="Summarise exported continuous-turn session: P50/P95/P99, failures, drops")
    live_report.add_argument("--session", required=True)
    live_report.add_argument("--output", required=True)
    offline = commands.add_parser("offline", help="Offline assurance: host snapshot and guarded bilateral trial")
    offline_commands = offline.add_subparsers(dest="offline_action", required=True)
    offline_snapshot = offline_commands.add_parser("snapshot", help="Record host routes and model integrity without inference")
    offline_snapshot.add_argument("--models-dir", default="models")
    offline_snapshot.add_argument("--output", required=True)
    offline_trial = offline_commands.add_parser("trial", help="Run en→zh + zh→en with Python non-loopback socket guard")
    offline_trial.add_argument("--models-dir", default="models")
    offline_trial.add_argument("--en-audio", required=True, help="English consented WAV")
    offline_trial.add_argument("--zh-audio", required=True, help="Mandarin consented WAV")
    offline_trial.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    offline_trial.add_argument("--no-tts", action="store_true")
    offline_trial.add_argument("--require-tts", action="store_true", help="Fail when output speech is unavailable")
    offline_trial.add_argument("--disconnected-confirmed", action="store_true", help="User attests all WAN uplinks were disabled; not independently verified")
    offline_trial.add_argument("--output", required=True)
    offline_preflight = offline_commands.add_parser("preflight", help="Validate speech WAVs, model hashes and local TTS voices before disconnecting")
    offline_preflight.add_argument("--models-dir", default="models")
    offline_preflight.add_argument("--en-audio", required=True)
    offline_preflight.add_argument("--zh-audio", required=True)
    offline_preflight.add_argument("--no-tts", action="store_true", help="Do not require system speech voices")
    offline_preflight.add_argument("--output", required=True, help="Create-only JSON preflight report")
    voices = commands.add_parser("voices", help="List macOS voices; does not require AI models")
    speak = commands.add_parser("speak", help="Create and optionally play native speech (no AI models needed)")
    speak.add_argument("--target", choices=["en", "zh"], required=True)
    speak.add_argument("--text", required=True)
    speak.add_argument("--output", default="speech-test.wav")
    speak.add_argument("--voice", default=None)
    speak.add_argument("--rate", type=int, default=175)
    speak.add_argument("--play", action="store_true")
    for command in (serve, translate):
        command.add_argument("--voice-en", default=None, help="Override installed English system voice")
        command.add_argument("--voice-zh", default=None, help="Override installed Mandarin system voice")
        command.add_argument("--speech-rate", type=int, default=175)
    benchmark = commands.add_parser("benchmark", help="Feasibility dataset template and local evaluation")
    subs = benchmark.add_subparsers(dest="bench_action", required=True)
    init = subs.add_parser("init", help="Create 20 empty record slots per direction (no fabricated data)")
    init.add_argument("--dataset", default="benchmarks/utterances.csv")
    init.add_argument("--per-direction", type=int, default=20)
    run = subs.add_parser("run", help="Evaluate completed CSV records with local models")
    run.add_argument("--dataset", default="benchmarks/utterances.csv")
    run.add_argument("--models-dir", default="models")
    run.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    run.add_argument("--report", default="benchmark-report.json")
    run.add_argument("--allow-incomplete", action="store_true")
    calibrate = commands.add_parser("calibrate", help="Acoustic hardware loopback calibration")
    calibrate.add_argument("--chirp-out", default=None, help="Generate calibration chirp WAV file")
    calibrate.add_argument("--test-latency", action="store_true", help="Run synthetic loopback validation")
    metrics = commands.add_parser("metrics", help="Evaluate S2ST scientific metrics (AL, LAAL, BLEU)")
    metrics.add_argument("--delays", nargs="+", type=float, default=[200.0, 450.0, 700.0], help="Delays in ms")
    metrics.add_argument("--src-len", type=int, default=10, help="Source length")
    metrics.add_argument("--tgt-len", type=int, default=8, help="Target length")
    metrics.add_argument("--ref", default="", help="Reference translation")
    metrics.add_argument("--hyp", default="", help="Hypothesis translation")
    guard = commands.add_parser("guardrails", help="Verify text against autoregressive hallucination patterns")
    guard.add_argument("--text", required=True, help="Text to verify")
    args = parser.parse_args(argv)
    if args.action == "models" and args.model_action == "locate":
        from .model_locator import manifest_candidates, verified_candidates, select_model_root, ModelDiscoveryError
        roots = [Path(x).expanduser() for x in args.search_root] if args.search_root else None
        try:
            if not args.print_path:
                print(json.dumps({"candidates": verified_candidates(roots)}, indent=2))
            selected = select_model_root(roots)
        except ModelDiscoveryError as error:
            raise SystemExit(f"Model discovery failed: {error}") from error
        print(str(selected) if args.print_path else f"SELECTED VERIFIED MODEL DIRECTORY: {selected}")
        return
    if getattr(args, "models_dir", None) == "auto":
        from .model_locator import resolve_model_root, ModelDiscoveryError
        try:
            args.models_dir = resolve_model_root("auto")
        except ModelDiscoveryError as error:
            raise SystemExit(f"Model discovery failed: {error}") from error
        print(f"Using existing verified models: {args.models_dir}")
    if args.action == "doctor":
        doctor(args.models_dir)
        return
    if args.action == "live-report":
        from .live_report import report_file
        path, result = report_file(args.session, args.output)
        print("Live session summary:", path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.action == "offline":
        from .offline import snapshot, _write_exclusive
        if args.offline_action == "preflight":
            from .preflight import preflight
            record = preflight(args.models_dir, args.en_audio, args.zh_audio,
                               require_tts=not args.no_tts)
            result_path = _write_exclusive(args.output, record)
            print(f"Offline experiment preflight evidence: {result_path}")
            print(json.dumps({"preflight_ready": record["preflight_ready"],
                              "model_verification": record["model_verification"],
                              "wav_valid": {item["label"]: item["valid_audio"] for item in record["inputs"]},
                              "voices": record["macos_voices"]}, indent=2))
            if not record["preflight_ready"]:
                raise SystemExit("Preflight not ready; see the evidence JSON for missing inputs/voices")
            return
        if args.offline_action == "snapshot":
            record = snapshot(args.models_dir)
            print("Offline host observation:", _write_exclusive(args.output, record))
            return
        from .models import verify_manifest
        from .offline import PythonEgressGuard, run_bilateral_trial
        if args.no_tts and args.require_tts:
            parser.error("--no-tts conflicts with --require-tts")
        offline_env()
        problems = verify_manifest(args.models_dir)
        if problems:
            raise SystemExit("Model cache verification failed: " + "; ".join(problems[:3]))
        from .audio import load_speech
        for path in [args.en_audio, args.zh_audio]:
            load_speech(path)  # verify before invoking models / activating guarded trial
        guard = PythonEgressGuard()
        record = snapshot(args.models_dir)
        record.update(run_bilateral_trial(
            lambda: create_runtime(args.models_dir, args.device, not args.no_tts,
                                   tts_policy="best-effort"),
            {"en": args.en_audio, "zh": args.zh_audio}, guard))
        record["user_attests_disconnected_wan"] = bool(args.disconnected_confirmed)
        record["functional_pass"] = record["functional_pass"] and (not args.require_tts or
            all(x["tts_status"] == "generated" for x in record["trials"]))
        record["self_attested_no_wan_trial_pass"] = bool(args.disconnected_confirmed and record["functional_pass"] and record["python_egress_gate_pass"])
        out = _write_exclusive(args.output, record)
        print("Offline trial evidence:", out)
        print(json.dumps({k:record[k] for k in ("functional_pass", "python_egress_gate_pass", "user_attests_disconnected_wan")}, indent=2))
        if not record["functional_pass"] or not record["python_egress_gate_pass"]:
            raise SystemExit("Offline trial has a failed gate; see evidence JSON")
        return
    if args.action == "voices":
        from .adapters import MacOSSayTTS
        tts = MacOSSayTTS()
        selected = {}
        for lang in ("en", "zh"):
            try:
                selected[lang] = tts.voice_for(lang)
            except RuntimeError:
                selected[lang] = None
        print(json.dumps({"available": [{"voice": n, "locale": loc} for n, loc in tts.available],
                          "selected": selected}, ensure_ascii=False, indent=2))
        return
    if args.action == "speak":
        from .adapters import MacOSSayTTS, play_wav_macos
        overrides = {args.target: args.voice} if args.voice else {}
        tts = MacOSSayTTS(voice_en=overrides.get("en"),
                          voice_zh=overrides.get("zh"), speech_rate=args.rate)
        output = tts.synthesize(args.text, args.target, args.output)
        print(f"Native speech WAV created: {output} (voice {tts.voice_for(args.target)})")
        if args.play:
            play_wav_macos(output)
        return
    if args.action == "benchmark" and args.bench_action == "init":
        from .benchmark import init_manifest
        print("Created empty research dataset:", init_manifest(args.dataset, args.per_direction))
        return
    if args.action == "prepare":
        # Preparing is the ONLY path permitted to contact Hugging Face.
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        from .models import prepare as run_prepare
        print(f"Model manifest written to: {run_prepare(args.models_dir, args.whisper)}")
        return
    if args.action == "verify":
        from .models import verify_manifest
        failures = verify_manifest(args.models_dir)
        if failures:
            for item in failures:
                print("FAIL", item)
            raise SystemExit(1)
        print("PASS All files listed in the model manifest match their SHA256 hashes (not a signature).")
        return
    offline_env()
    if args.action == "benchmark" and args.bench_action == "run":
        from .models import verify_manifest
        from .benchmark import evaluate_manifest, save_report
        failures = verify_manifest(args.models_dir)
        if failures:
            raise SystemExit("Model cache verification failed: " + "; ".join(failures[:3]))
        engine = create_runtime(args.models_dir, args.device, False)
        evidence = evaluate_manifest(args.dataset, engine,
                                     model_manifest_path=str(Path(args.models_dir) / "models.lock.json"))
        path = save_report(evidence, args.report)
        print("Evidence saved:", path)
        print(json.dumps(evidence["summary"], ensure_ascii=False, indent=2))
        if not args.allow_incomplete and any(
            group["pending_or_missing"] or group["failed"] for group in evidence["summary"].values()
        ):
            raise SystemExit("Benchmark is incomplete or has failed trials; inspect report (exit code 2)")
        return
    if args.action == "translate":
        from .core import evidence_record
        from .models import verify_manifest
        failures = verify_manifest(args.models_dir)
        if failures:
            raise SystemExit("Model cache verification failed: " + "; ".join(failures[:3]))
        engine = create_runtime(args.models_dir, args.device, not args.no_tts,
                                voice_en=args.voice_en, voice_zh=args.voice_zh,
                                speech_rate=args.speech_rate, tts_policy="strict")
        result = engine.process(args.audio, args.source, args.target,
                                None if args.no_tts else args.output_audio)
        record = evidence_record(result, args.audio)
        record["execution_mode"] = "local-models"
        record["tts_backend"] = "macos-say" if not args.no_tts else "disabled"
        record["inference_device_requested"] = args.device
        record["model_manifest"] = str(Path(args.models_dir).expanduser().resolve() / "models.lock.json")
        Path(args.evidence).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(record, ensure_ascii=False, indent=2))
        return
    if args.action == "calibrate":
        import numpy as np
        from .calibration import generate_calibration_chirp, detect_loopback_latency_ms, evaluate_latency_budget
        from .audio import write_wav_bytes
        chirp = generate_calibration_chirp()
        if args.chirp_out:
            Path(args.chirp_out).write_bytes(write_wav_bytes(chirp, 16000))
            print(f"Calibration chirp written: {args.chirp_out}")
        if args.test_latency:
            # Synthetic 120ms offset loopback verification
            offset = int(16000 * 0.12)
            padded = np.pad(chirp, (offset, 1600), mode='constant')
            delay_ms, conf = detect_loopback_latency_ms(chirp, padded)
            budget = evaluate_latency_budget(delay_ms)
            print(json.dumps({"measured_delay_ms": delay_ms, "confidence": conf, "budget": budget}, indent=2))
        return
    if args.action == "metrics":
        from .metrics import compute_al, compute_laal, sentence_bleu
        al = compute_al(args.delays, args.src_len, args.tgt_len)
        laal = compute_laal(args.delays, args.src_len, args.tgt_len)
        result = {"al_ms": round(al, 2), "laal_ms": round(laal, 2)}
        if args.ref and args.hyp:
            result["bleu"] = sentence_bleu(args.ref, args.hyp)
        print(json.dumps(result, indent=2))
        return
    if args.action == "guardrails":
        from .guardrails import is_phantom_hallucination, sanitize_hypothesis
        is_hallucination, reason = is_phantom_hallucination(args.text)
        sanitized = sanitize_hypothesis(args.text)
        print(json.dumps({"text": args.text, "is_hallucination": is_hallucination, "reason": reason, "sanitized": sanitized}, indent=2))
        return
    if args.action == "serve":
        from .models import verify_manifest
        failures = verify_manifest(args.models_dir)
        if failures:
            raise SystemExit("Model cache verification failed: " + "; ".join(failures[:3]))
        if not 1 <= args.port <= 65535:
            parser.error("--port must be 1..65535")
        if args.python_guard:
            from .offline import PythonEgressGuard
            guard = PythonEgressGuard().activate()
            print("Python non-loopback socket guard active (NOT an OS firewall)")
        import uvicorn
        from .web import create_app
        app = create_app(lambda: create_runtime(args.models_dir, args.device, not args.no_tts,
                                                 voice_en=args.voice_en,
                                                 voice_zh=args.voice_zh,
                                                 speech_rate=args.speech_rate,
                                                 tts_policy="best-effort"))
        print(f"Open locally: http://127.0.0.1:{args.port}/ (no public network binding)")
        uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
