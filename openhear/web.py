"""Bounded, local-only browser interface. No cloud, CDN, or recording retention."""
import base64
import json
import tempfile
import platform
import threading
from pathlib import Path

from .audio import MAX_AUDIO_BYTES, validate_uploaded_wav


class BusyError(Exception):
    pass


class Runtime:
    def __init__(self, factory):
        self.factory = factory
        self.bridge = None
        self.lock = threading.Lock()

    def preview(self, wav: bytes, source: str):
        if not self.lock.acquire(blocking=False):
            raise BusyError("Another local inference request is in progress")
        try:
            if self.bridge is None:
                self.bridge = self.factory()
            with tempfile.TemporaryDirectory(prefix="openhear-preview-") as folder:
                audio = Path(folder) / "preview.wav"
                audio.write_bytes(wav)
                import time
                started = time.perf_counter()
                transcript = self.bridge.asr.transcribe(str(audio), source).strip()
                return {"schema_version": "0.4", "transcript": transcript,
                        "source_language": source, "provisional": True,
                        "timings_ms": {"asr": round((time.perf_counter()-started)*1000, 2)},
                        "warning": "Unstable whole-window Whisper preview; not causal/token streaming"}
        finally:
            self.lock.release()

    def translate(self, wav: bytes, source: str, target: str):
        if not self.lock.acquire(blocking=False):
            raise BusyError("Another local inference request is in progress")
        try:
            if self.bridge is None:
                self.bridge = self.factory()  # lazily load local model weights
            with tempfile.TemporaryDirectory(prefix="openhear-") as folder:
                audio = Path(folder) / "input.wav"
                audio.write_bytes(wav)
                target_audio = Path(folder) / "output.wav" if self.bridge.tts is not None else None
                result = self.bridge.process(str(audio), source, target,
                                             str(target_audio) if target_audio else None)
                audio_bytes = target_audio.read_bytes() if result.tts_status == "generated" and target_audio and target_audio.is_file() else None
                if audio_bytes is not None and len(audio_bytes) > 8 * 1024 * 1024:
                    raise RuntimeError("Generated speech exceeds audio response limit")
                encoded = base64.b64encode(audio_bytes).decode() if audio_bytes else None

                # SOTA anti-hallucination and conversational backchannel verification
                from .guardrails import is_phantom_hallucination, sanitize_hypothesis
                from .dialogue import is_backchannel
                is_hal, hal_reason = is_phantom_hallucination(result.transcript)
                backchannel = is_backchannel(result.transcript, source)
                sanitized_transcript = sanitize_hypothesis(result.transcript) if is_hal else result.transcript

                return {"schema_version": "0.4", "request_id": result.request_id,
                        "source_language": source, "target_language": target,
                        "transcript": sanitized_transcript,
                        "raw_transcript": result.transcript,
                        "translation": result.translation,
                        "timings_ms": result.timings_ms,
                        "latency_note": "Software processing only; NOT calibrated physical mouth-to-ear latency",
                        "audio_base64": encoded,
                        "audio_mime": "audio/wav" if encoded else None,
                        "tts_status": result.tts_status,
                        "tts_error": result.tts_error,
                        "status": result.status,
                        "mode": "local-models",
                        "is_hallucination": is_hal,
                        "hallucination_reason": hal_reason,
                        "is_backchannel": backchannel,
                        "guardrail_status": "sanitized" if is_hal else "clean"}
        finally:
            self.lock.release()


def create_app(runtime_factory):
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Request
    from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
    from starlette.concurrency import run_in_threadpool

    runtime = Runtime(runtime_factory)
    app = FastAPI(title="OpenHear Bridge M2-alpha", docs_url=None, redoc_url=None, openapi_url=None)
    html_path = Path(__file__).parent / "static" / "index.html"

    @app.get("/static/{name}")
    def ui_assets(name: str):
        allowed = {"style.css": "text/css; charset=utf-8",
                   "app.js": "text/javascript; charset=utf-8",
                   "live.js": "text/javascript; charset=utf-8",
                   "earbud-3d.js": "text/javascript; charset=utf-8",
                   "capture-worklet.js": "text/javascript; charset=utf-8"}
        if name not in allowed:
            raise HTTPException(404, "Unknown asset")
        return FileResponse(html_path.parent / name, media_type=allowed[name],
                            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"})

    @app.get("/static/fonts/{name}")
    def font_assets(name: str):
        allowed = {"FixelDisplay-Bold.woff2": "font/woff2",
                   "FixelDisplay-SemiBold.woff2": "font/woff2",
                   "FixelText-Bold.woff2": "font/woff2",
                   "FixelText-SemiBold.woff2": "font/woff2",
                   "FixelText-Medium.woff2": "font/woff2",
                   "FixelText-Regular.woff2": "font/woff2"}
        if name not in allowed:
            raise HTTPException(404, "Unknown font asset")
        return FileResponse(html_path.parent / "fonts" / name, media_type=allowed[name],
                            headers={"Cache-Control": "public, max-age=31536000, immutable", "X-Content-Type-Options": "nosniff"})

    @app.get("/", response_class=HTMLResponse)
    def home():
        return HTMLResponse(html_path.read_text(encoding="utf-8"), headers={"Cache-Control": "no-store", "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; media-src blob: 'self'; connect-src 'self'; worker-src 'self'; object-src 'none'", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer", "Permissions-Policy": "microphone=(self)"})

    @app.get("/api/health")
    def health():
        return {"status": "ready" if runtime.bridge is not None else "not_loaded",
                "network": "no model downloads in server mode",
                "tts": bool(runtime.bridge and runtime.bridge.tts)}

    @app.get("/api/voices")
    def voices():
        # Voice status is independent of model downloads and first inference.
        if platform.system() != "Darwin":
            return {"platform": platform.system(), "available": [], "message": "Native TTS requires macOS"}
        try:
            from .adapters import MacOSSayTTS
            tts = MacOSSayTTS()
            preferred = {}
            for lang in ("en", "zh"):
                try:
                    preferred[lang] = tts.voice_for(lang)
                except RuntimeError:
                    preferred[lang] = None
            return {"platform": "Darwin", "available": [{"name": n, "locale": loc}
                                                       for n, loc in tts.available
                                                       if loc.lower().startswith(("en", "zh"))],
                    "preferred": preferred}
        except Exception as error:
            return {"platform": "Darwin", "available": [], "message": str(error)}

    @app.get("/api/stream/config")
    def stream_config():
        from .stream import VADPolicy
        return JSONResponse({"vad": VADPolicy().public(),
                             "mode": "vad-segmented-best-effort",
                             "warning": "Partial ASR is full-window provisional, and final speech is generated after each turn; not simultaneous interpreting"},
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/calibrate/chirp")
    def calibrate_chirp():
        from .calibration import generate_calibration_chirp
        from .audio import write_wav_bytes
        chirp = generate_calibration_chirp()
        return Response(content=write_wav_bytes(chirp, 16000), media_type="audio/wav",
                        headers={"Cache-Control": "no-store"})

    @app.post("/api/calibrate/measure")
    async def calibrate_measure(request: Request, audio: UploadFile = File(...)):
        from .calibration import generate_calibration_chirp, detect_loopback_latency_ms, evaluate_latency_budget
        from .audio import load_speech
        chirp = generate_calibration_chirp()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp:
            content = await audio.read()
            temp.write(content)
            temp.flush()
            try:
                rec = load_speech(temp.name)
                delay_ms, conf = detect_loopback_latency_ms(chirp, rec)
                budget = evaluate_latency_budget(delay_ms)
                return JSONResponse({"measured_delay_ms": delay_ms, "confidence": conf, "budget": budget},
                                    headers={"Cache-Control": "no-store"})
            except Exception as exc:
                raise HTTPException(422, f"Could not analyze loopback audio: {exc}") from exc

    @app.post("/api/lid")
    async def language_id(request: Request, text: str = Form(None)):
        from .lid import detect_script_language, auto_direction_for_input
        if text:
            src, conf = detect_script_language(text)
            s, t = auto_direction_for_input(text)
            return JSONResponse({"detected": src, "confidence": conf, "source": s, "target": t, "direction": f"{s}:{t}"},
                                headers={"Cache-Control": "no-store"})
        return JSONResponse({"detected": "en", "confidence": 0.5, "source": "en", "target": "zh", "direction": "en:zh"},
                            headers={"Cache-Control": "no-store"})

    @app.get("/api/benchmarks/summary")
    def benchmarks_summary():
        bench_file = Path(__file__).parent.parent / "benchmarks" / "benchmark-report.json"
        if bench_file.exists():
            try:
                data = json.loads(bench_file.read_text(encoding="utf-8"))
                return JSONResponse({"available": True, "summary": data.get("summary", {}),
                                     "captured_utc": data.get("captured_utc")},
                                    headers={"Cache-Control": "no-store"})
            except Exception:
                pass
        return JSONResponse({"available": False, "message": "No benchmark report yet"},
                            headers={"Cache-Control": "no-store"})

    @app.post("/api/preview")
    async def preview(request: Request, source: str = Form(...), audio: UploadFile = File(...)):
        origin = request.headers.get("origin")
        allowed = {f"http://127.0.0.1:{request.url.port}",
                   f"http://localhost:{request.url.port}"}
        if origin and origin not in allowed:
            raise HTTPException(403, "Requests must originate from the local OpenHear interface")
        if source not in ("en", "zh"):
            raise HTTPException(422, "Unsupported source language")
        try:
            chunks, n = [], 0
            while True:
                part = await audio.read(1024 * 1024)
                if not part:
                    break
                n += len(part)
                if n > MAX_AUDIO_BYTES:
                    raise HTTPException(413, "Audio exceeds 16 MiB")
                chunks.append(part)
            wav = b"".join(chunks)
            validate_uploaded_wav(wav)
            output = await run_in_threadpool(runtime.preview, wav, source)
            return JSONResponse(output, headers={"Cache-Control": "no-store",
                                                 "X-Content-Type-Options": "nosniff"})
        except BusyError as exc:
            raise HTTPException(429, str(exc)) from exc
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(503, f"Local preview failed: {exc}") from exc
        finally:
            await audio.close()

    @app.post("/api/translate")
    async def translation(request: Request, source: str = Form(...), target: str = Form(...), audio: UploadFile = File(...)):
        # Deny browser cross-origin requests, including malicious local-site submissions.
        origin = request.headers.get("origin")
        allowed = {f"http://127.0.0.1:{request.url.port}",
                   f"http://localhost:{request.url.port}"}
        if origin and origin not in allowed:
            raise HTTPException(403, "Requests must originate from the local OpenHear interface")
        if source not in ("en", "zh") or target not in ("en", "zh") or source == target:
            raise HTTPException(422, "Supported directions: en→zh and zh→en")
        try:
            # Stream into bounded memory, never accept an unbounded request body.
            chunks, n = [], 0
            while True:
                part = await audio.read(1024 * 1024)
                if not part:
                    break
                n += len(part)
                if n > MAX_AUDIO_BYTES:
                    raise HTTPException(413, "Audio exceeds 16 MiB")
                chunks.append(part)
            wav = b"".join(chunks)
            validate_uploaded_wav(wav)
            output = await run_in_threadpool(runtime.translate, wav, source, target)
            return JSONResponse(output, headers={"Cache-Control": "no-store",
                                                 "X-Content-Type-Options": "nosniff"})
        except BusyError as exc:
            raise HTTPException(429, str(exc)) from exc
        except HTTPException:
            raise
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(422, str(exc)) from exc
        except RuntimeError as exc:
            # No full tracebacks or source audio in browser. Server logs are off.
            raise HTTPException(503, f"Local model/TTS runtime failed: {exc}") from exc
        finally:
            await audio.close()

    return app
