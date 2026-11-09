# OpenHear Bridge 0.4.0 — Architecture and Trust Boundaries

## Model and licence boundary

Model checkpoints are prepared **online only** by the explicit `prepare` CLI. Inference (`serve`, `translate`, `benchmark run`, `offline trial`) loads local paths with Hugging Face offline environment flags and checks a file-hash manifest. No model weights, medical voice profiles, recordings or third-party SDK files ship in the ZIP. A SHA-256 manifest is not a signed software supply-chain attestation.

## Two supported capture modes

1. **Push-to-talk (M1.1 preserved)**: consented browser microphone → client WAV → loopback multipart HTTP → duration/decoding gates → Whisper ASR → Marian MT → macOS `say` TTS → user-controlled playback. The browser may export audio and JSON explicitly; server temporary files are removed on completion.
2. **Continuous turn-based (M2-alpha)**: consented browser microphone → AudioWorklet (fallback ScriptProcessor) → local RMS threshold and pre-roll buffer → provisional repeated full-window ASR previews → silence-or-max-duration commit → same final M1.1 inference path. A bounded queue (at most two pending turns) prevents unbounded backlog. Every completed turn is labelled, shown in history, and eligible for user-initiated playback. Automatic playback is opt-in and not guaranteed to avoid mic feedback.

Provisional previews and final results are serialized against one model inference lock. They may contend; previews can be dropped. The browser retries a bounded number of final requests on 429 and reports failure if still busy. This is **not** causal token-level S2ST, separate speaker diarisation, medically validated intelligibility or group translation.

## Evidence chain

- Source WAV validation limits each file to a short 29-second maximum and at most 16 MiB; the continuous mode commits turns after up to about 11 seconds.
- `Bridge.process` records `perf_counter` ASR/MT/TTS timings. Browser live reports additionally record `pause_to_result_ms` and `request_ms`. These are not calibrated microphone-to-earbud delays.
- Recent 20 live turns retain browser-held text and audio URLs. Up to 500 metadata-only audit records support longer research sessions. Old browser history gets purged; exports require a deliberate user action.
- `openhear live-report` aggregates directional P50/P95/P99, TTS completion, failures and dropped turns. `audit_overflow>0` marks the report incomplete for all-turn analysis.
- `openhear offline snapshot` observes model hashes and OS network state; `offline trial` guards *Python socket operations* and runs actual bilateral inference while retaining input/output hashes but not transcript text.

## Security and limits

The service listens only on loopback; browser assets have no CDN; same-origin POST checks reject common cross-origin form/JS requests. Local malware, privileged applications, extensions and untrusted user accounts are outside this research prototype's security boundary. The Python egress guard patches Python socket APIs only: it does **not** constrain native C extensions, macOS services, subprocesses or the browser. Host-native egress verification must be conducted independently.

## Research progression

M1.1 bilateral speech output → M1.2 offline functional evidence → M2-alpha segmented continuous-turn experience → M2-beta streaming semantic commits and validated feedback suppression (proposed, not implemented) → future multi-user LAN routing (not implemented) → separate governance review before accessibility/medical research.
