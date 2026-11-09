# M2.1 Mac acceptance protocol — 8 November 2026

**Scope:** Continuation of the existing OpenHear Bridge M2-alpha. The new operations patch fixes model-location ambiguity and supports repeatable experiment preflight; it **does not change Whisper, Marian, VAD, voice generation, or safety claims**. No Mac access is available to the build environment.

## Model cache: no new download

The user's previously verified model manifest is nested inside a folder with spaces, while the M2 `models` symbolic link is broken. Instead of recreating links or downloading new checkpoints, run:

```bash
python -m openhear.cli models locate --search-root "$HOME/Downloads"
python -m openhear.cli verify --models-dir auto
bash scripts/mac-launch.sh
```

Auto discovery searches a direct `OPENHEAR_MODEL_DIR` environment override (if set), `./models` and `~/Downloads`. It reads **only** folders containing `models.lock.json`; selected files must match SHA-256. If multiple different **valid** model manifests exist, the command refuses to choose arbitrarily: pass an explicit `--models-dir` with the intended path. Duplicate copies of the identical manifest are acceptable. File hashes are not signatures, licence clearance or proof of safety.

## M1.1 integration and M2 capture (manual on Mac)

1. Open `http://127.0.0.1:8765` after launch. Confirm consent and macOS microphone permission.
2. In push-to-talk mode, record a short original **English** utterance, choose English→Mandarin, translate, review displayed transcript, translation and **playable Mandarin WAV**. Save the recorded source WAV as `english.wav`; do **not** substitute the macOS-generated TTS test WAV.
3. Switch direction, record a short **Mandarin** utterance, review translation and **playable English WAV**, save the source as `mandarin.wav`.
4. Test **continuous conversation mode** with at least 10 turns in each direction. Speak, pause, monitor provisional captions, and confirm final history/voice output. Export JSON after each direction, then use `python -m openhear.cli live-report --session <actual filename> --output <new evidence path>`. Do not claim causal S2ST; this is turn-based VAD.
5. Document user-perceived audibility and correctness. The exported JSON alone does **not** prove audible audio or meaning fidelity.

## Disconnected bilateral trial

While connected, place the two actual microphone WAVs in the project root or supply absolute paths. **Stop the browser server with Ctrl+C** before running:

```bash
bash scripts/mac-offline-trial.sh "${PWD}/english.wav" "${PWD}/mandarin.wav"
```

The script verifies the already cached models, checks both source WAVs and installed English/Chinese system voices, writes a connected snapshot, then asks the operator to disconnect every WAN route. It records a second snapshot and invokes the existing Python-socket-guarded trial.

**Do not** enter `DISCONNECTED` until Wi-Fi, tethering, Ethernet and other uplinks are genuinely disabled. The resulting flag is an **operator attestation**. A controlled host-wide packet trace and process attribution are still required to support any claim of zero external network activity.

## Acceptance matrix

| Gate | Target | Evidence | State before user executes |
|---|---|---|---|
| M2.1 packaging regression | Old 96 + new operations tests pass | Build test log | Build environment only |
| Cache integrity | All listed hashes match | `verify` output | Previously demonstrated on Mac, new discovery untested |
| Full bilingual speech-to-speech | 1+ genuine microphone trial per direction with independently heard output | Saved WAV, UI screenshot, operator review | Pending |
| Continuous turn mode | ≥10 turns per direction, errors and queue drops retained | Exported live JSON and report | Pending |
| Functional no-WAN | Both translations and TTS under actual physical disconnection | Bilateral trial JSON plus operator note | Pending |
| Host-wide zero egress | Independent, process-attributed network observation | Network capture / firewall logs | Not implemented here |
| Performance & impact | P50/P95/P99 and blinded adequacy on preregistered corpus | Bench reports | Pending |

Suggested gates are **research objectives, not measured achievements**. Keep reports and WAVs private; the project code is nonmedical and should not be used for clinical consent, legal translation or safety-critical instructions.
