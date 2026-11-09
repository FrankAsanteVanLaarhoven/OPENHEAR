# M2-alpha Streaming / Continuous Turn Evaluation

## Experimental question

Can local VAD-assisted continuous capture improve practical bilingual turn-taking while preserving explicit speaker turns, transparent delays and a bounded failure rate over a controlled push-to-talk baseline?

The implementation uses **RMS VAD**, **periodic whole-window Whisper previews**, **final ASR → Marian MT → macOS TTS after pause**. It is *not* a direct simultaneous streaming S2ST model. Preliminary predictions are hypotheses, not results.

## Baselines and design

Use the existing M1.1 push-to-talk mode as a matched baseline. Randomize mode order for an adequately sized speaker-balanced session; pre-register utterance scripts, Mandarin/English accents, microphone distance, background noise and model hashes. Compare at least 20 utterances per direction for feasibility and 100 turns for initial reliability/queue stress. These are suggested feasibility counts, not a statistical power calculation.

Observe four measurement timestamps with the same monotonic clock: `speech_onset`, `vad_pause_commit`, `translation_response`, `audio_play_start`. The current browser already records **pause-to-result** and HTTP request times, but **not calibrated acoustic playback onset**; additional audio-coupler instrumentation is required for scientific mouth-to-ear latency. Preserve segmentation mistakes and dropped queues.

### Variable definitions

- **False trigger:** non-speech/audio playback selected as a speech segment.
- **Missed utterance:** a ground-truth spoken turn not captured as a committed segment.
- **Boundary truncation:** first or final lexical units absent from the committed audio.
- **Oversegmentation:** one intended sentence becomes multiple translated turns.
- **Undersegmentation:** multiple distinct turns merged into one segment.
- **Preview revision:** a changing provisional ASR hypothesis; never present such text as final.
- **Turn drop:** finalised WAV excluded because bounded inference queue reached capacity (visible count).

## Acceptance matrix

| Endpoint | Initial target (not measured) | Study artefact |
|---|---|---|
| Duplex English ↔ Mandarin | ≥20 translated turns per direction, voice output review | Recorded WAVs and blinded meaning assessment |
| Functional disconnected inference | 2 directions, Python guard clean, host WAN disabled | M1.2 offline trial JSON, independent link evidence |
| Pause-to-result time | P95 ≤3 s quiet baseline | Turn evidence JSON with monotonic ms |
| Turn drop | ≤2% across 100 scheduled turns | Complete live history, drop/timeout counts |
| Semantic safety | Report negation, quantities, names, code-switch and repair turns separately | Independent native-speaker adjudication |
| Provisional-caption revision | Rate per utterance (no preset accuracy claim) | Additional instrumented preview traces needed |

## Ablations

1. Push-to-talk versus adaptive silence end-pointing.
2. Quiet, 20/10/0 dB SNR noise conditions; varying microphone distances.
3. Short (2 s) versus long (8–11 s) utterances and sentence-final disambiguation.
4. Auto-play off versus on to detect feedback and unintended self-translation.
5. Fixed 750 ms silence versus adaptive 750/520 ms thresholds.
6. CPU versus MPS with memory, thermal and energy observations.
7. Interrupting previous translated playback with new speech, while separately checking whether emitted audio itself triggers VAD.

## Known failures by design

Whisper is noncausal in this adapter; previews rerun ASR on buffered audio and may be unstable. A simple RMS threshold has no speaker diarisation, overlap separation or language identification. A maximum of two pending segments can lead to intentional drops during sustained heavy load. `AudioWorklet` support depends on browser/OS; a legacy ScriptProcessor fallback is provided. Optional autoplay may be blocked and may feed the mic. Neither echo cancellation nor voice playback has been clinically or acoustically validated.

Record these failures as research outcomes, not as evidence to remove test cases. Next refinements must be selected from actual recordings and user studies rather than changing thresholds until a small test happens to pass.
