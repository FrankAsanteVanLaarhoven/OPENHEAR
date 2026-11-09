# OpenHear Bridge M1.1 — Preregistered Feasibility Protocol (Proposal)

**Research question:** Can an offline, modular ASR→MT→TTS system provide useful English↔Mandarin translation and audible output on the target Apple Silicon Mac within a measurable latency and quality envelope?

**Hypothesis H1 (not yet tested):** Under quiet, short-utterance conditions, at least 20 real trials per language direction will complete without cloud inference; audio synthesis will produce a WAV played through the chosen output when compatible voices are installed. The two previous single-utterance *text* smoke tests are not sufficient for statistical claims.

## Study design

1. Record 20 distinct consented utterances in English and 20 in Mandarin for a feasibility baseline. Use balanced speakers and accents where possible; for stronger generalisation use more speakers and speaker-disjoint splits. A sample of 20/direction is **not** a power analysis and not sufficient to prove a 99% service reliability rate.
2. Generate independent reference transcripts and translations, blinded to system outputs, and record critical errors (negation, medication terms, numbers, names, directions). Do not derive references from the model being scored.
3. Fix microphone position, output device, input level, firmware and operating system. Record first/warm-run status and CPU/MPS mode. Make at least one matched CPU/MPS comparison.
4. Repeat at quiet, 20/10/0 dB SNR in later studies. Do not mix noisy and clean trial percentiles without stratification.
5. Run `benchmark run`; save the JSON report, `models.lock.json` fingerprint, UI evidence and failure logs. Use exact checkpoint revision hashes.
6. Independently measure actual acoustic timing using a synchronized external audio capture/microphone and output coupler; never substitute model processing P95 for mouth-to-ear P95.

## Proposed SMART acceptance gates

| KPI | Proposed feasibility gate | Evidence and caveat |
|---|---|---|
| K1 · Coverage | ≥20 actual, consented test utterances per direction (40 total) in 14 days | No fabricated samples; report speech/accent composition |
| K2 · Speech output | 100% of correctly configured test sessions create playable English and Mandarin WAV files (both directions) | Voice/volume and earbud device manually verified; this is a target |
| K3 · Runtime latency | Calculate P50/P95/P99 for ASR, MT, TTS and processing total per direction | Exploratory software measurements only; no absolute 3-second guarantee |
| K4 · Translation fidelity | Report reference-backed chrF++, human adequacy and critical-error counts per direction | chrF++ is not synonymous with semantic correctness; pre-register a later acceptance threshold |
| K5 · Functional offline | Demonstrate both directions with external links disconnected, after model preload | A disconnected trial proves functional offline operation, not no attempted egress under connected conditions |
| K6 · Reliability | Record all timeouts/errors; later target ≥99% on sufficiently large, preregistered repeated tests | A 20-example set cannot establish a tight 99% interval |

**Failure protocol:** Keep silence/clip errors, unsupported devices, failed outputs, transcription hallucinations and partial/incomplete benchmark rows; never remove a failed case from the denominator to improve KPIs. If Mandarin voice is missing, treat generated audio gate as HOLD while preserving valid text translation. If model compatibility fails on MPS, re-run CPU as a diagnostic, keeping both traces.

## Phases

- **M1.1 / Week 1:** Install voices and verify sound output in both directions; collect first smoke traces, update hardware profile.
- **M1.1 / Week 2:** Acquire 20/20 utterances and independent references, run benchmarks and disconnect WAN for functional offline checks.
- **M2 (after acceptance):** Replace buffered `ScriptProcessorNode` with AudioWorklet; implement incremental ASR commit, VAD, structured per-segment timings and feedback-safe audio path. Measure latency-quality frontier.
- **M3:** Multiple consented speakers on local encrypted LAN, each with an identified mic, speaker attribution, per-language audio routing and realistic overlap handling.

## Alternative hypotheses, limitations and research stance

Translation quality might degrade on code-switched names, Mandarin tones or noisier devices even if software processing is fast. The system may be useful as a captioning assistant yet fail spoken conversational usability due to capture/utterance waiting. Existing Timekettle, iFLYTEK and other products have related capabilities; an original contribution would require controlled comparative evidence and a systematic prior-art assessment. Clinical hearing assistance and safety-critical robotic commands are out of scope.
