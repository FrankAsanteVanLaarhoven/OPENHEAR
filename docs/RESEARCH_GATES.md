# OpenHear Bridge Research Gate Register

**Current release:** M1.2/M2-alpha (0.4.0). No research benchmark result is asserted.

| Gate | Test | Status at build | Owner evidence required |
|---|---|---|---|
| G0 Existing core | M0/M1/M1.1 suite | Prior 61 tests preserved; green | Repeat on target Mac |
| G1 New regression suite | +35 network/stream/API/report tests | Passed in build environment | Repeat on target Mac |
| G2 Model provenance | Full SHA256 manifest | Code and test present; local weights absent from archive | `verify` on Mac |
| G3 Bidirectional voice output | English/Mandarin native WAV | Previously demonstrated as isolated Mac `speak` tests | Browser end-to-end speech evidence |
| G4 M1.2 disconnected local execution | Bilateral guarded trial | Implemented; unexecuted on Mac | Genuine WAVs, host link state, JSON |
| G5 Independent OS network audit | Host packet / firewall evidence | Not implemented | Authorized OS-level measurement |
| G6 Live VAD segmented UI | Browser continuous microphone | Implemented, static/API tests; browser automation blocked in build environment | Real browser + microphone exercise |
| G7 Provisional ASR / adaptive commit | Local whole-window preview and variable hangover | Programmatic tests only | Provisional versus final real utterance evidence |
| G8 S2ST turn playback | WAV + HTML audio controls | Implemented; target-device live playback not tested | Per-turn audio reviewed through output device |
| G9 Research quality + latency | 40-slot corpus + longer stress study | Design + benchmark harness only | P50/P95/P99, error counts, confidence intervals |

**Decision:** Software evaluation GO. HOLD on published SOTA, medical/hearing-aid claims, safety-critical translation and commercial deployment until model/data/patent licensing, human studies, regulatory purpose, and independent performance evidence are resolved.
