# M1.2 Offline Assurance — Acceptance Protocol

## Hypothesis

For the frozen local checkpoint, source WAV, system voices, hardware and OS, both English→Mandarin and Mandarin→English requests complete on an intentionally **WAN-disconnected** Mac without the Python application attempting a non-loopback connection.

This *does not* imply OS-wide egress prohibition: macOS native TTS (`say`), subprocesses, native C libraries, browser extensions, OS services and the browser itself fall outside the Python monkeypatch scope.

## Preconditions

1. Obtain informed permission for each voice recording; avoid storing identifiable audio in source control.
2. Pin Mac hardware model, macOS version, Python environment, local model manifest hashes and native voices.
3. Create separate 5–10 s actual speech WAVs for English and Mandarin, using the GUI `Save recorded WAV` control. Do not substitute fixture or synthetic dummy audio when measuring speech translation.
4. Ensure both work online first; note TTS actually audible through the selected device. `verify` must pass.
5. Use distinct evidence filenames; commands refuse overwrite. Protect evidence and recorded speech.

## Experiment: disconnected functional trial

```bash
python -m openhear.cli offline snapshot --models-dir models --output evidence/online-state.json
```

Manually disable **all** uplinks: Wi-Fi, ethernet (including docks), cellular/phone tethering, VPN and any other bridge. This must be established outside the application; `route get default` is only a supporting observation. `127.0.0.1` still works locally. Then:

```bash
python -m openhear.cli offline snapshot --models-dir models --output evidence/offline-state.json
python -m openhear.cli offline trial --models-dir models \
  --en-audio english.wav --zh-audio mandarin.wav \
  --device auto --require-tts --disconnected-confirmed \
  --output evidence/offline-trial.json
```

**Pass at the functional test level only if:** both directions complete, both TTS statuses are `generated`, the Python guard records zero attempted non-loopback calls, and the operator attested disconnection. `model_verification` must also pass. Keep all errors and unexpected resource failures in the report. With `--require-tts`, speech output problems fail the functional trial even if text is available.

The report labels an attested trial as `self_attested_no_wan_trial_pass`; this is not an externally audited claim. `functional_pass` in the JSON means local ASR/MT(/required TTS) completion, not proof of WAN disconnection.

## Independent network measurement

An IT-authorized supervisor should collect a process-attributed network trace or OS-level egress policy record while the host is connected but remote egress is blocked/monitored. Observe successful **and attempted** TCP, UDP and DNS activity. Document tools, timestamps, capture interface, filter, coverage, process IDs and OS controls. Do not publish packets belonging to other applications. A Python `socket` guard cannot substitute for this experiment.

## Failure categories and refinements

| Failure | Interpretation | Next action |
|---|---|---|
| `models.lock.json` absent | Preparation incomplete | Re-run `prepare` while connected; verify hashes |
| Native voice absent | Text inference may succeed, speech fails | Install/select correct local macOS voice |
| Guard reports blocked connection | Actual Python nonloopback call attempted | Trace initiating library and remove network dependency; do not disable guard merely to pass |
| Slow MPS operation / CPU fallback | Runtime feasibility issue | Repeat on `--device cpu`; retain both results |
| Default route remains | May be stale interface, VPN, or actual route | Independently inspect network links; do not infer disconnection from one command |
| App works offline, no audit trace | Limited positive evidence | Claim functional disconnected test only, not certified zero egress |

## Preservation

Record UTC date/time, package version, model manifest hash, path hashes, per-direction timings and failures. If files are copied to a shared repository, strip private transcript content and host-identifying network information; retain an internal controlled-access audit record where required.
