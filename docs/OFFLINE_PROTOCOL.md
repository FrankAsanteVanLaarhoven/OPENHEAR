# OpenHear Bridge — Offline Integrity Check (macOS)

The application is **offline-first by construction**, but neither `HF_HUB_OFFLINE=1` nor local model loading proves the entire process made no outbound network attempts. Use two distinct verification classes:

## A. Functional no-WAN test (minimum M1.1 gate)

1. While connected, download and verify models once: `python -m openhear.cli verify --models-dir models`.
2. Record macOS version, model hashes, voice names, peripheral device identifiers and the timestamp. Use `python -m openhear.cli doctor --models-dir models`.
3. Disconnect **all** uplinks: Wi-Fi, Ethernet, USB tethering, Bluetooth cellular tethering and VPN (if applicable). Use the GUI; do not assume Wi-Fi is the only route. On macOS, `scutil --nwi` and `route get default` can provide supporting system observations but are not independently signed proof.
4. Start the service with existing local weights: `python -m openhear.cli serve --models-dir models --device auto`.
5. Open `http://127.0.0.1:8765` in the same Mac browser. Translate one real recording in **both** directions with audible speech. Loopback does not require internet access.
6. Capture the date, direction, output WAV, response metadata, device state, and any errors. Keep the transcript private and report the outcome as **"functional offline inference under a disconnected host"**, not as a certified no-network claim.

## B. Network-egress audit (separate independent test)

Conduct an online-but-monitored trial with approval from your IT/security team: instrument the Python process using an OS-level outbound firewall/logger or packet capture, including DNS and UDP. Observe the actual process and permitted local loopback traffic. Identify any attempted remote connection and separately record whether it succeeded. Do **not** claim zero egress from a self-reported web label or from a quiet packet trace taken while all interfaces are disabled.

Use a trusted instrument appropriate to your environment and avoid changing system-wide packet-filter rules through an unreviewed installer. University-managed endpoints may require authorization to run a network trace. Record monitor name, scope, filter rules, date, OS and test duration. Never publish identifying data from other apps captured in a host-wide packet trace.

## Important limits

- Browser extensions, OS updates, system dictation, native speech voices and other processes are not controlled by the OpenHear Python application.
- The local FastAPI server intentionally uses `127.0.0.1` to receive speech, which is network IPC but **not external WAN traffic**.
- Source audio is ephemeral on the Python server; user-directed browser downloads and saved evidence are deliberately persistent until deleted.
- `models.lock.json` contains per-file checksums and model commit identifiers but is not a signed supply-chain attestation or a licence audit.
