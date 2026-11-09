#!/bin/bash
# Operator-guided research experiment. Does not disable or audit host network traffic.
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "Usage: bash scripts/mac-offline-trial.sh /path/english.wav /path/mandarin.wav" >&2
  exit 2
fi
EN_WAV="${1}"
ZH_WAV="${2}"
HERE="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
cd "$HERE"
if [[ ! -f .venv/bin/activate ]]; then
  echo 'Missing virtual environment .venv; install the project first.' >&2
  exit 2
fi
# shellcheck source=/dev/null
source .venv/bin/activate
MODEL_DIR="$(python -m openhear.cli models locate --print-path)"
RUN_DIR="evidence/offline-$(date -u +%Y%m%dT%H%M%SZ)-$$"
mkdir -m 700 -p "$RUN_DIR"
printf '\nEvidence directory: %s\n' "$RUN_DIR"
python -m openhear.cli offline preflight --models-dir "$MODEL_DIR" \
  --en-audio "$EN_WAV" --zh-audio "$ZH_WAV" \
  --output "$RUN_DIR/preflight-connected.json"
python -m openhear.cli offline snapshot --models-dir "$MODEL_DIR" \
  --output "$RUN_DIR/snapshot-connected.json"
printf '\nDisconnect Wi-Fi, Ethernet, tethering, VPN and other WAN interfaces.\n'
printf 'This script does not disable them for you. Offline claims require independent audit.\n'
printf 'Type DISCONNECTED only after verifying all WAN paths are disabled: '
IFS= read -r confirmation
if [[ "$confirmation" != 'DISCONNECTED' ]]; then
  echo 'No disconnected trial performed. Preflight evidence retained.' >&2
  exit 3
fi
python -m openhear.cli offline snapshot --models-dir "$MODEL_DIR" \
  --output "$RUN_DIR/snapshot-disconnected.json"
python -m openhear.cli offline trial --models-dir "$MODEL_DIR" \
  --en-audio "$EN_WAV" --zh-audio "$ZH_WAV" \
  --device auto --require-tts --disconnected-confirmed \
  --output "$RUN_DIR/bilateral-disconnected.json"
printf '\nTrial completed. Evidence: %s\n' "$RUN_DIR"
printf 'Interpret results as operator-attested functional offline evidence, NOT proof of zero OS egress.\n'
