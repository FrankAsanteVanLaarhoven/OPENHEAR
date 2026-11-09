#!/bin/bash
# Verified model selection, no re-download and no symlink edits.
set -euo pipefail
HERE="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)"
cd "$HERE"
if [[ ! -f .venv/bin/activate ]]; then
  echo 'Virtual environment missing. Run: python3 -m venv .venv && source .venv/bin/activate && python -m pip install -e ".[all,test]"' >&2
  exit 2
fi
# shellcheck source=/dev/null
source .venv/bin/activate
MODEL_DIR="$(python -m openhear.cli models locate --print-path)"
printf 'Using verified model cache: %s\n' "$MODEL_DIR"
python -m openhear.cli verify --models-dir "$MODEL_DIR"
exec python -m openhear.cli serve --models-dir "$MODEL_DIR" --device auto "$@"
