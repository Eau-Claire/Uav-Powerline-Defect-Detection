#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ ! -x "$ROOT/.venv/bin/python" ]]; then
  echo "Run bash setup.sh first." >&2
  exit 1
fi
for name in deployment.json model.ncnn.param model.ncnn.bin; do
  if [[ ! -s "$ROOT/models/pi_ncnn_model/$name" ]]; then
    echo "Missing exported model: $name. Export on desktop and copy the runtime bundle first." >&2
    exit 1
  fi
done
exec "$ROOT/.venv/bin/python" "$ROOT/detect.py" --model "$ROOT/models/pi_ncnn_model" "$@"
