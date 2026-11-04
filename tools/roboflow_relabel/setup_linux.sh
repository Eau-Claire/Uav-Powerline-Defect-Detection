#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PYTHON_BIN="${PYTHON_BIN:-python3}"
"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3,10), "Python 3.10+ required"'
"$PYTHON_BIN" -m venv .venv-roboflow
.venv-roboflow/bin/python -m pip install --upgrade pip
.venv-roboflow/bin/python -m pip install -r tools/roboflow_relabel/requirements-handoff.txt
if [[ ! -f .env ]]; then
  cp tools/roboflow_relabel/.env.example .env
  chmod 600 .env
  echo "Created .env. Insert a newly issued Roboflow key before running."
fi
.venv-roboflow/bin/python tools/roboflow_relabel/doctor.py || \
  echo "Setup completed, but preflight still has FAIL items. Fix them before running."
