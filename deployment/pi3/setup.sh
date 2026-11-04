#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ "$(uname -m)" != aarch64 ]]; then
  echo "This setup targets Raspberry Pi OS 64-bit (aarch64)." >&2
  exit 1
fi
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-numpy python3-opencv
python3 -m venv --system-site-packages .venv
# Require a wheel: avoid an accidental long source build on a 1 GB Pi.
.venv/bin/python -m pip install --only-binary=:all: ncnn
.venv/bin/python -c 'import cv2, numpy, ncnn; print("Runtime imports OK")'
.venv/bin/python -m pip freeze > runtime-versions.txt
printf 'Setup complete. Run: bash run.sh --image /path/to/image.jpg\n'
