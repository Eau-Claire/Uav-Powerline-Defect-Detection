#!/bin/bash
# One-shot setup + unattended run of legacy/notebooks/v6p5/v6p5-22_20261015-v6-vs-v5-teachers-anywhere.ipynb on a rented
# Ubuntu GPU box (e.g. AIDev RTX 3090). Safe to re-run: every step is skipped when already done, training resumes.
#
#   git clone -b v6p5-plateau-yolo26n https://github.com/Eau-Claire/Uav-Powerline-Defect-Detection.git
#   cd Uav-Powerline-Defect-Detection && echo 'CAMBER_API_KEY=...' > .env && bash scripts/run_v6p5_22.sh
#
# Then disconnect freely; watch with:  tail -f artifacts/v6p5_v6_teachers/progress.log
# Options: DRY_RUN=1 (1 epoch on 3% of the images, ~10 min) | V6P5_DATA_DIR=/big/disk (datasets, needs ~70 GB free)
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
NB=legacy/notebooks/v6p5/v6p5-22_20261015-v6-vs-v5-teachers-anywhere.ipynb
OUT=artifacts/v6p5_v6_teachers$([ "${DRY_RUN:-0}" = 1 ] && echo _dryrun || true)
log() { echo "[$(date +%H:%M)] $*"; }

# 1. checks: GPU, disk, Camber key
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || { echo "no NVIDIA GPU / driver"; exit 1; }
FREE=$(df -BG --output=avail "${V6P5_DATA_DIR:-$REPO}" | tail -1 | tr -dc 0-9)
log "free disk: ${FREE} GB (dataset 12 GB + image cache 55 GB; < 70 GB -> no cache, slower epochs)"
grep -q '^CAMBER_API_KEY=' .env 2>/dev/null || { echo "put CAMBER_API_KEY=... in $REPO/.env (dataset + weights come from Camber)"; exit 1; }

# 2. Python env (venv in the repo) + pinned ultralytics
if [ ! -x .venv/bin/python ]; then
  log "creating .venv"
  python3 -m venv .venv || { sudo apt-get update -qq && sudo apt-get install -y -qq python3-venv && python3 -m venv .venv; }
fi
. .venv/bin/activate
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || {
  log "installing torch (CUDA 12.6 wheels)"; pip install -q --upgrade pip
  pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu126; }
python -c "import ultralytics, sys; sys.exit(ultralytics.__version__ != '8.4.172')" 2>/dev/null || {
  log "installing ultralytics 8.4.172 + notebook runner"
  pip install -q ultralytics==8.4.172 ncnn pnnx pandas pyyaml opencv-python-headless nbconvert ipykernel; }
python -c "import torch, ultralytics; print('torch', torch.__version__, '| cuda', torch.cuda.get_device_name(0), '| ultralytics', ultralytics.__version__)"

# 3. Camber CLI (downloads dataset v5 + weights, uploads results)
command -v camber >/dev/null || [ -x "$HOME/.camber/bin/camber" ] || {
  log "installing Camber CLI"; curl -sL https://cli.cambercloud.com/install-v2.sh | bash; }

# 4. run the notebook in the background (survives SSH disconnect); executed copy + log land in $OUT
mkdir -p "$OUT"
if pgrep -f "nbconvert.*v6p5-22" >/dev/null; then log "already running -> tail -f $OUT/progress.log"; exit 0; fi
log "starting notebook (DRY_RUN=${DRY_RUN:-0}); follow: tail -f $OUT/progress.log"
cd "$(dirname "$NB")"
setsid nohup bash -c "DRY_RUN=${DRY_RUN:-0} '$REPO/.venv/bin/jupyter' nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
  --output-dir '$REPO/$OUT' --output v6p5-22_executed '$(basename "$NB")' > '$REPO/$OUT/nbconvert.log' 2>&1; \
  echo \"[\$(date +%H:%M)] notebook exit \$?\" >> '$REPO/$OUT/progress.log'" < /dev/null > /dev/null 2>&1 &
sleep 20; tail -5 "$REPO/$OUT/progress.log" 2>/dev/null || tail -5 "$REPO/$OUT/nbconvert.log"
