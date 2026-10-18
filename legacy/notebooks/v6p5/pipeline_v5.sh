#!/bin/bash
# V6.5 v5 unattended pipeline (2026-10-13):
#   wait for the open-vocab insulator proposals -> fuse (thresholds chosen on the GT benchmark) -> build datasets/v6p5_v5
#   -> execute notebook v6p5-19 (teacher -> student distill -> NCNN -> eval) -> upload results to Camber
#   -> commit + push code -> stop the Lightning studio.
# Upload, push and stop run even when an earlier step fails, so the GPU is never left idle.
# Start: setsid nohup legacy/notebooks/v6p5/pipeline_v5.sh > artifacts/v6p5_v5/pipeline.log 2>&1 < /dev/null &
set -u
PROJECT=/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection
DATASETS=/teamspace/studios/this_studio/datasets
OV=$PROJECT/data/label_audit/insulator_ov
NB=$PROJECT/legacy/notebooks/v6p5/v6p5-19_20261013-v5-insulator-filled-teacher-distill-yolo26n.ipynb
OUT=$PROJECT/artifacts/v6p5_v5
REMOTE=stash://bunpmc/projects/yolov11sdi/checkpoints/v6p5_v5_20261013
CAMBER=/teamspace/studios/this_studio/.camber/bin/camber
STATUS=failed
mkdir -p "$OUT"
log() { echo "[$(date -u +%H:%MZ)] $*"; }

finalize() {
  log "finalize (status=$STATUS)"
  cd "$PROJECT"
  set -a; . ./.env; set +a  # CAMBER_API_KEY (never printed)
  echo "$STATUS $(date -u +%FT%TZ)" > "$OUT/PIPELINE_STATUS.txt"
  up() { [ -e "$1" ] && "$CAMBER" stash cp ${3:-} "$1" "$REMOTE/$2" > /dev/null 2>&1 && log "uploaded $2" || log "upload skipped/failed: $2"; }
  for stage in teacher student_distill; do
    W=$OUT/runs/$stage
    up "$W/weights/best.pt" "$stage/best.pt"
    up "$W/weights/last.pt" "$stage/last.pt"
    up "$W/results.csv" "$stage/results.csv"
    up "$W/args.yaml" "$stage/args.yaml"
  done
  up "$OUT/runs/student_distill/weights/best_ncnn_model" "student_distill/best_ncnn_model" -r
  up "$OUT/eval.csv" "eval.csv"
  up "$OUT/v6p5-19_executed.ipynb" "v6p5-19_executed.ipynb"
  up "$DATASETS/v6p5_v5/manifest.json" "dataset_v6p5_v5_manifest.json"
  up "$OV/fuse_choice.json" "insulator_fuse_choice.json"
  up "$OUT/PIPELINE_STATUS.txt" "PIPELINE_STATUS.txt"
  up "$OUT/pipeline.log" "pipeline.log"

  git add -f scripts/label_audit_build.py data/label_audit/README.md data/label_audit/review_decisions.csv \
    data/label_audit/insulator_ov/run.py data/label_audit/insulator_ov/fuse.py data/label_audit/insulator_ov/start.sh \
    data/label_audit/insulator_ov/fuse_choice.json data/label_audit/insulator_ov/insulator_add.csv \
    data/label_audit/person_all/run.py data/label_audit/preflight/fo_person_sim.py data/label_audit/preflight/dry_run_nb18.py \
    legacy/notebooks/v6p5/v6p5-17_20261012-6cls-v3-teacher-distill-yolo26n.ipynb \
    legacy/notebooks/v6p5/v6p5-17a_20261012-sam-tighten-check.ipynb \
    legacy/notebooks/v6p5/v6p5-18_20261012-v4-person-teacher-distill-yolo26n.ipynb \
    legacy/notebooks/v6p5/v6p5-19_20261013-v5-insulator-filled-teacher-distill-yolo26n.ipynb \
    legacy/notebooks/v6p5/pipeline_v5.sh 2>/dev/null
  git commit -q -m "V6.5 v5: 7-class shared dataset (insulator filled, person), unattended train pipeline

Dataset v6p5_v5 = v4 + insulator boxes for unlabeled images (Grounding DINO + OWLv2
agreement, thresholds picked on a 1,000-image GT benchmark). Pipeline result: $STATUS.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && log "committed" || log "nothing to commit"
  git push -q 2>&1 | tail -2; log "push done"
  log "uploading final log, then stopping the studio"
  up "$OUT/pipeline.log" "pipeline.log"
  python -c "from lightning_sdk import Studio; Studio().stop()" 2>&1 | tail -1
}
trap finalize EXIT

# 1. wait for the open-vocab proposals
log "waiting for insulator_ov/run.py"
until grep -q "val done" "$OV/log.txt" 2>/dev/null; do
  pgrep -f "[r]un\.py" > /dev/null || { grep -q "val done" "$OV/log.txt" || { log "run.py died, restarting"; "$OV/start.sh"; sleep 60; }; }
  sleep 60
done
log "proposals done: $(wc -l < "$OV/pred_train.jsonl") train, $(wc -l < "$OV/pred_val.jsonl") val"

# 2. fuse + build v5
cd "$OV" && python fuse.py --auto || exit 1
cd /teamspace/studios/this_studio
if [ ! -f "$DATASETS/v6p5_v5/manifest.json" ]; then
  rm -rf "$DATASETS/v6p5_v5"
  python "$PROJECT/scripts/label_audit_build.py" --src datasets/v6p5_roboflow_ablation/data.yaml \
    --audit "$PROJECT/data/label_audit/yolo11n_adamw" --person "$PROJECT/data/label_audit/coco_person" \
    --review "$PROJECT/data/label_audit/review_decisions.csv" --person-all "$PROJECT/data/label_audit/person_all" \
    --tighten "$PROJECT/data/label_audit/sam_tighten/sam_boxes.csv" --add-insulator "$OV/insulator_add.csv" \
    --keep-insulator --out "$DATASETS/v6p5_v5" > "$OUT/build_v5.log" 2>&1 || { log "build failed"; tail -5 "$OUT/build_v5.log"; exit 1; }
fi
python - <<'EOF' || exit 1
import json, os
from pathlib import Path
D = Path('/teamspace/studios/this_studio/datasets'); n = 0
for s in ('train', 'val'):  # reuse the v3 image cache (same images, imgsz 640)
    for f in (D/'v6p5_6cls_v3/images'/s).glob('*.npy'):
        dst = D/'v6p5_v5/images'/s/f.name
        if not dst.exists(): os.symlink(os.path.realpath(f), dst); n += 1
m = json.load(open(D/'v6p5_v5/manifest.json'))
print('cache links', n, '| stats', {k: {a: b for a, b in v.items() if a.startswith(('I_', 'box:insulator', 'images'))} for k, v in m['stats'].items()})
EOF
log "dataset v6p5_v5 ready"

# 3. train + export + eval (watchdog/resume live inside the notebook)
cd "$(dirname "$NB")"
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 \
  --output-dir "$OUT" --output v6p5-19_executed "$NB" > "$OUT/nbconvert.log" 2>&1 || { log "notebook failed"; tail -20 "$OUT/nbconvert.log"; exit 1; }
log "notebook done"
python - <<'EOF'
import pandas as pd
d = pd.read_csv('/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection/artifacts/v6p5_v5/eval.csv')
print(d[d['class'].isin(['MEAN_5', 'insulator', 'person'])].pivot_table(index=['val', 'class'], columns='model', values='AP50').round(3).to_string())
EOF
STATUS=ok
