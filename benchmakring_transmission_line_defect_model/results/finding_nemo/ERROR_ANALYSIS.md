# Finding Nemo error analysis

This workflow diagnoses the frozen 1,931-image Test without changing its images
or labels. It uses the Finding Nemo YOLO26m checkpoint with SHA-256
`18576ce73960dee09cf572744323dc5415a4b9822fc67661d5cdbc058a773d7c`.

## 1. Cache Test predictions once

From the repository root:

```bash
python benchmakring_transmission_line_defect_model/scripts/diagnose_finding_nemo.py \
  predict-test --device 0 --imgsz 640
```

Use `--device cpu` when CUDA is unavailable. The same cache is scored against
both `data/test_main_v1` and `data/finding_nemo_human_reviewed_6cls`, preventing
model variance from being mistaken for a label-review effect.

## 2. Produce the error audit

```bash
python benchmakring_transmission_line_defect_model/scripts/diagnose_finding_nemo.py analyze
```

FP/FN counts default to confidence 0.25 (`--analysis-conf` changes it), while
the prediction cache remains at confidence 0.001 so it can still support AP.

Outputs under `error_analysis_yolo26m/`:

- `error_summary.json`: raw/reviewed aggregate and per-class counts.
- `errors.csv`: each FN, FP, localization error and class confusion.
- `raw_vs_reviewed_gt.csv`: boxes added, removed, reclassified or moved by review.
- `gallery/`: annotated examples grouped by error type. Green boxes are reviewed
  GT; colored boxes are predictions at confidence 0.25 or above.

The diagnostic matching threshold is IoU 0.5. These counts explain errors but
do not replace the COCO-style AP values produced by the benchmark evaluator.

## 3. Select inference resolution on Validation

Run this on a GPU machine because a full sweep is slow on the local Intel CPU:

```bash
python benchmakring_transmission_line_defect_model/scripts/diagnose_finding_nemo.py \
  sweep-val --device 0 --sizes 640 960 1280 --tta
```

The command is resumable and writes `resolution_sweep_val.csv` after every
configuration. Select the resolution/TTA setting from Validation, lock it, then
evaluate that one setting once on the frozen human-reviewed Test. Do not select
the setting from Test results.

Interpretation guide:

- Many small-object FNs: try higher resolution or tiling.
- Raw-only FPs that disappear with reviewed GT: likely missing raw annotations.
- Raw-only localization errors: likely old box conventions.
- `insulator`/`insulator_broken` confusion: inspect taxonomy consistency.
- Persistent reviewed `broken_strand` errors: real small/thin-object or domain
  difficulty; improve training data/model rather than editing Test to predictions.
