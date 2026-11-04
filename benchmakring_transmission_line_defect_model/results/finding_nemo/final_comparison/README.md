# Finding Nemo: best-validation vs test

One row per model in finding_nemo_final_best.csv. All metrics are fractions (0–1), gaps are percentage points.
Best epoch is inferred from validation mAP50–95 in CSV, not chosen using test. CSV rounding can hide ties; candidates are preserved.
Test metrics are the supplied evaluations of best.pt; checkpoint bytes were not downloaded or re-evaluated.
RT-DETR training precision is a mislabeled COCO AP proxy; it is excluded from val_precision. Training recall is COCO AR@100.
RT-DETR val COCOeval and test custom matching protocols differ; their gaps are descriptive, not a pure generalization estimate.
nemo_rtdetr100 is normalized to rtdetr101 using checkpoint path and local test_metrics.csv.
Training CSVs downloaded from stash://bunpmc/projects/finding_nemo/. Local vs remote numerical equality is recorded per model.

R101 local CSV stops at epoch 32 while stash reaches epoch 37; both have their best AP50–95 at epoch 7. Final table uses stash logs.
| Model | Best epoch candidates | Val AP50 | Val AP50–95 | Test AP50 | Test AP50–95 | Test MEAN5 | Δ95 pp |
|---|---|---:|---:|---:|---:|---:|---:|
| yolo11n | 40 | 0.7747 | 0.4340 | 0.7404 | 0.4051 | 0.7053 | -2.89 |
| yolo11m | 49 | 0.8245 | 0.4641 | 0.7717 | 0.4261 | 0.7379 | -3.79 |
| yolo26n | 50 | 0.7986 | 0.4431 | 0.7740 | 0.4258 | 0.7434 | -1.73 |
| yolo26m | 44;45 | 0.8320 | 0.4730 | 0.8086 | 0.4516 | 0.7807 | -2.14 |
| rtdetr50 | 7 | 0.7495 | 0.4076 | 0.7814 | 0.4366 | 0.7493 | +2.90 |
| rtdetr101 | 7 | 0.8355 | 0.4715 | 0.7942 | 0.4365 | 0.7702 | -3.50 |
