# Power-Line Defect Detection Benchmark

Controlled evaluation workspace for six detector configurations:

- YOLO11n and YOLO11m;
- YOLO26n and YOLO26m;
- RT-DETR-R50vd and RT-DETR-R101vd.

The directory contains the original Raw-versus-Curated experiment, the newer Finding Nemo human-reviewed training line, independent and operational test protocols, evaluation scripts, and compact result artifacts. Large datasets, run folders, and checkpoints are stored locally or on Camber and are excluded from Git.

## Taxonomy

| ID | Class |
|---:|---|
| 0 | `insulator` |
| 1 | `insulator_broken` |
| 2 | `insulator_flashover` |
| 3 | `bird_nest` |
| 4 | `foreign_object` |
| 5 | `broken_strand` |

`MEAN_5` is mean AP50 over classes 1–5.

## Layout

```text
configs/                  benchmark and dataset configuration
src/                      trainer, evaluator, dataset checks, and reporting
scripts/                  data preparation, training, evaluation, audit, and restore tools
results_yolo_model/       original YOLO Raw/Curated results
results_rtdetr_best/      original RT-DETR best-checkpoint results
results_rtdetr_last/      original RT-DETR last-checkpoint results
results_test_all/         combined historical test export
results/finding_nemo/     current Finding Nemo logs, audits, and evaluations
FINDING_NEMO_TEST_AUDIT_V5.md
                           roles and provenance for the v5 audit bundle
benchmark_pipeline.ipynb  historical interactive orchestration notebook
```

## Original Raw-versus-Curated benchmark

The original factorial experiment compares two training regimes across six architectures. The raw train has 18,334 image files and the curated train has 17,903. Validation has 2,272 images.

The 431 removed training images are recorded in `data/curated_combined/changes.csv` when the dataset is materialized:

- 100 synthetic person-paste samples (`C_person`);
- 239 synthetic foreign-object samples whose pasted target was never reliably fitted (`D_syn_fo`);
- 92 quarantined non-synthetic samples (`E_quarantine`).

Historical result folders use several evaluation configurations. Do not infer independence from a YAML key named `test`: older configs mapped `test` to `images/val_clean`. Check the recorded dataset YAML and split path for every result before citing it.

## Finding Nemo

Dataset and training artifacts:

```text
stash://bunpmc/projects/finding_nemo/
```

Rows maximizing validation mAP50 in the current training CSVs:

| Model | Epoch | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 50 | 0.81868 | 0.70205 | 0.75589 | 0.77845 | 0.43100 |
| YOLO11m | 44 | 0.83181 | 0.74897 | 0.78822 | 0.82595 | 0.46268 |
| YOLO26n | 44 | 0.82758 | 0.72476 | 0.77276 | 0.80260 | 0.44087 |
| YOLO26m | 45 | 0.83286 | 0.77667 | 0.80378 | 0.83204 | 0.47297 |
| RT-DETR-R50 | 9 | — | — | — | 0.75340 | 0.39690 |
| RT-DETR-R101 | 7 | — | — | — | 0.83550 | 0.47150 |

For RT-DETR, the training CSV fields named Precision and Recall are actually COCO AP@[.50:.95] and AR@100. They must not be used as Precision/Recall or combined into F1. Comparable P/R/F1 comes from the unified evaluator.

## Test protocol

| Dataset | Role |
|---|---|
| `test_main_v1` | Primary independent raw benchmark, 1,931 images |
| `test_operational_min8_v1` | Secondary UAV-operational subset using a fixed 8 px minimum-side image exclusion rule |
| v2–v5 | Exploratory annotation studies using model-assisted proposals |

The v2–v5 sets are diagnostic. Because evaluated model families contributed proposals, these results cannot replace `test_main_v1` without complete independent human verification and a new frozen corrected-test version.

Latest v5 exploratory result (`test_operational_min8_v5_full_consensus_assisted`):

| Model | Precision | Recall | F1 | mAP50 | mAP50:95 | MEAN_5 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26m | 0.8562 | 0.7689 | 0.8102 | **0.8284** | **0.4783** | **0.7999** |
| RT-DETR-R101 | **0.8606** | **0.7800** | **0.8183** | 0.8102 | 0.4610 | 0.7817 |
| RT-DETR-R50 | 0.8371 | 0.7657 | 0.7998 | 0.7955 | 0.4413 | 0.7684 |
| YOLO26n | 0.8150 | 0.7468 | 0.7795 | 0.7916 | 0.4464 | 0.7586 |
| YOLO11m | 0.8047 | 0.7365 | 0.7691 | 0.7874 | 0.4464 | 0.7505 |
| YOLO11n | 0.7925 | 0.6838 | 0.7341 | 0.7550 | 0.4279 | 0.7141 |

Canonical local table: [`results/finding_nemo/operational_min8_v5_all/benchmark_summary.csv`](results/finding_nemo/operational_min8_v5_all/benchmark_summary.csv).

Portable bundle:

```text
stash://bunpmc/projects/finding_nemo/audits/finding_nemo_test_audit_v5_bundle.tar
SHA-256: 5B9C59DEA1B5B7D0257A3CB938EB9F9A7A7AD9C304C8BB95B2E3B7A91D3CE51A
```

## Restore and inspect

```bash
camber stash cp \
  stash://bunpmc/projects/finding_nemo/audits/finding_nemo_test_audit_v5_bundle.tar .
tar -xf finding_nemo_test_audit_v5_bundle.tar
```

Fetch current lightweight training logs with:

```bash
python scripts/fetch_nemo_logs.py
```

The repository includes scripts to fetch selected checkpoints, evaluate them, build operational subsets, review targeted errors, apply approved corrections, and aggregate final results. Run scripts from the repository root unless the script help states otherwise.

## Training

YOLO Finding Nemo training is implemented by `scripts/train_finding_nemo.py`. It uses AdamW, cosine learning-rate decay, 50 epochs, 640 px input, AMP, seed 42, and periodic Camber synchronization.

RT-DETR uses the PekingU R50vd/R101vd checkpoints and a separate training/evaluation path. Its CSV schema mimics Ultralytics for aggregation; review metric semantics before using any named column.

## Evaluation rules

1. Select checkpoints using validation only and state the selection metric (`mAP50`, `mAP50:95`, or another criterion).
2. Evaluate the selected checkpoint once on the frozen test set.
3. Do not tune confidence, NMS, input size, labels, or checkpoints against the independent test.
4. Keep primary, operational, corrected, and model-assisted exploratory results in separate tables.
5. Record dataset YAML, image manifest/hash, checkpoint hash, evaluator version, confidence/NMS settings, and evaluation time.
6. Calculate `MEAN_5` from per-class AP50; it is not present in ordinary training CSVs.

## Quick checks

From the repository root:

```bash
pytest -q
python benchmakring_transmission_line_defect_model/scripts/prepare_datasets.py --help
python benchmakring_transmission_line_defect_model/scripts/evaluate_nemo_selected_checkpoints.py --help
```

See [`FINDING_NEMO_TEST_AUDIT_V5.md`](FINDING_NEMO_TEST_AUDIT_V5.md) and [`../docs/NHAT_KY_2026-11-03.md`](../docs/NHAT_KY_2026-11-03.md) for the latest audit record.
