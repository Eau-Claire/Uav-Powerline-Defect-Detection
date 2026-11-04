# UAV Power-Line Defect Detection

[![tests](https://github.com/Eau-Claire/Uav-Powerline-Defect-Detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Eau-Claire/Uav-Powerline-Defect-Detection/actions/workflows/tests.yml)

Reusable data-centric research toolkit for power-line defect detection in UAV imagery. The repository contains dataset ingestion and curation, leakage controls, training and evaluation utilities, the controlled Raw-versus-Curated benchmark, Finding Nemo audit results, historical experiment evidence, and optional deployment runtimes.

## Taxonomy and headline metric

| ID | Class | Meaning |
|---:|---|---|
| 0 | `insulator` | Healthy or complete insulator |
| 1 | `insulator_broken` | Broken or missing insulator elements |
| 2 | `insulator_flashover` | Flashover or arc damage |
| 3 | `bird_nest` | Bird nest on transmission infrastructure |
| 4 | `foreign_object` | Kite, balloon, plastic, vegetation, or another foreign object |
| 5 | `broken_strand` | Broken or fractured conductor strand |

`MEAN_5` is the mean AP50 over defect classes 1–5; healthy `insulator` is excluded.

## Current Finding Nemo results

Current training artifacts are mirrored under:

```text
stash://bunpmc/projects/finding_nemo/
```

The canonical record for the newest test audit is
[`FINDING_NEMO_TEST_AUDIT_V5.md`](benchmakring_transmission_line_defect_model/FINDING_NEMO_TEST_AUDIT_V5.md).
The validation table below is selected independently from each model's training CSV by maximum
`metrics/mAP50(B)`; the later v5 table is the newest audited test result.

The following rows maximize `metrics/mAP50(B)` in each current training CSV:

| Model | Epoch | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 50 | 0.81868 | 0.70205 | 0.75589 | 0.77845 | 0.43100 |
| YOLO11m | 44 | 0.83181 | 0.74897 | 0.78822 | 0.82595 | 0.46268 |
| YOLO26n | 44 | 0.82758 | 0.72476 | 0.77276 | 0.80260 | 0.44087 |
| YOLO26m | 45 | 0.83286 | 0.77667 | 0.80378 | 0.83204 | 0.47297 |
| RT-DETR-R50 | 9 | — | — | — | 0.75340 | 0.39690 |
| RT-DETR-R101 | 7 | — | — | — | 0.83550 | 0.47150 |

The RT-DETR CSV uses an Ultralytics-shaped schema, but its fields named `metrics/precision(B)` and `metrics/recall(B)` contain COCO AP@[.50:.95] and AR@100. They are not Precision and Recall, and an F1 derived from them is invalid. Use the unified evaluator for comparable RT-DETR P/R/F1.

### Evaluation roles

- `test_main_v1`: primary independent raw benchmark, 1,931 images.
- `test_operational_min8_v1`: secondary operational subset; excludes an image when any GT box has a minimum side below 8 px at 640 scale.
- v2–v5: exploratory model-assisted annotation studies. They diagnose missing or inconsistent ground truth but must not replace the independent benchmark without full human verification.

The latest exploratory set, `test_operational_min8_v5_full_consensus_assisted`, adds 215 missing healthy-insulator boxes proposed by all four YOLO checkpoints at mean confidence ≥ 0.50.

| Model | Precision | Recall | F1 | mAP50 | mAP50:95 | MEAN_5 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26m | 0.8562 | 0.7689 | 0.8102 | **0.8284** | **0.4783** | **0.7999** |
| RT-DETR-R101 | **0.8606** | **0.7800** | **0.8183** | 0.8102 | 0.4610 | 0.7817 |
| RT-DETR-R50 | 0.8371 | 0.7657 | 0.7998 | 0.7955 | 0.4413 | 0.7684 |
| YOLO26n | 0.8150 | 0.7468 | 0.7795 | 0.7916 | 0.4464 | 0.7586 |
| YOLO11m | 0.8047 | 0.7365 | 0.7691 | 0.7874 | 0.4464 | 0.7505 |
| YOLO11n | 0.7925 | 0.6838 | 0.7341 | 0.7550 | 0.4279 | 0.7141 |

Source: [`operational_min8_v5_all`](benchmakring_transmission_line_defect_model/results/finding_nemo/operational_min8_v5_all/). Canonical protocol, provenance, and caveats: [`FINDING_NEMO_TEST_AUDIT_V5.md`](benchmakring_transmission_line_defect_model/FINDING_NEMO_TEST_AUDIT_V5.md).

Portable audit bundle:

```text
stash://bunpmc/projects/finding_nemo/audits/finding_nemo_test_audit_v5_bundle.tar
SHA-256: 5B9C59DEA1B5B7D0257A3CB938EB9F9A7A7AD9C304C8BB95B2E3B7A91D3CE51A
```

## Repository layout

```text
src/yolov11sdi/                         reusable Python package
scripts/                                pipeline, build, audit, train, evaluation, and export CLIs
configs/                                canonical pipeline and experiment configuration
tests/                                  unit, integrity, split, and offline smoke tests
notebooks/                              current ordered pipeline notebooks
legacy/notebooks/                       preserved V6.1–V6.5 experiment history
benchmakring_transmission_line_defect_model/
                                        controlled benchmark and Finding Nemo work
artifacts/                              compact manifests, reports, reviews, and paper evidence
state/                                  resumable pipeline state
data/README.md                          local data contract; bulk data is ignored
deployment/pi3/                         Raspberry Pi runtime and alert client
deployment/modal/                       Modal API and local IP-camera bridge
docs/                                   data sources, versioning, migration notes, and journals
```

`legacy/` is intentionally retained. It is part of the reproducibility record and documents how dataset, annotation, and training decisions evolved.

### Source-code map

There are two active source trees with different responsibilities:

- [`src/yolov11sdi/`](src/yolov11sdi/) is the reusable project package used by the root CLIs and current notebooks: ingestion, composition, deduplication, QA, state, reporting, and training helpers.
- [`benchmakring_transmission_line_defect_model/src/`](benchmakring_transmission_line_defect_model/src/) is the self-contained Raw-versus-Curated benchmark engine: benchmark configuration, dataset verification, training, evaluation, and report generation.

Files below [`artifacts/paper_evidence_20261026/sources/`](artifacts/paper_evidence_20261026/sources/) are immutable evidence snapshots referenced by `source_manifest.json`; they are not imported as runtime source code. The evidence report links to those captured files explicitly.

Bulk datasets, checkpoints, virtual environments, caches, generated galleries, and credentials are excluded by `.gitignore`; large persistent artifacts belong on Camber.

## Reusable pipeline

The core package supports:

- heterogeneous source ingestion and six-class mapping;
- normalized YOLO/COCO annotation conversion;
- exact and perceptual-hash duplicate checks;
- deterministic split and immutable freeze manifests;
- model-assisted label audit with human-review gates;
- resumable dataset stages and training runs;
- common evaluation, per-class metrics, hardware profiling, and reports.

Raw training contains 18,334 image files. The curated build contains 17,903; the 431 removed training images are 100 synthetic person-paste samples, 239 synthetic foreign-object samples whose pasted target was not reliably fitted, and 92 quarantined non-synthetic samples.

## Setup

Python 3.12 is the reference research environment.

```bash
git clone https://github.com/Eau-Claire/Uav-Powerline-Defect-Detection.git
cd Uav-Powerline-Defect-Detection
python -m venv .venv-research
source .venv-research/bin/activate
pip install -e ".[dev,notebook]"
pip install -e ".[stage]"   # staging and QA
pip install -e ".[train]"   # training and evaluation
pytest -q
```

Common commands:

```bash
python scripts/status.py
python scripts/status.py --remote
python scripts/run_stage.py parent
python scripts/run_stage.py external
python scripts/run_stage.py compose
python scripts/verify_artifacts.py
```

Current notebooks live in `notebooks/`. Historical notebooks are indexed by [`legacy/notebooks/INDEX.csv`](legacy/notebooks/INDEX.csv).

## Camber

Authenticate with `camber login` or export `CAMBER_API_KEY`.

```bash
camber stash ls -r stash://bunpmc/projects/finding_nemo/
camber stash cp \
  stash://bunpmc/projects/finding_nemo/audits/finding_nemo_test_audit_v5_bundle.tar .
```

Never commit `.env`, API keys, downloaded datasets, or model weights.

## Deployment

Deployment code is optional and separated from the reusable research package.

- [`deployment/pi3/`](deployment/pi3/README.md): Raspberry Pi ONNX runtime, alert delivery, packaging, and device setup.
- [`deployment/modal/`](deployment/modal/README.md): working Modal inference API, upload UI, and local iPhone/IP-camera MJPEG bridge.

The Modal runtime exposes `POST /detect` and `GET /health`. Its local bridge samples frames from a LAN camera and sends individual images to Modal; it does not upload the full video stream.

```bash
cd deployment/modal
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/fetch_model.sh
python run_demo.py
```

Read the deployment README before invoking cloud inference; it documents authentication, cost behavior, tests, and shutdown.

## Verification

Core tests:

```bash
pytest -q
```

Modal runtime tests without cloud inference:

```bash
cd deployment/modal
python -m unittest -v test_app.py test_local_web.py test_run_demo.py
node --check static/app.js
node --test test_stream.cjs
```

## Data and reporting policy

- Never use TEST for training, threshold selection, checkpoint selection, or iterative tuning.
- Keep `test_main_v1` as the primary independent benchmark.
- Report model-assisted v2–v5 results as exploratory until independently human-verified and frozen under a new version.
- Store provenance, correction logs, split manifests, hashes, and near-duplicate decisions with published datasets.
- State the exact evaluator and checkpoint-selection metric. Do not relabel COCO AP/AR fields as Precision/Recall.

## Documentation

- Dataset sources and citations: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)
- Versioning: [`docs/VERSIONING.md`](docs/VERSIONING.md)
- Finding Nemo journal: [`docs/NHAT_KY_2026-11-03.md`](docs/NHAT_KY_2026-11-03.md)
- Paper evidence: [`artifacts/paper_evidence_20261026/`](artifacts/paper_evidence_20261026/README.md)
- Benchmark details: [`benchmakring_transmission_line_defect_model/README.md`](benchmakring_transmission_line_defect_model/README.md)

## Status — 2026-11-04

- Finding Nemo training logs and checkpoints are mirrored on Camber.
- `test_main_v1` remains the independent benchmark.
- operational min-8 v5 is the latest exploratory audit result and still requires independent human verification.
- YOLO26m leads the v5 exploratory evaluation in mAP50, mAP50:95, and MEAN_5.
- deployment runtimes are isolated under `deployment/`; standalone demo apps have been removed.
- `legacy/` remains preserved as research history.
