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

## Integrated datasets and references

The 22,618-image six-class benchmark is a harmonized collection built from the following six public sources. Source labels were converted to the taxonomy above; these datasets are not treated as mutually exclusive class partitions. Cite the applicable original sources when using the integrated data or reporting derived results.

| Source | Contribution to the integrated taxonomy | Primary paper or dataset citation | Public access |
|---|---|---|---|
| **ZHENG1600** | `insulator`, `insulator_broken`, `insulator_flashover` | Zheng et al., “Insulator-Defect Detection Algorithm Based on Improved YOLOv7,” *Sensors* 22(22):8801, 2022. [doi:10.3390/s22228801](https://doi.org/10.3390/s22228801) | [Dataset Ninja mirror and citation](https://github.com/dataset-ninja/insulator-defect-detection) |
| **IDID** | `insulator`, `insulator_broken`, `insulator_flashover` | Lewis and Kulkarni, “Insulator Defect Detection” [dataset], IEEE DataPort. [doi:10.21227/vkdw-x769](https://doi.org/10.21227/vkdw-x769) | IEEE DataPort; access may require accepting its terms |
| **CPLID** | whole-insulator context and broken-insulator instances; synthetic defects are training-only | Tao et al., “Detection of Power Line Insulator Defects Using Aerial Images Analyzed With Convolutional Neural Networks,” *IEEE TSMC: Systems*. [doi:10.1109/TSMC.2018.2871750](https://doi.org/10.1109/TSMC.2018.2871750) | [InsulatorData/InsulatorDataSet](https://github.com/InsulatorData/InsulatorDataSet) |
| **VPMBGI** | broken-glass instances mapped to `insulator_broken` | Vietnamese Public Merged dataset of Broken Glass Insulator; no single canonical paper, so cite the upstream sources documented by the repository | [phd-benel/VPMBGI](https://github.com/phd-benel/VPMBGI) |
| **InsuFault-Net** | `insulator`, damaged-insulator, and flashover instances; integrated as training data | “InsuFault-Net: A Deep Learning-Based Insulator Defect Detection Algorithm for Transmission Lines” and its accompanying dataset repository | Upstream availability and provenance are recorded in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) |
| **FOTL / EFOD_Drone** | `bird_nest` and `foreign_object` | Gao et al., “YOLOv11-Based UAV Foreign Object Detection for Power Transmission Lines,” 2025 | Upstream availability and the FOTL overlay lineage are recorded in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) |

The controlled Raw-versus-Curated benchmark uses only the six-source integrated collection above. Later research branches also studied Open Images V7, MPCD, the Power Equipment Image Dataset, and a Roboflow transmission dataset; those sources are augmentation, ablation, or external-stage inputs and must not be described as constituents of the reported six-source benchmark.

Additional source citations:

- **Open Images V7:** Kuznetsova et al., “The Open Images Dataset V4,” *IJCV*, 2020. [arXiv:1811.00982](https://arxiv.org/abs/1811.00982); [dataset site](https://storage.googleapis.com/openimages/web/index.html).
- **MPCD:** Benelmostafa and Medromi, “PowerLine-MTYOLO,” *Drones* 9(7):505, 2025. [doi:10.3390/drones9070505](https://doi.org/10.3390/drones9070505); [repository](https://github.com/phd-benel/PowerLine-MTYOLO).
- **Power Equipment Image Dataset:** Xiong et al., “Object recognition for power equipment via human-level concept learning,” *IET Generation, Transmission & Distribution* 15(10):1578–1587, 2021. [doi:10.1049/gtd2.12088](https://doi.org/10.1049/gtd2.12088); [repository](https://github.com/xiongsiheng/Power-equipment-image-dataset).

### Related datasets reviewed but not merged

These references informed taxonomy or external-evaluation planning, but their images are not constituents of the reported six-source benchmark:

- **Insulator-DET:** Li et al., “TLINet: A defects detection method for insulators of overhead transmission lines using partially transformer block,” *PLOS ONE*, 2025.
- **BGI:** public broken-glass insulator data and a predecessor/overlap source for VPMBGI; it is not merged independently.
- **CPMID:** “Cross-scale recognition of dense insulators and defects in complex power grid environments,” *Engineering Applications of Artificial Intelligence*, 2025. [doi:10.1016/j.engappai.2025.113283](https://doi.org/10.1016/j.engappai.2025.113283).
- **TLID:** Hu et al., “Towards Defect Detection of Transmission Line Insulator: A Dataset, Benchmarks and Challenges,” ICPRE 2025. [doi:10.1109/ICPRE67300.2025.11274094](https://doi.org/10.1109/ICPRE67300.2025.11274094).
- **Porcelain Disk Insulator 2026:** external-domain dataset, Mendeley Data, CC BY 4.0. [doi:10.17632/5pwrd7gmw2.1](https://doi.org/10.17632/5pwrd7gmw2.1).
- **MPID:** reference dataset accompanying “APF-YOLOv8: Enhancing Multiscale Detection and Intra-Class Variance Handling for UAV-Based Insulator Power Line Inspections.” [Zenodo dataset](https://doi.org/10.5281/zenodo.14604384).
- **InsPLAD:** Vieira e Silva et al., “InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images,” 2023. [doi:10.1080/01431161.2023.2283900](https://doi.org/10.1080/01431161.2023.2283900); [repository](https://github.com/andreluizbvs/InsPLAD).
- **TTPLA:** Abdelfattah et al., “TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines,” ACCV 2020.
- **TL-Defect5K:** Jia, “Mamba-enhanced detection transformer with selective state space models for multi-category defect detection in transmission lines,” *Discover Computing* 29(1):489, 2026. [paper](https://link.springer.com/article/10.1007/s10791-026-10353-0); [repository](https://github.com/Dali8710/TL-Defect5K).
- **Roboflow transmission-detection v1:** CC BY 4.0; used only in the `v6p5-11` ablation, with `cable_defectueux` mapped to `broken_strand` and `rubbish` mapped to `foreign_object`.

Their exact intended role, mapping restrictions, access notes, and redistribution cautions are maintained in [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).

## Dataset versions

All counts below are read from the YOLO label files. Image counts include background images; box counts are totals over the six classes.

| Dataset | Role | Split | Images | Boxes |
|---|---|---|---:|---:|
| `finding_nemo_raw` (= `v6p5_roboflow_ablation`) | Raw training arm: V6.2E train, 500 synthetic composites and 351 Roboflow images | train / val | 18,334 / 2,272 | 25,807 / 2,903 |
| `curated_combined` | Curated arm: Modules I–IV applied to train only | train / val / val_clean | 17,903 / 2,272 / 2,271 | 23,204 / 2,903 / 2,863 |
| `finding_nemo_human_reviewed_6cls` | Finding Nemo line: curated data after cross-label proposals and human review | train / val / test | 17,903 / 2,271 / 1,931 | 25,811 / 3,204 / 2,850 |
| `test_main_v1` | Primary independent test: V6.2E TEST, deduplicated, person-as-FO images removed | test | 1,931 | 2,562 |
| `test_operational_min8_v1` | Secondary test: drops images with any box shorter than 8 px at 640 | test | 1,871 | — |

Boxes per class:

| Class | Raw train | Curated train | Reviewed train | val_clean | Reviewed val | test_main_v1 | Reviewed test |
|---|---:|---:|---:|---:|---:|---:|---:|
| `insulator` | 8,665 | 6,724 | 9,078 | 778 | 1,087 | 1,207 | 1,478 |
| `insulator_broken` | 6,236 | 6,236 | 6,424 | 791 | 810 | 809 | 823 |
| `insulator_flashover` | 1,249 | 1,248 | 1,253 | 163 | 163 | 124 | 125 |
| `bird_nest` | 6,136 | 6,110 | 6,160 | 821 | 830 | 177 | 178 |
| `foreign_object` | 1,491 | 955 | 961 | 58 | 62 | 111 | 112 |
| `broken_strand` | 2,030 | 1,931 | 1,935 | 252 | 252 | 134 | 134 |

Lineage:

- `val_clean` is the raw validation set with rules A, C, G and H applied (2,272 → 2,271 images). It contains the same images as `val`.
- The human-reviewed set takes its train split from `curated_combined/train`, its val split from `val_clean` and its test split from `test_main_v1`. On each split, a curated YOLO26m teacher and a Roboflow workflow proposed extra boxes, and only human-accepted proposals were applied ([`tools/roboflow_relabel/`](tools/roboflow_relabel/)). Its test split therefore differs from the independent `test_main_v1`.
- The curation rules are documented in [`benchmakring_transmission_line_defect_model/scripts/build_curation_module_ablation.py`](benchmakring_transmission_line_defect_model/scripts/build_curation_module_ablation.py): Module I = A, B, F; Module II = C, D; Module III = E, G; Module IV = H.

## Results

Every table names both the evaluation set and the checkpoint-selection protocol. Do
not compare numbers as if they came from the same checkpoint merely because the
model name is the same. In particular, the Finding Nemo test tables below use the
explicit `last`/epoch-45/epoch-15 checkpoints recorded in their source reports;
they do not use the training-log rows with maximum validation mAP50.

### 1. Raw versus Curated, seed 42 (evaluated on `val_clean`)

`configs/dataset_curated.yaml` maps `test` to `images/val_clean`, so these are **not** results on an independent test set: `val_clean` holds the validation images used for checkpoint selection. YOLO rows come from [`results_yolo_model/`](benchmakring_transmission_line_defect_model/results_yolo_model/); RT-DETR rows use the last checkpoint from [`results_rtdetr_last/`](benchmakring_transmission_line_defect_model/results_rtdetr_last/).

| Model | F1 raw | F1 curated | mAP50 raw | mAP50 curated | Δ mAP50 | mAP50:95 raw | mAP50:95 curated |
|---|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0.7472 | 0.7650 | 0.7564 | 0.7911 | +0.0347 | 0.4090 | 0.4305 |
| YOLO11m | 0.7827 | 0.7893 | 0.7977 | 0.8099 | +0.0122 | 0.4391 | 0.4565 |
| YOLO26n | 0.7659 | 0.7836 | 0.7806 | 0.8119 | +0.0313 | 0.4222 | 0.4402 |
| YOLO26m | 0.7962 | 0.8186 | 0.8090 | **0.8323** | +0.0233 | 0.4480 | **0.4691** |
| RT-DETR-R50 | 0.8090 | 0.8365 | 0.7823 | 0.8017 | +0.0194 | 0.4301 | 0.4392 |
| RT-DETR-R101 | 0.8154 | **0.8426** | 0.7875 | 0.8157 | +0.0282 | 0.4351 | 0.4505 |

### 2. Finding Nemo line (trained on `finding_nemo_human_reviewed_6cls`)

Training artifacts are mirrored under `stash://bunpmc/projects/finding_nemo/`. Results live in [`results_ablation/finding_nemo/`](benchmakring_transmission_line_defect_model/results_ablation/finding_nemo/).

**Primary independent test, `test_main_v1` (1,931 images).** Unified-evaluator
results for the fixed checkpoints named in the `Checkpoint` column
([`selected_checkpoints_eval_test_main_v1`](benchmakring_transmission_line_defect_model/results_ablation/finding_nemo/selected_checkpoints_eval_test_main_v1/)).
These are not evaluations of the maximum-validation-mAP50 epochs listed later:

| Model | Checkpoint | Precision | Recall | F1 | mAP50 | mAP50:95 | MEAN_5 | Latency (ms) | FPS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO11n | last epoch | 0.7665 | 0.6611 | 0.7099 | 0.7072 | 0.3823 | 0.6988 | 11.97 | 83.5 |
| YOLO11m | epoch 45 | 0.7753 | 0.7155 | 0.7442 | 0.7431 | 0.4039 | 0.7309 | 16.28 | 61.4 |
| YOLO26n | epoch 45 | 0.7899 | 0.7234 | 0.7552 | 0.7435 | 0.4012 | 0.7392 | 12.87 | 77.7 |
| YOLO26m | epoch 45 | 0.8274 | 0.7467 | 0.7850 | **0.7806** | **0.4296** | **0.7779** | 16.89 | 59.2 |
| RT-DETR-R50 | epoch 15 | 0.8073 | 0.7489 | 0.7770 | 0.7499 | 0.3991 | 0.7472 | 37.50 | 26.7 |
| RT-DETR-R101 | epoch 15 | **0.8346** | **0.7632** | **0.7973** | 0.7714 | 0.4198 | 0.7705 | 64.30 | 15.6 |

**Human-reviewed test (1,931 images, 2,850 boxes)** ([`selected_checkpoints_eval`](benchmakring_transmission_line_defect_model/results_ablation/finding_nemo/selected_checkpoints_eval/)). Same checkpoints; labels include human-accepted cross-label additions, so report separately from `test_main_v1`:

| Model | Precision | Recall | F1 | mAP50 | mAP50:95 | MEAN_5 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 0.7953 | 0.6627 | 0.7230 | 0.7382 | 0.4074 | 0.7034 |
| YOLO11m | 0.8056 | 0.7141 | 0.7571 | 0.7697 | 0.4250 | 0.7349 |
| YOLO26n | 0.8193 | 0.7232 | 0.7683 | 0.7727 | 0.4246 | 0.7419 |
| YOLO26m | 0.8591 | 0.7472 | 0.7993 | **0.8092** | **0.4535** | **0.7814** |
| RT-DETR-R50 | 0.8343 | 0.7417 | 0.7853 | 0.7769 | 0.4245 | 0.7497 |
| RT-DETR-R101 | **0.8635** | **0.7588** | **0.8078** | 0.7974 | 0.4437 | 0.7724 |

**Training-log validation maxima (not the checkpoints used in the test tables).**
Each row below is selected independently by maximum `metrics/mAP50(B)` in its
training CSV. It is a validation-log lookup, not a test result:

| Model | Epoch | Precision | Recall | F1 | mAP50 | mAP50:95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO11n | 50 | 0.81868 | 0.70205 | 0.75589 | 0.77845 | 0.43100 |
| YOLO11m | 44 | 0.83181 | 0.74897 | 0.78822 | 0.82595 | 0.46268 |
| YOLO26n | 44 | 0.82758 | 0.72476 | 0.77276 | 0.80260 | 0.44087 |
| YOLO26m | 45 | 0.83286 | 0.77667 | 0.80378 | 0.83204 | 0.47297 |
| RT-DETR-R50 | 9 | — | — | — | 0.75340 | 0.39690 |
| RT-DETR-R101 | 7 | — | — | — | 0.83550 | 0.47150 |

The RT-DETR CSV uses an Ultralytics-shaped schema, but its `metrics/precision(B)` and `metrics/recall(B)` fields contain COCO AP@[.50:.95] and AR@100. They are not Precision and Recall, and an F1 derived from them is invalid. Use the unified evaluator for comparable RT-DETR P/R/F1.

To produce a matched test table for these maximum-mAP50 epochs, the exact saved
checkpoints from epochs 50, 44, 44, 45, 9 and 7 must be evaluated with the same
unified evaluator on the same frozen test split. Test metrics from a different
epoch or from `best.pt` (selected by validation mAP50:95) must not be relabeled as
results for these rows.

**Exploratory test audits (v2–v5).** These sets add boxes proposed by the evaluated model families and must not replace `test_main_v1` without independent human verification. The latest, `test_operational_min8_v5_full_consensus_assisted`, adds 215 healthy-insulator boxes proposed by all four YOLO checkpoints at mean confidence ≥ 0.50 (214 of them fall inside the min-8 subset):

| Model | Precision | Recall | F1 | mAP50 | mAP50:95 | MEAN_5 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26m | 0.8562 | 0.7689 | 0.8102 | **0.8284** | **0.4783** | **0.7999** |
| RT-DETR-R101 | **0.8606** | **0.7800** | **0.8183** | 0.8102 | 0.4610 | 0.7817 |
| RT-DETR-R50 | 0.8371 | 0.7657 | 0.7998 | 0.7955 | 0.4413 | 0.7684 |
| YOLO26n | 0.8150 | 0.7468 | 0.7795 | 0.7916 | 0.4464 | 0.7586 |
| YOLO11m | 0.8047 | 0.7365 | 0.7691 | 0.7874 | 0.4464 | 0.7505 |
| YOLO11n | 0.7925 | 0.6838 | 0.7341 | 0.7550 | 0.4279 | 0.7141 |

Source: [`operational_min8_v5_all`](benchmakring_transmission_line_defect_model/results_ablation/finding_nemo/operational_min8_v5_all/). Protocol and caveats: [`FINDING_NEMO_TEST_AUDIT_V5.md`](benchmakring_transmission_line_defect_model/FINDING_NEMO_TEST_AUDIT_V5.md).

```text
stash://bunpmc/projects/finding_nemo/audits/finding_nemo_test_audit_v5_bundle.tar
SHA-256: 5B9C59DEA1B5B7D0257A3CB938EB9F9A7A7AD9C304C8BB95B2E3B7A91D3CE51A
```

### 3. Curation-module ablation (YOLO26m)

Each ablation dataset is rebuilt from the raw train with only the selected modules ([`build_curation_module_ablation.py`](benchmakring_transmission_line_defect_model/scripts/build_curation_module_ablation.py), [`build_curation_loo_ablation.py`](benchmakring_transmission_line_defect_model/scripts/build_curation_loo_ablation.py)). Validation is frozen across runs. The only canonical held-out ablation test is `finding_nemo_human_reviewed_6cls/images/test` (1,931 images, 2,850 boxes); every generated ablation `data.yaml` exposes this split as `test`.

**Leave-one-out on `val_clean`**, from [`docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md`](docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md) (run folders are not in Git):

| Configuration | Modules | Train images | Precision | Recall | mAP50 | mAP50:95 |
|---|---|---:|---:|---:|---:|---:|
| Reference | I, II | 18,052 | 0.832 | 0.791 | 0.835 | 0.453 |
| w/o Module II | I, III, IV | 18,242 | 0.816 | 0.792 | 0.829 | 0.456 |
| w/o Module III | I, II, IV | 18,052 | 0.813 | 0.786 | 0.824 | 0.449 |
| w/o Module I | II, III, IV | 17,960 | 0.803 | 0.790 | 0.805 | 0.435 |

The three LOO runs differ by exactly one module and can be compared directly: dropping Module I costs the most (mAP50 0.805), mainly through `foreign_object` (AP50 0.575 vs 0.646–0.649) and `broken_strand` (0.725 vs 0.766–0.768). The reference lacks Modules III and IV, so its gap to a LOO run is not a single-module effect. A run with all four modules and a run without Module IV have not been trained.

**Historical exploratory v5 audit — not the canonical ablation test.** The
following results used model-assisted v5 labels
([`results_ablation/finding_destiny_test_audit_v5/`](benchmakring_transmission_line_defect_model/results_ablation/finding_destiny_test_audit_v5/)).
They are retained for provenance but must not be reported as the final ablation
test results:

| Run | Modules | Precision | Recall | mAP50 | mAP50:95 |
|---|---|---:|---:|---:|---:|
| `dori_m1_abf` | I | 0.8456 | 0.7372 | 0.7898 | 0.4333 |
| `dori_m2_cd` | I, II | 0.8441 | 0.7344 | 0.7926 | 0.4321 |
| `dori_m3_eg` | I, II, III | 0.8425 | 0.7451 | 0.7914 | 0.4312 |
| `destiny_m234` | II, III, IV | 0.8546 | 0.7161 | 0.7896 | 0.4105 |
| `destiny_m134` | I, III, IV | 0.8510 | 0.7357 | 0.7935 | 0.4231 |
| `destiny_m124` | I, II, IV | 0.8656 | 0.7169 | 0.7967 | 0.4141 |

The cumulative and leave-one-out checkpoints must be re-evaluated on
`finding_nemo_human_reviewed_6cls/images/test` with the same unified evaluator
before a canonical ablation test table is reported. The checkpoints and their
training metadata are stored under `stash://bunpmc/projects/finding_ablation/`.

### 4. Seed replication

| Seed | Status |
|---|---|
| 42 | Complete: table 1 above |
| 36 | Raw YOLO runs evaluated on the 1,931-image v5 audit test ([`results_seed/finding_seed36_test_audit_v5/`](benchmakring_transmission_line_defect_model/results_seed/finding_seed36_test_audit_v5/)): YOLO26m mAP50 0.8032, YOLO26n 0.7644, YOLO11m 0.7439, YOLO11n 0.7331. Curated metrics were removed as wrong (commit `ceeb9fc`) and need re-evaluation; RT-DETR runs are in progress. `summary.csv` still lists the stale curated rows. |
| 85 | Notebook ready ([`legacy/seeds/`](legacy/seeds/)); not yet run |

The seed-36/85 notebooks train the curated arm on `finding_nemo_human_reviewed_6cls`, whereas seed 42 used `curated_combined`. Compare seeds with this difference in mind.

## Figures

[`figures/`](figures/README.md) regenerates every result chart and method diagram from the files above into `figures/out/` (PDF + 300 dpi PNG):

```bash
.venv/bin/python figures/make_all.py
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
figures/                                scripts that regenerate result charts and method diagrams
tools/roboflow_relabel/                 cross-label proposals and human-review tooling
deployment/                             placeholder; deployment code is kept outside this repository
docs/                                   data sources, versioning, migration notes, reports, and journals
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

Deployment runtimes (Raspberry Pi client, Docker API, Modal service) are maintained outside this repository. The `deployment/` folder is kept as an empty placeholder and its contents are ignored by Git.

## Verification

```bash
pytest -q
```

## Data and reporting policy

- Never use TEST for training, threshold selection, checkpoint selection, or iterative tuning.
- Keep `test_main_v1` as the primary independent benchmark.
- Report model-assisted v2–v5 results as exploratory until independently human-verified and frozen under a new version.
- Name the evaluation set in every table. `val_clean` contains the validation images and is not an independent test set.
- Store provenance, correction logs, split manifests, hashes, and near-duplicate decisions with published datasets.
- State the exact evaluator and checkpoint-selection metric. Do not relabel COCO AP/AR fields as Precision/Recall.

## Documentation

- Dataset sources and citations: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md)
- Versioning: [`docs/VERSIONING.md`](docs/VERSIONING.md)
- Finding Nemo journal: [`docs/NHAT_KY_2026-11-03.md`](docs/NHAT_KY_2026-11-03.md)
- Leave-one-out curation ablation: [`docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md`](docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md)
- Figures and diagrams: [`figures/README.md`](figures/README.md)
- Paper evidence: [`artifacts/paper_evidence_20261026/`](artifacts/paper_evidence_20261026/README.md)
- Benchmark details: [`benchmakring_transmission_line_defect_model/README.md`](benchmakring_transmission_line_defect_model/README.md)

## Model references

- **YOLO11 and YOLO26:** Ultralytics. [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics). Experiments in this repository pin the exact package version in their environment or run metadata.
- **RT-DETR:** Zhao et al., “DETRs Beat YOLOs on Real-time Object Detection,” CVPR 2024. [arXiv:2304.08069](https://arxiv.org/abs/2304.08069).
- **SAM 2:** Ravi et al., “SAM 2: Segment Anything in Images and Videos,” 2024. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714).

## Status — 2026-11-15

- `test_main_v1` remains the primary independent benchmark; on it, YOLO26m leads mAP50 (0.7806), mAP50:95 (0.4296) and MEAN_5 (0.7779), and RT-DETR-R101 leads F1 (0.7973).
- The seed-42 Raw-versus-Curated comparison was evaluated on `val_clean`, not on an independent test set.
- Curation-module ablation: leave-one-out on `val_clean` is complete; add-one-in results are on Camber and still to be imported.
- Seed 36: raw YOLO runs evaluated; curated runs need re-evaluation; RT-DETR runs in progress. Seed 85 notebook is ready.
- Operational min-8 v5 is the latest exploratory audit and still requires independent human verification.
- Deployment code now lives outside this repository; `legacy/` remains preserved as research history.
