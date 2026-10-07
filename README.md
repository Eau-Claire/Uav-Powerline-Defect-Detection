# UAV Powerline Defect Detection — `yolov11sdi` (V6.2E → V6.5)

[![tests](https://github.com/Eau-Claire/Uav-Powerline-Defect-Detection/actions/workflows/tests.yml/badge.svg)](https://github.com/Eau-Claire/Uav-Powerline-Defect-Detection/actions/workflows/tests.yml)

Research pipeline for **StructDETR: Structural-Consistency-Aware Detection
Transformer for UAV-Based Electrical Infrastructure Inspection**. It replaces
a chain of one-off Kaggle notebooks with one Python package that:

* runs **locally first** (Jupyter or CLI), and the **same code** runs on Kaggle (generated launchers) and Camber;
* is **resumable**: every expensive stage checkpoints to `state/<stage>.json` and reuses verified results;
* keeps **lineage**: deterministic content-derived tags, immutable freeze manifests, frozen VAL, untouched TEST.

> Current experiment: **V6.5 data ablation** — does additional data help the weak
> classes (foreign_object, broken_strand) at YOLO11n@640 before spending compute
> on YOLO11m@1024 / RT-DETR? It is **not** the final V6.5-vs-V6.4 benchmark.

## Research lineage

| Version | Tag | Notes |
|---|---|---|
| V6.2E (canonical parent, immutable) | `v6p2e_23562a45b343` | train 17,483 / val 2,272 / test 2,012; base `56b7ea194ee5`, FOTL overlay `ee2d3c4497fd`, parent `v6p1_5d0b1149891c` |
| V6.3 ROI-A | `v6p3_roi_a_8b9ed089d547` | ×1.2 box scaling — did not fix the localization gap, rejected |
| V6.4 clean | `v6p4_clean_official` (bundle) | not exactly replayable from the bundle; not used as base |
| V6.5 external stage | `v6p5_ext_<12hex>` | **not produced yet (unverified)** |
| V6.5 compose | `v6p5_compose_<12hex>` | **not produced yet** — last Kaggle run: `Compose candidates: []` |

## Classes (fixed order — part of the dataset contract)

| id | class | meaning / note |
|---|---|---|
| 0 | `insulator` | whole insulator — base object the detector anchors on |
| 1 | `insulator_broken` | broken / missing disc, broken glass |
| 2 | `insulator_flashover` | flashover / arc burn marks |
| 3 | `bird_nest` | bird nest on tower / line (may later fold into foreign_object) |
| 4 | `foreign_object` | kite, balloon, plastic bag, etc. on or near the line |
| 5 | `broken_strand` | broken / fractured conductor strand, cable discontinuity |

## Data sources

In V6.2E: **ZHENG1600, IDID, CPLID, VPMBGI, InsuFault, FOTL/EFOD_Drone**.
Added by V6.5: **Open Images V7** (foreign-object appearance only), **MPCD** (broken_strand),
**Power Equipment Image Dataset** (`dx_dg/yw/nw`; `dx_sg` review-only).
Studied but not merged: Insulator-DET, BGI, CPMID, TLID, Porcelain Disk Insulator 2026, MPID,
InsPLAD, TTPLA, TL-Defect5K.
**Full citations, DOIs and dataset links: [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md).**

## Results so far (YOLO11n@640, frozen V6.2E VAL)

| dataset | mAP50 | mAP50:95 | gap |
|---|---|---|---|
| V6.2E original | 0.768 | 0.413 | 0.355 |
| V6.3-ROI-A candidate (×1.2 boxes) | 0.794 | 0.431 | 0.364 |

Per class (V6.2E): insulator .934/.638 · broken .800/.407 · flashover .755/.326 ·
bird_nest .831/.455 · foreign_object .537/.295 · broken_strand .749/.356 (AP50/AP50:95).
YOLO11m@1024 (V6.2E, 40 ep): 0.815 / 0.457 / gap 0.358. V6.4 post-clean (YOLO11m@1024, RAW VAL): ≈0.839 / 0.472 / gap 0.367.

### Diagnosis

* The model **recognizes the right classes, but boxes are loose**: the mAP50 → mAP50:95 gap stays ≥ 0.3
  for every version (and ~0.36 even with more capacity / resolution).
* mAP improves slowly between epochs.
* Classification is fine → predictions are **conservative** → many **missed objects** → low recall,
  which also drags strict-IoU (localization) metrics down.
* Earlier analysis: tiny boxes do not explain the gap; predicted/GT area ratio ≈ 1; ROI-A box scaling
  did not fix it → class-specific **box semantics / extent consistency** is the main suspect
  (flashover, broken_strand), while foreign_object is mainly a **recall / coverage** problem —
  the target of the V6.5 data ablation.

### Data checklist (per source)

`pytest -v tests/test_data_checklist.py` runs one test per step:

- [x] raw sources pinned (version / id / repo + file)
- [x] canonicalize taxonomy (all mappings land in 0..5; `dx_sg` never mapped)
- [x] canonicalize bbox (VOC px → normalized YOLO, clipped, degenerate boxes dropped)
- [x] QA per source (review manifest per source, external_eval never → TRAIN, nothing pre-approved)
- [x] provenance preserved (source, split, original path, sha256, license; synthetic keeps both lineages)
- [x] dedup / near-dup grouping (exact SHA vs every V6.2E split, pHash ≤ 4 grouping)
- [x] split (V6.2E counts verified vs manifest; deterministic external split ≈ 80/20)
- [x] freeze (content-derived tag, policy flags, immutable registry)

These tests check the **code paths**; whether the real V6.5 sources pass is only known after
`notebooks/03` runs (results land in `state/dedup.json` and the freeze).

## Pipeline

```mermaid
flowchart TD
    P[V6.2E immutable<br/>v6p2e_23562a45b343] -->|stream, no extraction| PP[parent<br/>≤1200 TRAIN backgrounds<br/>VAL+TEST pHash cache]
    OI[Open Images V7 TRAIN<br/>detections only + GrabCut] --> OIC[openimages<br/>masked crops, 250/class]
    M[MPCD] --> MS[mpcd<br/>broken → broken_strand]
    PE[Power Equipment HF zip] --> PES[power_equipment<br/>dx_dg/yw/nw mapped<br/>dx_sg review-only]
    PP --> SY[synthetic<br/>relation-aware foreign_object<br/>100/subtype, TRAIN only]
    OIC --> SY
    PP --> DD[dedup<br/>exact SHA vs all V6.2E<br/>pHash ham≤4 vs VAL/TEST]
    MS --> DD
    PES --> DD
    SY --> EF[external_freeze<br/>v6p5_ext_hash]
    DD --> EF
    MS --> EF
    PES --> EF
    EF --> CQ{compose<br/>human KEEP/DROP}
    CQ -- pending --> NR[needs_review<br/>fill decisions CSV]
    NR --> CQ
    CQ -- resolved --> CT[v6p5_compose_hash<br/>index.synthetic_archive]
    CT --> Y[yolo11n_ablation<br/>YOLO11n@640, 50 ep<br/>frozen V6.2E VAL]
    Y --> EV[evaluate<br/>vs V6.2E baseline + gate]
    EV -->|gate pass + human promotion| CAN[canonical V6.5 freeze]
    CAN --> Y11M[YOLO11m@1024]
    CAN --> RT[RT-DETR-R50 B0 → B1/B2/B3]
```

## Directory structure

```text
configs/            YAML configs (base -> v6p2e -> v6p5_external -> v6p5_compose -> yolo11n_ablation)
notebooks/          01..06: ordered steps to train on the newest dataset (generated by scripts/build_notebooks.py)
src/yolov11sdi/     package: config, paths, environment, state, hashing, io_utils,
                    archive_utils, camber, pipeline, dataset/, qa/, training/, reporting/
scripts/            run_stage.py, status.py, sync_camber.py, verify_artifacts.py,
                    export_kaggle.py, build_notebooks.py
tests/              unit tests + offline end-to-end smoke test
data/               bulky regenerable data (not committed) — see data/README.md
artifacts/          remote_mirror/ (Camber hierarchy), freezes/, manifests/, qa/, compose/, exports/
runs/               yolo11n_640/, yolo11m_1024/, rtdetr_r50/ (weights, results.csv, run_state.json)
state/              <stage>.json — resume/progress records (commit these)
logs/               logs/<stage>/<timestamp>.log
legacy/notebooks/   historical Kaggle notebooks (read-only)
dist/kaggle/        generated Kaggle launchers
docs/               DATA_SOURCES.md (citations), VERSIONING.md, MIGRATION_PLAN.md, context/
```

## Local setup

```bash
git clone https://github.com/Eau-Claire/Uav-Powerline-Defect-Detection.git
cd Uav-Powerline-Defect-Detection
python -m venv .venv-research
source .venv-research/bin/activate
pip install -e ".[dev,notebook]"          # core + tests + Jupyter
pip install -e ".[stage]"                 # data staging: imagehash, OpenCV, FiftyOne 1.22.1, gdown, HF hub
pip install -e ".[train]"                 # GPU training: ultralytics 8.4.172 (+ torch)
python -m ipykernel install --user --name yolov11sdi --display-name "yolov11sdi (.venv-research)"
pytest
```

(`.venv/` is the existing FiftyOne viewer environment; the research venv is `.venv-research/`.)

### Camber

Install the CLI (`curl -sL https://cli.cambercloud.com/install-v2.sh | bash`) or use
`drone-modal/.tools/camber` via `export CAMBER_BIN=...`. Authenticate with
`camber login` **or** `export CAMBER_API_KEY=...` (see `.env.example`; never commit `.env`).
Camber is optional for local work: without it outputs stay local and can be pushed later
with `scripts/sync_camber.py`. Remote paths are unchanged from the Kaggle era.

## Jupyter workflow

```bash
jupyter lab notebooks/
```

Only the ordered steps needed to train on the newest dataset. Each notebook starts with a
**"DATASET TRÊN CAMBER"** cell: the exact Camber input/output paths, the tags in use
(`EXTERNAL_STAGE`, `COMPOSE`; `None` = automatic) and a check that Camber is reachable
(storage is primarily Camber; `REQUIRE_CAMBER=False` for a purely local run).

| Step | Notebook | Camber input → output |
|---|---|---|
| 01 | `01_setup_and_check` | — (env, deps, Camber login, status) |
| 02 | `02_prepare_parent_v6p2e` | `datasets/v6p2e/v6p2e_23562a45b343/` + base/overlay archives → `v6p5_external_stage/parent_cache/` (pHash cache) |
| 03 | `03_build_external_stage_v6p5` | Open Images / MPCD / Power Equipment → `v6p5_external_stage/v6p5_ext_<12hex>/` + `mpcd/`, `power_equipment/` |
| 04 | `04_review_and_compose_v6p5` | `v6p5_ext_<12hex>` → `compose_qa/v6p5_compose_<12hex>/` (+ `compose_qa/review_decisions/`) |
| 05 | `05_train_yolo11n_640_ablation` | `v6p5_compose_<12hex>` → `checkpoints/v6p5_data_ablation/v6p5_y11n_ablation_<12hex>/` |
| 06 | `06_evaluate_vs_v6p2e_baseline` | run report → comparison CSV next to the run on Camber |

YOLO11m@1024 / RT-DETR notebooks are added only after canonical V6.5 is frozen
(configs already exist: `configs/yolo11m_1024.yaml`, `configs/rtdetr_b0.yaml`).
Old Kaggle notebooks: `legacy/notebooks/` (renamed `<version>-<NN>_<date>_<content>[__FINAL|__EXECUTED]`, see its `INDEX.csv`).

Notebooks only call `yolov11sdi.pipeline.run_stage(...)`; set `FORCE_REBUILD=True` in a
notebook only when you really want to rebuild.

## CLI

```bash
python scripts/status.py                    # stage table, disk, checkpoint, pending reviews
python scripts/status.py --remote           # + read-only Camber tag listing
python scripts/run_stage.py parent          # default = resume/reuse
python scripts/run_stage.py external        # parent .. external_freeze in order
python scripts/run_stage.py compose         # exit code 2 = needs human review
python scripts/run_stage.py yolo11n_ablation
python scripts/run_stage.py evaluate
python scripts/run_stage.py synthetic --force
python scripts/run_stage.py dedup --dry-run
python scripts/verify_artifacts.py          # sha256 + archive integrity vs freezes/state
```

Stages: `parent, openimages, mpcd, power_equipment, synthetic, dedup, external_freeze,
compose, yolo11n_ablation, evaluate`.

## Kaggle workflow

```bash
python scripts/export_kaggle.py    # -> dist/kaggle/KAGGLE_0{3,4,5}_*.ipynb (same step numbers as notebooks/)
```

Each launcher embeds a zip of `src/` + `configs/` (re-export after every change), installs only
the needed extras, reads `CAMBER_API_KEY` from Kaggle Secrets, uses `/kaggle/working/yolov11sdi`
purely as a cache and uploads persistent outputs to Camber. Config hashes and dataset tags are
identical to local runs (verified). After a Kaggle restart, "Run all": training restores
`last.pt` + `run_state.json` from `checkpoints/v6p5_data_ablation/<run_tag>/` and resumes.

## Checkpoint / resume

* `state/<stage>.json`: `status` (running/complete/failed/needs_review/stale), config hash,
  input + output SHA256, progress, summary, history of stale records.
* A complete stage is reused when its stage-specific config hash, input hashes and outputs match.
  Changing a semantic config key marks the old record `stale` (kept in history) and reruns.
  `runtime:` keys never affect hashes.
* Incremental checkpoints: pHash (partial CSV), background pool (deterministic files),
  Open Images (per class), synthetic (per subtype), downloads (verified local mirror),
  archives (deterministic, digest-matched).
* Training: `runs/<family>/<experiment>__<compose_tag>/` with `weights/{best,last}.pt`,
  `results.csv`, `args.yaml`, `experiment_manifest.json`, `run_state.json`. `last.pt` +
  matching train hash ⇒ `resume=True`. A material change (model/imgsz/epochs/batch/seed/
  dataset/compose) moves the old run to `<run>__stale_<hash>` — never deleted.
  Checkpoints sync to Camber every `training.sync_every_epochs` (best effort).

### Recover after a kernel crash / reboot

```bash
python scripts/status.py                       # see what is running/failed
python scripts/run_stage.py <same stage>       # resumes from the last checkpoint
```

## Tests

```bash
pytest            # integrity checks + data checklist + offline end-to-end smoke test (34 tests)
```

The smoke test runs every stage on tiny fake data (downloads and GPU training stubbed):
parent streaming, pHash, direct overlays, synthetic, dedup, freeze, review gate, compose,
dataset build with TEST/VAL guards, report and evaluate, plus resume/reuse.
GitHub Actions runs it on every push (`.github/workflows/tests.yml`).

## Versioning

See `docs/VERSIONING.md`. Short version: code = git tag `vX.Y.Z` + commit recorded everywhere;
data = content-addressed tags; experiment ID `v6p5-abl01-yolo11n-640-e50`; each run's
`run_state.json → sessions[]` lists which commit trained which epochs.

## Artifact naming

| Artifact | Name | Identity |
|---|---|---|
| External stage | `v6p5_ext_<12hex>` | SHA256 over sorted `relpath:sha256` of the overlay dir |
| Compose | `v6p5_compose_<12hex>` | SHA256 of {stage tag, overlay SHA, review decisions, approved synthetic archive SHA, taxonomy} |
| Training dataset | `v6p5_ablation_ds_<12hex>` | SHA256 of the dataset manifest (split/file/origin/sha) + compose tag |
| Remote run | `v6p5_y11n_ablation_<12hex>` | legacy formula {compose tag, model, imgsz, epochs, train count} |

Never timestamps. Local mirror: `artifacts/remote_mirror/<path after stash://bunpmc/projects/yolov11sdi/>`.
Manifests store project-relative paths (`data/...`, `artifacts/...`).

## Data policy

* **TEST is immutable** and never materialized for training. **VAL is frozen** (exact V6.2E VAL SHA set).
* Open Images V7 TRAIN (Balloon, Kite, Plastic bag, Bird, Person) = **appearance source only**;
  `annotation_download=detections_only`, `mask_strategy=local GrabCut foreground mask from
  detection bbox`, `openimages_segmentation_zip_downloaded=false`, `SYNTH_REQUIRE_MASK=True`.
  Generic objects become `foreign_object` only when composited near an insulator on internal
  TRAIN backgrounds (TRAIN only).
* MPCD broken/fracture/discontinuity → `broken_strand`. PowerEquipment `dx_dg→broken_strand`,
  `yw→foreign_object`, `nw→bird_nest`, `dx_sg` review-only. Non-train source splits →
  `external_eval`, never into V6.2E val/test.
* Low disk: streaming tar reads, bounded pool, early archive deletion, detections-only Open Images.

## QA / review policy

* Review manifest `v6p5_candidate_review_manifest.csv` (review_id, candidate_type, overlay_image,
  overlay_label, source, target_split, mapped_class_names, subtype, recommended_action, decision,
  reviewer, comment). Decisions go into `artifacts/qa/v6p5_review_decisions.csv`: `KEEP`, `DROP`, blank.
* `REVIEW_THEN_KEEP` is a recommendation; compose stops with `needs_review` while any TRAIN
  candidate is blank. `compose.auto_approve_recommended` exists only as an explicit, recorded override.
* Exact SHA duplicates vs V6.2E = hard gate. pHash candidates vs VAL/TEST must be reviewed
  (`compose.allow_phash_candidates` recorded in the freeze).
* Synthetic samples intentionally use internal TRAIN backgrounds and are not counted as leakage.
* Only KEEP'ed direct rows enter training (`training.direct_filter_by_review`; see MIGRATION_PLAN C3).

## Current status (2026-10-07)

Nothing in V6.5 is verified complete. The local Camber CLI is not logged in, so remote
artifacts could not be checked; the last Kaggle output showed no compose candidate.
Next: authenticate Camber → `python scripts/status.py --remote` → decide whether to reuse an
existing remote `v6p5_ext_*` (`compose.stage_tag`) or run `python scripts/run_stage.py external`.

---

## FiftyOne viewer for the Power Equipment dataset (existing tool)

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
yay -S mongodb-bin && sudo systemctl enable --now mongodb
./run_fiftyone.sh            # http://127.0.0.1:5151
```

`Power-equipment-image-dataset/` is a clone of the upstream repo (not vendored here, 1.3 GB):

```bash
git clone https://github.com/xiongsiheng/Power-equipment-image-dataset.git
cd Power-equipment-image-dataset && git apply --ignore-whitespace ../tools/power_equipment/local_src_changes.patch
```

Helper scripts: `Power-equipment-image-dataset/src/{data_stat,unique_types_extractor,copy_example_images}.py`.
The V6.5 pipeline itself downloads the Power Equipment zip from Hugging Face, not this clone.
