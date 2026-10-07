# Migration plan: Kaggle notebooks → local-first resumable project

Date: 2026-10-07. Source of truth for settings: the last Kaggle notebook
`KAGGLE_V6P5_STAGE_COMPOSE_RECOVERY_DISKSAFE.ipynb` (copied to
`legacy/notebooks/v6p5/`) and `KAGGLE_V6P5_CAMBER_YOLO11N_640_DATA_ABLATION_TRAIN.ipynb`.
Historical context: `docs/context/chatgpt_context.md`.

## 1. Inventory before migration

| Item | Where | Finding |
|---|---|---|
| Repo notebooks | `notebooks/` | **empty**, no V6.x code in the repo |
| V6.1–V6.5 Kaggle notebooks | `~/Downloads` (outside the repo) | 39 relevant; 34 unique copied to `legacy/notebooks/` |
| FiftyOne viewer | `view_fiftyone.py`, `run_fiftyone.sh`, `.venv` | kept as-is (separate tool) |
| Power Equipment clone | `Power-equipment-image-dataset/` | local raw copy + helper scripts; kept. The pipeline uses the **HF zip** (`sxiong/...`) so tags match Kaggle |
| `drone-modal/`, `sci-demo/` | demo apps | untouched; `drone-modal/.tools/camber` is a Camber CLI binary (v1.0.40, not logged in) |
| Checkpoints | `drone-modal/models/baseline_v5_seed42_best.pt` | V5 demo model only, unrelated to V6.5 |
| Frozen artifacts / manifests | none locally | everything lives on Camber Stash |

### Duplicate implementations found
* Camber helpers (`run`, `stash_ls`, `stash_cp`, `stash_put`) re-implemented in every notebook + every worker string → `camber.py`.
* `sha256_file`, `read_yolo_label`, `write_yolo_label`, `xyxy_to_yolo`, `make_contact`, tag extraction → `hashing.py`, `io_utils.py`, `qa/contacts.py`.
* V6.2E reconstruction exists 3× (staging stream, compose, train worker full-extract) → `dataset/v6p2e.py` (one streaming reader).
* Synthetic composition 2× with different settings (launcher POS90/NEG30 vs DISKSAFE 100) → `dataset/synthetic.py` (DISKSAFE).

### Hard-coded Kaggle paths
`/kaggle/working/...`, `/dev/shm/...`, `/tmp/v65_*` in every notebook → only `paths.py` knows
`/kaggle/working/yolov11sdi` (Kaggle) and `/tmp/yolov11sdi` (Camber worker).
One intentional exception: `power_equipment.split_key_prefix` (see C4).

### Camber paths (unchanged)
`datasets/v6p2e/v6p2e_23562a45b343/`, `datasets/curated_v6_rs1280_56b7ea194ee5.tar`,
`datasets/v6p1/v6p1_5d0b1149891c/v6p1_fotl_overlay_ee2d3c4497fd.tar.gz`,
`datasets/v6p5_external_stage/{v6p5_ext_*,mpcd,power_equipment,compose_qa,jobs}`,
`checkpoints/v6p5_data_ablation/`.

## 2. Verified experiment state (do not over-claim)

* Last real notebook output: `Compose candidates: []` → `RuntimeError` (training blocked).
* No notebook output proves any `v6p5_ext_*`, `v6p5_compose_*` or V6.5 training run exists.
* The local Camber CLI is not authenticated, so the remote state could not be checked.
  Run `python scripts/status.py --remote` after `camber login` / exporting `CAMBER_API_KEY`.
* Every stage therefore starts as `not run (unverified)`.

## 3. Conflicts and deliberate deviations

| # | Topic | Legacy | Now | Why |
|---|---|---|---|---|
| C1 | V6.2E freeze identity key | freeze uses `final_tag` (written as `v6p2d.freeze.json`, stored remotely as `v6p2e.freeze.json`) | validator accepts `final_tag` or `tag`, checks lineage + counts vs config, **fails on any mismatch** | frozen artifact is authoritative |
| C2 | Hard negatives | earlier launcher: POS 90 + NEG 30 per subtype | DISKSAFE value kept: 100 positives/subtype; hard negatives `enabled: false` (future config) | your spec + latest notebook |
| C3 | Direct rows in training | train worker added **every** direct TRAIN row from the fixed archives, ignoring review | `training.direct_filter_by_review: true` → only KEEP rows (compose writes `*.direct_approved_manifest.csv`) | "Do not auto-promote external data". Set `false` to reproduce legacy |
| C4 | PowerEquipment split key | `deterministic_fraction(str(absolute_kaggle_path))` | key rebuilt as `<split_key_prefix>/<relpath>` with the legacy Kaggle prefix | same split on every machine; **no resplit** |
| C5 | Crop / synthetic file ids | hashed absolute paths; one RNG across subtypes | ids from OI image id / background SHA; per-subtype RNG `(seed, subtype)` | portable + resumable per subtype. A rebuild will **not** reproduce a legacy `v6p5_ext_*` tag if one exists remotely → reuse it via `compose.stage_tag` |
| C6 | Archive bytes | `tarfile.add` (gzip mtime, uid) → non-reproducible | deterministic tar.gz/zip (sorted, mtime 0) | digest-matched reuse; compose tag (includes archive SHA) becomes reproducible |
| C7 | pHash failures | silently dropped | still excluded from the cache, but recorded in `parent_phash_failures.csv` | honest QA |
| C8 | Existing compose tag on Camber | raised an error | skip re-upload (content-addressed, identical) unless `allow_existing_compose_tag` | idempotent reruns |
| C9 | Train dataset | full extraction of the 3 parent archives | members streamed straight into the dataset dir | Errno 28 |
| C10 | Ext/compose freeze | overwritten on re-run | first freeze for a tag is kept (immutable), also copied to `artifacts/freezes/` | lineage |
| C11 | Exact-SHA gate | enforced at compose | still enforced at compose (dedup only reports) | legacy semantics |

## 4. Target structure and mapping

| Legacy cell(s) | New location |
|---|---|
| DISKSAFE §0 config | `configs/*.yaml` |
| §1 install + auth | `environment.load_camber_api_key`, `camber.py`, `pyproject` extras |
| §4 parent stream + pool + pHash | `dataset/v6p2e.py`, stage `parent` |
| §5 Open Images GrabCut | `dataset/openimages.py`, stage `openimages` |
| §6 MPCD / §7 PowerEquipment | `dataset/mpcd.py`, `dataset/power_equipment.py`, `dataset/direct_overlay.py` |
| §9 synthetic | `dataset/synthetic.py`, stage `synthetic` |
| §10 contacts + dedup | `qa/contacts.py`, `dataset/dedup.py`, stage `dedup` |
| §11 review + freeze + upload | `qa/review.py`, `dataset/external_stage.py`, stage `external_freeze` |
| §12–15 compose gates, archive, tag, index | `dataset/compose.py`, stage `compose` |
| Train worker A–G | `training/dataset_builder.py`, `training/ultralytics_runner.py`, `training/metrics.py`, `reporting/compare.py`, stages `yolo11n_ablation`, `evaluate` |

## 5. Steps executed

1. Inspected repo + `~/Downloads` notebooks; extracted code of every V6.5 notebook.
2. Wrote package `src/yolov11sdi`, YAML configs, CLI scripts, thin notebooks (generated by `scripts/build_notebooks.py`).
3. Copied (not moved — they live outside the repo) legacy notebooks to `legacy/notebooks/`.
4. Tests: unit tests for the 14 required checks + an offline end-to-end smoke test of all stages.
5. Generated Kaggle launchers with `scripts/export_kaggle.py`; verified the embedded bundle gives the same config hash as local.
