# Versioning and traceability

Four independent version axes. Each run records all four, so any result can
be traced back to exact code, config, data and checkpoint.

## 1. Code version (git)

* Package version in `pyproject.toml` / `src/yolov11sdi/__init__.py` (currently `0.2.0`), semver.
* Git tag per code release: `v0.1.0`, `v0.2.0`, ... Bump MINOR when stage
  behaviour or an artifact contract changes, PATCH for fixes that cannot change outputs.
* Every state file, freeze and run records `git_commit` (suffix `+dirty` when
  uncommitted changes existed — commit before long training runs).

## 2. Dataset version (content-addressed)

| Version | Format | Example |
|---|---|---|
| parent | `v6p2e_<12hex>` | `v6p2e_23562a45b343` (immutable) |
| external stage | `v6p5_ext_<12hex>` | digest of the overlay content |
| compose (reviewed) | `v6p5_compose_<12hex>` | digest of stage + decisions + approved archive |
| training dataset | `v6p5_ablation_ds_<12hex>` | digest of the materialized dataset manifest |

Same content ⇒ same tag on any machine (local / Kaggle / Camber). Never timestamps.

### V6.5 dataset line (built by scripts, not by pipeline stages)

After the Roboflow ablation, V6.5 data was iterated with standalone scripts. These are sequential
names, not content hashes; each build writes a `labels_sha256` / build log as its identity.

| Dataset | Built by | Change vs parent |
|---|---|---|
| `v6p5_v2` … `v6p5_v4` | notebooks `v6p5-16..18` | 6-class v3 with SAM-tightened boxes; v4 adds `person` |
| `v6p5_v5` | notebook `v6p5-19` | 7 classes, insulator filled — base of `v5_orig` / `n_v5` |
| `v6p5_v6` | `scripts/build_v6.py` | foreign_object reviewed (`data/label_audit/fo_review_v6`) |
| `v6p5_v7` | `scripts/build_v7.py` | automatic box fixes from teacher (no measurable effect) |
| `v6p5_v8` | `scripts/build_v8.py` | rule L: −170 train images near-duplicating val; rule M: `val_clean_v8` fixes |
| `v6p5_v9` | `scripts/build_v9.py` | rule N: cluster-aware `test_fo` split (145 img / 150 FO boxes); see `artifacts/build_v9.log` |

Rule: a model may only be scored on a split it has not seen. v5/v6/v8-trained models have seen
`test_fo`; models trained before v8 must be scored on `val_clean_v8` minus `leaked_val.txt`.
Datasets live on Camber (`datasets/v6p5_v*`); package with `scripts/package_dataset.py` (sha256 recorded).

## 3. Experiment ID (human readable, in config)

```
<dataset version>-<ablation id>-<model>-<imgsz>-e<epochs>
v6p5-abl01-yolo11n-640-e50      V6.5 data ablation #01 (current)
v6p5c-yolo11m-1024-e40          on canonical V6.5 (future)
v6p5c-b0-rtdetr-r50-640         StructDETR B0 baseline (future); later b1/b2/b3
```

`v6p5` = candidate/ablation data, `v6p5c` = canonical frozen V6.5.
A new ablation (e.g. foreign-only sources) = new config with `abl02`; never
reuse an ID for a different setup.

Script/notebook experiments after `v6p5-11` are identified by their notebook number
(`v6p5-NN`, see `legacy/notebooks/INDEX.csv`) and model names such as `v5_orig`, `n_v5`, `soup_g_v6`;
results sit in `artifacts/<experiment>/`. The 2×6 benchmark names runs `<raw|curated>_<model>`
(e.g. `curated_yolo26n`) under `benchmakring_transmission_line_defect_model/runs/`.

## 4. Run directory

```
runs/<family>/<experiment id>__<compose tag>/
    weights/{best,last}.pt  results.csv  args.yaml
    experiment_manifest.json   # config_hash, dataset/compose tag, code_commit, code_version
    run_state.json             # epochs, best metric, sessions[]
```

`run_state.json → sessions[]` has one entry per (re)start:
`{code_commit, code_version, host, environment, action: fresh|resume, start_epoch, end_epoch}`.
To reproduce or inspect a run: `git checkout <code_commit>` and load the same config.

Remote copy (legacy-compatible): `checkpoints/v6p5_data_ablation/v6p5_y11n_ablation_<12hex>/`.

## Recommended loop

```bash
git commit -am "..."                 # clean tree => commit recorded without +dirty
python scripts/run_stage.py yolo11n_ablation
git add state/ artifacts/freezes artifacts/manifests artifacts/qa/v6p5_review_decisions.csv
git commit -m "run v6p5-abl01: <result>"
git tag exp/v6p5-abl01               # optional: mark the commit of a finished experiment
```
