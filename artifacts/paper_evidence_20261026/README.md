# Paper evidence bundle — 2026-10-26

This bundle records a local code-and-artifact audit; it is not a benchmark rerun. `sources/` contains byte-preserving copies of 42 source files in their repository-relative layout, and `source_manifest.json` stores each SHA-256. Notebook `.txt` exports include saved source and output. Full datasets and checkpoints are not included. Missing values are stated explicitly and are never inferred as executed results.

## 1. Dataset inventory and class counts

Saved output in [`sources/benchmakring_transmission_line_defect_model/benchmark_pipeline.ipynb`](sources/benchmakring_transmission_line_defect_model/benchmark_pipeline.ipynb), cell 2, reports raw train 18,334 images / 18,334 label files, curated train 17,903 / 17,903, raw validation 2,272, and curated `val_clean` 2,271. The counter is `verify_dataset_structure` in [`sources/benchmakring_transmission_line_defect_model/src/dataset.py`](sources/benchmakring_transmission_line_defect_model/src/dataset.py). It counts JPG/PNG and TXT files; it does not hash images or detect empty labels. These are file counts, not proof of 18,334 or 17,903 unique image contents. Legacy notebook V6.5-17 also records 17,903 train, 2,272 validation, and 2,271 `val_clean` files.

`sources/artifacts/v6p5_train_local/20261009T024019Z_roboflow/build.json` records 18,334 training files, 351 Roboflow training additions, 74 external-evaluation images, 968 dropped items, and base dataset `v6p5_compose_4b9a452e24e2`. The name of a YAML file alone is insufficient to call those 18,334 files the original canonical V6.2E dataset.

No script or output in this bundle reproduces the manuscript's exact 25,807 / 23,204 box-count table. The actual number of empty label files is unknown. The manuscript's class-image totals are 17,969 raw, 17,061 curated, 2,154 validation, and 1,904 test. Their difference from unique image totals cannot be interpreted as background counts because images may contain multiple classes.

`inventory_dataset.py` is a newly supplied audit utility, not a historical benchmark script:

```bash
python inventory_dataset.py /path/to/raw_combined --out /path/to/inventory_raw
python inventory_dataset.py /path/to/curated_combined --out /path/to/inventory_curated
```

It emits `images.csv`, `splits.csv`, `classes.csv`, and `summary.json`; distinguishes missing, empty, and invalid labels; counts one class-image occurrence per image/class and one box per valid YOLO row; and computes byte-level image SHA-256 uniqueness. It has not been run against the full datasets because they are absent from this checkout.

## 2. Evaluation logs and split identity

`evaluation_inventory.csv` combines 12 records from the historical YOLO and RT-DETR result JSON files while preserving Windows paths, split keys, timestamps, checkpoint paths when present, accuracy, and hardware measurements.

- Eight YOLO records from 2026-10-23 point to `configs/dataset_curated.yaml`. Its current `val` is `images/val`, while `test` is `images/val_clean`. The old JSON does not record checkpoint paths; a README statement that it used `best.pt` is not checkpoint metadata.
- Four RT-DETR records from 2026-10-25 point to `data/curated_combined/data_val_clean.yaml` and `runs/<run>/weights/last.pt`. That runtime YAML is absent, so its contents cannot be inferred from its filename.
- No evaluation image manifest or hash set proves that all 12 runs used identical images and labels. A `split_key=test` field alone does not establish an independent test.
- [`sources/benchmakring_transmission_line_defect_model/src/evaluator.py`](sources/benchmakring_transmission_line_defect_model/src/evaluator.py) computes YOLO F1 as the harmonic mean of mean precision and mean recall, but HF F1 as mean per-class F1. A unified definition is required.
- A current notebook cell points to `v6p2e_test/dataset.yaml`, but the saved code calls `.open()` on a string and does not prove old records were evaluated on that new path.

## 3. FPS and resources

The manuscript's hardware table uses raw-run measurements. Curated YOLO26m is 61.4 FPS / 16.29 ms; raw YOLO26m is 63.6 FPS / 15.72 ms. Curated mAP50 0.8323 must not be paired with raw-run 63.6 FPS as if they came from one run.

In [`sources/benchmakring_transmission_line_defect_model/src/evaluator.py`](sources/benchmakring_transmission_line_defect_model/src/evaluator.py), `profile_hardware` runs YOLO on a zero tensor `[1,3,640,640]` with 50 warmups and 300 timed iterations. The HF branch uses batch 2 and real split images, prepares processor output outside timing, synchronizes around forward passes, and has no separate warmup. These protocols are not equivalent. HF FLOPs are THOP estimates and may omit unsupported operators.

The README names an RTX 4060, but historical result JSON does not store GPU name, CUDA/PyTorch version, or precision. Exact runtime logs are required to verify hardware for each run.

## 4. Training configurations

YOLO uses [`sources/benchmakring_transmission_line_defect_model/configs/benchmark_config.yaml`](sources/benchmakring_transmission_line_defect_model/configs/benchmark_config.yaml) and `train_single_run` in [`sources/benchmakring_transmission_line_defect_model/src/trainer.py`](sources/benchmakring_transmission_line_defect_model/src/trainer.py): 640 px, 50 epochs, AdamW, `lr0=.001`, `lrf=.01`, cosine schedule, 3 warmup epochs, weight decay `.0005`, seed 42, AMP, and device 0. Nano batch is 16 and medium batch is 8. Declared accumulation 1/2 is printed but not passed to training, and `nbs` is unset. Exact `args.yaml` from all eight runs is required to establish effective batch size.

The evaluator reconstructs RT-DETR state dictionaries with PekingU R50vd or R101vd; run key `rtdetr100` maps to R101. These are not interchangeable with Ultralytics `rtdetr-l.pt` or `rtdetr-x.pt` entries elsewhere in the config.

The saved PekingU training notebook records AdamW with backbone LR `1e-4`, head LR `1e-3`, weight decay `1e-4`, cosine minimum LR `1e-6`, R50 batch 8 with accumulation 2, and 50 default epochs. Its validation loader uses `images/val`, and it has no three-epoch warmup. Exact four-run logs and checkpoint mappings are still required. Legacy V3 teacher/student settings belong to another experiment and must not substitute for benchmark metadata.

## 5. Curation and duplicate control

The primary `scripts/label_audit_build.py` implements A hflip, B add-nest, C person, D synthetic-FO, E quarantine, F tiny, G review, and H tighten. Rules A/C/G/H may affect `val_clean`; B/D/E/F are primarily training operations. The exact V3 code revision, arguments, and output records are required because later code added P/I rules and changed SAM policy.

The saved V6.5-17a notebook reports train box adjustments of 1,141 insulators and 453 foreign objects, plus 27 and 2 on `val_clean`, with additional bird-nest candidates. This is an analysis table, not proof that every candidate was applied. A182, approximately 55% synthetic removal, and approximately 9% person screening remain proposal descriptions until recomputed from the exact build's `changes.csv`, `review_decisions.csv`, `sam_boxes.csv`, and manifest.

Duplicate control has three distinct scopes:

1. Legacy V6.2E freeze code checks cross-split SHA-256, pHash, dHash, embeddings, groups, quarantine actions, and preservation of test hashes. Its local notebook has no executed output; the final acceptance block is a specification, not a run log.
2. `sources/artifacts/v6p4_post_clean_audit/` records zero exact groups and 135 pHash candidate pairs: 86 train–validation, 46 train–test, and 3 validation–test. These are candidates, not 135 confirmed leaks; the bundle lacks a complete disposition record.
3. [`sources/src/yolov11sdi/dataset/dedup.py`](sources/src/yolov11sdi/dataset/dedup.py) checks direct external images against the canonical manifest with SHA-256 and against validation/test with pHash. It does not prove that every benchmark build ran those checks, nor does it treat synthetic inputs as direct external images.

## 6. Diffusion augmentation

Diffusion augmentation reached prototype stage but is not evidenced as benchmark training input. Both prototype scripts use `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. Metadata contains 14 foreign-object rows and 16 strand rows; `diffusion_inventory.json` summarizes statuses. `status=ok` means generation completed, not that a human accepted the image.

Original contact sheets are `artifacts/diffusion_fo_proto/sheet.jpg` and `artifacts/diffusion_strand_proto/sheet.jpg`. The 2026-10-15 session log estimates about 25% usable foreign-object samples and 0/16 usable strand samples and states that they were excluded from training. No per-image approval manifest is included. The defensible description is: exploratory synthesis prototype, excluded from reported training; integration into the benchmark remains future work.
