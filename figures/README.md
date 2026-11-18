# Paper figures

Scripts that turn the committed benchmark results into publication figures. Every figure is written as a vector PDF (for LaTeX) and a 300 dpi PNG (for Word or slides), grouped by topic:

```text
out/01_dataset/          fig01–fig03, fig21
out/02_raw_vs_curated/   fig04–fig06
out/03_benchmark/        fig07, fig08, fig09, fig12 (one set per protocol)
out/04_analysis/         fig10, fig11, fig13, fig14, fig17
out/05_ablation/         fig15, fig16, fig18, fig19, fig20
out/06_method/           diag01–diag07 (method and project diagrams)
```

```bash
.venv/bin/python figures/make_all.py                       # all figures
.venv/bin/python figures/make_all.py --with-train-sizes    # + object sizes of the 17,903 train images
.venv/bin/python figures/fig07_benchmark.py --only test_main_v1   # one script, one protocol
```

Requirements: `matplotlib`, `pandas`, `numpy`, `Pillow`. All four are already in the repository `.venv`.

## Figure index

| File | Shows | Source |
|---|---|---|
| `fig01_dataset_distribution` | Boxes per class in curated train, val, val_clean and test_main_v1 | `data/curated_combined/manifest.json`, `data/test_main_v1/distribution.csv` |
| `fig02_curation_actions` | Images touched by each curation rule (A, C, D, E, G, H) | `data/curated_combined/manifest.json` |
| `fig03_object_size_<split>` | Short box side after letterbox to 640 px, per class, with 8 px and 32 px marks | YOLO labels and image headers of the split |
| `fig04_raw_vs_curated_val_clean` | Raw vs Curated, six architectures, seed 42 (mAP50, mAP50-95, MEAN_5) | `results_yolo_model/`, `results_rtdetr_best/` |
| `fig05_per_class_gain_val_clean` | Per-class AP change, Curated − Raw | same as fig04 |
| `fig06_seed36_raw_vs_curated` | Seed-36 replication (skipped for now, see below) | `results_seed/finding_seed36_test_audit_v5/` |
| `fig07_benchmark_<protocol>` | P, R, F1, mAP50, mAP50-95 and MEAN_5 of the six selected checkpoints | `results_ablation/finding_nemo/<eval folder>/` |
| `fig08_tradeoff_<protocol>` | mAP50-95 vs latency and vs GFLOPs | same as fig07 |
| `fig09_per_class_<protocol>` | Per-class AP50 and AP50-95 heatmaps | same as fig07 |
| `fig10_val_test_gap` | Validation vs human-reviewed test vs test_main_v1 | `selected_checkpoints_eval{_val,,_test_main_v1}` |
| `fig11_training_curves` | Validation mAP and box loss per epoch, selected epoch marked | `final_comparison/finding_nemo_all_epochs.csv` |
| `fig12_pr_points_<protocol>` | Precision–recall operating points with iso-F1 curves | same as fig07 |
| `fig13_test_protocols` | Score of each detector across test protocols and audit versions | `selected_checkpoints_eval*`, `operational_min8_v{3,4,5}_all` |
| `fig14_audit_yolo26m_per_class` | YOLO26m per-class AP50 across operational min-8 v1, v3, v4, v5 | `operational_min8_yolo26m`, `operational_min8_v{3,4,5}_all` |
| `fig15_curation_ablation_test` | Cumulative and leave-one-out curation modules, YOLO26m, 1,931-image test | `results_ablation/finding_destiny_test_audit_v5/` |
| `fig16_loo_ablation_val_clean` | Leave-one-out ablation on val_clean | `data/loo_val_clean.csv` (copied from the LOO report) |
| `fig17_hardware_profile` | Params, GFLOPs, latency, FPS and VRAM | `selected_checkpoints_eval_test_main_v1` |
| `fig18_v6p5_synthetic_data` | YOLO11n: v6.2E baseline vs +500 synthetic images, per class (val) | `artifacts/v6p5_train_local/20261008T181605Z_6b0eac5f/*report.json` |
| `fig19_v6p5_recipe_curves` | Validation curves of the v6p5 training-recipe runs (SGD, MuSGD, AdamW cosine, AdamW plateau) | `artifacts/v6p5_train_local/*/runs/*/results.csv` |
| `fig20_v6p5_5cls_cleaning_<split>` | YOLO26n 5-class: original vs cleaned labels (`val` and `val_clean`) | `artifacts/v6p5_train_local/20261010T133406Z_*/y26n_5cls_eval.csv` |

Method diagrams (`diagrams.py`):

| File | Shows |
|---|---|
| `diag01_pipeline` | End-to-end method: sources → V6.2E → curation (train only) → six detectors × two arms → selection on val → evaluate once |
| `diag02_curation_modules` | Modules I–IV with rules A–H, their inputs (judge audit, human review, SAM) and per-rule image counts |
| `diag03_ablation_design` | Cumulative stages m0–m4 and leave-one-out runs, fixed hyper-parameters, evaluation sets |
| `diag04_test_protocol` | How test_main_v1 is built from V6.2E TEST, the operational min-8 rule and the v2–v5 exploratory audits |
| `diag05_audit_loop` | Targeted label-audit loop used to produce v2–v5 |
| `diag06_project_pipeline` | Whole project: sources → V6.2E / finding_nemo_raw / test_main_v1 → curation → cross-label → human review → three training lines → evaluation → Pi 3 and server deployment (`diagrams_project.py`) |
| `diag07_cross_label_review` | How `finding_nemo_human_reviewed_6cls` is built: YOLO26m teacher + Roboflow proposals → human review → apply accepted only |

`fig21_dataset_versions` (`diagrams_project.py`) compares boxes per class across Raw (`finding_nemo_raw`), Curated (`curated_combined`) and Reviewed (`finding_nemo_human_reviewed_6cls`) for train, validation and test. All counts in diag06, diag07 and fig21 are read from the label files at run time.

The module and rule definitions in the diagrams follow `scripts/build_curation_module_ablation.py` and `build_curation_loo_ablation.py`. Where `docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md` disagrees, the scripts are followed. The report describes G as "non-grid drop" and Module 4 as canonical harmonisation; in the scripts, G is the other human-reviewed box drops and Module IV is H, SAM tightening. The LOO runs keep Module IV.

`<protocol>` is one of:
- `test_main_v1`: the primary independent test.
- `reviewed_test`: the human-reviewed 6-class test.
- `val`: validation.
- `op_min8_v5`: exploratory and model-assisted.

## Caveats to keep in the captions

- **Evaluation sets differ between figure groups.**
  - fig04 and fig05 use `val_clean`. `configs/dataset_curated.yaml` maps `test` to `images/val_clean`.
  - fig07–fig14 and fig17 use the Finding Nemo protocols.
  - fig15 uses the 1,931-image test set of the v5 audit bundle.
  - fig16 uses `val_clean`.

  Do not combine numbers from different groups in one table.
- **Operational min-8 v3–v5 audits (fig13 and fig14) are exploratory.** The models being evaluated proposed some of the labels, so these results cannot replace `test_main_v1`.
- **MEAN_5 excludes `insulator`.** It is the mean AP50 over classes 1–5. Adding insulator boxes in v5 therefore leaves MEAN_5 unchanged.
- **The RT-DETR training log uses Ultralytics column names with different meanings.** fig11 plots only the RT-DETR mAP columns and leaves RT-DETR out of the loss panels.
- **fig06 is skipped on purpose.** Commit `ceeb9fc` removed the curated seed-36 `metrics.json` files as wrong, but `summary.csv` still holds the stale curated rows. The script uses only runs that still have a `metrics.json`. It will draw the figure again once the curated runs are re-evaluated.
- **The v6p5 ablations (fig18–fig20) are exploratory.** They predate the controlled benchmark, use different datasets and schedules, and several runs stopped early (marked × in fig19). In fig20 the original-label run stopped at epoch 9 and the cleaned run at epoch 66, so part of that gap comes from training length.
- **The distillation and teacher runs are not plotted.** These are `artifacts/v6p5_v2` through `v6p5_v7` and `exp_student_res`. Their class sets (5 vs 6), datasets and schedules all differ, so no fair comparison is possible.
- **`data/loo_val_clean.csv` is copied by hand.** It comes from tables 3.1 and 3.2 of `docs/RESEARCH_REPORT_LOO_CURATION_ABLATION.md`. Update it if the report changes.

## Style

`common.py` holds the shared palette, model order, fonts and figure sizes:
- **Width:** single column is 3.5 in and double column is 7.16 in.
- **Colour:** each model family has one colour (YOLO11 blue, YOLO26 orange, RT-DETR aqua). The first three slots of the reference palette passed the colour-vision check for every pair.
- **Model size:** filled markers and solid lines mark the large variant (m / R101). Hollow markers, dashed lines and hatched bars mark the small variant (n / R50). Model identity therefore never depends on colour alone.
- **Heatmaps:** sequential blue for scores, red–gray–blue for changes.
- **Fonts:** embedded as TrueType (`pdf.fonttype 42`), which journal submission systems accept.

To add a figure, write a new `figNN_*.py` with a `main()` function and add it to `JOBS` in `make_all.py`.
