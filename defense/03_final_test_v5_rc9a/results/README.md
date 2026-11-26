# Finding Nemo operational min-8 V5 benchmark

Matched evaluation of 12 fixed checkpoints (Raw and Curated training across six architectures) on `test_operational_min8_v5_full_consensus_assisted`.

- Protocol: 1,871 images and 2,467 boxes.
- Role: operational, model-assisted exploratory benchmark; it does not replace the independent `test_main_v1` benchmark.
- Overall metrics: `benchmark_summary.csv`.
- Per-class metrics: `per_class_summary.csv`.
- Per-checkpoint evaluator outputs: `raw_*.json` and `curated_*.json`.
- Publication figures: `figures/`.
- Report-ready Markdown and tables: `report/`.
- Per-class Raw/Curated figure: `figures/figure_rc3_per_class_raw_curated_v5.{png,pdf}`.
- Per-class report section: `report/section_5_4_per_class_v5_ready_to_paste.md`.
- Validation mAP50 trajectories: `figures/figure_validation_map50_raw_curated.{png,pdf}`.
- Validation-curve maxima and source logs: `report/validation_map50_maxima.csv`.
- Combined 12-run dynamics: `figures/figure_training_dynamics_12_runs.{png,pdf}`.
- Data-centric workflow: `figures/figure_data_centric_pipeline.{png,pdf}`.

The executed orchestration notebook is preserved as research history at [`legacy/notebooks/v6p7/v6p7-07_20261119-finding-nemo-raw-curated-operational-min8-v5-eval.ipynb`](../../../../legacy/notebooks/v6p7/v6p7-07_20261119-finding-nemo-raw-curated-operational-min8-v5-eval.ipynb). Rebuild report tables and summary figures with [`scripts/build_report_rc_tables_figures.py`](../../../scripts/build_report_rc_tables_figures.py).
