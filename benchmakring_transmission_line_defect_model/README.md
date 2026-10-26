# Controlled 2x6 Factorial Benchmark: Raw vs Curated Data on Power-Line Defect Detection

**Research Title:** *A Data-Centric Benchmark for Multi-Class Power-Line Defect Detection: Dataset Integration, Quality Curation, and Model Evaluation for UAV and Server Deployment*  
**Authors:** Phạm Hoàng Minh Châu, Đỗ Trung Quân, Nguyễn Nhật Thông  
**Mentor:** Đặng Ngọc Minh Đức  

---

## 1. Overview
This codebase provides a reproducible, automated benchmarking framework implementing the **Controlled 2x6 Factorial Experimental Design** specified in the research proposal. It systematically isolates the empirical impact of **Data Curation** (Raw vs 8-Rule Curated Dataset) versus **Model Scaling** across three major object detector families:

1. **NMS-based CNNs:** `YOLO11n` and `YOLO11m`
2. **NMS-free CNNs:** `YOLO26n` and `YOLO26m`
3. **NMS-free Vision Transformers:** `RT-DETR-R50` and `RT-DETR-R101`

Total: **12 Experimental Variants** evaluated on the frozen benchmark testbed ($2,012$ images).

---

## 2. Directory Layout

```text
benchmakring_transmission_line_defect_model/
├── configs/
│   ├── benchmark_config.yaml      # Master config (models, batch sizes, hyperparameters)
│   ├── dataset_raw.yaml           # YOLO configuration for Raw Combined (V6.2E uncurated)
│   └── dataset_curated.yaml       # YOLO configuration for Curated Combined (8-rule cleaned)
├── data/
│   ├── raw_combined/              # Raw dataset (21,767 images, original uncurated labels)
│   └── curated_combined/          # Curated dataset (v6p5_6cls_v3 with SAM tightened boxes)
├── src/
│   ├── __init__.py
│   ├── config.py                  # Config loader and taxonomy helpers
│   ├── dataset.py                 # Dataset verification and Camber sync tools
│   ├── trainer.py                 # BenchmarkTrainer (resumable, OOM-safe for RTX 4060)
│   ├── evaluator.py               # BenchmarkEvaluator (mAP, per-class, latency, FPS profiling)
│   └── report.py                  # BenchmarkReporter (2x6 Factorial Matrix & RQ generator)
├── results/
│   ├── benchmark_test_results.json# Raw test evaluation metrics
│   ├── benchmark_summary.csv      # Flat tabular summary
│   ├── factorial_benchmark_matrix.csv
│   └── factorial_benchmark_report.md
├── benchmark_pipeline.ipynb       # Interactive Jupyter Notebook to orchestrate everything
└── README.md
```

---

## 3. Key Capabilities

### A. Automatic Resumption (Không bao giờ train lại từ đầu)
* Every run tracks its progress inside `runs/benchmark_state.json`.
* If a training run is interrupted (power loss, system reboot, manual cancel), re-running the cell automatically detects `last.pt` and calls Ultralytics with `resume=True`, seamlessly picking up from the interrupted epoch.
* If a run has already finished (`best.pt` exists and test results are logged), it is automatically skipped (`[SKIP]`).

### B. Hardware-Optimized for RTX 4060 (8GB VRAM)
* **Gradient Accumulation:** Maintains an effective batch size of 16 across all models while tailoring physical batch size to avoid GPU Out-Of-Memory (OOM):
  - `yolo11n` / `yolo26n`: `batch=16, accumulate=1`
  - `yolo11m` / `yolo26m`: `batch=8, accumulate=2`
  - `rtdetr_r50`: `batch=4, accumulate=4`
  - `rtdetr_r101`: `batch=2, accumulate=8`
* **AMP (Automatic Mixed Precision):** Enabled by default for faster training and minimal VRAM usage.

### C. Comprehensive Proposal Metrics
* **Accuracy:** Precision, Recall, F1-Score, mAP@0.5, mAP@0.5:0.95, per-class AP, and headline `MEAN_5` defect score.
* **Deployment Profiling:** Latency (ms/frame), Throughput (FPS), Parameters (M), GFLOPS, Peak GPU VRAM (MB).

---

## 4. Quick Start via Jupyter Notebook

Open and run `benchmark_pipeline.ipynb`:

```bash
jupyter lab benchmark_pipeline.ipynb
```

The notebook guides you through 5 simple steps:
1. **Cell 1:** Verify RTX 4060 GPU and import modules.
2. **Cell 2:** Check and link dataset paths.
3. **Cell 3:** Train models (run individually or launch all 12).
4. **Cell 4:** Evaluate all completed checkpoints on the frozen Test set.
5. **Cell 5:** Generate 2x6 Factorial Matrix, plots, and markdown report.

## 5. Upload YOLO Runs to Camber Stash

The uploader sends every file under the `raw_yolo*` and `curated_yolo*` run folders to
`stash://bunpmc/projects/<run-folder>/`, preserving each folder's internal paths. It
defaults to all matching folders; use `--run` to select specific runs. Preview the
files and destination paths before uploading:

```bash
python scripts/upload_yolo_runs_to_camber.py --dry-run
python scripts/upload_yolo_runs_to_camber.py --run raw_yolo11n curated_yolo11n --dry-run
```

After checking the dry run, omit `--dry-run` to upload. The Camber CLI must be
installed and authenticated with `CAMBER_API_KEY` or `camber login`.
