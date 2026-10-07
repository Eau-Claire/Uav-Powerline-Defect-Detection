# Legacy Kaggle notebooks (read-only history)

**Do not run these for new work.** The logic now lives in `src/yolov11sdi` and runs through
`notebooks/01..06` or `scripts/run_stage.py`.

## Naming

```
<version>/<version>-<NN>_<YYYYMMDD-HHMM>_<content>[__FINAL|__EXECUTED].ipynb
```

* `NN`: chronological order inside the version (by the time Kaggle saved the file).
* `YYYYMMDD-HHMM`: saved time (local) — the original files were downloaded from Kaggle.
* `__FINAL`: last revision of its chain (earlier revisions without a suffix are superseded fixes).
* `__EXECUTED`: a Kaggle run **with saved outputs** — these are the evidence of what happened.
* `INDEX.csv`: new name ↔ original Kaggle file name, sha256, role, notes, where the logic lives now.

## Key files

| File | Why it matters |
|---|---|
| `v6p2/v6p2-05_..._v6p2e-freeze-auto-quarantine__FINAL` | latest V6.2E freeze notebook (`v6p2e_23562a45b343`). Which of variants 04/05 produced the frozen tag is **not recorded** in saved outputs |
| `v6p2/v6p2-06_..._v6p2e-yolo11-baselines-2gpu__FINAL` | V6.2E baselines: YOLO11n@640 0.768/0.413, YOLO11m@1024 0.815/0.457 |
| `v6p3/v6p3-08_..._roi-a-...__FINAL` | ROI-A ×1.2 box scaling (`v6p3_roi_a_8b9ed089d547`) — rejected |
| `v6p4/v6p4-06_..._camber-bootstrap-post-clean-audit__EXECUTED` | V6.4 post-clean RAW VAL results (mAP50≈0.839, 50:95≈0.472, per class) |
| `v6p5/v6p5-07_..._yolo11n-640-ablation-train-launcher__EXECUTED` | last V6.5 train attempt: `Compose candidates: []` → blocked |
| `v6p5/v6p5-10_..._stage-compose-recovery-disksafe__FINAL` | **authoritative V6.5 settings** copied into `configs/*.yaml` |

Two Kaggle downloads had misleading names and were re-filed by content:
`v6p4-post-clean-dataset-gate.ipynb` → `v6p4-06…__EXECUTED`;
`v6p4-post-clean-dataset-gate (1).ipynb` was actually the V6.5 YOLO11n train run → `v6p5-07…__EXECUTED`.
