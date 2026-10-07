# Legacy Kaggle notebooks (read-only history)

Copied from `~/Downloads` on 2026-10-07 (originals untouched). Byte-identical
`(1)/(2)` duplicates were skipped. **Do not run these for new work** — the
logic now lives in `src/yolov11sdi` and runs through `scripts/run_stage.py` or
the generated launchers in `dist/kaggle/`.

| Folder | What it documents |
|---|---|
| `v6p1/` | V6.1 merge/audit/freeze, first clean YOLO baselines |
| `v6p2/` | V6.2 → V6.2E rebalance, strict leakage audit, manual review + freeze (`v6p2e_23562a45b343`), V6.2E YOLO11 baselines, bbox/localization audits |
| `v6p3/` | canonical rebuild, ROI-A (`v6p3_roi_a_8b9ed089d547`, ×1.2 box scaling — rejected) |
| `v6p4/` | V6.4 post-clean dataset gate / strict-IoU evaluation |
| `v6p5/` | V6.5 external staging (low disk), Camber offload, compose QA, YOLO11n data-ablation launcher, `..._STAGE_COMPOSE_RECOVERY_DISKSAFE` (**authoritative V6.5 settings**) |

Last known V6.5 outcome (from saved outputs): the YOLO11n launcher printed
`Compose candidates: []` — no compose artifact was proven to exist.
