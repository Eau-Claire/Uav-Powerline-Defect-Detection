# Finding Nemo test audit v5

The portable dataset/audit bundle is stored at:

`stash://bunpmc/projects/finding_nemo/audits/finding_nemo_test_audit_v5_bundle.tar`

- Size: 789,935,104 bytes (0.736 GiB)
- SHA-256: `5B9C59DEA1B5B7D0257A3CB938EB9F9A7A7AD9C304C8BB95B2E3B7A91D3CE51A`
- Contents: frozen `test_main_v1`, operational min-8 subsets, exploratory v2-v5 labels,
  audit previews/decisions, evaluation CSV/JSON/Markdown reports, and reproduction scripts.

## Result roles

- `test_main_v1`: primary independent raw benchmark (1,931 images).
- `test_operational_min8_v1`: secondary UAV-operational subset; excludes an entire
  image if any GT box has a minimum side below 8 px after scaling to 640.
- v2-v5: model-assisted exploratory annotation studies. These must not replace the
  independent benchmark without full human verification because evaluated models
  contributed proposals.

## Latest v5 result

`test_operational_min8_v5_full_consensus_assisted` adds 215 missing healthy-insulator
boxes proposed by four-of-four YOLO checkpoints at mean confidence >= 0.50, on top of
the reviewed broken-strand convention changes.

| Model | mAP50 | mAP50-95 | MEAN_5 |
|---|---:|---:|---:|
| YOLO26m | 0.8284 | 0.4783 | 0.7999 |
| RT-DETR-R101 | 0.8102 | 0.4610 | 0.7817 |
| RT-DETR-R50 | 0.7955 | 0.4413 | 0.7684 |
| YOLO26n | 0.7916 | 0.4464 | 0.7586 |
| YOLO11m | 0.7874 | 0.4464 | 0.7505 |
| YOLO11n | 0.7550 | 0.4279 | 0.7141 |

## Restore

```bash
camber stash cp stash://bunpmc/projects/finding_nemo/audits/finding_nemo_test_audit_v5_bundle.tar .
tar -xf finding_nemo_test_audit_v5_bundle.tar
```
