# data/ (not committed)

Bulky, regenerable working data. Override the location with
`YOLOV11SDI_DATA_ROOT=/path/to/big/disk`.

| Dir | Content | Lifetime |
|---|---|---|
| `cache/remote/` | large Camber archives mirrored with the stash hierarchy (V6.2E base/overlay TAR, reconciled ZIP) | deleted after use when `runtime.clean_parent_archives_after_qa: true` |
| `parent/` | `train_background_pool/` (≤1200 TRAIN images), `parent_val_test_phash.csv.gz` | keep |
| `external/` | extracted MPCD, Power Equipment, Open Images masked crops (+ `PROVENANCE.json` with archive SHA256) | keep |
| `working/` | `v6p5_overlay/` (defines the `v6p5_ext_*` digest), synthetic manifests, compose extracts, training datasets | regenerable |

Small, valuable records live elsewhere: `state/`, `artifacts/freezes/`,
`artifacts/manifests/`, `artifacts/qa/v6p5_review_decisions.csv`.
