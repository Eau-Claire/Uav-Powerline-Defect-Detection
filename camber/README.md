# Camber model mirror

This directory is a compact local mirror of reusable model artifacts from
`stash://bunpmc/`. It deliberately excludes datasets, dataset archives,
intermediate epoch checkpoints, resumable training states, and legacy curated
runs.

Downloaded files live under `artifacts/` and are ignored by Git. The generated
`inventory.csv` is also ignored because it records a point-in-time remote
snapshot.

Layout:

- `artifacts/benchmark/curated_latest/`: latest Finding Nemo curated benchmark models.
- `artifacts/benchmark/raw/`: matching raw-trained baseline models.
- `artifacts/ablations/`: add-one-in and leave-one-out models and metrics.
- `artifacts/seeds/raw/`: raw seed 36/85 model snapshots.
- `artifacts/active_runs/`: current BN-fixed RT-DETR seed runs.

Refresh the mirror with:

```bash
python camber/sync_models.py
```

The sync is additive/overwrite-safe: it downloads selected files to their
canonical local paths, removes only incomplete `.part` downloads, and never
deletes completed local artifacts or remote data.
