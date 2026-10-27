# Curation stress test v1

This challenge set measures performance on curation-sensitive cases. It is an additional evaluation set, not a replacement for the immutable V6.2E test set.

## Selection contract

- Source population: the immutable V6.2E test split only.
- Selection uses ground-truth class presence, normalized box area, image aspect ratio and object density.
- No prediction, confidence, loss, checkpoint identity or model metric may be used during selection.
- Default size: 600 images, with at least 80 images containing each of the five defect classes.
- Seed: `20261027`.
- Exact duplicate image records are collapsed by SHA-256 in the stress subset; the canonical frozen TEST preserves them.
- The generated `manifest.csv` is frozen by SHA-256 in `freeze.json` before any model evaluation.
- Raw-trained and curated-trained checkpoints must use the same images, labels, resolution and evaluation settings.

The score deliberately emphasizes small objects, oversized boxes requiring review, multi-defect images, dense annotations and non-square imagery. It also gives disclosed priority to `bird_nest` and `foreign_object`, because missing nests and invalid foreign-object annotations are explicit curation-rule targets. Every selected row records these reasons in `selection_reasons`. Results must therefore be reported as a curation stress test and must not be described as performance on the deployment population.

Only images containing at least one of the five defect classes are eligible. The current frozen source stores the context-only `insulator` examples separately, so the generated v1 subset contains no class-0 ground truth. Use `MEAN_5` and the five defect-class AP values as the stress-test metrics; do not report six-class aggregate mAP for this subset.

## Build

```powershell
python scripts/build_curation_stress_test.py `
  --dataset-yaml D:\path\to\v6p2e_test\dataset.yaml `
  --split test `
  --out-dir data\curation_stress_test_v1 `
  --target-images 600 `
  --min-images-per-class 80 `
  --seed 20261027 `
  --materialize hardlink
```

Generated records:

- `manifest.csv`: immutable image membership and ground-truth-only selection features.
- `distribution.csv`: image and box counts by class.
- `freeze.json`: selection parameters and manifest hash.
- `dataset.yaml`: YOLO evaluation configuration when images are materialized.

Commit the three small records before evaluating any checkpoint. Dataset images and labels remain ignored by Git.

## Reporting

Report three views separately:

1. Immutable V6.2E test results.
2. Curation stress test results.
3. Per-class AP50/AP50-95 with the image and box counts from `distribution.csv`.

The defensible claim is that curated training improves performance on curation-sensitive cases. Do not use this set alone to claim universal or deployment-wide superiority.
