# Test protocol: raw vs curated

## Test sets

| Role | Set | Images | Definition |
|---|---|---:|---|
| **Main** | `data/test_main_v1` | 1,931 | Frozen V6.2E TEST, deduplicated by SHA-256, minus 31 images where a `foreign_object` box is a person |
| Reference | `data/v6p2e_frozen_test` | 2,012 | Frozen V6.2E TEST as released (50 duplicate rows, person-as-FO kept) |
| Appendix | `data/curation_stress_test_v1` | 600 | Curation-sensitive stress set (`CURATION_STRESS_TEST_PROTOCOL.md`); never the headline |

Describe the main set as "the V6.2E TEST, deduplicated, person-as-foreign-object images excluded". Do **not** call it a clean or audited test set: no human audit was done.

Main set distribution (`test_main_v1/distribution.csv`):

| Class | Images | Boxes |
|---|---:|---:|
| insulator | 662 | 1,207 |
| insulator_broken | 661 | 809 |
| insulator_flashover | 109 | 124 |
| bird_nest | 166 | 177 |
| foreign_object | 103 | 111 |
| broken_strand | 121 | 134 |
| background (no boxes) | 145 | — |

## Why this set is fair to both arms

- **No leakage.** V6.2E TEST was kept unchanged from V6.1, and V6.2E is audited for exact and near duplicates against TRAIN/VAL (`v6p2e.freeze.json`: `exact_cross_split_overlap = 0`). Curated training (`v6p5`) is V6.2E TRAIN plus 500 synthetic images built only on TRAIN backgrounds (`test_modified = false`).
- **No selection.** Every unique image is kept except the 31 reviewed exclusions. No sampling, difficulty score, class bonus or model output is used.
- **No label edits.** Labels are the original V6.2E labels. None of the curation rules is applied (no flips, no added nests, no tiny-box removal, no SAM tightening). The raw box convention, if anything, disadvantages curated models on AP50-95.
- **Exact duplicates.** The 50 duplicate rows are collapsed, keeping the canonical (non-`__dupN`) file. 48 of these pairs differ only by box jitter (separately annotated identical video frames), and one differs by an extra `insulator` box.

## Person-as-foreign-object exclusion

This project defines a foreign object as a real object, **not a person** (`data/label_audit/README.md`). The FOTL source labels some utility workers as `foreign_object`, and does so inconsistently: some images box only one of several visible workers.

All 134 `foreign_object` images in the TEST were reviewed visually, and each FO box was classified by its main content (`data/clean_test_v2/fo_person_review.csv`):

| Verdict | Images |
|---|---:|
| no_person | 103 |
| person_only | 29 |
| mixed (one person box and one real object) | 2 |

- 48 of 161 FO boxes (30%) are persons.
- The 31 person_only and mixed images are removed as whole images, with no labels edited. Removing only the person boxes would leave people in the images as unlabelled negatives. That is curation rule C applied to the test set, and it would penalise raw models.
- The reference set keeps these images. Report `foreign_object` AP on both sets.

## Label-noise estimate (limitation)

A model-based visual review covered 100 random images drawn from the 1,000-image stratified subset `clean_test_v2/final` (seed `20261028`). The reviewer is not a domain expert. Per-image verdicts are in `data/clean_test_v2/noise_review.csv`.

| Verdict | Images |
|---|---:|
| ok | 67 |
| unsure (mostly tiny broken, flashover or nest boxes) | 27 |
| person_fo (now excluded) | 5 |
| missing (unlabelled frayed strands) | 1 |

Report in Limitations: labels are not human-audited; clear errors ≈ 1% of sampled images; up to 27% are uncertain at review resolution. Label noise affects all 12 checkpoints equally, so the raw-vs-curated comparison stays valid, but absolute deployment accuracy is not established.

## Build

```powershell
python scripts/build_clean_test.py full `
  --dataset-yaml data\v6p2e_frozen_test\dataset.yaml `
  --exclude-csv data\clean_test_v2\fo_person_review.csv `
  --out-dir data\test_main_v1
```

`freeze.json` records the source, the duplicate collapse, the exclusion review hash and the manifest hash (`a5fcd477…78a72`). Commit the small records before evaluating any checkpoint.

`data/clean_test_v2` (the 1,000-image stratified subset and its blind-audit package, built with `build_clean_test.py sample`) is kept only as the frame for the reviews above. It was meant for a human audit that was not carried out.

## Reporting

- Evaluate all 12 checkpoints on `test_main_v1/dataset.yaml` and on the reference set, with identical imgsz, conf and IoU settings.
- Report 6-class mAP50/mAP50-95, MEAN_5 and per-class AP50/AP50-95, with counts from `distribution.csv`.
- Give bootstrap 95% CIs over images, and a paired bootstrap for curated minus raw on the same resamples.
- Do not cite the earlier `val_clean` numbers (`per_class_benchmark.csv`). That split carries curated labels and is the curated run's model-selection split.
