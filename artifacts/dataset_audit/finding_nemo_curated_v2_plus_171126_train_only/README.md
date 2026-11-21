# Dataset audit — Finding Nemo curated v2 + 171126 train-only

Audited dataset:
`benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls_v2_171126`

Recommended canonical name:
`finding_nemo_curated_v2_plus_171126_train_only`

## Structural result

- train: 21,227 images / 21,227 labels
- val: 2,271 images / 2,271 labels
- test: 1,931 images / 1,931 labels
- imported images present: 3,342 / 3,363 (21 verified leakage images removed)
- unreadable images: 0
- exact duplicate image groups: 0
- exact cross-split duplicates: 0
- malformed/out-of-range label rows: 1

On 2026-11-17, 39 verified strong near-duplicate images were removed from the
train split together with their labels and matching Ultralytics `.npy` disk
caches. This comprises 21 images imported by the 171126 merge and 18 images
inherited from the curated base. Validation and test files were not changed.
The removal manifest, including pre-removal SHA-256 hashes, is stored in
`removed_cross_split_duplicates_20261117.csv`.

## Findings requiring review

1. **Cross-split leakage resolved:** all 39 train-side images confirmed by ORB
   geometric verification were removed. The historical candidate reports are
   retained as pre-removal evidence; their listed train files no longer exist.
2. **One test box slightly exceeds the image boundary:**
   `labels/test/e48eaf92d4e99e99980dc2fae11cb94a16031285929be21af756dc658f6fbbd6.txt`,
   line 1. Do not silently edit a frozen test label; review and version the test
   set if it is corrected.
3. **Class distribution shift remains:** the added data disproportionately
   increases `foreign_object`, while adding no `insulator_flashover` or
   `broken_strand` boxes.
4. There are 1,305 empty train labels after leakage removal.
5. There are 83 train boxes below the Rule F 4-pixel threshold. These require a
   separate pruning decision and were not silently changed by leakage cleanup.

## Files

- `summary.json`: full structural and box audit
- `invalid_labels.csv`: invalid/out-of-range label rows
- `suspicious_labels.csv`: empty labels, duplicate boxes, and very large boxes
- `exact_duplicates.csv`: exact SHA-256 duplicate groups (empty/no findings)
- `same_dhash.csv`: same perceptual dHash groups
- `cross_split_visual_similarity.csv`: pre-removal ORB verification evidence
- `imported_cross_split_leakage_exclusions.csv`: pre-removal imported leakage list
- `removed_cross_split_duplicates_20261117.csv`: removal manifest and hashes

No validation/test files or annotations were changed. The remaining invalid
row belongs to the frozen test set and is intentionally documented only.
