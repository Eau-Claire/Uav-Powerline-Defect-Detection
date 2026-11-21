# V7.1 171126 imported-annotation audit

Dataset audited:
`benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls_v2_171126`

Scope: the 3,342 imported train images that remain after cross-split leakage
removal, compared with the remaining 17,885 curated-base train images.
Validation and test were not modified.

## Conclusion

The imported data is not ready for another training run. The audit found a
deterministic merge bug, a material box-convention shift, tiny boxes that
violate Rule F, and model-supported missing-label candidates. These findings
are consistent with the fine-tune run underperforming the seed85 checkpoint.

## Critical finding: contradictory insulator labels

All 402 remaining imported `insulator_broken` boxes have an exactly identical
`insulator` box in the same label file (IoU = 1.0), across 385 images. The merge
routine expanded a damage box to the containing insulator extent but retained
the original normal-insulator annotation. Same-class deduplication could not
remove this cross-class contradiction.

Evidence: `cross_class_overlaps.csv`.

This must not be repaired by blindly deleting one class until the intended
`insulator_broken` box convention is confirmed. The imported broken-box median
area is 0.06195, versus 0.01436 in the curated base, indicating that expansion
to the complete insulator also changed the annotation convention.

## Model-assisted missing-label audit

Judge: `weights/v7p1_seed85_curated_yolo26m_best.pt`

Rule-B-compatible screening used confidence >= 0.60 and maximum IoU < 0.30
against every existing ground-truth box in an image.

- 171 candidate missing boxes across 154 images
- 122 `insulator_broken` candidates in Powerline Project 3
- 15 `foreign_object` and 13 `insulator` candidates in power line inspection
- 10 `foreign_object`, 6 `broken_strand`, 3 `bird_nest`, and 2 `insulator`
  candidates in Powerline Project 3
- 2 high-confidence class-conflict candidates at IoU >= 0.50

The candidates are supporting evidence, not automatic ground truth. Review
`potential_missing_boxes.csv` and `class_conflicts.csv` before adding labels.

The power-line-inspection source taxonomy has no insulator class. Therefore,
its mapped labels cannot provide complete six-class annotations when an
insulator is visible. The 13 high-confidence insulator candidates confirm this
partial-label risk.

## Box-convention differences

Median normalized box area:

| Class | Curated base | Imported | Relative shift |
|---|---:|---:|---:|
| insulator | 0.02634 | 0.00372 | imported ~7.1x smaller |
| insulator_broken | 0.01436 | 0.06195 | imported ~4.3x larger |
| bird_nest | 0.00726 | 0.03182 | imported ~4.4x larger |
| foreign_object | 0.03657 | 0.02505 (PLI) / 0.01305 (PP3) | imported smaller |

Full per-class width, height, area, aspect-ratio, and border statistics are in
`class_geometry.csv`.

## Structural and distribution findings

- Imported images: 3,342; imported boxes: 8,293; empty imported labels: 0.
- Imported boxes by class: 4,281 insulator, 402 insulator_broken, 469 bird_nest,
  3,141 foreign_object, 0 insulator_flashover, 0 broken_strand.
- 70 imported boxes have width or height below 4 pixels; the curated remainder
  has 13. The imported boxes violate the stated Rule F threshold.
- 2,820 ambiguous source annotations were omitted inside images that were still
  imported: 2,732 from Powerline Project 3 and 88 from power line inspection.
- The original merge report additionally records 1,141 source images skipped
  because no mapped usable boxes remained.
- No malformed imported annotations were reported.
- Seed85 produced only 1,419 predictions at confidence >= 0.25 on imported
  images and matched 504/8,293 GT boxes by same class at IoU >= 0.50. This low
  agreement reflects both domain/convention shift and incomplete labels; it
  also means judge disagreement alone must not be used for deletion.

## Recommended repair order

1. Recover the original incoming annotations for the 385 broken-insulator
   conflict images and decide whether the canonical class uses the local damage
   box or the full insulator box. Remove the paired normal class only after that
   convention is fixed.
2. Run six-class cross-labeling on every imported image, not only source-class
   mapping, then human-review the 171 high-confidence missing candidates.
3. Apply Rule F to the 70 imported tiny boxes.
4. Review class-specific box constraints, especially broken insulator and bird
   nest, and tighten/normalize them consistently with the curated base.
5. Re-audit, then start a fresh training run from seed85. Do not use the current
   fine-tuned checkpoint as the new baseline.

## Files

- `summary.json`: group/source counts and judge totals
- `potential_missing_boxes.csv`: 171 Rule-B candidates
- `class_conflicts.csv`: high-confidence prediction/GT class conflicts
- `cross_class_overlaps.csv`: deterministic imported GT contradictions
- `class_geometry.csv`: per-source, per-class geometry statistics
- `image_inventory.csv`: image provenance, dimensions, and box counts
