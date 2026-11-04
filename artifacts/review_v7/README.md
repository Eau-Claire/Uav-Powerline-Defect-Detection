# V7 review: decisions requiring human input

The automated V7 pass (`datasets/v6p5_v7`; details in `changes_v7.csv`) corrected 331 clearly misaligned boxes using teacher predictions (`confidence ≥ 0.8`, `IoU 0.5–0.75`), merged five crack groups into one box per damaged insulator disc, and added 301 missing insulator labels. Validation labels were not changed automatically, avoiding inflated validation metrics.

## A. Annotation-convention decisions

These choices have the largest effect on the mAP50-to-mAP50:95 gap:

1. `insulator_broken`: use one box per damaged disc (recommended), or one box per crack?
2. `broken_strand`: tightly box the frayed region (recommended), or box the entire conductor segment? Existing 640×640 sources use the full segment, whereas 640×480 sources use tight boxes.
3. `insulator_flashover`: box the entire burned disc (recommended), or only the visible burn mark?
4. Should a drone/UAV visible in the image count as `foreign_object`?

## B. Train-to-validation leakage (`val_train_near_dups.json`)

There are 148 `val_clean_v6` images (6.5%) with near-duplicates in training. Manual inspection confirmed 8/8 sampled pairs as true duplicates, including Roboflow augmentations. The recommendation for V8 is to delete the duplicated training images while leaving validation unchanged. Another 409 validation images have near-duplicates within validation, so the current bootstrap confidence interval is probably too narrow.

## C. `review_queue.csv`

The queue contains 283 rows. Column `suggest` records the preliminary recommendation after sample inspection; column `decision` is for the reviewer. Contact sheets are named `*_NN.jpg`; the displayed `#id` matches column `id`. White boxes identify the candidate under review, and colored boxes show existing labels.

| Group | Count | Preliminary recommendation | Required action |
|---|---:|---|---|
| validation `bird_nest` missing label | 13 | **add**: approximately 11/13 appear to be real unlabeled nests | Quickly confirm one sheet |
| validation FO missed GT | 12 | **keep**: labels look correct; the model misses kites, bags, and balloons | None |
| validation `bird_nest` missed GT | 42 | **keep**: most are real nests obscured by equipment | None |
| validation `insulator_broken` missed GT | 82 | Many boxes cover glare, tiny chips, clamps, or ties (`#64`, `#77`) | Domain-expert keep/drop review |
| validation `broken_strand` missed GT | 27 | Depends on decision A2 | Review after A2 is settled |
| validation flashover missed GT | 11 | Some images contain pre-drawn red circles (`#24`, `#25`), which may become a shortcut | Keep/drop and flag marked images |
| training flashover / FO | 90 | Low priority | Review later |

Recommended order: section A (5 minutes), `bird_nest` additions (2 minutes), then the 82 `insulator_broken` cells across four sheets (about 20 minutes). The remaining review is optional.
