# Label audit (2026-10-10)

Model-assisted audit of `datasets/v6p5_roboflow_ablation` (6 cls) -> cleaned 5-class `datasets/v6p5_5cls_clean`.
Auditor model: yolo11n AdamW 6cls best.pt (mAP50 .750 / 50-95 .396), conf 0.05, imgsz 640, CPU.

Pipeline
    scripts/label_audit_predict.py   -> yolo11n_adamw/pred_{train,val}.jsonl
    scripts/label_audit_analyze.py   -> yolo11n_adamw/audit/{issues.csv,summary.json}
    scripts/label_audit_flip.py      -> yolo11n_adamw/audit/flips.csv
    coco_person/run.py               -> coco_person/pred_*.jsonl (COCO yolo11n, person, FO images only)
    scripts/label_audit_sheets.py    -> yolo11n_adamw/sheets/*.jpg (visual review)
    scripts/label_audit_build.py     -> datasets/v6p5_5cls_clean (+ changes.csv, manifest.json)

Findings (visually verified on samples)
- 208 images (182 train / 26 val), almost all bird_nest, have labels mirrored horizontally (gx ~ 1-px, r=0.99).
  Not EXIF. 12/12 sampled verified. -> rule A.
- foreign_object contains workers (people): ~9% of train FO boxes, 15% of val FO; only some workers per image
  are labelled. V6.5 synthetic also pastes `person` as FO. -> rule C (person conf >= 0.6; at 0.5 ~1/6 hits
  were real kites/balloons/bags).
- Synthetic FO pastes: median side 36 px, 55% never fitted by the model even on train. -> rule D.
- broken_strand / FO GT unseen by the model (even on train): ~half sampled show no visible defect
  (distant thin lines). -> rule E (quarantine image, list in changes.csv for human review).
- insulator class: defect sources never label insulator; the model learned it (2/764 defect-only val images
  get an insulator prediction) -> cannot be filled by this model. Dropped (5-class).
- Box extent: pred/GT disagreement is two-sided (area ratio median 1.08-1.18, pred>GT 55-66%) -> noise,
  not a convention shift; GT boxes NOT replaced. insulator_broken unseen GT are mostly real small chips -> kept.

Limits: train predictions come from a model trained on train -> under-reports train errors.
Next: 2-fold cross-prediction on GPU, then re-run analyze/build.

## v2 (2026-10-11) -> datasets/v6p5_5cls_v2
Goal: mAP50 > 0.82 with a model that runs on Raspberry Pi. Same rules as v1 plus rule G (visual review,
`review_decisions.csv`, sheets in `review_sheets/`). FO definition used everywhere:
real foreign object; NOT a person, electrical equipment, vegetation/vines, empty region, or a drawing.
- val_clean: all 85 FO boxes reviewed (not only model misses): 27 dropped (17 persons, 10 no object/vegetation),
  1 drawing image removed -> 57 FO boxes.
- train: 44 FO boxes overlapping COCO person at conf 0.3-0.6 reviewed -> 25 persons dropped, 19 real FO kept.
- train rule-E quarantine reviewed box by box: FO 96 (38 restored, 52 dropped as person/vegetation with the image
  kept as negative, 6 still quarantined); broken_strand 274 (186 restored, 88 far-view boxes with no visible
  defect still quarantined).
Result: train 17,903 images; broken_strand 1,713 -> 1,931 boxes; FO 927 -> 955 boxes.
Training: legacy/notebooks/v6p5/v6p5-16_20261011-yolo26m-teacher-distill-yolo26n-v2.ipynb
(YOLO26m teacher -> distilled YOLO26n -> NCNN export -> eval on val + val_clean).

## v3 (2026-10-12) -> datasets/v6p5_6cls_v3 (shared dataset for YOLO, RT-DETR, StructDETR)
v2 rules + `--keep-insulator` (6 classes, ids unchanged) + rule H (`--tighten sam_tighten/sam_boxes.csv`).
- insulator is labelled only in far-view sources: 5,914 close-up train images of broken/flashover insulators have no
  insulator box, and V6.4 finds insulators in 0/30 of them (10/10 in labelled sources). Its AP measures the source,
  so the headline metric stays MEAN_5 (5 defect classes); insulator is reported separately.
- rule H: SAM 2.1-b prompted with each GT box; the GT is replaced by the mask's bounding box when that box lies
  inside the GT (2% slack, shrink only) and 0.5 <= IoU < 0.85. Applied to insulator + foreign_object only:
  on bird_nest SAM cuts off the twigs at the nest edge, and on broken / flashover / broken_strand (defect regions
  on an object) it grows to the whole shed / wire. Contact sheets checked visually; check notebook:
  legacy/notebooks/v6p5/v6p5-17a_20261012-sam-tighten-check.ipynb.
Result: train 1,594 boxes tightened (insulator 1,141, FO 453), val_clean 148; val untouched; box counts unchanged.
`annotations/instances_{train,val,val_clean}.json` = COCO view of the same labels (category id = YOLO id).
Training: legacy/notebooks/v6p5/v6p5-17_20261012-6cls-v3-teacher-distill-yolo26n.ipynb.

## v4 (2026-10-12) -> datasets/v6p5_v4 (7 classes; replaces v3 as the shared dataset for YOLO, RT-DETR, StructDETR)
Classes: 0 insulator, 1 insulator_broken, 2 insulator_flashover, 3 bird_nest, 4 foreign_object, 5 broken_strand, 6 person.
Why: v3 results (val_clean MEAN_5: teacher 0.800 vs v2 teacher 0.822; student NCNN 0.751). Learning the partial insulator
labels (far-view sources only) cost ~2 points on the defect classes, and person was the largest foreign_object
false-positive source (teacher v3 on val_clean @conf 0.25: 38 TP, 34 FP, of which ~8-10 are people, 14 are box-extent
mismatches and ~5 are vines).
- insulator KEPT in the dataset (StructDETR needs it as context) and SAM-tightened (rule H, insulator only), but the
  Pi YOLO trains with `classes=[1..6]` (insulator ignored), as in v2. Insulator labels are still missing in close-up
  images; YOLOE (yoloe-11l-seg, 2 prompt sets) cannot fill them. Idea for StructDETR: prompt SAM with the
  broken/flashover box (it grows to the whole insulator body) and review.
- rule P: person class from YOLO11x COCO (person_all/, imgsz 960). Only conf >= 0.8 and covering <= 20% of any GT box:
  at 0.5-0.8, COCO labels close-up/top-down insulators and cable joints as person (checked on contact sheets).
  train 274, val 16, val_clean 34 person boxes. Person is its own alert type ("person on/near the line"), never FO.
  Person labels are auto (recall-limited at conf 0.8): review them if person AP matters.
- rule H no longer applied to foreign_object: SAM cut the strings/tails off, so the FO box convention became mixed.
- review_decisions.csv: new `add_box` decision (with w,h); 1 missing FO added in val_clean (balloon in the wires).
Defect labels = v6p5_5cls_v2 exactly (ids +1), apart from that one added FO box.
Known: 159 near-duplicate image pairs inside val (dHash <= 6), 16 val images with a near-duplicate in train.
Training: legacy/notebooks/v6p5/v6p5-18_20261012-v4-person-teacher-distill-yolo26n.ipynb (teacher v2 + new head rows
for insulator/person, student v3 + person row; remap checked: identical predictions on 20 val images).

## v5 (2026-10-13) -> datasets/v6p5_v5 (v4 + insulator filled; the shared dataset from now on)
The insulator class is labelled only in far-view sources: 15,249 train / 1,950 val images had no insulator box.
V6.4, YOLOE and SAM point prompts could not fill them; Grounding DINO base ("insulator") + OWLv2 base
("a photo of an insulator") can. insulator_ov/run.py runs both on every image without an insulator label and on
1,000 train images that have one (benchmark). insulator_ov/fuse.py keeps a Grounding DINO box after NMS when it is
not a group box (holds >= 2 other boxes), not ~the whole image, not on a bird_nest / FO / person GT box, not wrapping
a smaller kept box, and is confirmed by OWLv2 (IoU >= 0.5) or has a high Grounding DINO score. `--auto` picks the
thresholds with the highest benchmark recall at precision >= 0.85 (fuse_choice.json); if no setting reaches 0.75
precision nothing is added. Rule I (`--add-insulator insulator_add.csv`) adds the boxes to train + val_clean
(val stays frozen). Known failure mode on contact sheets: a whole lattice tower boxed as "insulator".
Training: legacy/notebooks/v6p5/v6p5-19_20261013-v5-insulator-filled-teacher-distill-yolo26n.ipynb, run unattended
by pipeline_v5.sh (results -> stash://bunpmc/projects/yolov11sdi/checkpoints/v6p5_v5_20261013/, then the studio stops).

## v6 (2026-10-15) -> datasets/v6p5_v6 (v5 + foreign_object review; built by scripts/build_v6.py from v5)
Why: hard mining with teacher v5 on train (legacy/notebooks/v6p5/v6p5-21_20261015-v6-data-hard-mining.ipynb,
artifacts/v6p5_v6_data/) showed contradictory FO labels in train (workers and vines labelled FO in some images, not in
others) and that the 161 leftover v65 synthetic FO pastes caused 32% of the FO misses (pasted birds labelled FO).
Misses are driven by object size (small boxes miss 12-37%), not by lighting measured inside the box.
FO definition from now on: a real foreign object, including hornet nests and vegetation lying on or climbing a pole,
line or equipment; NOT a person, a bird nest, plants growing on the ground, electrical equipment, an empty region or a
drawing.
fo_review_v6/: make_sheets.py (all 955 train FO GT + 66 unlabelled teacher FO boxes, COCO yolo26s person overlap) ->
visual review of every tile -> decisions_src.py (decisions + user answers, readable) -> decisions.csv;
user_sheets.py / fix_view.py = pages the user checked (person boxes completed with COCO proposals, G034 kite vs man,
one nest split in two boxes merged); apply.py applies decisions.csv to one image's labels (asserts one FO line per box).
- rule J (train): 34 FO -> person (+8 missed persons added), 7 FO -> bird_nest (+1 merged nest), 23 boxes dropped
  (5 plants on the ground, 12 empty/equipment, 4 unclear, 1 group of people, 1 drawing), 6 missed FO added, all 161
  v65 synthetic images removed. FO train 955 -> 734, person 264 -> 306.
- rule K: the v2 audit had dropped FO boxes as "vegetation" (review_decisions.csv, centres only). old_veg_decisions.csv:
  4 val_clean boxes are vegetation on a structure -> restored from the untouched `val` labels in val_clean_v6;
  10 train images held such a box (size unknown) -> removed.
- val and val_clean are v5 unchanged; val_clean_v6 (FO 58 -> 62) is the main metric set from now on.
Baseline v5 on val_clean_v6 (preflight/system_two_tier*.py --split val_clean_v6): Pi n 0.778, teacher 0.822,
two-tier FO@800 @0.05 0.8285 (CI95 0.798-0.860, P>0.82 74%), FO@800 vs 640 +0.0087 (CI95 +0.0009..+0.019).

## v8 / v9 (2026-10-15/16) -> datasets/v6p5_v8, datasets/v6p5_v9 (scripts/build_v8.py, build_v9.py)
- v8 (from v6): rule L removes 170 train images that near-duplicate 148 val images (leak); rule M: val_clean_v8 =
  val_clean_v6 + 11 missed bird_nest (review_v7 ids 140-145, 147-151) - 2 insulator_broken boxes not on an insulator
  (ids 64, 77). leaked_val.txt: score v5/v6-trained models on val_clean_v8 without these 148 images.
- v9 (from v8): rule N holds out test_fo: whole near-duplicate clusters (dHash, Hamming <= 6) holding FO, drawn at random
  until >= 150 FO boxes. Only v9-trained models may be scored on test_fo.
- Conventions decided by the user (2026-10-16), apply to every future label edit: insulator_broken = 1 box per damaged
  shed (only ~3% of boxes look like per-crack groups: 215 train / 16 val); broken_strand = tight around the visible
  fraying + ~5%; insulator_flashover = the whole burnt shed; a drone is not foreign_object.
- Open: 80 small dubious insulator_broken val boxes (review_v8_broken/verify_broken.csv, user verdicts); upper bound if
  all were label errors: broken 0.847 -> 0.926, two-tier MEAN_5 0.817 -> 0.833 (scripts/eval_v8.py).
