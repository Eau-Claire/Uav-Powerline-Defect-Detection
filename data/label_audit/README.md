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
  train 274, val 16, val_clean 34 person boxes. Person is never an alert.
- rule H no longer applied to foreign_object: SAM cut the strings/tails off, so the FO box convention became mixed.
- review_decisions.csv: new `add_box` decision (with w,h); 1 missing FO added in val_clean (balloon in the wires).
Defect labels = v6p5_5cls_v2 exactly (ids +1), apart from that one added FO box.
Known: 159 near-duplicate image pairs inside val (dHash <= 6), 16 val images with a near-duplicate in train.
Training: legacy/notebooks/v6p5/v6p5-18_20261012-v4-person-teacher-distill-yolo26n.ipynb (teacher v2 + new head rows
for insulator/person, student v3 + person row; remap checked: identical predictions on 20 val images).
