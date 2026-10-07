# ChatGPT Context — CAPSTONE / StructDETR / V6.5

Last updated: 2026-10-07

## User preferences
- Main language: Vietnamese, informal.
- Prefers direct, practical, step-by-step help.
- For notebooks: provide a real `.ipynb`.
- If a notebook errors, patch the actual latest notebook and compile all code cells before returning it.
- Strong emphasis on reproducibility, leakage control, provenance, Camber persistence, and immutable dataset versions.
- Keep TEST locked during development unless doing final evaluation.

## Project
Title: **StructDETR: Structural-Consistency-Aware Detection Transformer for UAV-Based Electrical Infrastructure Inspection**

Ablation:
- B0 = RT-DETR-R50
- B1 = + Frequency
- B2 = + Structural
- B3 = Full

Practical baselines:
- YOLO11n@640 edge
- YOLO11m@1024 cloud
- RT-DETR-R50@640

Do not present YOLO11m@1024 vs RT-DETR@640 as a strict same-resolution comparison.

## Fixed six-class taxonomy
```text
0 insulator
1 insulator_broken
2 insulator_flashover
3 bird_nest
4 foreign_object
5 broken_strand
```

## Main accepted dataset before V6.4
V6.2E tag:
```text
v6p2e_23562a45b343
```

Split:
```text
Train 17,483
Val    2,272
Test   2,012
```

Camber:
```text
stash://bunpmc/projects/yolov11sdi/datasets/v6p2e/v6p2e_23562a45b343/
```

Expected:
```text
v6p2e.freeze.json
v6p2e_split_manifest.csv
v6p2e_reconciled_labels.zip
```

Base:
```text
stash://bunpmc/projects/yolov11sdi/datasets/curated_v6_rs1280_56b7ea194ee5.tar
```

FOTL overlay:
```text
stash://bunpmc/projects/yolov11sdi/datasets/v6p1/v6p1_5d0b1149891c/v6p1_fotl_overlay_ee2d3c4497fd.tar.gz
```

Leakage wording:
> No detected exact or near-duplicate cross-split leakage under the applied provenance, SHA256, flip-aware perceptual-hash, ResNet18 embedding, filename-lineage, and connected-group audit protocol.

Do not claim mathematically guaranteed zero leakage.

## V6.2E YOLO baselines
YOLO11n@640, 50 epochs:
```text
mAP50     0.768
mAP50:95  0.413
gap       0.355
```

Per class:
```text
insulator            .934 / .638
insulator_broken     .800 / .407
insulator_flashover  .755 / .326
bird_nest            .831 / .455
foreign_object       .537 / .295
broken_strand        .749 / .356
```

YOLO11m@1024, 40 epochs:
```text
mAP50     0.815
mAP50:95  0.457
gap       0.358
```

Important conclusion:
- more capacity/resolution improved AP;
- strict localization gap remained ~0.36.

## Geometry / localization findings
Tiny boxes do not explain the gap.

Important observations:
- broken_strand median area is large, not tiny.
- flashover also not mainly tiny.
- center-only pass@0.75 was much higher than actual for broken / flashover / strand.
- median predicted/GT area ratio near 1, so “predictions always too small” is not the issue.
- class-specific bbox semantics / extent consistency are major suspects.

## ROI-A
Tag:
```text
v6p3_roi_a_8b9ed089d547
```

Policy:
- broken ×1.20 + min side 20px
- flash ×1.20 + min side 20px
- strand ×1.20
- others unchanged
- TEST untouched

Result:
```text
mAP50     ~0.7942
mAP50:95  ~0.4306
gap       ~0.3636
```

Conclusion:
- global bbox scaling does not solve the localization gap.
- do not continue blind ×1.2 / ×1.4 box scaling.

## V6.4 current state
Camber dataset bundle:
```text
stash://bunpmc/projects/yolov11m/datasets/v6p4_clean/v6p4_clean_official
```

Camber checkpoint:
```text
stash://bunpmc/projects/yolov11m/checkpoints/v6p4_clean/v6p4_clean_official/yolo11m_v6p4_clean_official_best.pt
```

V6.4 pipeline reportedly included:
- 79 Missing-GT cases reviewed
- 14 broken-strand Loose Span cases / Defect-Level Cluster policy
- ~600 hard-example crops
- YOLO11m@1024

Important provenance caveat:
- current V6.4 bundle does not fully replay the exact training runtime;
- do not claim exact training reproducibility from the current bundle.

## V6.4 post-clean evaluation
Approx current results on frozen RAW VAL:
```text
mAP50     ~0.839
mAP50:95  ~0.472
gap       ~0.367
```

Approx per-class:
```text
insulator:
AP50 ~0.977
AP75 ~0.803
gap  ~0.268
miss ~1.2%
extent/shape error ~9.9%

insulator_broken:
AP50 ~0.873
AP75 ~0.429
gap  ~0.413
miss ~5.7%
extent/shape error ~26.3%

insulator_flashover:
AP50 ~0.858
AP75 ~0.252
gap  ~0.480
miss ~4.9%
extent/shape error ~36.2%

bird_nest:
AP50 ~0.880
AP75 ~0.597
gap  ~0.340
miss ~3.0%
extent/shape error ~16.6%

foreign_object:
AP50 ~0.656
AP75 ~0.337
gap  ~0.280
miss ~18.4%
extent/shape error ~29.6%

broken_strand:
AP50 ~0.788
AP75 ~0.270
gap  ~0.420
miss ~5.6%
extent/shape error ~46.8%
```

Interpretation:
- insulator is already close to target, leave mostly untouched.
- foreign_object main problem = coverage/detection recall.
- broken_strand main problem = bbox geometry/semantics, not simple recall.
- flashover main problem = strict localization / annotation semantic ambiguity.
- bird_nest reasonably strong.

## V6.5 strategy

### foreign_object
User wants to add likely hazardous objects even if generic source images are not already on power lines.

Generic appearance classes:
```text
Balloon
Kite
Plastic bag
Bird
Person
```

Primary source:
**Open Images V7**

Use TRAIN only.

Do not directly map arbitrary full-image:
```text
Person -> foreign_object
Bird -> foreign_object
```

Instead:
- use object crop/mask as appearance source;
- composite onto internal power-line TRAIN backgrounds;
- positive relation: near/on electrical structure -> foreign_object;
- hard negative: generic object far from structure -> background.

Keep subtype metadata:
```text
foreign_subtype:
balloon
kite
plastic_bag
bird
person
```

### broken_strand
Must use conductor/power-line-specific sources.

#### MPCD / PowerLine-MTYOLO
Map broken / fracture / discontinuity to `broken_strand`.
Useful because it also has cable segmentation masks.

Desired semantics:
```text
partial/frayed strand:
bbox around local damaged region + limited conductor context

complete discontinuity:
bbox around gap / broken ends

do NOT:
box an arbitrary long conductor span
```

#### Power Equipment Image Dataset
Mappings:
```text
dx_dg -> broken_strand
yw    -> foreign_object
nw    -> bird_nest
dx_sg -> review-only
```

`dx_sg` is strand loosening and must not be blindly merged into broken_strand.

## Source table
| Dataset | Use | Paper | Paper link | Dataset link |
|---|---|---|---|---|
| Open Images V7 | foreign_object appearance source | Kuznetsova et al., Open Images Dataset V4, IJCV 2020 | https://arxiv.org/abs/1811.00982 | https://storage.googleapis.com/openimages/web/index.html |
| MPCD / PowerLine-MTYOLO | broken_strand | Benelmostafa & Medromi, Drones 2025 | https://doi.org/10.3390/drones9070505 | https://github.com/phd-benel/PowerLine-MTYOLO |
| Power Equipment Image Dataset | dx_dg→broken_strand, yw→foreign_object, nw→bird_nest, dx_sg review-only | Xiong et al., IET GTD 2021 | https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/gtd2.12088 | https://github.com/xiongsiheng/Power-equipment-image-dataset |

## Camber usage
Kaggle repeatedly ran out of disk.

Current policy:
**Kaggle = launcher/controller only. Heavy data work = Camber compute. Large outputs = Camber Stash.**

Camber secret:
```python
from kaggle_secrets import UserSecretsClient
key = UserSecretsClient().get_secret("CAMBER_API_KEY")
os.environ["CAMBER_API_KEY"] = key
```

Always pass:
```python
env=os.environ.copy()
```
to subprocess calls.

V6.5 stage root:
```text
stash://bunpmc/projects/yolov11sdi/datasets/v6p5_external_stage
```

Jobs:
```text
stash://bunpmc/projects/yolov11sdi/datasets/v6p5_external_stage/jobs
```

## Recent notebooks

### V6.5 zero-disk offload
```text
KAGGLE_V6P5_CAMBER_ZERO_DISK_OFFLOAD.ipynb
```

Purpose:
- free Kaggle disk
- write workers to `/dev/shm`
- offload Open Images / MPCD / Power Equipment / parent background staging to Camber
- keep big artifacts on Stash

Expected source staging:
```text
open_images/
mpcd/
power_equipment/
parent_background_pool/
external_stage_manifest.json
```

### V6.5 compose + QA
```text
KAGGLE_V6P5_CAMBER_COMPOSE_QA_LAUNCHER.ipynb
```

Purpose:
- relation-aware synthetic foreign-object composition
- hard-negative composition
- QA/contact sheets
- review manifest
- exact SHA audit
- produce tag:
  v6p5_compose_<hash>

Expected root:
```text
stash://bunpmc/projects/yolov11sdi/datasets/v6p5_external_stage/compose_qa/
```

### V6.5 YOLO11n data ablation
```text
KAGGLE_V6P5_CAMBER_YOLO11N_640_DATA_ABLATION_TRAIN.ipynb
```

Experiment:
```text
V6.2E
+ MPCD train
+ Power Equipment train
+ synthetic foreign_object
-> YOLO11n@640, 50 epochs
-> V6.2E frozen VAL
```

This is a data-ablation experiment, NOT final canonical V6.5-vs-V6.4 comparison.

Promotion gate:
```text
global mAP50:95 does not regress > 1 point
foreign_object AP50 improves >= 2 points
foreign_object AP50:95 improves >= 1 point
broken_strand AP50:95 improves >= 1 point
```

If pass:
```text
review / promotion
-> canonical V6.5
-> YOLO11m@1024
-> RT-DETR B0
-> B1 Frequency
-> B2 Structural
-> B3 Full
```

## Current blocking issue
Latest training notebook output:
```text
Compose candidates: []
```

Error:
```text
RuntimeError:
Không tìm thấy compose candidate.
Chạy KAGGLE_V6P5_CAMBER_COMPOSE_QA_LAUNCHER trước.
```

Meaning:
- compose stage has not successfully produced a visible `v6p5_compose_<hash>` candidate on Camber;
- training must not start until compose output exists.

Immediate next action:
1. list:
   ```text
   stash://bunpmc/projects/yolov11sdi/datasets/v6p5_external_stage/
   ```
2. verify source staging exists;
3. verify compose worker under `/jobs`;
4. verify compose worker actually ran to completion;
5. verify:
   ```text
   compose_qa/v6p5_compose_<hash>/
   ```
6. then rerun YOLO11n data-ablation notebook.

Do not invent a compose tag and do not silently skip QA.

## Research guardrails
- Keep TEST locked.
- Do not auto-replace GT with predictions.
- Do not globally scale boxes again.
- Keep full source lineage.
- Do not collapse provenance into a generic `base`.
- Generic foreign-object data must be relation-aware.
- broken_strand must remain power-line/conductor-specific.
- final StructDETR experiments should begin only after dataset state is sufficiently frozen that improvements cannot be dismissed as annotation drift.
