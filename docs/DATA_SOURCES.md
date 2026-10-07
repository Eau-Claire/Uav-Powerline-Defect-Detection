# Data sources and citations

Status legend: **V6.2E** = inside the canonical parent dataset · **V6.5** = external
expansion handled by this repo · **candidate** = studied, not merged · **eval-only** = only for
external evaluation · **reference** = domain/reference data, not merged into the defect taxonomy.

Links marked `TODO` were not recorded in the project notes; fill them from the official page
rather than guessing.

## Sources currently used

| Dataset | Status | Used for (→ class) | Paper | Dataset link |
|---|---|---|---|---|
| **ZHENG1600** (Insulator-Defect Detection) | V6.2E | insulator, broken, flashover (insulator specialist) | Zheng et al., "Insulator-Defect Detection Algorithm Based on Improved YOLOv7", *Sensors* 22(22):8801, 2022. [doi:10.3390/s22228801](https://doi.org/10.3390/s22228801) | Dataset Ninja / Supervisely release — `TODO` |
| **IDID** (Insulator Defect Image Dataset) | V6.2E | insulator, broken, flashover | Dataset publication: Lewis & Kulkarni, "Insulator Defect Detection", IEEE DataPort. [doi:10.21227/vkdw-x769](https://doi.org/10.21227/vkdw-x769) | IEEE DataPort (requires accepting competition rules) |
| **CPLID** | V6.2E | whole-insulator context + broken; its synthetic defects are train-only | Tao et al., "Detection of Power Line Insulator Defects Using Aerial Images Analyzed With Convolutional Neural Networks", *IEEE TSMC*. [doi:10.1109/TSMC.2018.2871750](https://doi.org/10.1109/TSMC.2018.2871750) | official `InsulatorDataSet` repository — `TODO` link |
| **VPMBGI** | V6.2E | insulator_broken only (broken glass) | No canonical paper — merged Vietnamese public broken-glass dataset; cite its upstream sources listed in the repo | VPMBGI official repository — `TODO` |
| **InsuFault** | V6.2E (train-only) | insulator, damaged, flashover | "InsuFault-Net: A Deep Learning-Based Insulator Defect Detection Algorithm for Transmission Lines" (repository) | official repository — `TODO` |
| **FOTL / EFOD_Drone** | V6.2E (FOTL overlay `ee2d3c4497fd`) | bird_nest, foreign_object (kite, balloon, plastic) | Gao et al., "YOLOv11-Based UAV Foreign Object Detection for Power Transmission Lines", MDPI, 2025 — DOI `TODO` | MDPI article supplementary / `TODO` |
| **Open Images V7** | V6.5 | foreign_object **appearance only**: Balloon, Kite, Plastic bag, Bird, Person crops (TRAIN, detections + local GrabCut) composited onto power-line TRAIN backgrounds | Kuznetsova et al., "The Open Images Dataset V4", *IJCV* 2020. [arXiv:1811.00982](https://arxiv.org/abs/1811.00982) | [storage.googleapis.com/openimages/web](https://storage.googleapis.com/openimages/web/index.html) |
| **MPCD** (Merged Public Power Cable Dataset) | V6.5 | broken_strand (fractured strands / discontinuities); 1,871 images, 1,906 broken boxes, 2,501 cable masks | Benelmostafa & Medromi, "PowerLine-MTYOLO: A Multitask YOLO Model for Simultaneous Cable Segmentation and Broken Strand Detection", *Drones* 9(7):505, 2025. [doi:10.3390/drones9070505](https://doi.org/10.3390/drones9070505) | [github.com/phd-benel/PowerLine-MTYOLO](https://github.com/phd-benel/PowerLine-MTYOLO) · Google Drive id `1KyciMmwL2_p_-mwckHFqG5fkBA1jzATv` |
| **Power Equipment Image Dataset** | V6.5 | transmission-line subset: `dx_dg→broken_strand`, `yw→foreign_object`, `nw→bird_nest`; `dx_sg` review-only | Xiong et al., "Object recognition for power equipment via human-level concept learning", *IET GTD* 15(10):1578–1587, 2021. [doi:10.1049/gtd2.12088](https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/gtd2.12088) | [github.com/xiongsiheng/Power-equipment-image-dataset](https://github.com/xiongsiheng/Power-equipment-image-dataset) · HF `sxiong/Power-equipment-image-dataset` |

## Studied, not merged

| Dataset | Status | Why / possible role | Paper | Dataset link |
|---|---|---|---|---|
| **Insulator-DET** | candidate (not used yet) | insulator + broken disc / glass loss + pollution flashover; 2,150 images, 9 classes | Li et al., "TLINet", *PLOS ONE*, 2025 (also benchmarks IDID/CPLID) — DOI `TODO` | PLOS ONE / PMC — `TODO` |
| **BGI** | candidate | broken glass only; predecessor of VPMBGI (overlap) | — | BGI official repository — `TODO` |
| **CPMID** | candidate (optional, train-only) | dense / multi-scale insulators + defects; lineage overlap risk | "Cross-scale recognition of dense insulators and defects in complex power grid environments", *Eng. Appl. of AI*. [doi:10.1016/j.engappai.2025.113283](https://doi.org/10.1016/j.engappai.2025.113283) | ScienceDirect |
| **TLID** | eval-only | structural insulator damage — good external evaluation set | Hu et al., "Towards Defect Detection of Transmission Line Insulator: A Dataset, Benchmarks and Challenges", ICPRE 2025. [doi:10.1109/ICPRE67300.2025.11274094](https://doi.org/10.1109/ICPRE67300.2025.11274094) | official TLID repository (request access) |
| **Porcelain Disk Insulator 2026** | eval-only | external-domain test; 255 UAV images, 973 instances, 4 condition classes, CC BY 4.0. *broken* usable; *Dirty/Flashed* must NOT be mapped blindly to flashover | Mendeley Data | [doi:10.17632/5pwrd7gmw2.1](https://doi.org/10.17632/5pwrd7gmw2.1) |
| **MPID** | reference | glass / porcelain / composite insulator localization, not a localized-defect benchmark; 4,807 images / 7,850 instances | "APF-YOLOv8: Enhancing Multiscale Detection and Intra-Class Variance Handling for UAV-Based Insulator Power Line Inspections" (PMC, F1000Research) | Zenodo [doi:10.5281/zenodo.14604384](https://doi.org/10.5281/zenodo.14604384) |
| **InsPLAD** | reference | general power-line asset inspection; do not merge into the defect-region taxonomy | Vieira e Silva et al., "InsPLAD: A Dataset and Benchmark for Power Line Asset Inspection in UAV Images", 2023. [doi:10.1080/01431161.2023.2283900](https://doi.org/10.1080/01431161.2023.2283900) | repository cited in the paper |
| **TTPLA** | reference | tower + power-line detection/segmentation, not defects | Abdelfattah et al., "TTPLA: An Aerial-Image Dataset for Detection and Segmentation of Transmission Towers and Power Lines", ACCV 2020 (CVF Open Access) | official TTPLA repository — `TODO` |
| **TL-Defect5K** | candidate | damper-defect, insulator, damper, nest, grading-ring | Jia, "Mamba-enhanced detection transformer with selective state space models for multi-category defect detection in transmission lines", *Discover Computing* 29(1):489, 2026. [link](https://link.springer.com/article/10.1007/s10791-026-10353-0) | [github.com/Dali8710/TL-Defect5K](https://github.com/Dali8710/TL-Defect5K) |

## Merge rules

* Never map a generic class straight into the taxonomy (`Person`, `Bird` → foreign_object only via
  relation-aware compositing near an electrical structure).
* broken_strand must stay conductor-specific (MPCD, Power Equipment `dx_dg`). `dx_sg` (strand
  loosening) and Porcelain *Dirty/Flashed* are not merged without a written mapping decision.
* Every merged source goes through the checklist in `tests/test_data_checklist.py`:
  raw sources → canonical taxonomy → canonical bbox → QA per source → provenance → dedup /
  near-dup grouping → split → freeze.
* Licenses differ per source (e.g. MPCD repository AGPL-3.0, Power Equipment MIT repository,
  Porcelain CC BY 4.0). Check redistribution terms before publishing any derived images.
