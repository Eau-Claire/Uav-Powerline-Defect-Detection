## 5.2.2 Matched results on operational min-8 audit V5

All twelve fixed checkpoints (Raw and Curated training across six architectures) were evaluated with the same evaluator on `test_operational_min8_v5_full_consensus_assisted`. The set contains 1,871 images and 2,467 boxes after the fixed minimum-side exclusion and V5 consensus-assisted annotation updates. Because evaluated model families contributed proposals to V5, these results are an operational exploratory comparison rather than a replacement for the independent `test_main_v1` benchmark.

**Table RC-9a.** Matched Raw-versus-Curated detection results on operational min-8 audit V5.

| Train   | Model        |   Precision |   Recall |     F1 |   mAP50 |   mAP50:95 |   MEAN_5 |
|:--------|:-------------|------------:|---------:|-------:|--------:|-----------:|---------:|
| Raw     | YOLO11n      |      0.8138 |   0.6601 | 0.7289 |  0.7234 |     0.3983 |   0.7104 |
| Raw     | YOLO11m      |      0.8037 |   0.7283 | 0.7641 |  0.7715 |     0.4281 |   0.7678 |
| Raw     | YOLO26n      |      0.8198 |   0.7132 | 0.7628 |  0.7704 |     0.4274 |   0.7667 |
| Raw     | YOLO26m      |      0.8576 |   0.7357 | 0.7920 |  0.7943 |     0.4463 |   0.7935 |
| Raw     | RT-DETR-R50  |      0.8454 |   0.7316 | 0.7844 |  0.7403 |     0.4075 |   0.7359 |
| Raw     | RT-DETR-R101 |      0.9056 |   0.4686 | 0.6176 |  0.7433 |     0.3987 |   0.7441 |
| Curated | YOLO11n      |      0.7912 |   0.6844 | 0.7340 |  0.7551 |     0.4277 |   0.7141 |
| Curated | YOLO11m      |      0.8054 |   0.7364 | 0.7694 |  0.7875 |     0.4464 |   0.7506 |
| Curated | YOLO26n      |      0.8193 |   0.7443 | 0.7800 |  0.7915 |     0.4465 |   0.7585 |
| Curated | YOLO26m      |      0.8584 |   0.7673 | 0.8103 |  0.8284 |     0.4780 |   0.7999 |
| Curated | RT-DETR-R50  |      0.8339 |   0.7699 | 0.8006 |  0.7972 |     0.4408 |   0.7702 |
| Curated | RT-DETR-R101 |      0.8665 |   0.7766 | 0.8191 |  0.8079 |     0.4606 |   0.7790 |

**Table RC-9b.** Curated-minus-Raw differences on operational min-8 audit V5, in percentage points.

| Model        |   Delta F1 (pp) |   Delta mAP50 (pp) |   Delta mAP50:95 (pp) |   Delta MEAN_5 (pp) |
|:-------------|----------------:|-------------------:|----------------------:|--------------------:|
| YOLO11n      |           +0.51 |              +3.17 |                 +2.94 |               +0.37 |
| YOLO11m      |           +0.53 |              +1.60 |                 +1.83 |               -1.72 |
| YOLO26n      |           +1.72 |              +2.11 |                 +1.91 |               -0.82 |
| YOLO26m      |           +1.83 |              +3.41 |                 +3.17 |               +0.64 |
| RT-DETR-R50  |           +1.62 |              +5.69 |                 +3.33 |               +3.43 |
| RT-DETR-R101 |          +20.15 |              +6.46 |                 +6.19 |               +3.49 |

Curated YOLO26m achieves the highest mAP50 (0.8284), mAP50:95 (0.4780), and MEAN_5 (0.7999). Curated RT-DETR-R101 has the highest F1 (0.8191), while Raw RT-DETR-R101 has high precision but substantially lower recall at the reported operating point. The matched results therefore support YOLO26m as the strongest overall V5 detector under threshold-independent AP metrics, while the V5 provenance remains explicit.

**Figure RC-1.** Matched Raw-versus-Curated F1, mAP50, and mAP50:95 on operational min-8 audit V5. This is a model-assisted exploratory protocol, not the primary independent test.
