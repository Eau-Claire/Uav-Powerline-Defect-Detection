# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `dataset.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\test_main_v3_model_aligned_exploratory\dataset.yaml)
- **Split**: `test`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-03 09:46:14`

## Overall Benchmark Summary

| Model   | Description        |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:--------|:-------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26m | YOLO26m (epoch 45) |      0.8327 |   0.7591 |     0.7942 |    0.7968 |         0.4423 |   0.7974 |          16.46 |  60.8 |
| yolo11m | YOLO11m (epoch 45) |      0.7828 |   0.7267 |     0.7537 |    0.7585 |         0.4132 |   0.7494 |          16.82 |  59.4 |

## Per-Class AP@0.5 Breakdown

| Model   |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:--------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26m |           0.7939 |                  0.8286 |                     0.7578 |           0.9229 |                0.7377 |               0.7402 |
| yolo11m |           0.8038 |                  0.7976 |                     0.6596 |           0.8834 |                0.6982 |               0.7082 |

## Hardware Profile

| Model   |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:--------|-------------:|---------:|---------------:|------:|------------:|
| yolo26m |        21.78 |    75    |          16.46 |  60.8 |       270.4 |
| yolo11m |        20.06 |    68.32 |          16.82 |  59.4 |       346.7 |
