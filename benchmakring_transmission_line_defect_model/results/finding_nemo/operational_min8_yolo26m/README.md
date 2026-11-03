# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `dataset.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\test_operational_min8_v1\dataset.yaml)
- **Split**: `test`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-03 09:15:01`

## Overall Benchmark Summary

| Model   | Description        |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:--------|:-------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26m | YOLO26m (epoch 45) |      0.8243 |   0.7552 |     0.7882 |     0.785 |         0.4399 |   0.7804 |          15.91 |  62.9 |

## Per-Class AP@0.5 Breakdown

| Model   |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:--------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26m |            0.808 |                  0.8329 |                     0.7567 |           0.9332 |                0.7364 |               0.6427 |

## Hardware Profile

| Model   |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:--------|-------------:|---------:|---------------:|------:|------------:|
| yolo26m |        21.78 |       75 |          15.91 |  62.9 |       270.4 |
