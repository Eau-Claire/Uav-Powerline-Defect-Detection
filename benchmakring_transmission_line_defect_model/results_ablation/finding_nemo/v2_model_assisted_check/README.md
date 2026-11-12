# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `dataset.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\test_main_v2_model_assisted_exploratory\dataset.yaml)
- **Split**: `test`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-03 09:40:21`

## Overall Benchmark Summary

| Model   | Description        |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:--------|:-------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26m | YOLO26m (epoch 45) |      0.8292 |   0.7502 |     0.7877 |    0.7858 |         0.4361 |   0.7842 |          16.75 |  59.7 |
| yolo11m | YOLO11m (epoch 45) |      0.7804 |   0.7193 |     0.7486 |    0.7488 |         0.4083 |   0.7378 |          16.71 |  59.9 |

## Per-Class AP@0.5 Breakdown

| Model   |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:--------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26m |           0.7939 |                  0.8286 |                     0.7578 |           0.9229 |                0.7377 |                0.674 |
| yolo11m |           0.8038 |                  0.7976 |                     0.6596 |           0.8834 |                0.6982 |                0.65  |

## Hardware Profile

| Model   |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:--------|-------------:|---------:|---------------:|------:|------------:|
| yolo26m |        21.78 |    75    |          16.75 |  59.7 |       270.4 |
| yolo11m |        20.06 |    68.32 |          16.71 |  59.9 |       346.7 |
