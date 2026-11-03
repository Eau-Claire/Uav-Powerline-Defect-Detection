# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `dataset.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\test_operational_min8_v5_full_consensus_assisted\dataset.yaml)
- **Split**: `test`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-03 13:36:39`

## Overall Benchmark Summary

| Model     | Description                              |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:----------|:-----------------------------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26n   | YOLO26n (epoch 45 - Best validation mAP) |      0.815  |   0.7468 |     0.7795 |    0.7916 |         0.4464 |   0.7586 |          14.77 |  67.7 |
| yolo26m   | YOLO26m (epoch 45)                       |      0.8562 |   0.7689 |     0.8102 |    0.8284 |         0.4783 |   0.7999 |          16.41 |  60.9 |
| yolo11n   | YOLO11n (last epoch)                     |      0.7925 |   0.6838 |     0.7341 |    0.755  |         0.4279 |   0.7141 |          12.06 |  83   |
| yolo11m   | YOLO11m (epoch 45)                       |      0.8047 |   0.7365 |     0.7691 |    0.7874 |         0.4464 |   0.7505 |          16.47 |  60.7 |
| rtdetr50  | RT-DETR-R50vd (epoch 15)                 |      0.8371 |   0.7657 |     0.7998 |    0.7955 |         0.4413 |   0.7684 |          36.7  |  27.2 |
| rtdetr101 | RT-DETR-R101vd (epoch 15)                |      0.8606 |   0.78   |     0.8183 |    0.8102 |         0.461  |   0.7817 |          63.51 |  15.7 |

## Per-Class AP@0.5 Breakdown

| Model     |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:----------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26n   |           0.9566 |                  0.8053 |                     0.6374 |           0.8729 |                0.7492 |               0.7283 |
| yolo26m   |           0.9713 |                  0.8329 |                     0.7567 |           0.9332 |                0.7364 |               0.7402 |
| yolo11n   |           0.9597 |                  0.7557 |                     0.5779 |           0.887  |                0.6886 |               0.6611 |
| yolo11m   |           0.972  |                  0.802  |                     0.6534 |           0.891  |                0.6977 |               0.7085 |
| rtdetr50  |           0.9314 |                  0.8154 |                     0.6747 |           0.9032 |                0.722  |               0.7266 |
| rtdetr101 |           0.9522 |                  0.8284 |                     0.7271 |           0.9255 |                0.7149 |               0.7127 |

## Hardware Profile

| Model     |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:----------|-------------:|---------:|---------------:|------:|------------:|
| yolo26n   |         2.51 |     5.9  |          14.77 |  67.7 |        74.3 |
| yolo26m   |        21.78 |    75    |          16.41 |  60.9 |       280.5 |
| yolo11n   |         2.59 |     6.5  |          12.06 |  83   |        76.2 |
| yolo11m   |        20.06 |    68.32 |          16.47 |  60.7 |       284.7 |
| rtdetr50  |        42.74 |   137    |          36.7  |  27.2 |       827.9 |
| rtdetr101 |        76.4  |   259.81 |          63.51 |  15.7 |       822.4 |
