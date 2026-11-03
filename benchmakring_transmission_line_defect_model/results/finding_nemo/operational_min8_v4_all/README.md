# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `dataset.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\test_operational_min8_v4_consensus_assisted\dataset.yaml)
- **Split**: `test`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-03 11:57:20`

## Overall Benchmark Summary

| Model     | Description                              |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:----------|:-----------------------------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26n   | YOLO26n (epoch 45 - Best validation mAP) |      0.7895 |   0.7443 |     0.7662 |    0.7649 |         0.4214 |   0.7589 |          12.13 |  82.4 |
| yolo26m   | YOLO26m (epoch 45)                       |      0.826  |   0.7668 |     0.7953 |    0.8025 |         0.4539 |   0.8    |          16.83 |  59.4 |
| yolo11n   | YOLO11n (last epoch)                     |      0.7624 |   0.6843 |     0.7212 |    0.7267 |         0.401  |   0.7136 |          13.68 |  73.1 |
| yolo11m   | YOLO11m (epoch 45)                       |      0.7797 |   0.7362 |     0.7573 |    0.764  |         0.425  |   0.7508 |          18.69 |  53.5 |
| rtdetr50  | RT-DETR-R50vd (epoch 15)                 |      0.8115 |   0.7668 |     0.7885 |    0.7678 |         0.4129 |   0.7684 |          36.6  |  27.3 |
| rtdetr101 | RT-DETR-R101vd (epoch 15)                |      0.8347 |   0.7809 |     0.8069 |    0.7838 |         0.4349 |   0.7817 |          60.73 |  16.5 |

## Per-Class AP@0.5 Breakdown

| Model     |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:----------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26n   |           0.7948 |                  0.8055 |                     0.6374 |           0.8748 |                0.7484 |               0.7283 |
| yolo26m   |           0.8149 |                  0.8329 |                     0.7567 |           0.933  |                0.7374 |               0.7402 |
| yolo11n   |           0.7926 |                  0.755  |                     0.5778 |           0.8849 |                0.689  |               0.6611 |
| yolo11m   |           0.8301 |                  0.802  |                     0.6534 |           0.8929 |                0.697  |               0.7085 |
| rtdetr50  |           0.7647 |                  0.8154 |                     0.6747 |           0.9032 |                0.722  |               0.7266 |
| rtdetr101 |           0.7939 |                  0.8284 |                     0.7271 |           0.9255 |                0.715  |               0.7127 |

## Hardware Profile

| Model     |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:----------|-------------:|---------:|---------------:|------:|------------:|
| yolo26n   |         2.51 |     5.9  |          12.13 |  82.4 |        74.3 |
| yolo26m   |        21.78 |    75    |          16.83 |  59.4 |       280.5 |
| yolo11n   |         2.59 |     6.5  |          13.68 |  73.1 |        76.2 |
| yolo11m   |        20.06 |    68.32 |          18.69 |  53.5 |       284.7 |
| rtdetr50  |        42.74 |   137    |          36.6  |  27.3 |       827.9 |
| rtdetr101 |        76.4  |   259.81 |          60.73 |  16.5 |       822.4 |
