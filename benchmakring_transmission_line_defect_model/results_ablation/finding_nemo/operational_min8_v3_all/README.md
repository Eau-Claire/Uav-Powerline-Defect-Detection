# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `dataset.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\test_operational_min8_v3_model_aligned\dataset.yaml)
- **Split**: `test`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-03 10:02:00`

## Overall Benchmark Summary

| Model     | Description                              |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:----------|:-----------------------------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26n   | YOLO26n (epoch 45 - Best validation mAP) |      0.7866 |   0.7448 |     0.7652 |    0.7635 |         0.4205 |   0.7586 |          10.55 |  94.8 |
| yolo26m   | YOLO26m (epoch 45)                       |      0.8293 |   0.7669 |     0.7969 |    0.8012 |         0.4526 |   0.7999 |          16.11 |  62.1 |
| yolo11n   | YOLO11n (last epoch)                     |      0.7638 |   0.6819 |     0.7205 |    0.7256 |         0.4004 |   0.7141 |           9.88 | 101.2 |
| yolo11m   | YOLO11m (epoch 45)                       |      0.7768 |   0.735  |     0.7553 |    0.7627 |         0.4235 |   0.7505 |          16.83 |  59.4 |
| rtdetr50  | RT-DETR-R50vd (epoch 15)                 |      0.8099 |   0.768  |     0.7884 |    0.7667 |         0.4123 |   0.7684 |          36.49 |  27.4 |
| rtdetr101 | RT-DETR-R101vd (epoch 15)                |      0.8376 |   0.7803 |     0.808  |    0.7825 |         0.4341 |   0.7817 |          62.09 |  16.1 |

## Per-Class AP@0.5 Breakdown

| Model     |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:----------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26n   |           0.7881 |                  0.8053 |                     0.6374 |           0.8729 |                0.7492 |               0.7283 |
| yolo26m   |           0.808  |                  0.8329 |                     0.7567 |           0.9332 |                0.7364 |               0.7402 |
| yolo11n   |           0.7835 |                  0.7557 |                     0.5779 |           0.887  |                0.6886 |               0.6611 |
| yolo11m   |           0.8234 |                  0.802  |                     0.6534 |           0.891  |                0.6977 |               0.7085 |
| rtdetr50  |           0.7586 |                  0.8154 |                     0.6747 |           0.9032 |                0.722  |               0.7266 |
| rtdetr101 |           0.7862 |                  0.8284 |                     0.7271 |           0.9255 |                0.7149 |               0.7127 |

## Hardware Profile

| Model     |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:----------|-------------:|---------:|---------------:|------:|------------:|
| yolo26n   |         2.51 |     5.9  |          10.55 |  94.8 |        74.3 |
| yolo26m   |        21.78 |    75    |          16.11 |  62.1 |       280.5 |
| yolo11n   |         2.59 |     6.5  |           9.88 | 101.2 |        76.2 |
| yolo11m   |        20.06 |    68.32 |          16.83 |  59.4 |       285.8 |
| rtdetr50  |        42.74 |   137    |          36.49 |  27.4 |       829   |
| rtdetr101 |        76.4  |   259.81 |          62.09 |  16.1 |       822.4 |
