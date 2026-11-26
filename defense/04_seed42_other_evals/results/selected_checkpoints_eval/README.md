# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `data.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\finding_nemo_human_reviewed_6cls\data.yaml)
- **Split**: `test`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-02 16:39:14`

## Overall Benchmark Summary

| Model     | Description                              |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:----------|:-----------------------------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26n   | YOLO26n (epoch 45 - Best validation mAP) |      0.8193 |   0.7232 |     0.7683 |    0.7727 |         0.4246 |   0.7419 |          18.4  |  54.4 |
| yolo26m   | YOLO26m (epoch 45)                       |      0.8591 |   0.7472 |     0.7993 |    0.8092 |         0.4535 |   0.7814 |          16.76 |  59.7 |
| yolo11n   | YOLO11n (last epoch)                     |      0.7953 |   0.6627 |     0.723  |    0.7382 |         0.4074 |   0.7034 |          11.55 |  86.6 |
| yolo11m   | YOLO11m (epoch 45)                       |      0.8056 |   0.7141 |     0.7571 |    0.7697 |         0.425  |   0.7349 |          16.61 |  60.2 |
| rtdetr50  | RT-DETR-R50vd (epoch 15)                 |      0.8343 |   0.7417 |     0.7853 |    0.7769 |         0.4245 |   0.7497 |          38.24 |  26.2 |
| rtdetr101 | RT-DETR-R101vd (epoch 15)                |      0.8635 |   0.7588 |     0.8078 |    0.7974 |         0.4437 |   0.7724 |          63.96 |  15.6 |

## Per-Class AP@0.5 Breakdown

| Model     |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:----------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26n   |           0.9269 |                  0.8023 |                     0.6517 |           0.8634 |                0.7446 |               0.6473 |
| yolo26m   |           0.9484 |                  0.8309 |                     0.7673 |           0.926  |                0.7399 |               0.6427 |
| yolo11n   |           0.912  |                  0.7582 |                     0.5953 |           0.8828 |                0.6866 |               0.5941 |
| yolo11m   |           0.9436 |                  0.8019 |                     0.67   |           0.8892 |                0.6977 |               0.6159 |
| rtdetr50  |           0.9124 |                  0.8123 |                     0.6847 |           0.9045 |                0.7211 |               0.6261 |
| rtdetr101 |           0.922  |                  0.8304 |                     0.7296 |           0.9157 |                0.7138 |               0.6727 |

## Hardware Profile

| Model     |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:----------|-------------:|---------:|---------------:|------:|------------:|
| yolo26n   |         2.51 |     5.9  |          18.4  |  54.4 |        74.3 |
| yolo26m   |        21.78 |    75    |          16.76 |  59.7 |       280.5 |
| yolo11n   |         2.59 |     6.5  |          11.55 |  86.6 |        76.2 |
| yolo11m   |        20.06 |    68.32 |          16.61 |  60.2 |       288.5 |
| rtdetr50  |        42.74 |   137    |          38.24 |  26.2 |       843.6 |
| rtdetr101 |        76.4  |   259.81 |          63.96 |  15.6 |       821.8 |
