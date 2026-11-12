# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `data.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\finding_nemo_human_reviewed_6cls\data.yaml)
- **Split**: `val`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-02 17:18:40`

## Overall Benchmark Summary

| Model     | Description                              |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:----------|:-----------------------------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26n   | YOLO26n (epoch 45 - Best validation mAP) |      0.8297 |   0.7267 |     0.7748 |    0.802  |         0.4411 |   0.7789 |          16.62 |  60.2 |
| yolo26m   | YOLO26m (epoch 45)                       |      0.8286 |   0.7778 |     0.8024 |    0.8312 |         0.4726 |   0.8099 |          29.73 |  33.6 |
| yolo11n   | YOLO11n (last epoch)                     |      0.8141 |   0.7033 |     0.7547 |    0.7777 |         0.4308 |   0.7519 |          21.46 |  46.6 |
| yolo11m   | YOLO11m (epoch 45)                       |      0.8287 |   0.7493 |     0.787  |    0.8234 |         0.4636 |   0.8009 |          27.94 |  35.8 |
| rtdetr50  | RT-DETR-R50vd (epoch 15)                 |      0.8438 |   0.7517 |     0.7951 |    0.807  |         0.4541 |   0.7858 |         126.24 |   7.9 |
| rtdetr101 | RT-DETR-R101vd (epoch 15)                |      0.8656 |   0.7809 |     0.8211 |    0.8269 |         0.472  |   0.8074 |          92.8  |  10.8 |

## Per-Class AP@0.5 Breakdown

| Model     |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:----------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26n   |           0.9177 |                  0.7806 |                     0.7602 |           0.9063 |                0.7059 |               0.7415 |
| yolo26m   |           0.9377 |                  0.8241 |                     0.8004 |           0.9262 |                0.7262 |               0.7724 |
| yolo11n   |           0.9066 |                  0.7343 |                     0.7056 |           0.9004 |                0.6825 |               0.7368 |
| yolo11m   |           0.9356 |                  0.8074 |                     0.755  |           0.9218 |                0.7401 |               0.7804 |
| rtdetr50  |           0.9132 |                  0.795  |                     0.7345 |           0.9111 |                0.7122 |               0.776  |
| rtdetr101 |           0.9239 |                  0.8275 |                     0.7737 |           0.9059 |                0.7532 |               0.7769 |

## Hardware Profile

| Model     |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:----------|-------------:|---------:|---------------:|------:|------------:|
| yolo26n   |         2.51 |     5.9  |          16.62 |  60.2 |        74.3 |
| yolo26m   |        21.78 |    75    |          29.73 |  33.6 |       280.5 |
| yolo11n   |         2.59 |     6.5  |          21.46 |  46.6 |        76   |
| yolo11m   |        20.06 |    68.32 |          27.94 |  35.8 |       288.4 |
| rtdetr50  |        42.74 |   137    |         126.24 |   7.9 |       655   |
| rtdetr101 |        76.4  |   259.81 |          92.8  |  10.8 |       798.4 |
