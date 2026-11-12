# Finding Nemo: Selected Checkpoints Test Benchmark

- **Dataset Evaluated**: `dataset.yaml` (C:\Users\intern\Documents\SEP490\train\benchmakring_transmission_line_defect_model\data\test_main_v1\dataset.yaml)
- **Split**: `test`
- **Images Evaluated**: `Full Dataset`
- **Device**: `0`
- **Evaluated At**: `2026-11-02 17:28:52`

## Overall Benchmark Summary

| Model     | Description                              |   Precision |   Recall |   F1-Score |   mAP@0.5 |   mAP@0.5:0.95 |   MEAN_5 |   Latency (ms) |   FPS |
|:----------|:-----------------------------------------|------------:|---------:|-----------:|----------:|---------------:|---------:|---------------:|------:|
| yolo26n   | YOLO26n (epoch 45 - Best validation mAP) |      0.7899 |   0.7234 |     0.7552 |    0.7435 |         0.4012 |   0.7392 |          12.87 |  77.7 |
| yolo26m   | YOLO26m (epoch 45)                       |      0.8274 |   0.7467 |     0.785  |    0.7806 |         0.4296 |   0.7779 |          16.89 |  59.2 |
| yolo11n   | YOLO11n (last epoch)                     |      0.7665 |   0.6611 |     0.7099 |    0.7072 |         0.3823 |   0.6988 |          11.97 |  83.5 |
| yolo11m   | YOLO11m (epoch 45)                       |      0.7753 |   0.7155 |     0.7442 |    0.7431 |         0.4039 |   0.7309 |          16.28 |  61.4 |
| rtdetr50  | RT-DETR-R50vd (epoch 15)                 |      0.8073 |   0.7489 |     0.777  |    0.7499 |         0.3991 |   0.7472 |          37.5  |  26.7 |
| rtdetr101 | RT-DETR-R101vd (epoch 15)                |      0.8346 |   0.7632 |     0.7973 |    0.7714 |         0.4198 |   0.7705 |          64.3  |  15.6 |

## Per-Class AP@0.5 Breakdown

| Model     |   insulator_AP50 |   insulator_broken_AP50 |   insulator_flashover_AP50 |   bird_nest_AP50 |   foreign_object_AP50 |   broken_strand_AP50 |
|:----------|-----------------:|------------------------:|---------------------------:|-----------------:|----------------------:|---------------------:|
| yolo26n   |           0.7651 |                  0.8009 |                     0.6375 |           0.8619 |                0.7484 |               0.6473 |
| yolo26m   |           0.7939 |                  0.8286 |                     0.7578 |           0.9229 |                0.7377 |               0.6427 |
| yolo11n   |           0.7489 |                  0.7516 |                     0.5772 |           0.8802 |                0.6909 |               0.5941 |
| yolo11m   |           0.8038 |                  0.7976 |                     0.6596 |           0.8834 |                0.6982 |               0.6159 |
| rtdetr50  |           0.7636 |                  0.8108 |                     0.6746 |           0.903  |                0.7215 |               0.6261 |
| rtdetr101 |           0.7758 |                  0.8252 |                     0.7263 |           0.9141 |                0.7142 |               0.6727 |

## Hardware Profile

| Model     |   Params (M) |   GFLOPS |   Latency (ms) |   FPS |   VRAM (MB) |
|:----------|-------------:|---------:|---------------:|------:|------------:|
| yolo26n   |         2.51 |     5.9  |          12.87 |  77.7 |        74.3 |
| yolo26m   |        21.78 |    75    |          16.89 |  59.2 |       270.4 |
| yolo11n   |         2.59 |     6.5  |          11.97 |  83.5 |       201.4 |
| yolo11m   |        20.06 |    68.32 |          16.28 |  61.4 |       259.3 |
| rtdetr50  |        42.74 |   137    |          37.5  |  26.7 |       757.7 |
| rtdetr101 |        76.4  |   259.81 |          64.3  |  15.6 |      1067.6 |
