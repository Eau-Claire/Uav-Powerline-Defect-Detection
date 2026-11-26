# Kiểm thử AI MF04 — 2026-11-22_1323

## Môi trường

- **run_at**: 2026-11-22_1323
- **endpoint**: Modal app powerline-yolo26m-benchmark / env main / Detector.predict_jpeg (SDK)
- **gpu**: Tesla T4 (Modal), max_containers=1, scaledown_window=60 s
- **model**: seed85_curated_yolo26m
- **checkpoint_sha256**: 8682baa083ba5148d378762ddba5c5176cafaedd72b0c2eb369d3ea992924f16
- **thresholds_sha256**: 733d02d77ca9f031
- **dataset**: finding_nemo_human_reviewed_6cls
- **client**: Linux x86_64, Python 3.14.8, modal 1.5.5, Pillow 13.0.0
- **container_resources_at_start**: {'gpu': 'Tesla T4', 'gpu_util_pct': 0.0, 'gpu_mem_used_mib': 497.0, 'gpu_mem_total_mib': 15360.0, 'torch_allocated_mib': 196.3, 'torch_max_allocated_mib': 277.1, 'rss_mib': 4942.2, 'max_rss_mib': 4976.3, 'served': 346, 'uptime_s': 286.9, 'model_load_s': 2.9}

## Ngưỡng chấp nhận (định trước)

- Xử lý AI phía server (T1 decode → T4 hậu xử lý): p95 < 3000 ms (Report 5, Scope).
- Thời gian một ảnh phía client (round-trip, không tính cold start): trung bình ≤ 5000 ms (Report 5 §2.1.4).
- Cold start được đo và báo riêng. T5 (DB/storage) và T6 (hàng đợi MF04) không thuộc phạm vi lần đo này.

## Tổng hợp

| ID | Hạng mục | Kết quả |
|---|---|---|
| AI-PERF-00 | Cold start (container khởi động + nạp model + lần predict đầu) | **Measured** |
| AI-PERF-01 | 1 ảnh 640×640, 30 lần liên tiếp | **Passed** |
| AI-PERF-02 | 1 ảnh 1920×1080, 20 lần liên tiếp | **Passed** |
| AI-PERF-03 | 1 ảnh 3840×2160, 20 lần liên tiếp | **Passed** |
| AI-PERF-04 | 10 ảnh khác nhau gửi liên tiếp | **Passed** |
| AI-PERF-05 | 50 ảnh gửi đồng thời | **Passed** |
| AI-PERF-06 | 100 ảnh liên tục | **Passed** |
| AI-ACC-VAL | Độ chính xác endpoint đã deploy trên val (2271 ảnh) | **Measured** |
| AI-ACC-TEST | Độ chính xác endpoint đã deploy trên test (1931 ảnh) | **Measured** |
| AI-ROB-01 | Ảnh thiếu sáng (độ sáng × 0,35) | **Measured** |
| AI-ROB-02 | Ảnh mờ chuyển động (kernel ngang 5 px) | **Measured** |
| AI-ROB-03 | Background phức tạp | **Measured** |
| AI-ROB-04 | Vật thể nhỏ: recall theo kích thước box (cạnh ngắn ở 640 px), val | **Measured** |
| AI-ROB-05 | Ảnh không có đối tượng thuộc 6 class (ảnh nền trong val) | **Measured** |
| AI-ROB-06 | Ảnh hỏng / sai định dạng / quá lớn | **Passed** |
| AI-PERF-07 | Chạy liên tục 30.0 phút | **Failed** |
| AI-PERF-08 | Restart AI worker khi còn job | **Measured** |
| AI-OUT-01 | Kiểm tra output: class thuộc 6 class, confidence ∈ [0,1], bbox hợp lệ trong ảnh | **Passed** |

## AI-PERF-00 — Cold start (container khởi động + nạp model + lần predict đầu)

- Tiêu chí: Chỉ đo, báo riêng; không tính vào trung bình
- Kết quả: **Measured**
- Ghi chú: Modal scale-to-zero sau 60 s không có request

```json
{
  "roundtrip_ms": 2807.5,
  "predict_ms": 33.9,
  "error": null,
  "model_load_s": 2.8597
}
```

## AI-PERF-01 — 1 ảnh 640×640, 30 lần liên tiếp

- Tiêu chí: 0 lỗi; xử lý AI p95 < 3000 ms; round-trip trung bình ≤ 5000 ms
- Kết quả: **Passed**

```json
{
  "size": "640x640",
  "jpeg_kib": 43,
  "errors": 0,
  "server_ai_ms": {
    "n": 30,
    "mean": 41.7,
    "p50": 41.6,
    "p95": 42.1,
    "min": 41.4,
    "max": 44.3
  },
  "roundtrip_ms": {
    "n": 30,
    "mean": 544.1,
    "p50": 517.3,
    "p95": 643.9,
    "min": 504.9,
    "max": 743.9
  },
  "stages_mean_ms": {
    "decode_ms": 2.1021,
    "preprocess_ms": 0.9478,
    "inference_ms": 36.8,
    "nms_ms": 0.5357,
    "filter_ms": 0.0071
  }
}
```

## AI-PERF-02 — 1 ảnh 1920×1080, 20 lần liên tiếp

- Tiêu chí: 0 lỗi; xử lý AI p95 < 3000 ms; round-trip trung bình ≤ 5000 ms
- Kết quả: **Passed**

```json
{
  "size": "1920x1080",
  "jpeg_kib": 135,
  "errors": 0,
  "server_ai_ms": {
    "n": 20,
    "mean": 38.7,
    "p50": 38.7,
    "p95": 39.5,
    "min": 38.3,
    "max": 40.0
  },
  "roundtrip_ms": {
    "n": 20,
    "mean": 597.9,
    "p50": 518.7,
    "p95": 752.7,
    "min": 511.8,
    "max": 753.2
  },
  "stages_mean_ms": {
    "decode_ms": 9.1511,
    "preprocess_ms": 1.5657,
    "inference_ms": 24.0,
    "nms_ms": 0.5329,
    "filter_ms": 0.0077
  }
}
```

## AI-PERF-03 — 1 ảnh 3840×2160, 20 lần liên tiếp

- Tiêu chí: 0 lỗi; xử lý AI p95 < 3000 ms; round-trip trung bình ≤ 5000 ms
- Kết quả: **Passed**

```json
{
  "size": "3840x2160",
  "jpeg_kib": 388,
  "errors": 0,
  "server_ai_ms": {
    "n": 20,
    "mean": 93.5,
    "p50": 93.8,
    "p95": 95.2,
    "min": 91.7,
    "max": 96.0
  },
  "roundtrip_ms": {
    "n": 20,
    "mean": 724.1,
    "p50": 756.8,
    "p95": 1026.0,
    "min": 519.2,
    "max": 1066.3
  },
  "stages_mean_ms": {
    "decode_ms": 46.3,
    "preprocess_ms": 1.9005,
    "inference_ms": 23.9,
    "nms_ms": 0.5234,
    "filter_ms": 0.0079
  }
}
```

## AI-PERF-04 — 10 ảnh khác nhau gửi liên tiếp

- Tiêu chí: 0 lỗi; round-trip trung bình ≤ 5000 ms
- Kết quả: **Passed**

```json
{
  "images": 10,
  "total_s": 5.4496,
  "images_per_s": 1.835,
  "errors": 0,
  "roundtrip_ms": {
    "n": 10,
    "mean": 542.9,
    "p50": 515.2,
    "p95": 672.2,
    "min": 501.8,
    "max": 763.3
  }
}
```

## AI-PERF-05 — 50 ảnh gửi đồng thời

- Tiêu chí: 0 lỗi; round-trip p95 ≤ 5000 ms
- Kết quả: **Passed**
- Ghi chú: Một container GPU xử lý tuần tự; thời gian chờ hàng đợi ước lượng = round-trip − xử lý AI − overhead mạng đo ở AI-PERF-04

```json
{
  "requests": 50,
  "errors": 0,
  "total_s": 4.1088,
  "throughput_img_s": 12.2,
  "roundtrip_ms": {
    "n": 50,
    "mean": 3685.4,
    "p50": 3804.4,
    "p95": 3919.1,
    "min": 2836.0,
    "max": 4022.5
  },
  "est_queue_wait_ms": {
    "n": 50,
    "mean": 3182.4,
    "p50": 3301.0,
    "p95": 3416.5,
    "min": 2321.1,
    "max": 3518.0
  },
  "server_ai_ms": {
    "n": 50,
    "mean": 28.1,
    "p50": 27.5,
    "p95": 37.0,
    "min": 17.8,
    "max": 45.6
  }
}
```

## AI-PERF-06 — 100 ảnh liên tục

- Tiêu chí: 0 lỗi; RSS tăng ≤ 10%; latency 10 ảnh cuối ≤ 1,2× 10 ảnh đầu
- Kết quả: **Passed**

```json
{
  "images": 100,
  "errors": 0,
  "total_s": 53.8,
  "images_per_s": 1.8589,
  "server_ai_ms": {
    "n": 100,
    "mean": 39.7,
    "p50": 36.0,
    "p95": 55.4,
    "min": 28.4,
    "max": 67.3
  },
  "first10_mean_ms": 41.8,
  "last10_mean_ms": 39.2,
  "rss_mib": [
    4929.7,
    4946.4
  ],
  "gpu_mem_used_mib": [
    497.0,
    497.0
  ],
  "torch_max_allocated_mib": 277.1
}
```

## AI-ACC-VAL — Độ chính xác endpoint đã deploy trên val (2271 ảnh)

- Tiêu chí: Chỉ đo (P/R/F1 @IoU 0.5, sau hậu xử lý); so với số offline trong thresholds.json
- Kết quả: **Measured**

```json
{
  "errors": 0,
  "micro": {
    "precision": 0.8964,
    "recall": 0.813,
    "f1": 0.8527,
    "tp": 2605,
    "fp": 301,
    "fn": 599,
    "macro_f1": 0.8039
  },
  "per_class_f1": {
    "insulator": 0.8717,
    "insulator_broken": 0.8227,
    "insulator_flashover": 0.7436,
    "bird_nest": 0.9126,
    "foreign_object": 0.7009,
    "broken_strand": 0.7718
  },
  "images_per_s": 14.7
}
```

## AI-ACC-TEST — Độ chính xác endpoint đã deploy trên test (1931 ảnh)

- Tiêu chí: Chỉ đo (P/R/F1 @IoU 0.5, sau hậu xử lý); so với số offline trong thresholds.json
- Kết quả: **Measured**

```json
{
  "errors": 29,
  "micro": {
    "precision": 0.899,
    "recall": 0.8018,
    "f1": 0.8477,
    "tp": 2262,
    "fp": 254,
    "fn": 559,
    "macro_f1": 0.7876
  },
  "per_class_f1": {
    "insulator": 0.8991,
    "insulator_broken": 0.8093,
    "insulator_flashover": 0.7113,
    "bird_nest": 0.8858,
    "foreign_object": 0.7586,
    "broken_strand": 0.6612
  },
  "images_per_s": 14.2
}
```

## AI-ROB-01 — Ảnh thiếu sáng (độ sáng × 0,35)

- Tiêu chí: Chỉ đo mức suy giảm so với ảnh gốc (có ground truth)
- Kết quả: **Measured**

```json
{
  "images": 150,
  "original": {
    "precision": 0.9027,
    "recall": 0.8608,
    "f1": 0.8812,
    "tp": 204,
    "fp": 22,
    "fn": 33,
    "macro_f1": 0.8332
  },
  "degraded": {
    "precision": 0.8967,
    "recall": 0.8059,
    "f1": 0.8489,
    "tp": 191,
    "fp": 22,
    "fn": 46,
    "macro_f1": 0.7873
  },
  "delta_recall": -0.0549,
  "delta_precision": -0.0059
}
```

## AI-ROB-02 — Ảnh mờ chuyển động (kernel ngang 5 px)

- Tiêu chí: Chỉ đo mức suy giảm so với ảnh gốc (có ground truth)
- Kết quả: **Measured**

```json
{
  "images": 150,
  "original": {
    "precision": 0.9027,
    "recall": 0.8608,
    "f1": 0.8812,
    "tp": 204,
    "fp": 22,
    "fn": 33,
    "macro_f1": 0.8332
  },
  "degraded": {
    "precision": 0.9,
    "recall": 0.8354,
    "f1": 0.8665,
    "tp": 198,
    "fp": 22,
    "fn": 39,
    "macro_f1": 0.8089
  },
  "delta_recall": -0.0253,
  "delta_precision": -0.0027
}
```

## AI-ROB-03 — Background phức tạp

- Tiêu chí: Không thực hiện
- Kết quả: **Measured**
- Ghi chú: Không có tập ảnh nền phức tạp được gán nhãn riêng; không tự tính mAP khi không có ground truth

## AI-ROB-04 — Vật thể nhỏ: recall theo kích thước box (cạnh ngắn ở 640 px), val

- Tiêu chí: Chỉ đo
- Kết quả: **Measured**

```json
{
  "broken_strand 16-32px": {
    "recall": 0.286,
    "n": 7
  },
  "broken_strand 32-64px": {
    "recall": 0.475,
    "n": 40
  },
  "broken_strand >=64px": {
    "recall": 0.805,
    "n": 205
  },
  "foreign_object 16-32px": {
    "recall": 0.8,
    "n": 5
  },
  "foreign_object 32-64px": {
    "recall": 0.778,
    "n": 9
  },
  "foreign_object >=64px": {
    "recall": 0.625,
    "n": 48
  },
  "insulator_broken 16-32px": {
    "recall": 0.639,
    "n": 166
  },
  "insulator_broken 32-64px": {
    "recall": 0.858,
    "n": 323
  },
  "insulator_broken <16px": {
    "recall": 0.156,
    "n": 45
  },
  "insulator_broken >=64px": {
    "recall": 0.848,
    "n": 276
  },
  "insulator_flashover 16-32px": {
    "recall": 0.5,
    "n": 10
  },
  "insulator_flashover 32-64px": {
    "recall": 0.667,
    "n": 60
  },
  "insulator_flashover >=64px": {
    "recall": 0.763,
    "n": 93
  }
}
```

## AI-ROB-05 — Ảnh không có đối tượng thuộc 6 class (ảnh nền trong val)

- Tiêu chí: Chỉ đo tỉ lệ ảnh có box sai
- Kết quả: **Measured**

```json
{
  "background_images": 174,
  "images_with_detection": 18,
  "false_boxes": 20,
  "rate": 0.1034
}
```

## AI-ROB-06 — Ảnh hỏng / sai định dạng / quá lớn

- Tiêu chí: File không hợp lệ bị từ chối có thông báo lỗi; request hợp lệ ngay sau vẫn thành công; container không khởi động lại
- Kết quả: **Passed**

```json
{
  "cases": [
    {
      "input": "truncated JPEG",
      "bytes": 6902,
      "accepted": false,
      "expected_accept": false,
      "error": "HTTPException: 415: Invalid or oversized image",
      "worker_alive": true
    },
    {
      "input": "text file",
      "bytes": 1300,
      "accepted": false,
      "expected_accept": false,
      "error": "HTTPException: 415: Invalid or oversized image",
      "worker_alive": true
    },
    {
      "input": "empty body",
      "bytes": 0,
      "accepted": false,
      "expected_accept": false,
      "error": "HTTPException: 415: Invalid or oversized image",
      "worker_alive": true
    },
    {
      "input": "GIF",
      "bytes": 144,
      "accepted": false,
      "expected_accept": false,
      "error": "HTTPException: 415: Use JPEG or PNG",
      "worker_alive": true
    },
    {
      "input": "24 MP JPEG (> 20 MP)",
      "bytes": 375626,
      "accepted": false,
      "expected_accept": false,
      "error": "HTTPException: 413: Image exceeds 20 MP",
      "worker_alive": true
    },
    {
      "input": "PNG > 10 MiB",
      "bytes": 17017968,
      "accepted": false,
      "expected_accept": false,
      "error": "HTTPException: 413: Image exceeds 10 MiB",
      "worker_alive": true
    },
    {
      "input": "valid PNG",
      "bytes": 104132,
      "accepted": true,
      "expected_accept": true,
      "error": "",
      "worker_alive": true
    }
  ],
  "same_container": true
}
```

## AI-PERF-07 — Chạy liên tục 30.0 phút

- Tiêu chí: Tỉ lệ lỗi < 1%; RSS cuối ≤ 1,1× RSS phút 2; p95 phút cuối ≤ 1,2× phút đầu; không khởi động lại container
- Kết quả: **Failed**

```json
{
  "minutes": 30.0,
  "requests": 2568,
  "errors": 0,
  "error_rate": 0.0,
  "images_per_s": 1.4267,
  "server_ai_ms": {
    "n": 2568,
    "mean": 41.2,
    "p50": 36.1,
    "p95": 54.1,
    "min": 24.3,
    "max": 3204.4
  },
  "p95_first_min": 52.9,
  "p95_last_min": 48.7,
  "rss_mib_min2": 4991.2,
  "rss_mib_end": 5119.8,
  "gpu_mem_used_mib_end": 466.0,
  "gpu_util_pct_samples": [
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    18.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0
  ],
  "container_restarts": 1
}
```

## AI-PERF-08 — Restart AI worker khi còn job

- Tiêu chí: Không thực hiện ở đây
- Kết quả: **Measured**
- Ghi chú: Cần bridge + RabbitMQ trên VPS (job store SQLite, retry, dedup EventId)

## AI-OUT-01 — Kiểm tra output: class thuộc 6 class, confidence ∈ [0,1], bbox hợp lệ trong ảnh

- Tiêu chí: 0 detection không hợp lệ trên mọi response thành công
- Kết quả: **Passed**

```json
{
  "responses_checked": 7421,
  "problems": 0,
  "examples": []
}
```
