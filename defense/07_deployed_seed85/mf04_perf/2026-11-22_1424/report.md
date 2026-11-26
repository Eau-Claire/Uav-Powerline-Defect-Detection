# Kiểm thử AI MF04 — 2026-11-22_1424

## Môi trường

- **run_at**: 2026-11-22_1424
- **endpoint**: Modal app powerline-yolo26m-benchmark / env main / Detector.predict_jpeg (SDK)
- **gpu**: Tesla T4 (Modal), max_containers=1, scaledown_window=60 s
- **model**: seed85_curated_yolo26m
- **checkpoint_sha256**: 8682baa083ba5148d378762ddba5c5176cafaedd72b0c2eb369d3ea992924f16
- **thresholds_sha256**: e69a40318bba4a5b
- **dataset**: finding_nemo_human_reviewed_6cls
- **client**: Linux x86_64, Python 3.14.8, modal 1.5.5, Pillow 13.0.0
- **container_resources_at_start**: {'gpu': 'Tesla T4', 'gpu_util_pct': 0.0, 'gpu_mem_used_mib': 491.0, 'gpu_mem_total_mib': 15360.0, 'torch_allocated_mib': 196.4, 'torch_max_allocated_mib': 277.1, 'rss_mib': 5062.9, 'max_rss_mib': 5062.9, 'served': 2, 'uptime_s': 112.3, 'model_load_s': 5.2}

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
| AI-PERF-07 | Chạy liên tục 30.0 phút | **Passed** |
| AI-PERF-08 | Restart AI worker khi còn job | **Measured** |
| AI-OUT-01 | Kiểm tra output: class thuộc 6 class, confidence ∈ [0,1], bbox hợp lệ trong ảnh | **Passed** |

## AI-PERF-00 — Cold start (container khởi động + nạp model + lần predict đầu)

- Tiêu chí: Chỉ đo, báo riêng; không tính vào trung bình
- Kết quả: **Measured**
- Ghi chú: Modal scale-to-zero sau 60 s không có request

```json
{
  "roundtrip_ms": 3131.6,
  "predict_ms": 44.9,
  "error": null,
  "model_load_s": 5.2155
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
    "mean": 43.1,
    "p50": 43.0,
    "p95": 43.3,
    "min": 42.5,
    "max": 48.0
  },
  "roundtrip_ms": {
    "n": 30,
    "mean": 603.4,
    "p50": 594.7,
    "p95": 660.9,
    "min": 556.2,
    "max": 886.2
  },
  "stages_mean_ms": {
    "decode_ms": 3.004,
    "preprocess_ms": 1.0123,
    "inference_ms": 36.9,
    "nms_ms": 0.6726,
    "filter_ms": 0.0093
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
    "mean": 55.9,
    "p50": 55.5,
    "p95": 58.1,
    "min": 54.2,
    "max": 60.9
  },
  "roundtrip_ms": {
    "n": 20,
    "mean": 791.5,
    "p50": 870.7,
    "p95": 978.4,
    "min": 580.2,
    "max": 1187.2
  },
  "stages_mean_ms": {
    "decode_ms": 12.1,
    "preprocess_ms": 1.9724,
    "inference_ms": 36.9,
    "nms_ms": 0.6687,
    "filter_ms": 0.0117
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
    "mean": 108.6,
    "p50": 102.0,
    "p95": 138.2,
    "min": 98.4,
    "max": 140.9
  },
  "roundtrip_ms": {
    "n": 20,
    "mean": 877.3,
    "p50": 790.1,
    "p95": 1336.4,
    "min": 642.1,
    "max": 1338.0
  },
  "stages_mean_ms": {
    "decode_ms": 50.4,
    "preprocess_ms": 2.3027,
    "inference_ms": 36.9,
    "nms_ms": 0.7187,
    "filter_ms": 0.0104
  }
}
```

## AI-PERF-04 — 10 ảnh khác nhau gửi liên tiếp

- Tiêu chí: 0 lỗi; round-trip trung bình ≤ 5000 ms
- Kết quả: **Passed**

```json
{
  "images": 10,
  "total_s": 6.3149,
  "images_per_s": 1.5836,
  "errors": 0,
  "roundtrip_ms": {
    "n": 10,
    "mean": 631.2,
    "p50": 605.2,
    "p95": 790.1,
    "min": 563.2,
    "max": 901.7
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
  "total_s": 5.1091,
  "throughput_img_s": 9.7864,
  "roundtrip_ms": {
    "n": 50,
    "mean": 3963.5,
    "p50": 3671.4,
    "p95": 4978.9,
    "min": 3374.9,
    "max": 5063.0
  },
  "est_queue_wait_ms": {
    "n": 50,
    "mean": 3371.9,
    "p50": 3080.4,
    "p95": 4387.3,
    "min": 2768.2,
    "max": 4466.7
  },
  "server_ai_ms": {
    "n": 50,
    "mean": 31.3,
    "p50": 28.3,
    "p95": 42.6,
    "min": 22.8,
    "max": 54.9
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
  "total_s": 61.7,
  "images_per_s": 1.6194,
  "server_ai_ms": {
    "n": 100,
    "mean": 48.1,
    "p50": 44.4,
    "p95": 57.8,
    "min": 41.2,
    "max": 82.4
  },
  "first10_mean_ms": 52.8,
  "last10_mean_ms": 46.9,
  "rss_mib": [
    5083.9,
    5104.8
  ],
  "gpu_mem_used_mib": [
    491.0,
    491.0
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
    "precision": 0.8914,
    "recall": 0.8405,
    "f1": 0.8652,
    "tp": 2693,
    "fp": 328,
    "fn": 511,
    "macro_f1": 0.8241
  },
  "per_class_f1": {
    "insulator": 0.8933,
    "insulator_broken": 0.8275,
    "insulator_flashover": 0.781,
    "bird_nest": 0.9131,
    "foreign_object": 0.7478,
    "broken_strand": 0.7819
  },
  "images_per_s": 8.9345
}
```

## AI-ACC-TEST — Độ chính xác endpoint đã deploy trên test (1931 ảnh)

- Tiêu chí: Chỉ đo (P/R/F1 @IoU 0.5, sau hậu xử lý); so với số offline trong thresholds.json
- Kết quả: **Measured**

```json
{
  "errors": 0,
  "micro": {
    "precision": 0.8918,
    "recall": 0.8211,
    "f1": 0.855,
    "tp": 2340,
    "fp": 284,
    "fn": 510,
    "macro_f1": 0.7964
  },
  "per_class_f1": {
    "insulator": 0.9059,
    "insulator_broken": 0.8124,
    "insulator_flashover": 0.7583,
    "bird_nest": 0.8812,
    "foreign_object": 0.7647,
    "broken_strand": 0.6559
  },
  "images_per_s": 5.6684
}
```

## AI-ROB-01 — Ảnh thiếu sáng (độ sáng × 0,35)

- Tiêu chí: Chỉ đo mức suy giảm so với ảnh gốc (có ground truth)
- Kết quả: **Measured**

```json
{
  "images": 150,
  "original": {
    "precision": 0.8952,
    "recall": 0.865,
    "f1": 0.8798,
    "tp": 205,
    "fp": 24,
    "fn": 32,
    "macro_f1": 0.7884
  },
  "degraded": {
    "precision": 0.907,
    "recall": 0.8228,
    "f1": 0.8628,
    "tp": 195,
    "fp": 20,
    "fn": 42,
    "macro_f1": 0.7697
  },
  "delta_recall": -0.0422,
  "delta_precision": 0.0118
}
```

## AI-ROB-02 — Ảnh mờ chuyển động (kernel ngang 5 px)

- Tiêu chí: Chỉ đo mức suy giảm so với ảnh gốc (có ground truth)
- Kết quả: **Measured**

```json
{
  "images": 150,
  "original": {
    "precision": 0.8952,
    "recall": 0.865,
    "f1": 0.8798,
    "tp": 205,
    "fp": 24,
    "fn": 32,
    "macro_f1": 0.7884
  },
  "degraded": {
    "precision": 0.8938,
    "recall": 0.8523,
    "f1": 0.8726,
    "tp": 202,
    "fp": 24,
    "fn": 35,
    "macro_f1": 0.8226
  },
  "delta_recall": -0.0127,
  "delta_precision": -0.0014
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
    "recall": 0.5,
    "n": 40
  },
  "broken_strand >=64px": {
    "recall": 0.82,
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
    "recall": 0.667,
    "n": 48
  },
  "insulator_broken 16-32px": {
    "recall": 0.675,
    "n": 166
  },
  "insulator_broken 32-64px": {
    "recall": 0.864,
    "n": 323
  },
  "insulator_broken <16px": {
    "recall": 0.178,
    "n": 45
  },
  "insulator_broken >=64px": {
    "recall": 0.848,
    "n": 276
  },
  "insulator_flashover 16-32px": {
    "recall": 0.7,
    "n": 10
  },
  "insulator_flashover 32-64px": {
    "recall": 0.683,
    "n": 60
  },
  "insulator_flashover >=64px": {
    "recall": 0.806,
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
  "images_with_detection": 17,
  "false_boxes": 19,
  "rate": 0.0977
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
      "error": "HTTPException: 415: Use JPEG, MPO, PNG",
      "worker_alive": true
    },
    {
      "input": "JPEG > 60 MP",
      "bytes": 945410,
      "accepted": false,
      "expected_accept": false,
      "error": "HTTPException: 413: Image exceeds 60 MP",
      "worker_alive": true
    },
    {
      "input": "body > 50 MiB",
      "bytes": 52428804,
      "accepted": false,
      "expected_accept": false,
      "error": "HTTPException: 413: Image exceeds 50 MiB",
      "worker_alive": true
    },
    {
      "input": "valid PNG",
      "bytes": 104132,
      "accepted": true,
      "expected_accept": true,
      "error": "",
      "worker_alive": true
    },
    {
      "input": "valid MPO (DJI)",
      "bytes": 12805,
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
- Kết quả: **Passed**

```json
{
  "minutes": 30.0,
  "requests": 2933,
  "errors": 0,
  "error_rate": 0.0,
  "images_per_s": 1.6294,
  "server_ai_ms": {
    "n": 2933,
    "mean": 48.6,
    "p50": 45.0,
    "p95": 59.1,
    "min": 40.2,
    "max": 111.3
  },
  "p95_first_min": 66.6,
  "p95_last_min": 58.5,
  "rss_mib_min2": 5150.8,
  "rss_mib_end": 5176.4,
  "gpu_mem_used_mib_end": 491.0,
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
    0.0
  ],
  "container_restarts": 0
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
  "responses_checked": 7815,
  "problems": 0,
  "examples": []
}
```
