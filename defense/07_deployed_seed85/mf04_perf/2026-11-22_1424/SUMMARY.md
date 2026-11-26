# Kiểm thử AI MF04 — tổng hợp (22/11/2026)

Phạm vi: dịch vụ AI của MF04 (endpoint Modal `powerline-yolo26m-benchmark`, model `seed85_curated_yolo26m`), từ lúc nhận ảnh đến lúc trả bounding box. Không gồm lưu DB, hàng đợi của backend (T5, T6), end-to-end job và restart worker (AI-PERF-08). Các mục này cần chạy cùng backend trên VPS.

- Lần 1 (`2026-11-22_1323`): bản deploy trước khi sửa. Lần này phát hiện 3 lỗi.
- Lần 2 (`2026-11-22_1424`, thư mục này): bản đã sửa. Đây là số liệu chính thức.
- Dữ liệu: `finding_nemo_human_reviewed_6cls`, nhãn cross-label đã được người accept (val 2.271 ảnh / 3.204 box, test 1.931 ảnh / 2.850 box). Cùng ảnh với Finding Nemo V5 nhưng **khác nhãn** (V5 có 2.769 box), nên số ở đây không phải số trên V5.
- Môi trường: GPU Tesla T4 (Modal), `max_containers=1`, `scaledown_window=60 s`. Client gọi qua Modal SDK từ Việt Nam đến region us-east.

## Ngưỡng chấp nhận (định trước khi chạy)

- Xử lý AI phía server (decode → preprocess → inference → NMS → lọc): p95 < 3.000 ms (Report 5, Scope).
- Thời gian một ảnh phía client (round-trip, không tính cold start): trung bình ≤ 5.000 ms (Report 5 §2.1.4).

## 1. Kết quả hệ thống

| ID | Kịch bản | Kết quả | Số liệu chính |
|---|---|---|---|
| AI-PERF-00 | Cold start (đo riêng, `cold_start.json`) | Đo | **9,3–10,5 s** round-trip (2/2 mẫu, container mới phục vụ đúng 1 request): khởi động container khoảng 3–4 s + nạp model 2,7 s + lần predict đầu 2,4 s (CUDA làm nóng) + mạng |
| AI-PERF-01 | 640×640, 30 lần | **Passed** | Server p95 43 ms (inference 36,9 ms); round-trip trung bình 603 ms |
| AI-PERF-02 | 1920×1080, 20 lần | **Passed** | Server p95 58 ms (decode 12 ms); round-trip trung bình 792 ms |
| AI-PERF-03 | 3840×2160, 20 lần | **Passed** | Server p95 138 ms (decode 50 ms); round-trip trung bình 877 ms |
| AI-PERF-04 | 10 ảnh liên tiếp | **Passed** | Round-trip trung bình 631 ms, 1,6 ảnh/s |
| AI-PERF-05 | 50 ảnh đồng thời | **Passed (sát ngưỡng)** | Round-trip p95 **4.979 ms** / ngưỡng 5.000; chờ hàng đợi ước lượng p95 4,4 s; 9,8 ảnh/s |
| AI-PERF-06 | 100 ảnh liên tục | **Passed** | 0 lỗi; RSS 5.084 → 5.105 MiB; VRAM 491 MiB không đổi |
| AI-PERF-07 | Liên tục 30 phút | **Passed** | 2.933 request, 0 lỗi; p95 phút đầu 67 ms, phút cuối 58 ms; RSS +0,5%; không khởi động lại container |
| AI-PERF-08 | Restart worker khi còn job | Không thực hiện | Cần bridge + RabbitMQ trên VPS |
| AI-ROB-06 | Ảnh hỏng / sai định dạng / quá lớn | **Passed** | Bị từ chối với 413/415; request hợp lệ ngay sau vẫn chạy; nhận PNG và **MPO (DJI)** |
| AI-OUT-01 | Kiểm tra output | **Passed** | 7.815 response, 0 detection sai class / confidence / bbox |

AI-PERF-00 trong `report.md` (3,1 s) **không hợp lệ**: Modal chưa tắt container sau 90 s rảnh, nên request đó không phải cold start. Script đã được sửa để chờ lâu hơn và tự đánh dấu phép đo không hợp lệ. Số đúng là phép đo riêng ở bảng trên, chỉ gọi khi `modal container list` đã trống.

Cold start (khoảng 10 s) vượt ngưỡng 5 s/ảnh, nhưng chỉ xảy ra với ảnh đầu tiên sau khi endpoint rảnh trên 60 s. Ngưỡng 5 s áp dụng cho trạng thái chạy bình thường, xem khuyến nghị 3.

Thời gian round-trip chủ yếu là mạng (khoảng 550 ms Việt Nam ↔ us-east). Xử lý AI trên server chỉ 43–138 ms, thấp hơn ngưỡng 3 s nhiều.

### Lỗi phát hiện ở lần 1 và đã sửa

| Lỗi | Bằng chứng (lần 1) | Sau khi sửa (lần 2) |
|---|---|---|
| Predict từng ảnh dùng letterbox chữ nhật, khác cách train và tune ngưỡng | Val recall 0,813 so với 0,840 offline; 147 ảnh ngang bị thiếu box | Letterbox vuông: val P 0,891 / R 0,841, **khớp số offline** |
| Từ chối ảnh MPO (JPEG của flycam DJI) | 6 ảnh test bị lỗi 415 | Nhận JPEG, MPO, PNG |
| Giới hạn 10 MiB / 20 MP quá chặt với ảnh UAV | 23 ảnh test (19–47 MB, 50 MP) bị lỗi 413 | Giới hạn 50 MiB / 60 MP; test: 0 lỗi |
| (Lần 1) PERF-07 Failed | Modal thay container giữa chừng, 1 request chờ 258 s | Lần 2 không lặp lại; rủi ro vẫn còn, xem khuyến nghị |

Ảnh ở lần 1 còn được xoay theo EXIF, để box khớp với ảnh trình duyệt hiển thị.

## 2. Độ chính xác của endpoint đã deploy (IoU 0,5, sau hậu xử lý)

| Split | Precision | Recall | F1 | Macro-F1 | Lỗi request |
|---|---|---|---|---|---|
| val | 0,891 | 0,841 | 0,865 | 0,824 | 0 |
| test | 0,892 | 0,821 | 0,855 | 0,796 | 0 (lần 1: 29) |

F1 theo class trên test: insulator 0,906 · insulator_broken 0,812 · flashover 0,758 · bird_nest 0,881 · foreign_object 0,765 · **broken_strand 0,656**.

## 3. Leakage train ↔ val/test và ảnh hưởng lên điểm

Audit bằng SHA-256 và dHash (`yolov11/artifacts/dataset_audit/leakage_finding_nemo_human_reviewed_6cls`):
- Trùng chính xác: **0**.
- Gần trùng với train (dHash ≤ 4 bit): **53 ảnh test (đều thuộc V5)** và 80 ảnh val.
- Kiểm tra bằng mắt: các cặp có khoảng cách ≤ 2 là leakage cấp cảnh, gồm cùng ảnh nền dán foreign_object khác nhau và khung hình liên tiếp của cùng một video. Ở khoảng cách 3–4 có lẫn cặp chỉ giống bố cục, nên con số ≤ 4 là cận trên.

| Test, bỏ ảnh gần trùng train | Ảnh | F1 | Macro-F1 | foreign_object F1 |
|---|---|---|---|---|
| Tất cả | 1.931 | 0,855 | 0,796 | **0,765** |
| Bỏ d ≤ 2 | 1.911 | 0,855 | 0,794 | 0,754 |
| Bỏ d ≤ 4 | 1.878 | 0,853 | 0,787 | **0,719** |
| Chỉ 53 ảnh bị gắn cờ | 53 | 0,933 | — | 0,973 (18/19) |

Kết luận: điểm tổng gần như không đổi. F1 foreign_object giảm 0,046 khi bỏ 53 ảnh, nhưng các ảnh này chứa khoảng 19/111 box FO và là ảnh dễ, nên mức giảm **không thể quy hết cho leakage**. Đây là sensitivity analysis, không phải ước lượng nhân quả.

## 4. Điểm yếu của model và hướng cải thiện

| Phát hiện | Số liệu | Hướng cải thiện |
|---|---|---|
| Vật thể nhỏ (AI-ROB-04, val) | insulator_broken < 16 px: recall **0,18**; broken_strand 32–64 px: **0,50**, 16–32 px: 0,29 | Tăng độ phân giải: thử nghiệm imgsz 960 (notebook `v7p2_20261122_yolo26m_imgsz960_local`) |
| broken_strand yếu nhất | F1 test 0,656 | Như trên, kèm xem lại label các box nhỏ |
| flashover bị sót | 39/163 box val bị sót, gần như không nhầm class; hạ ngưỡng không giúp | Review label và bổ sung dữ liệu flashover |
| Ảnh thiếu sáng (AI-ROB-01) | Recall 0,865 → **0,823** (−4,2 điểm), precision giữ nguyên | Augmentation độ sáng mạnh hơn khi train |
| Ảnh mờ chuyển động (AI-ROB-02) | Recall 0,865 → 0,852 (−1,3 điểm) | Chấp nhận được |
| Ảnh nền không có lỗi (AI-ROB-05) | 17/174 ảnh (9,8%) có box sai | Thêm ảnh nền (hard negative) vào train |
| foreign_object | Thí nghiệm V7.2: FO cỡ lớn từ nguồn ngoài sửa được lỗi của V7.1 nhưng chỉ hòa seed85; điểm test bị thổi phồng do leakage | Tách nhóm ảnh cùng nền vào cùng một split; thu thập FO thật |
| AI-ROB-03 nền phức tạp | Chưa có tập ảnh có nhãn | Không tự tính mAP khi không có ground truth |

## 5. Khuyến nghị vận hành

1. **Một container GPU duy nhất là điểm yếu về độ tin cậy.** Ở lần 1, Modal thay container và một request chờ 258 s. Bridge có timeout 120 s và thử lại tối đa 3 lần, nên job không mất nhưng chậm 4–5 phút. Có thể thêm GPU dự phòng (`gpu=['T4', 'L4']`) hoặc `max_containers=2`; cả hai đều tăng chi phí, cần nhóm quyết định.
2. **50 ảnh đồng thời đã sát ngưỡng 5 s** (p95 4,98 s), vì một container xử lý tuần tự. Upload dồn dập hơn mức này cần thêm container hoặc nhận rằng thời gian chờ hàng đợi sẽ tăng.
3. **Cold start khoảng 10 s.** Có thể giảm khoảng 2,4 s bằng cách chạy một lần predict giả ngay khi container khởi động (`@modal.enter`), không tốn thêm chi phí. Muốn loại bỏ hẳn thì giữ một container luôn chạy (`min_containers=1`), đổi lại phải trả GPU T4 24/24.
4. Đổi checkpoint thì phải chạy lại `tune_thresholds.py` trên val và chạy lại bộ test này.

## Tái lập

```bash
cd uavpms-server/modal
python perf/mf04_ai_tests.py --dataset <finding_nemo_human_reviewed_6cls> --soak-minutes 30
python perf/leakage_impact.py --dataset <ds> --split test --pairs <leakage_pairs.csv> --out perf/results/<run>
```

Từng request nằm trong `calls.csv`, detection theo ảnh trong `detections_{val,test}.jsonl`, số liệu đầy đủ trong `results.json` và `leakage_impact_*.json`.
