# Gói bằng chứng cho paper — 2026-10-26

Đây là kiểm tra code và artifacts local, không phải một lần chạy lại benchmark. `sources/` giữ bản sao 42 file nguồn theo cấu trúc repo; `source_manifest.json` lưu SHA-256 từng file. Notebook có thêm bản `.txt` gồm source và output đã lưu. Không chứa checkpoint/dataset đầy đủ. Các giá trị thiếu được ghi rõ, không suy diễn thành kết quả đã chạy.

## 1. Dataset inventory / class counts

**Đã xác nhận từ output lưu:** [`sources/benchmakring_transmission_line_defect_model/benchmark_pipeline.ipynb`](sources/benchmakring_transmission_line_defect_model/benchmark_pipeline.ipynb), cell index 2: raw train 18,334 ảnh / 18,334 label files; curated train 17,903 / 17,903; raw val 2,272; curated val_clean 2,271. Script đếm là [`sources/benchmakring_transmission_line_defect_model/src/dataset.py`](sources/benchmakring_transmission_line_defect_model/src/dataset.py), hàm `verify_dataset_structure`; script đếm file jpg/png và txt, KHÔNG hash ảnh và KHÔNG kiểm tra label rỗng. Vì vậy đây là số file, không phải bằng chứng 18,334/17,903 nội dung ảnh duy nhất. Notebook legacy v6p5-17 cũng lưu 17,903 train, 2,272 val, 2,271 val_clean.

`artifacts/v6p5_train_local/20261009T024019Z_roboflow/build.json`: train 18,334, Roboflow thêm 351 train và 74 external_eval, 968 dropped; base_dataset v6p5_compose_4b9a452e24e2. Không được gọi 18,334 là canonical V6.2E nguyên bản chỉ dựa vào tên YAML.

Chưa tìm thấy script/output tạo đúng bảng class-count 25,807 / 23,204 box của bản thảo. Chưa biết số label rỗng thực tế. Tổng class-image trong bản thảo là 17,969 raw, 17,061 curated, 2,154 val, 1,904 test. Không thể lấy hiệu với tổng ảnh để suy số background vì ảnh có thể chứa nhiều lớp.

**Script mới cung cấp:** `inventory_dataset.py`, không phải script lịch sử. Chạy trên từng dataset thật:

```bash
python inventory_dataset.py /path/to/raw_combined --out /path/to/inventory_raw
python inventory_dataset.py /path/to/curated_combined --out /path/to/inventory_curated
```

Xuất images.csv (hash ảnh, tình trạng nhãn), splits.csv, classes.csv, summary.json. Class images đếm một lần/ảnh/lớp, boxes đếm mỗi dòng YOLO hợp lệ. Phân biệt missing label, empty label, invalid label; số ảnh duy nhất SHA-256 là theo byte file. Chưa chạy trên dataset thật vì checkout không chứa dữ liệu đó.

## 2. Evaluation logs và split

`evaluation_inventory.csv` tổng hợp cả 12 records từ `results_yolo_model/benchmark_test_results.json` và `results_rtdetr/benchmark_test_results.json`, giữ đường dẫn Windows gốc, split key, thời gian, checkpoint nếu có, accuracy/hardware. CSV summary gốc cũng có trong sources.

- YOLO: 8 records ngày 2026-10-23 đều trỏ configs/dataset_curated.yaml. YAML hiện có `val: images/val`, `test: images/val_clean`. Không lưu checkpoint_path trong JSON cũ; README nói best.pt, không phải bằng chứng từ checkpoint metadata.
- RT-DETR: 4 records ngày 2026-10-25 đều trỏ data/curated_combined/data_val_clean.yaml, checkpoint `runs/<run>/weights/last.pt`. Chính file YAML runtime đó không có local; không khẳng định nội dung chỉ từ tên file.
- Chưa có manifest/hash danh sách ảnh evaluation nên chưa chứng minh 12 runs chấm cùng ảnh và nhãn. split_key=test không đủ chứng minh test độc lập.
- Code: [`sources/benchmakring_transmission_line_defect_model/src/evaluator.py`](sources/benchmakring_transmission_line_defect_model/src/evaluator.py): `evaluate_checkpoint` (YOLO), `_evaluate_huggingface_rtdetr` (HF), `evaluate_all` (xuất summary). YOLO F1=harmonic(mean P, mean R), HF F1=mean(class F1), cần thống nhất.
- Notebook hiện tại cell 4 đã trỏ v6p2e_test/dataset.yaml, nhưng dùng string.open() nên cần sửa thành Path; nó không chứng minh các records cũ được chấm trên đường dẫn mới.

## 3. FPS và tài nguyên

Bảng hardware trong LaTeX lấy toàn bộ **raw runs**. Curated YOLO26m: 61.4 FPS / 16.29 ms; raw: 63.6 FPS / 15.72 ms. mAP50 0.8323 là curated, không ghép thành cùng run với 63.6 FPS.

Code [`sources/benchmakring_transmission_line_defect_model/src/evaluator.py`](sources/benchmakring_transmission_line_defect_model/src/evaluator.py), hàm `profile_hardware`: YOLO predict tensor zeros [1,3,640,640], 50 warmup, 300 iterations theo benchmark_config.yaml; tính thời gian tổng rồi chia số lượt; GPU index 0. Nhánh HF: batch 2, ảnh thật toàn split, processor chuẩn bị ngoài vùng timing, GPU synchronize trước/sau forward; không có warmup riêng; ms/image là tổng forward time/số ảnh. Hai protocol không đồng nhất. HF FLOPs là THOP estimate, không mặc định đầy đủ mọi operator.

README nói RTX 4060, nhưng result JSON không lưu GPU name, CUDA/torch version hoặc precision. Chưa xác nhận hardware từng run bằng log runtime. Checkpoint RT-DETR có trong inventory; checkpoint YOLO cũ chỉ có mô tả README best.pt.

## 4. Training configurations

YOLO dùng chung [`sources/benchmakring_transmission_line_defect_model/configs/benchmark_config.yaml`](sources/benchmakring_transmission_line_defect_model/configs/benchmark_config.yaml) và [`sources/benchmakring_transmission_line_defect_model/src/trainer.py`](sources/benchmakring_transmission_line_defect_model/src/trainer.py), hàm `train_single_run`: imgsz640, epochs50, AdamW lr0=.001 lrf=.01 cosine, warmup3, weight_decay=.0005, seed42, AMP, device0. n batch16, m batch8. accumulate 1/2 được khai báo nhưng chỉ in ra, KHÔNG truyền vào train; nbs không đặt. Cần args.yaml của đúng 8 runs để chốt effective batch thực tế.

RT-DETR benchmark state dict được evaluator dựng bằng PekingU/rtdetr_r50vd hoặc rtdetr_r101vd (run key rtdetr100 thực tế ánh xạ R101). Không đồng nhất chúng với rtdetr-l.pt / rtdetr-x.pt trong config Ultralytics.

Notebook `kaggle-train-rtdetr-peking_used.ipynb` và bản text đính kèm chứa training loop thực tế được lưu: AdamW backbone1e-4/head1e-3, weight_decay1e-4, cosine eta_min1e-6, batch8 accum2 trong call R50, default epochs50; xem call R101 trong notebook. Không có warmup3 trong loop đó. Loader validation images/val. Notebook Kaggle không đủ chứng minh tên GPU đã chạy; cần log đúng 4 runs và mapping checkpoint xuất ra với last.pt.

Không lấy args.yaml của legacy v3 teacher/student để thay cho benchmark: legacy v3 output ghi NVIDIA L4, teacher40 epochs/student60 epochs, đây là thí nghiệm khác.

## 5. Eight-rule curation / duplicate control

Code chính `scripts/label_audit_build.py`: A hflip; B add_nest; C person; D syn_fo; E quarantine; F tiny; G review; H tighten. A/C/G/H có áp dụng cho val_clean; B/D/E/F chủ yếu train. Cần đối chiếu đúng phiên bản code và args của build v3; code hiện tại đã phát triển thêm P/I và thay đổi chính sách SAM.

Bằng chứng SAM: output notebook v6p5-17a lưu bảng train insulator thay1,141, FO453; val_clean insulator27, FO2; cũng có bird_nest candidates. Đây là bảng phân tích SAM, không tự chứng minh tất cả candidate được áp dụng. Các con số A182 train/26val, ~55% synthetic, ~9% person là mô tả proposal, chưa tái tính từ changes.csv của đúng build trong gói này. Chưa có log đủ A–H và số quyết định human review v3 để xác nhận toàn bộ. Cần manifest.json, changes.csv, review_decisions.csv, sam_boxes.csv đúng build; không thay bằng số review v6/v7.

Dedup có HAI phạm vi:
1. Legacy v6p2-05 freeze: code kiểm tra SHA-256 cả train–val/train–test/val–test; pHash/dHash/embedding, group handling, quarantine liên quan test; bảo toàn test hash set. Notebook local không có executed output; block số0 cuối notebook là acceptance criteria, không phải log chạy.
2. `artifacts/v6p4_post_clean_audit/` có audit result runtime V6.2E: exact groups0, pHash candidate pairs135 (train–val86, train–test46, val–test3). Đây là ứng viên, không phải 135 confirmed leaks; chưa có disposition đủ cho các cặp trong gói.
3. Pipeline [`sources/src/yolov11sdi/dataset/dedup.py`](sources/src/yolov11sdi/dataset/dedup.py): external direct images vs toàn canonical manifest bằng SHA-256; external vs VAL/TEST bằng pHash. Không kiểm tra synthetic như external leakage. Đây là phạm vi code, không bằng chứng tự động mọi bước đã chạy trong benchmark.

## 6. Diffusion augmentation

**Đã có prototype, chưa có bằng chứng được dùng train benchmark.** Model trong cả hai script: `diffusers/stable-diffusion-xl-1.0-inpainting-0.1`. scripts/proto_diffusion_fo.py và proto_diffusion_strand.py; metadata trong artifacts/diffusion_fo_proto/meta.csv (14 rows) và diffusion_strand_proto/meta.json (16 rows). Xem diffusion_inventory.json cho status counts. `status=ok` là trạng thái generation, không phải human acceptance.

Contact sheets gốc: artifacts/diffusion_fo_proto/sheet.jpg và artifacts/diffusion_strand_proto/sheet.jpg. Nhật ký docs/SESSION_2026-10-15_v6p5_plateau.md mục5 ghi FO ~25% dùng được, strand0/16, không dùng train; không có per-image human approval manifest trong gói. Mô tả phù hợp: exploratory synthesis prototype excluded from reported training; việc tích hợp augmentation vào benchmark là future work. Không nói diffusion chưa từng thực hiện, cũng không nói nó tạo ra cải thiện benchmark.
