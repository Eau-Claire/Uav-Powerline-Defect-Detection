# Xác minh trước khi cập nhật Report (23/11/2026)

Trả lời 5 mục kiểm tra cho các bảng/hình RC (§4.4, §5.1, §5.2.2, §5.4, §5.5). Nguồn RC nằm trên branch
`finding_nemo_v5_results`, thư mục `benchmakring_transmission_line_defect_model/results/finding_nemo/operational_min8_v5/`
(commit `f946f98`, `ce050c0`). Đoạn tiếng Anh trong khung có thể dán thẳng.

| Mức | Mục | Kết luận |
|---|---|---|
| Cao | §4.4 Table RC-9a: số box operational V5 | **Đã xác minh: 2.467 box** |
| Cao | §5.1 PyTorch / CUDA | **"PyTorch 2.1.4" là gõ nhầm của `2.14.1+cu126`**; môi trường từng nhóm run khác nhau |
| Cao | §5.2.2 12 checkpoint RC-9a | Cố định **trước** V5 (đúng). Hash đã tính. **Nhưng** checkpoint curated không phải `best.pt` và tiêu chí chọn chưa được ghi lại |
| Cao | §5.4 Figure RC-3 và Table RC-11 | **Cùng một đợt đánh giá V5** (12 checkpoint, 19/11/2026), không phải benchmark gốc; mọi số RC-11 tính lại khớp |
| TB | §5.5 Table RC-13 | Đã có phiên bản nhãn, seed, evaluator, phần cứng cho từng run (mục 5). Có 1 nhận định sai trong báo cáo ablation cũ |

## 1. §4.4 Operational V5: 2.467 box

- 1.871 ảnh, **2.467 box**, 124 ảnh nền. Theo lớp: insulator 1.130 · insulator_broken 804 · insulator_flashover 124 ·
  bird_nest 172 · foreign_object 111 · broken_strand 126.
- Ba nguồn khớp nhau: `data/test_operational_min8_v5_full_consensus_assisted/freeze.json` (distribution);
  trường `protocol` trong 12 file JSON kết quả ("1871 images, 2467 boxes"); và dựng lại độc lập từ nhãn
  v1 + log v3 + 215 box V5 rồi áp quy tắc min-8 (`tools/fn_v5_review/build_review.py`).
- Manifest SHA-256: `e8e2bb2ccf18205c63a734cfd57d162dfd23e4008e3483378426f7fc5dd9ea8b`.

## 2. §5.1 Môi trường (lấy từ log Ultralytics trong notebook đã chạy, không suy ra từ driver)

Chuỗi `torch-X+cuNNN` do PyTorch in ra chính là `torch.__version__`; hậu tố `+cuNNN` là CUDA mà PyTorch được build
(= `torch.version.cuda`), không phải phiên bản driver.

| Nhóm run | Phần cứng | Python | PyTorch (CUDA build) | Nguồn |
|---|---|---|---|---|
| Train curated Finding Nemo (YOLO) | RTX 4060 Laptop 8 GB | 3.12.2 | **2.14.1+cu126** | `legacy/notebooks/v6p5/v6p5-27_20261031-finding-nemo-6cls-cross-labeled-train.ipynb` |
| Benchmark gốc raw/curated | RTX 4060 Laptop | (không in) | (không in) | `artifacts/paper_evidence_20261026/.../benchmark_pipeline.txt` |
| Ablation `m3eg` | RTX 4060 Laptop | 3.12.2 | 2.14.1+cu126 | `legacy/notebooks/v6p7/v6p7-01_*` |
| Ablation `m124`, `m134`, `m234` | RTX 4050 Laptop 6 GB | 3.13.12 | 2.6.0+cu124 | `legacy/notebooks/v6p7/v6p7-02..04_*` |
| Seed 36/85 (Kaggle) | 2× Tesla T4 | | 2.11.0+cu130 (torchvision 0.26.0+cu130); RT-DETR: transformers 5.20.2 | `legacy/seeds/seed36.ipynb`, `train.log` trên stash |
| Curated RT-DETR Finding Nemo, raw checkpoints, ablation `m1_abf`/`m2_cd` | | | **không tìm thấy log** | |

Ultralytics 8.4.172 ở mọi run YOLO. Câu cũ "Python 3.12.2, PyTorch 2.1.4 (CUDA 12.6)" sửa thành:

> Experiments were run with Python 3.12.2, PyTorch 2.14.1 built for CUDA 12.6, and Ultralytics 8.4.172 on an NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). Additional seeds (36 and 85) were trained on 2× Tesla T4 with PyTorch 2.11.0 (CUDA 13.0 build). Three ablation runs used an RTX 4050 Laptop GPU with PyTorch 2.6.0 (CUDA 12.4 build).

## 3. §5.2.2 Mười hai checkpoint của Table RC-9a

| Regime | Model | Checkpoint (stash `bunpmc/projects/…`) | SHA-256 |
|---|---|---|---|
| Raw | YOLO11n | `raw_yolo11n/weights/best.pt` (5.5 MB) | `f490385b49c3033960809baec696778eb94cc9854a3dc2574248b593f2ee31a4` |
| Raw | YOLO11m | `raw_yolo11m/weights/best.pt` (40.5 MB) | `786cfd875bc0321bca006d34680cef55db8083ff717b925de7b799bea019f2da` |
| Raw | YOLO26n | `raw_yolo26n/weights/best.pt` (5.4 MB) | `f2cab932d289acc011001fa94ff59c3e3cc5c6d94cbe62343e20a04ff1d0416d` |
| Raw | YOLO26m | `raw_yolo26m/weights/best.pt` (44.0 MB) | `ad398adf69f28e041aee1559767a01d7e3c52306a1729dc76b69db9dae7182e0` |
| Curated | YOLO11n | `finding_nemo/yolo11n/weights/last.pt` (5.5 MB) | `0284db0cae7d4ea21176a69717cde3d43820f33c61d849469de6423b27d8819a` |
| Curated | YOLO11m | `finding_nemo/yolo11m/weights/epoch45.pt` (161.2 MB) | `4c273fd59d5de5241a4b810fa1f162d9fcd0a730c422304509b6b54d563b65d4` |
| Curated | YOLO26n | `finding_nemo/yolo26n/weights/epoch45.pt` (20.8 MB) | `93e1bc2dd126e6e43d2d007aaea6a9c3e24b953af8a0b368b347d17aa38a5bb5` |
| Curated | YOLO26m | `finding_nemo/yolo26m/weights/epoch45.pt` (175.1 MB) | `a1fc47b7fa1cbdbfd1ba430a67bd91bf4203fafae78e52bf75125c17c12e9089` |
| Raw | RT-DETR-R50 | `raw_rtdetr50/checkpoints/best.pt` (171.7 MB) | `410b9f0307e6497af815c2460850fe4f61ba4b055e21fde6b89ffb1bcb032336` |
| Raw | RT-DETR-R101 | `raw_rtdetr101/checkpoints/best.pt` (307.0 MB) | `10ee145654d0f4734cca197f624899f2a1e74cfb8408b03575ca16ff68956092` |
| Curated | RT-DETR-R50 | `finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/r50vd/checkpoints/epoch15.pt` (171.8 MB) | `01b17f925b6f55f4f6f309eed7f1fe898f8d08c3799778383ace34cdfbfdce87` |
| Curated | RT-DETR-R101 | `finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/r101vd/checkpoints/epoch15.pt` (307.0 MB) | `812b5d5694fe86fe3154d716ed69d237cbd67f32be3a93bfb7a6f4a20766e7c7` |

Hash tính trên file trong stash (stream, 23/11/2026). JSON kết quả RC-9a không lưu hash, nên chưa chứng minh được bản trên máy Windows lúc đánh giá trùng từng byte; đường dẫn và tên file thì khớp với `STASH` trong notebook đánh giá.
Kiểm tra chéo: hash `finding_nemo/yolo11n/weights/last.pt` trùng `last_sha256` trong `stash://bunpmc/projects/finding_nemo/training_state.json` (ghi lúc train 01/11/2026).

**Cố định trước V5: đúng.** Bộ checkpoint curated được đánh giá ngày 02/11/2026 (16:39–17:28, các thư mục
`selected_checkpoints_eval*`); V5 được tạo và commit ngày 03/11/2026 13:42 (`30e8bc0`). Checkpoint raw lấy từ
benchmark gốc tháng 10. Kết quả RC-9a được tính ngày 19/11/2026.

**Evaluator:** `benchmakring_transmission_line_defect_model/src/evaluator.py`, blob `6e79369eb84d9d07aa23c2339fe95af4f79d9039`
(không đổi từ `30e8bc0` đến nay); `imgsz` 640, split `test` của
`test_operational_min8_v5_full_consensus_assisted/dataset.yaml`; MEAN_5 = trung bình AP50 của 5 lớp lỗi. Mỗi
file `raw_*.json` / `curated_*.json` ghi đường dẫn checkpoint, dataset, thời điểm đánh giá.

**Cần công khai hai điểm:**
1. **Checkpoint curated không phải `best.pt`:** yolo11n `last`, yolo11m/26n/26m `epoch45`, RT-DETR `epoch15`
   ("user-selected", script `evaluate_nemo_selected_checkpoints.py`). Theo CSV huấn luyện, epoch tốt nhất trên val
   (mAP50-95) là yolo11n 40, yolo11m 49, yolo26n 50, yolo26m 44/45, RT-DETR 7: chỉ YOLO26m trùng. Checkpoint raw thì dùng
   `best.pt`. Hai regime **không cùng quy tắc chọn checkpoint**, và không có bản ghi tiêu chí chọn. Lần đánh giá
   test (16:39) diễn ra trước lần đánh giá val (17:18) cùng ngày, nên không thể chứng minh test không được dùng khi chọn.
2. **4 checkpoint YOLO curated chính là các model đề xuất 215 box insulator của V5** (voters
   `yolo26m|yolo26n|yolo11m|yolo11n`; audit dùng `yolo26m_epoch45.pt`).

> The twelve checkpoints were fixed before the V5 labels were created (curated: 2 November 2026; V5: 3 November 2026) and evaluated once with the same evaluator. Raw checkpoints are the `best.pt` files of the original benchmark. Curated checkpoints are the epochs selected for the Finding Nemo evaluation (epoch 45 for YOLO11m, YOLO26n and YOLO26m; the last epoch for YOLO11n; epoch 15 for both RT-DETR models), which differ from the validation-optimal epochs for four of the six models. The four curated YOLO checkpoints also proposed the 215 `insulator` boxes added in V5. Raw-versus-curated differences on V5 should therefore be read as descriptive, and the `insulator` class should not be used to compare the two regimes.

Khuyến nghị (nếu còn thời gian): đánh giá thêm checkpoint curated `best.pt` trên operational V5 bằng cùng evaluator,
báo cáo như kiểm tra độ nhạy cho quy tắc chọn checkpoint.

## 4. §5.4 Figure RC-3 và Table RC-11: cùng một đợt đánh giá V5

- **Figure RC-3** = `figures/figure_rc3_per_class_raw_curated_v5.pdf`, tạo bởi `scripts/plot_rc3_per_class_v5.py` từ
  `results/finding_nemo/operational_min8_v5/per_class_summary.csv`.
- **Table RC-11** (`report/section_5_4_per_class_v5_ready_to_paste.md`) dùng cùng file CSV đó. Đã tính lại toàn bộ:
  insulator AP50 +16,81 đến +21,31 pp; trung bình theo lớp +17,95 / +0,44 / −0,48 / +1,38 / +1,82 / +1,34;
  YOLO26m foreign_object 0,6993 → 0,7370, bird_nest 0,9110 → 0,9330 (AP50-95 +3,23), insulator_broken 0,8532 → 0,8324,
  broken_strand AP50 −0,75 / AP50-95 +2,48; RT-DETR-R101 cải thiện 5/6 lớp, foreign_object −2,24. **Tất cả khớp.**
- Không có số nào lấy từ benchmark gốc (test 2.012 ảnh).
- Lưu ý khi diễn giải: chênh lệch insulator Raw→Curated (+17,95 pp) **bị nhiễu bởi 215 box V5**, vốn do checkpoint
  curated đề xuất và theo quy ước gán nhãn curated. Kết luận Raw so với Curated nên dựa trên MEAN_5 và 5 lớp lỗi.

## 5. §5.5 Table RC-13: thông tin từng run ablation

Tất cả: YOLO26m, 50 epoch, 640 px, AdamW lr0 1e-3, cosine, warmup 3, AMP, batch 8 (nbs 64), **seed 42**, deterministic;
dữ liệu train `data/curation_module_ablation`; evaluator: Ultralytics `model.val` (split `test`).

| Run | Module dùng | Phần cứng / PyTorch | Chọn checkpoint trên | Báo cáo trên | Kết quả V5 (SHA-256 checkpoint) |
|---|---|---|---|---|---|
| `m2_cd` (reference) | I, II (theo báo cáo ablation) | không có log | `val` 2.272 ảnh / 2.903 box (theo báo cáo ablation, chưa có notebook) | `val_clean` 2.271 / 2.863 | `dori_m2_cd` `2fea6881…` |
| `m1_abf` | I | không có log | không có log | không có log | `dori_m1_abf` `45a0848f…` |
| `m3eg` | III | RTX 4060, 2.14.1+cu126 | `val` 2.272 / 2.903 | `val_clean` | `dori_m3_eg` `4a7c51ab…` |
| `m134` (bỏ II) | I, III, IV | RTX 4050, 2.6.0+cu124 | `val` 2.272 / 2.903 | `val_clean` | `destiny_m134` `d25b44f2…` |
| `m124` (bỏ III) | I, II, IV | RTX 4050, 2.6.0+cu124 | `val` 2.272 / 2.903 | `val_clean` | `destiny_m124` `bdd75596…` |
| `m234` (bỏ I) | II, III, IV | RTX 4050, 2.6.0+cu124 | `val` 2.272 / 2.903 | `val_clean` | `destiny_m234` `e4cc0ba4…` |

Kết quả trên V5 đầy đủ (1.931 ảnh, cùng checkpoint) và SHA-256 đầy đủ:
`benchmakring_transmission_line_defect_model/results_ablation/finding_destiny_test_audit_v5/summary.csv`.

**Sửa nhận định sai:** báo cáo ablation cũ (`curation_ablation_report.md`) gọi `val_clean` là
"Independent Ground-Truth Test Benchmark". Thực tế `val_clean` chỉ khác `val` (tập chọn checkpoint) 1 ảnh và 40 box
foreign_object, nên **không độc lập**.

> Ablation runs used YOLO26m with seed 42 and identical hyper-parameters. Checkpoints were selected on the validation split (2,272 images, 2,903 boxes) and reported on `val_clean` (2,271 of the same images, 2,863 boxes after removing person-as-foreign-object boxes), which is therefore not an independent test set. Runs were executed on two GPUs (RTX 4060 with PyTorch 2.14.1 for m3eg; RTX 4050 with PyTorch 2.6.0 for m124, m134 and m234). The same checkpoints were additionally evaluated on Finding Nemo V5 (1,931 images); the results are exploratory.
