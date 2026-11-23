# Finding Nemo

**Finding Nemo** là tên chung cho bộ dữ liệu 6 lớp đã curate + human review, các tập test và kết quả
đánh giá dùng cho paper, Report và model deploy (MF04). Mọi report về Finding Nemo được gom ở đây.
Cập nhật 23/11/2026.

Lớp: `insulator, insulator_broken, insulator_flashover, bird_nest, foreign_object, broken_strand`.
MEAN_5 = trung bình AP50 của 5 lớp lỗi (không tính `insulator`).

## 1. Các phiên bản dữ liệu

Thư mục gốc: `benchmakring_transmission_line_defect_model/data/`.

| Tên | Thư mục | Ảnh | Box | Nguồn gốc | Vai trò |
|---|---|---|---|---|---|
| Raw train | `finding_nemo_raw` | 18.334 | 25.807 | Gộp 6 nguồn | Train baseline |
| Curated train | `finding_nemo_human_reviewed_6cls/train` | 17.903 | 25.811 | 8 rule curation + cross-label + human review (2.607 accept / 231 reject / 11 unsure) | Train |
| Val gốc | `finding_nemo_raw/val` | 2.272 | 2.903 | | Lịch sử |
| `val_clean` (benchmark) | `curated_combined/labels/val_clean` | 2.271 | 2.863 | Val gốc + review 86 box foreign_object (57 giữ, 27 bỏ, 1 ảnh bỏ, 1 thêm), log: `data/label_audit/review_decisions.csv` | Ablation trong paper |
| `val_clean` (Finding Nemo) | `finding_nemo_human_reviewed_6cls/val` | 2.271 | 3.204 | `val_clean` + 341 box cross-label (YOLO26m curated seed 42 + Roboflow), **chỉ box người chọn accept**; không xóa box nào | Chọn checkpoint model deploy |
| Test gốc | `v6p2e_frozen_test` | 2.012 (1.962 duy nhất) | | Có 50 dòng ảnh trùng | **Không dùng** |
| Test v1 | `test_main_v1` | 1.931 | | 1.962 ảnh duy nhất, bỏ 31 ảnh khoanh người thành FO; không sửa nhãn | Phụ lục |
| Test cross-labelled | `finding_nemo_human_reviewed_6cls/test` | 1.931 | 2.850 | v1 + box cross-label được người accept | **Số MF04 / Report 5** |
| Test v3 | `test_main_v3_model_aligned_exploratory` | 1.931 | | v1 + sửa broken_strand (30 thay, 8 xóa), người duyệt từng ảnh | Cha của V5 |
| **Test V5** | `test_main_v5_full_consensus_assisted` | 1.931 | 2.769 | v3 (broken_strand đã duyệt) + 215 box `insulator` do 4/4 YOLO đồng thuận (không duyệt từng box; đã kiểm chứng ảnh hưởng, xem mục 2) | **Test chính của paper** |
| Operational V5 | `test_operational_min8_v5_full_consensus_assisted` | 1.871 | | V5, bỏ 60 ảnh có box < 8 px | Phân tích phụ |

Phân bố V5: insulator 1.422 · insulator_broken 809 · flashover 124 · bird_nest 177 · foreign_object 111 ·
broken_strand 126 · 124 ảnh nền. Nhãn V5 dựng lại được từ v1 + `test_main_v3_*/correction_log.csv` +
`test_main_v5_*/insulator_additions.csv` (đã đối chiếu khớp 100% với `freeze.json`).

**Không trộn hai bản `val_clean` (2.863 / 3.204) và không trộn test cross-labelled (2.850) với V5 (2.769):**
cùng ảnh nhưng khác nhãn, số liệu không so sánh trực tiếp được.

## 2. Bằng chứng human review

| Bước | Bằng chứng |
|---|---|
| Train | `artifacts/roboflow_relabel/curated_combined_serverless/human_review/` (2.865 mục), `tools/roboflow_relabel/HANDOFF.md` |
| val → val_clean | `data/label_audit/review_decisions.csv` (86 quyết định, split `val_clean`) |
| val_clean / test cross-label | `tools/roboflow_relabel/apply_eval_review.py` chỉ thêm box `accept`; `assemble_finding_nemo_dataset.py` dừng nếu split nào chưa `human_review_applied`; `DATASET_STATUS.json` có hash trạng thái cả 3 split. File quyết định (số reject) chỉ còn trên máy Windows, không có trên git/stash |
| Sửa broken_strand của V5 (v3) | `results_ablation/finding_nemo/targeted_audit_test_main_v1_nms50/review_progress.csv`: 75/75 ảnh lỗi broken_strand đã duyệt (71 box GT sai, 3 lỗi model, 1 sai lớp); **29/29 ảnh bị sửa trong v3 đều nằm trong số đã duyệt** |
| 215 box insulator của V5 | Không duyệt từng box (insulator lành; log: insulator 0/100). Kiểm chứng ảnh hưởng (operational v3 → V5, kết quả v3 lấy từ git `84d4ea7^`): AP50 insulator +0,149…+0,176 ở cả 6 model, **RT-DETR không đề xuất box cũng tăng tương đương** (+0,173 / +0,166) → box là insulator thật, không thiên vị model đề xuất; mAP50 +0,025…+0,029, thứ hạng không đổi, MEAN_5 không đổi |

## 3. Leakage

`artifacts/dataset_audit/leakage_finding_nemo_human_reviewed_6cls/` (SHA-256 + dHash ≤ 4):
trùng chính xác 0; gần trùng với train: 53 ảnh test, 80 ảnh val.
Bỏ 53 ảnh test: F1 0,855 → 0,853, F1 foreign_object 0,765 → 0,719. Đây là **sensitivity analysis**,
không phải ước lượng nhân quả: 53 ảnh này chứa khoảng 19/111 box foreign_object và là ảnh dễ.

## 4. Kết quả (thư mục `benchmakring_transmission_line_defect_model/results_ablation/finding_nemo/`)

| Thư mục | Tập đánh giá | Ghi chú |
|---|---|---|
| `selected_checkpoints_eval_val` | val (Finding Nemo) | 6 model curated, seed 42 |
| `selected_checkpoints_eval` | test cross-labelled (2.850 box) | 6 model curated, seed 42 |
| `selected_checkpoints_eval_test_main_v1` | test v1 | 6 model curated, seed 42 |
| `operational_min8_v5_all` | operational V5 (1.871 ảnh) | Bảng trong [test_audit_v5.md](test_audit_v5.md) |
| `operational_min8_yolo26m` | operational v1 | Chỉ YOLO26m |
| `final_comparison` | val vs test | Khoảng cách val–test từng model |
| `ERROR_ANALYSIS.md`, `targeted_audit_test_main_v1_nms50/` | | Phân tích lỗi, nguồn đề xuất của v3/V5 |

Model deploy MF04 (`seed85_curated_yolo26m`, test cross-labelled, IoU 0,5, sau hậu xử lý): P 0,892 ·
R 0,821 · F1 0,855 · macro-F1 0,796. Chi tiết: `deployment/modal/uavpms-server/modal/perf/results/`.

Benchmark 3 seed (42/36/85) × 6 kiến trúc × raw/curated trên V5 và v1: **chưa có**, chạy bằng
`tools/fn_v5_review/benchmark_seeds.py` khi R101 seed 36/85 train xong (FP32 + warmup 2000,
notebook `legacy/seeds/seed{36,85}-r101-warmup.ipynb`).

## 5. Report trong thư mục này

| File | Nội dung |
|---|---|
| [test_audit_v5.md](test_audit_v5.md) | Audit tập test V5, vai trò các phiên bản, bảng operational V5 |
| [VERIFICATION_2026-11-23.md](VERIFICATION_2026-11-23.md) | Xác minh RC-9a / RC-3 / RC-11 / RC-13, môi trường, hash 12 checkpoint |
| [REPORT_PAPER_FACTS.md](REPORT_PAPER_FACTS.md) | **Số liệu và đoạn văn sẵn để viết report / paper** |
| [curation_ablation_report.md](curation_ablation_report.md) | Ablation leave-one-out 4 module curation (YOLO26m, `val_clean` 2.863 box, 1 seed) |

Liên quan: [../NHAT_KY_2026-11-03.md](../NHAT_KY_2026-11-03.md) (nhật ký audit),
`artifacts/dataset_audit/` (audit dataset), `tools/fn_v5_review/` (script dựng lại / freeze / benchmark V5).

## 6. Sửa sai và dọn dẹp (23/11/2026)

- Số MF04 / Report 5 trước đây ghi là "test V5" → đúng là **test cross-labelled** (2.850 box).
- Paper dùng test 2.012 ảnh (có 50 dòng trùng) → chuyển sang V5 (1.931 ảnh), phụ lục v1.
- Đã xóa (còn trong lịch sử git, script tạo lại vẫn giữ): dữ liệu và kết quả thăm dò v2
  (`test_main_v2_*`), v4 (`test_main_v4_*`), operational v3/v4 (`test_operational_min8_v3_*`,
  `test_operational_min8_v4_*`) và các thư mục kết quả `v2_model_assisted_check`,
  `v3_model_aligned_check`, `operational_min8_v3_all`, `operational_min8_v4_all`. Đã bị V5 thay thế.
  `test_main_v3_*` được giữ vì là cha của V5.
