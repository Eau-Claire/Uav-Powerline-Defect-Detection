# Thông tin để viết Report và Paper (Finding Nemo)

Cập nhật 23/11/2026. Chỉ gồm số liệu đã kiểm chứng trong repo; mục chưa có ghi **[chờ]**.
Chi tiết nguồn gốc: [README.md](README.md). Đoạn tiếng Anh trong khung có thể dán thẳng vào paper/report.

## 0. Quy tắc khi ghi số

- Mỗi con số phải kèm **phiên bản nhãn**: V5 (2.769 box), cross-labelled (2.850 box), v1. Không trộn trong một bảng.
- Validation cũng có 2 phiên bản: `val_clean` 2.863 box (benchmark, ablation) và Finding Nemo 3.204 box (model deploy).
- Kết luận về **phát hiện lỗi** dùng MEAN_5 và AP từng lớp lỗi (không bị 215 box của V5 ảnh hưởng).

## 1. Dữ liệu

| Tập | Ảnh | Box | Nguồn gốc |
|---|---:|---:|---|
| Raw train | 18.334 | 25.807 | 6 nguồn: Zheng et al., IDID, CPLID, VPMBGI, InsuFault-Net, Gao et al. 2025 (tên dataset: xem mục 8) |
| Curated train | 17.903 | 25.811 | 8 rule curation + cross-label + human review (2.607 accept / 231 reject / 11 unsure) |
| `val_clean` (benchmark) | 2.271 | 2.863 | Val gốc 2.272 / 2.903 + review 86 box foreign_object (57 giữ, 27 bỏ, 1 ảnh bỏ, 1 thêm) |
| Val Finding Nemo | 2.271 | 3.204 | `val_clean` + 341 box cross-label được người accept (309 insulator, 19 broken, 9 bird_nest, 4 FO) |
| Test cross-labelled | 1.931 | 2.850 | v1 + box cross-label được người accept |
| **Test V5** | 1.931 | 2.769 | v1 + sửa broken_strand (30 thay, 8 xóa; **người duyệt**, 29/29 ảnh có log) + 215 box insulator do 4/4 YOLO đề xuất (**chưa người duyệt**) |
| Operational V5 | 1.871 | | V5, bỏ 60 ảnh có box < 8 px ở 640 |

Phân bố V5: insulator 1.422 · insulator_broken 809 · flashover 124 · bird_nest 177 · foreign_object 111 · broken_strand 126 · 124 ảnh nền.
Test gốc (`v6p2e_frozen_test`) có 2.012 dòng nhưng chỉ 1.962 ảnh duy nhất → không dùng.

> The primary test set, Finding Nemo V5 (1,931 images, 2,769 boxes), is derived from the original held-out split by removing 50 duplicate rows and 31 images in which a person was annotated as a foreign object. Its labels add two changes. First, `broken_strand` corrections (30 boxes replaced, 8 removed): a targeted audit selected 100 `broken_strand` images, and every corrected image was confirmed by a human reviewer as having an incorrect ground-truth box; the replacement boxes were proposed by the YOLO26m checkpoint. Second, 215 `insulator` boxes proposed by the consensus of four YOLO checkpoints (4/4 agreement, mean confidence ≥ 0.50). The same audit also selected 100 `insulator` images, but none were reviewed, so these 215 boxes were not verified by a person. They can favour those models on `insulator` and on overall mAP, but they do not affect the five defect classes. Defect-detection conclusions therefore rely on MEAN\_5 and per-class defect AP. Results on the unmodified labels (v1) are given in Appendix X.

> Two annotation versions of the validation set (2,271 images) are used. The ablation uses the curated-benchmark labels (2,863 boxes). The deployed model was selected on the Finding Nemo version (3,204 boxes), which adds 341 boxes proposed by cross-labelling (curated YOLO26m, seed 42, and a Roboflow model) and added only after explicit human acceptance; no boxes were removed. Results from the two versions are reported separately.

## 2. Leakage

Trùng chính xác 0; gần trùng với train (dHash ≤ 4): 53 ảnh test, 80 ảnh val.

> Exact-hash checks found no duplicates across splits. A perceptual-hash audit (dHash, Hamming distance ≤ 4) flagged 53 test and 80 validation images as near-duplicates of training images. Removing the 53 test images changes the overall F1 from 0.855 to 0.853 and the `foreign_object` F1 from 0.765 to 0.719. The removed images contain about 19 of the 111 `foreign_object` instances and are detected almost perfectly, so the drop cannot be attributed to leakage alone. We report it as a sensitivity analysis.

## 3. Huấn luyện và đánh giá

- YOLO11n/m, YOLO26n/m: Ultralytics 8.4.172, 50 epoch, 640 px, AdamW lr0 1e-3, cosine → 0,01, warmup 3 epoch, weight decay 5e-4, close_mosaic 10, AMP; batch 16 (n) / 8 (m).
- RT-DETR-R50/R101: PekingU, 50 epoch, 640 px, AdamW (backbone 1e-4, head 1e-3), effective batch 16. Seed 36/85 dùng transformers 5.20.2, DDP 2× T4; **R101 seed 36/85: FP32 + warmup tuyến tính 2.000 bước** (FP16 tràn gradient từ bước đầu ở cả hai seed). R50: AMP, không warmup.
- Phần cứng: seed 42 trên RTX 4060 (benchmark gốc); seed 36/85 trên Kaggle 2× T4; ablation trên RTX 4050 Laptop. Ghi đúng phiên bản Python/PyTorch thực tế (**không** ghi "PyTorch 2.1.4 / CUDA 12.6", phiên bản này không tồn tại).
- Chọn checkpoint chỉ trên validation. YOLO: fitness của Ultralytics; RT-DETR: val loss thấp nhất (`best.pt`). **[cần xác nhận]** nếu paper giữ câu "chọn theo mAP50-95 cho mọi model".
- Evaluator: `benchmakring_transmission_line_defect_model/src/evaluator.py` (bản hiện tại, có ánh xạ khóa checkpoint RT-DETR của transformers 5.x).

> MEAN\_5 is the mean AP$_{50}$ over the five defect classes (`insulator_broken`, `insulator_flashover`, `bird_nest`, `foreign_object`, `broken_strand`), excluding the healthy `insulator` class. Checkpoints were selected on the validation set only; the test set was evaluated once per final checkpoint and was not used for selection or tuning.

## 4. Kết quả đã có

**Seed 42 curated, operational V5 (1.871 ảnh)** — `results_ablation/finding_nemo/operational_min8_v5_all`:

| Model | P | R | F1 | mAP50 | mAP50-95 | MEAN_5 |
|---|---:|---:|---:|---:|---:|---:|
| YOLO26m | 0,856 | 0,769 | 0,810 | **0,828** | **0,478** | **0,800** |
| RT-DETR-R101 | 0,861 | 0,780 | 0,818 | 0,810 | 0,461 | 0,782 |
| RT-DETR-R50 | 0,837 | 0,766 | 0,800 | 0,796 | 0,441 | 0,768 |
| YOLO26n | 0,815 | 0,747 | 0,780 | 0,792 | 0,446 | 0,759 |
| YOLO11m | 0,805 | 0,737 | 0,769 | 0,787 | 0,446 | 0,751 |
| YOLO11n | 0,793 | 0,684 | 0,734 | 0,755 | 0,428 | 0,714 |

**Model deploy `seed85_curated_yolo26m`**:
- Val Finding Nemo (640): P 0,878 · R 0,771 · mAP50 0,841 · mAP50-95 0,480.
- Endpoint (IoU 0,5, sau hậu xử lý): val P 0,891 / R 0,841 / F1 0,865 / macro-F1 0,824; **test cross-labelled** P 0,892 / R 0,821 / F1 0,855 / macro-F1 0,796.
- F1 theo lớp (test cross-labelled): insulator 0,906 · insulator_broken 0,812 · flashover 0,758 · bird_nest 0,881 · foreign_object 0,765 · broken_strand 0,656.

**Thí nghiệm không qua gate** (gate: mAP50-95 val ≥ seed85 và AP foreign_object không giảm) → giữ seed85 @640:
- V7.1/V7.2 (thêm dữ liệu foreign_object): không đạt.
- imgsz 960: i960 mAP50 −0,012, mAP50-95 −0,010 (FO AP50-95 +0,009, flashover −0,047); seed85 fine-tune 960: mAP50-95 −0,060. Recall vật thể nhỏ không cải thiện.

**[chờ]** Benchmark 3 seed (42/36/85) × 6 kiến trúc × raw/curated trên V5 và v1, mean ± std, AP từng lớp:
`tools/fn_v5_review/benchmark_seeds.py` (xuất `table.tex`). Cần R101 seed 36/85 train xong và xác nhận checkpoint seed 42
(dùng các run trong `stash://bunpmc/projects/finding_nemo/`, train trên đúng bộ human-reviewed).

## 5. Ablation (đã có, cần viết lại phần kết luận)

YOLO26m, 1 seed (42), RTX 4050, đánh giá trên `val_clean` 2.863 box. mAP50: Reference (M1+M2) 0,835; bỏ Module II 0,829; bỏ Module III 0,824; bỏ Module I 0,805.

> Among the leave-one-out runs, excluding Module I is associated with the lowest mAP$_{50}$ (0.805), mainly on thin and small classes (`foreign_object` 0.575 vs 0.646–0.649, n = 58; `broken_strand` 0.725 vs 0.766–0.768). These results come from a single seed, are measured on the validation split that is also used for checkpoint selection, and differ by amounts comparable to seed-to-seed variation (about one point). They indicate tendencies rather than established module effects. No run includes all four modules or omits only Module IV, so the effect of SAM 2.1 refinement is not estimated.

## 6. Kiểm thử MF04 (Report)

- Dịch vụ AI (Modal T4): xử lý phía server p95 43 / 58 / 138 ms (640 / 1080p / 4K); round-trip 0,6–0,9 s; 50 ảnh đồng thời p95 4.979 ms (ngưỡng 5.000); 30 phút liên tục 2.933 request, 0 lỗi; cold start khoảng 10 s; 7.815 response, 0 output sai.
- Robustness: ảnh thiếu sáng recall 0,865 → 0,823; insulator_broken < 16 px recall 0,18; broken_strand 16–32 px recall 0,29.
- End-to-end trên VPS (sau khi sửa bridge): 10 ảnh p95 15,1 s; 50 ảnh đồng thời p95 21,2 s (trước sửa 92,5 s); mất kết nối Modal rồi phục hồi p95 67,2 s; ảnh 50 MP p95 18,8 s; label mapping 63/64 ảnh khớp IoU 1,0.
- Defect: 16, đã xử lý 13 (81,3%). Mở (BE): DEF-014 ảnh > 30 MB → 502; DEF-015 confirm qua route AI lỗi consumer, không DLQ; DEF-016 reanalysis nhân đôi detection.

## 7. Threats to validity

1. 215 box `insulator` của V5 do model đề xuất, **không được người duyệt** (log: insulator 0/100 ảnh đã duyệt), ảnh hưởng `insulator` và mAP tổng, không ảnh hưởng MEAN_5. Phần broken_strand của V5 thì đã được duyệt.
2. Near-duplicate train–test (53 ảnh) và train–val (80 ảnh).
3. Hai phiên bản validation; ablation đo trên chính tập dùng chọn checkpoint.
4. Ablation 1 seed, phần cứng khác benchmark.
5. Seed 42 dùng pipeline RT-DETR cũ; R101 seed 36/85 dùng FP32 + warmup.
6. broken_strand và vật thể nhỏ vẫn yếu; foreign_object chỉ 111 box trên test.

## 8. Data availability và việc còn lại

SHA-256: manifest `test_main_v1` `a5fcd477…`; operational V5 `e8e2bb2c…`; bundle audit V5 `5B9C59DE…`;
checkpoint deploy `8682baa0…`. Stash: `stash://bunpmc/projects/finding_nemo/`, `stash://bunpmc/projects/finding_seed/`.

| Việc | Trạng thái |
|---|---|
| R101 seed 36/85 (FP32 + warmup) | Đang train lại |
| Xác nhận checkpoint seed 42 | Chưa |
| Benchmark 3 seed trên V5 + v1 | Chờ 2 việc trên |
| Tên dataset: README ghi **"FOTL / EFOD_Drone"**, paper ghi **"FOLT-Drone"** | Đối chiếu với bài gốc Gao et al. 2025 rồi thống nhất |
| `file.bib` | Không có trong repo |
| Số box bị reject khi duyệt val/test cross-label | Chỉ có trên máy Windows (`HUMAN_REVIEW_STATUS.json`); không bắt buộc |
| "Report 3/4" cũ dùng test 2.012 ảnh | Đổi sang V5 hoặc ghi rõ là số cũ |
