# Phiên 2026-10-15: teacher plateau, audit dữ liệu, structural consistency

Máy: thuê RTX 3090 24 GB, 25 CPU, 53 GB RAM, Python 3.13, torch 2.14.1+cu130, ultralytics 8.4.172.
Branch: `v6p5-plateau-yolo26n`. Mọi thứ không nằm trong git ở trên Camber:
`stash://bunpmc/projects/yolov11sdi/checkpoints/v6p5_v7_session_20261015/`.
Khôi phục ở máy mới bằng `notebooks/08_resume_session_20261015.ipynb`.

Thước đo dùng xuyên suốt: **MEAN_5** = trung bình AP50 của 5 lớp lỗi (insulator_broken, insulator_flashover,
bird_nest, foreign_object, broken_strand), hệ 2 tầng (Pi n_v5 gắn cờ ảnh khi conf ≥ 0.05; server = teacher, FO lấy ở
imgsz 800, các lớp khác ở 640), trên `val_clean_v6`. Bootstrap ghép cặp 2000 vòng.

## 1. Notebook 22 (v6 vs v5 teachers), chạy xong 16:42
Fine-tune teacher v5 (YOLO26m) 20 epoch, lr 1e-4, mosaic, `val=False` (mAP giữa chừng hiện 0 là do đó, không phải lỗi).
- v5_orig 0.8295 · ft_v5 0.8198 · ft_v6 0.8128. Student n_v6 Pi-only 0.7697 so với n_v5 0.7785.
- Kết luận: chính bước fine-tune làm tụt (ft_v5, cùng data v5, cũng tụt), không phải data v6.
  Nguyên nhân: v5_orig đã kết thúc ở lr ~5e-5 không mosaic; train lại lr 1e-4 + mosaic + AdamW mới đẩy khỏi tối ưu.

## 2. v6p5-23: fine-tune nhẹ (`scripts/exp_v6p5_23_gentle_ft.py`, `artifacts/v6p5_v7_gentle/summary.md`)
lr 1e-5, không mosaic, 8 epoch, val mỗi epoch; gf = freeze=10; soup = WiSE-FT 0.5·v5_orig + 0.5·model.

| teacher | MEAN_5 | | teacher | MEAN_5 |
|---|---|---|---|---|
| v5_orig | **0.8283** | | gf_v6 | 0.8252 |
| soup_g_v6 | 0.8281 | | g_v6 | 0.8219 |
| soup_gf_v6 | 0.8266 | | g_v5 | 0.8206 |
| soup_ft_v6 | 0.8255 | | ft_v6 (nb22) | 0.8129 |

- **soup_g_v6 − soup_g_v5 = +0.0039, P(>0)=0.96**: nhãn v6 có ích khi bỏ phần hại do fine-tune.
- Không cách nào vượt v5_orig → teacher đã plateau với nhãn hiện tại.
- Student gn_v6 (distill từ v5_orig, lr 1e-5) 0.7776 ≈ n_v5 0.7785 (FO +0.011, strand +0.008, broken −0.018).

## 3. v6p5-24: mẹo suy luận (`scripts/exp_v6p5_24_inference.py`): tất cả KÉM baseline
FO@1024 −0.007, TTA −0.003…−0.007, WBF ensemble −0.008…−0.012. Giữ server như cũ (v5_orig, FO@800).

## 4. Audit dữ liệu
- `scripts/audit_v6_boxes.py` → `artifacts/audit_v6_boxes/` (flags.csv: 3924 cờ train, 1066 val; sheets/).
  Phát hiện: insulator_broken vẽ lúc 1 box/tán lúc 1 box/vết nứt; teacher v5 nhầm thợ điện → FO (conf 0.8–0.9);
  924 insulator thấy rõ nhưng chưa gán nhãn; ảnh flashover có vòng tròn đỏ vẽ sẵn (shortcut).
- **Rò rỉ**: 148 ảnh val_clean_v6 (6.5%) gần trùng ảnh train (pHash ≤ 6, xem tay 8/8 trùng thật, có bản augment
  Roboflow). `artifacts/review_v7/val_train_near_dups.json`. 409 ảnh val có bản trùng ngay trong val.
  **MEAN_5 không rò: v5_orig 0.8158, soup_g_v6 0.8165** (thay vì 0.828). Chưa đạt mục tiêu 0.82.
- v7 tự sửa box theo teacher (`scripts/build_v7.py`, `datasets/v6p5_v7`): 331 box chỉnh, 5 gộp, 301 insulator thêm.
  A/B g_v7 − g_v6 = +0.0003 → không có tác dụng. Val không bị sửa tự động (tránh thổi phồng điểm).
- Quy ước đã chốt với user: **box sát vùng lỗi nhìn thấy + chừa ~5% mỗi phía**. Nhưng SAM (`scripts/sam_tighten.py`)
  cho thấy box hiện tại đã sát (IoU trung vị 0.826 = đúng mức chỉ do padding) → không áp dụng hàng loạt.
- broken_strand (`scripts/strand_v8.py`): giả thuyết "box dài là sai quy ước" SAI. Box dài thường đúng (dây tưa dài,
  ảnh chụp gần). Vấn đề là **miền ảnh**: 77% box strand là ảnh điện thoại/mặt đất (640×640), chỉ 360 box ảnh drone
  (640×480, AP50 0.56). Val strand cũng 208/252 là ảnh mặt đất. Cần ảnh drone thật.
- Hàng đợi review: `artifacts/review_v7/README.md` + `review_queue.csv` (283 dòng, cột `suggest`).

## 5. Diffusion (SDXL-inpainting)
- FO (`scripts/proto_diffusion_fo.py`): ~25% dùng được (túi nilon, khăn); lỗi: vẽ ra chim, quầng oval xám, vật lơ lửng.
- broken_strand drone (`scripts/proto_diffusion_strand.py`): 0/16. Hough bắt nhầm xà thép; SDXL không vẽ được sợi tưa.
- Không dùng để train. Nếu tiếp: LoRA/ControlNet trên ~1900 vùng dây tưa thật; và cần test set drone thật trước.

## 6. Câu hỏi nghiên cứu (`scripts/struct_audit.py`, `artifacts/struct_audit/`)
"Can structural consistency provide additional evidence for identifying annotation errors and unreliable detections
beyond conventional confidence- and loss-based methods?" Study set = val_clean_v6 bỏ 148 ảnh rò (2123 ảnh).
- Part A (dự đoán không đáng tin, tự chấm bằng GT): logistic CV 5-fold, {conf} vs {conf + 7 đặc trưng cấu trúc}.
  AUROC 0.945 vs 0.945; FO tệ hơn (0.872→0.792, n=110). **Kết quả âm** với đặc trưng thủ công.
- Part B (lỗi gán nhãn): điểm cấu trúc tương quan yếu với conf (ρ 0.29), độc lập với disagreement (ρ −0.02);
  top-50 chỉ trùng 11/50 → tín hiệu khác biệt nhưng **chưa biết đúng hay sai**.
  Luật cứng "lỗi phải nằm trong insulator": 1213 box chỉ cấu trúc bắt; xem 24 mẫu: đa số là thiếu nhãn insulator /
  ảnh cận, không phải box lỗi sai.
- **Việc còn treo**: user xác minh mù 205 box (`verify_01..09.jpg`, điền `verify_blind.csv` cột verdict:
  correct | wrong_class:<cls> | bad_box | not_defect | duplicate | unsure), rồi ghép `verify_key.csv` → precision@k.

## 7. Việc tiếp theo (ưu tiên)
1. Điền `verify_blind.csv` (205 box) → trả lời RQ phần B.
2. Bỏ 148 ảnh train trùng val (v8), báo cáo MEAN_5 không rò + tách strand theo miền ảnh.
3. Sửa val: thêm ~13 bird_nest thiếu (review_v7), duyệt 82 insulator_broken đáng ngờ; quyết drone có là FO (mặc định không).
4. Thu ảnh drone thật có dây tưa (kể cả chưa gán nhãn → pre-label bằng teacher).
5. Sau khi nhãn ổn: train lại đầy đủ (không fine-tune từ v5_orig), đối chứng v6 cùng công thức.
6. Server: giữ v5_orig (hoặc soup_g_v6, ngang điểm) + FO@800; Pi: giữ n_v5 (gn_v6 ngang điểm).

GitHub: commit chưa push được từ máy thuê (không có credential). Bundle git nằm trên Camber (`repo.bundle`);
notebook 08 tự áp bundle nếu GitHub thiếu commit.
