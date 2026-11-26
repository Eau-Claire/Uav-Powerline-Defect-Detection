# Defense 1.1: notebook và kết quả cuối cùng

Thư mục này gom **chỉ những notebook và kết quả thực sự sinh ra số liệu trong Report/Paper**, lấy từ mọi branch
(`finding_v7p1_171126`, `finding_nemo_v5_results`, `finding_seed`, `v6p5-plateau-yolo26n`, `main`). Gom ngày 26/11/2026.

- Đây là **bản sao**. File gốc vẫn ở đường dẫn cũ; notebook tìm đường dẫn theo gốc repo, nên muốn chạy lại thì chạy bản gốc.
- Checkpoint và dataset không nằm trong git, chỉ có trên Camber stash (cột "Camber").
- Số liệu và đoạn văn để viết bài: [../docs/finding_nemo/REPORT_PAPER_FACTS.md](../docs/finding_nemo/REPORT_PAPER_FACTS.md).
  Xác minh checkpoint, môi trường, SHA-256: [../docs/finding_nemo/VERIFICATION_2026-11-23.md](../docs/finding_nemo/VERIFICATION_2026-11-23.md).
- **Test cuối đã chốt:** Finding Nemo V5 (operational), 1.871 ảnh / 2.467 box, manifest SHA-256 `e8e2bb2c…`.

## Bản đồ: mục trong report → thư mục

| Mục report | Thư mục | Notebook / script | Kết quả | Camber (`stash://bunpmc/projects/…`) |
|---|---|---|---|---|
| Table RC-7, RC-8a, RC-8, RC-9, RC-10; số FPS của Figure RC-3 (giai đoạn phát triển, **không phải test cuối**) | [01_dev_stage_benchmark_seed42](01_dev_stage_benchmark_seed42) | `benchmark_pipeline.ipynb` (YOLO raw + curated), `kaggle-train-rtdetr-peking_used.ipynb` (RT-DETR) | `results/results_yolo_model/` (YOLO), `results/results_rtdetr_last/` (RT-DETR, khớp RC-8a), `per_class_benchmark.csv` | `raw_<model>/` |
| Train model Curated seed 42 (12 checkpoint của RC-9a, phần curated) | [02_curated_training_seed42](02_curated_training_seed42) | `v6p5-27_*` (4 YOLO, RTX 4060), `v6p5-28_*` (RT-DETR, Kaggle), `scripts/train_finding_nemo.py` | `training_curves/<model>/results.csv`, `training_state.json` | `finding_nemo/<model>/`, `finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/` |
| **Table RC-9a, Table RC-11, Figure RC-4, Table RC-5** (test cuối V5) | [03_final_test_v5_rc9a](03_final_test_v5_rc9a) | `v6p7-07_*` (12 checkpoint cố định, 1 evaluator), `scripts/build_report_rc_tables_figures.py`, `scripts/plot_*.py` | `results/raw_*.json`, `curated_*.json`, `benchmark_summary.csv`, `per_class_summary.csv`, `results/report/` (bảng sẵn để dán), `results/figures/` | 12 checkpoint: xem bảng bên dưới; bundle test V5: `finding_nemo/audits/finding_nemo_test_audit_v5_bundle.tar` |
| Table RC-12, nguồn của nhãn V5, đánh giá seed 42 trên val / test cũ | [04_seed42_other_evals](04_seed42_other_evals) | `scripts/evaluate_nemo_selected_checkpoints.py`, `audit_targeted_errors.py`, `review_targeted_audit.py`, `audit_split_leakage.py` | `results/ERROR_ANALYSIS.md`, `targeted_audit_test_main_v1_nms50/` (log duyệt broken_strand, 215 box insulator), `selected_checkpoints_eval*` | `finding_nemo/` |
| **Table RC-13, §5.5** (ablation, exploratory) | [05_ablation_rc13](05_ablation_rc13) | `v6p7-01` (m1_abf, m2_cd, m3_eg), `v6p7-02/03/04` (m124, m134, m234), `v6p7-05` (chấm trên V5 đầy đủ), `scripts/eval_full4.py` (dòng đủ 4 module) | `results/finding_destiny_test_audit_v5/summary.csv`, `results/full_pipeline_v5full_eval.json`, `results/loo_val_clean.csv`, `curation_ablation_report.md` | `finding_dori/`, `finding_destiny/yolo26m_m_<124\|134\|234>/`, `finding_destiny/test_audit_v5_results/` |
| Seed 36 / 85 (benchmark 3 seed: **chưa xong**) | [06_seeds_36_85](06_seeds_36_85) | `seed36.ipynb`, `seed85.ipynb` (12 run mỗi seed, Kaggle 2× T4), `seed*-rerun-rt.ipynb` (RT-DETR chạy lại), `seed*-r101-warmup.ipynb` (R101 FP32 + warmup), `v6p7-06_*` | `results/finding_seed36_test_audit_v5/`; `scripts/benchmark_seeds.py` (để chạy khi đủ checkpoint) | `finding_seed/seed36/`, `finding_seed/seed85/`, `finding_seed/seed<36\|85>_rtdetr_bnfix/`, `finding_seed/seed<36\|85>_rtdetr101_warmup/` |
| Model deploy `seed85_curated_yolo26m`; §5.3, kiểm thử MF04 (Report §6), sensitivity analysis leakage | [07_deployed_seed85](07_deployed_seed85) | Train: `06_seeds_36_85/seed85.ipynb`; chấm test V5: `scripts/eval_seed85.py` | `results/seed85_v5_test_eval.json`; `mf04_perf/` (MF04; leakage: `2026-11-22_1424/SUMMARY.md` §3) | `finding_seed/seed85/curated_yolo26m/weights/best.pt` (SHA-256 `8682baa0…`) |
| Thí nghiệm sau deploy, **không qua gate** (giữ seed85) | [08_after_deploy_experiments](08_after_deploy_experiments) | `v7p1_*` (thêm dữ liệu FO), `v7p2_*fo_size*`, `v7p2_*imgsz960*` | `results/finding_v7p1_171126/`, `results/finding_v7p2_211126/` | `finding_v7p1_171126/`, `finding_v7p2_211126/` |

## 12 checkpoint của Table RC-9a

| Regime | Model | Camber (`stash://bunpmc/projects/…`) | SHA-256 (đầu) |
|---|---|---|---|
| Raw | YOLO11n / 11m / 26n / 26m | `raw_<model>/weights/best.pt` | `f490385b` / `786cfd87` / `f2cab932` / `ad398adf` |
| Raw | RT-DETR-R50 / R101 | `raw_rtdetr50\|raw_rtdetr101/checkpoints/best.pt` | `410b9f03` / `10ee1456` |
| Curated | YOLO11n | `finding_nemo/yolo11n/weights/last.pt` | `0284db0c` |
| Curated | YOLO11m / 26n / 26m | `finding_nemo/<model>/weights/epoch45.pt` | `4c273fd5` / `93e1bc2d` / `a1fc47b7` |
| Curated | RT-DETR-R50 / R101 | `finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/<r50vd\|r101vd>/checkpoints/epoch15.pt` | `01b17f92` / `812b5d56` |

Hash đầy đủ: VERIFICATION §3. Checkpoint curated được **chọn thủ công**, không phải `best.pt` (phải nói rõ trong bài).

## Những gì có trên các branch nhưng **không** đưa vào đây

| Thứ | Ở đâu | Lý do |
|---|---|---|
| Test v2, v3 (operational), v4 và kết quả của chúng | `main`, `finding_seed` | V5 đã thay thế (v3 đầy đủ vẫn giữ ở `data/` vì là cha của V5) |
| `finding_nemo_evaluate_raw_curated_human_reviewed_test.ipynb` | `v6p5-plateau-yolo26n` | Đánh giá trên test cross-labelled, đã thay bằng `v6p7-07` trên V5 |
| `figures/` (fig01–fig21, diag01–07) | `v6p5-plateau-yolo26n` | Bản nháp trước đợt đánh giá RC ngày 19/11; hình cuối nằm ở `03_final_test_v5_rc9a/results/figures/` |
| `results_rtdetr_best/`, `results_test_all/` | thư mục benchmark | Không khớp số trong report (RC-8a dùng `results_rtdetr_last`) |
| `legacy/notebooks/v6p1`–`v6p5` (trừ v6p5-27/28), `notebooks/00–08` | mọi branch | Giai đoạn dựng dữ liệu cũ, không sinh số trong report |

## Log tải từ Camber: [camber_logs](camber_logs)

Bản sao **các file nhỏ** (log, CSV, JSON, YAML; không có weights hay ảnh) của `stash://bunpmc/projects/finding_dori/`,
`finding_destiny/` và `finding_seed/`, giữ nguyên đường dẫn trên stash (`camber_logs/<project>/<run>/…`). Tải ngày 26/11/2026.
Các `train.log` > 1 MB được nén thành `train.log.gz` (đọc bằng `zcat`).

| Thư mục | Nội dung |
|---|---|
| `finding_dori/yolo26m_m1_a_b_f`, `yolo26m_m2_c_d`, `yolo26m_m3_e_g` | Ablation M1, M1+M2, M3: `args.yaml`, `data.yaml`, `dataset_manifest.json`, `results.csv` |
| `finding_destiny/yolo26m_m_124`, `_134`, `_234` | Ablation leave-one-out: `args.yaml`, `results.csv` |
| `finding_destiny/test_audit_v5_results` | Kết quả RC-13 trên V5 đầy đủ (giống `05_ablation_rc13/results/`) |
| `finding_seed/seed36`, `seed85` | 8 run YOLO mỗi seed: `train.log`, `results.csv`, `done.json` (epoch tốt nhất trên val); `seed85/curated_yolo26m` là model deploy |
| `finding_seed/seed<36\|85>_rtdetr_officialfix` | RT-DETR: **R50 raw/curated hoàn tất**; R101 hỏng (NaN / BatchNorm không finite) |
| `finding_seed/seed<36\|85>_rtdetr101_warmup` | R101 FP32 + warmup, đang chạy: seed 85 raw 30/50, curated 3/50; seed 36 raw dừng ở epoch 11, curated hỏng (NaN) |
| `finding_seed/seed36_rtdetr_ddpfix`, `_dnfix`, `seed36/*rtdetr100` | Lần thử RT-DETR cũ, đã bị thay; giữ để đối chiếu |

## Còn thiếu

- `m1_abf` và `m2_cd`: đã có cấu hình và `results.csv` (YOLO26m, seed 42, 50 epoch, batch 8, AdamW 1e-3, AMP, deterministic;
  chọn trên `val` raw, test = `val_clean`), nhưng **không có `train.log`**, nên không biết GPU và phiên bản PyTorch.
- Log môi trường của RT-DETR curated seed 42 và của các checkpoint raw: không tìm thấy (VERIFICATION §2).
- Benchmark 3 seed (42/36/85) trên V5: chờ R101 seed 36/85 train xong.
