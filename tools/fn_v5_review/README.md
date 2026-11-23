# Duyệt 215 box insulator do model đề xuất trong Finding Nemo V5

Finding Nemo V5 = `test_main_v1` + chỉnh broken_strand (log v3) + **215 box insulator do 4/4 model YOLO
cùng đề xuất**. Các box này đến từ chính các model được đánh giá, nên mỗi box phải được người xác nhận
trước khi V5 dùng làm test chính. Bộ duyệt này phục vụ việc đó.

## 1. Tạo bộ duyệt (đã chạy sẵn)

```bash
.venv/bin/python tools/fn_v5_review/build_review.py --out artifacts/fn_v5_review
```

Nhãn hiện có được dựng lại từ v1 + log v3 và đã đối chiếu khớp 100% với phân bố trong
`test_operational_min8_v5_full_consensus_assisted/freeze.json` (6 lớp + 124 ảnh nền).

Kết quả trong `artifacts/fn_v5_review/`:

| File | Nội dung |
|---|---|
| `index.html` | Trang duyệt: mở trực tiếp bằng trình duyệt (để cùng thư mục với `crops/`) |
| `crops/P001.jpg` … `P215.jpg` | Mỗi đề xuất: ảnh đầy đủ (trái) + phóng to (phải) |
| `sheets/sheet_01.jpg` … `sheet_36.jpg` | Contact sheet 6 đề xuất / trang, để xem nhanh hoặc in |
| `review.csv` | 1 dòng / box: `box_id, image_id, x_center, y_center, width, height, x1_px…y2_px, decision, adjusted_xyxy_px, reviewer, notes` |
| `kit_manifest.json` | SHA-256 của file nguồn, ảnh có nhiều đề xuất (P144, P145 cùng ảnh) |

Màu: **hồng = box đề xuất**; xanh lá insulator, đỏ insulator_broken, cam flashover, tím bird_nest,
xanh dương foreign_object, vàng broken_strand = nhãn đang có.
Độ tin cậy và model đề xuất **không hiển thị** cho người duyệt (vẫn lưu trong `review.csv` để truy vết).

## 2. Duyệt

Mở `artifacts/fn_v5_review/index.html`, điền tên người duyệt, rồi với mỗi box:

| Phím | Quyết định | Khi nào |
|---|---|---|
| **A** | accept | Là chuỗi sứ `insulator` thật, **chưa có nhãn**, box ôm sát (lệch ≤ ~10% mỗi cạnh) |
| **R** | reject | Không phải insulator; đã có box khác che phủ; là lỗi (phải là lớp khác); quá mờ/che khuất theo quy ước dataset |
| **J** | adjust | Insulator thật chưa có nhãn nhưng box sai rõ (quá rộng, quá hẹp, gộp 2 chuỗi): ghi box đúng `x1,y1,x2,y2` (pixel ảnh gốc) |
| **U** | unsure | Không quyết được: ghi lý do; phải giải quyết trước khi freeze |
| ← → / N | | Ảnh trước / sau / ảnh chưa quyết tiếp theo |

- Reject / adjust / unsure **bắt buộc có ghi chú**.
- Một chuỗi sứ = một box. Box bao 2 chuỗi (ví dụ P060) → adjust thành một chuỗi, hoặc reject.
- Ảnh chụp cận chỉ thấy một phần chuỗi sứ (ví dụ P056): theo quy ước của các nhãn insulator khác trong
  test; nếu dataset không gán nhãn trường hợp này thì reject.
- Quyết định lưu trong trình duyệt. Bấm **Export review.csv** khi xong (và thỉnh thoảng để backup), lưu đè
  `artifacts/fn_v5_review/review.csv`.
- Đề xuất cho paper: một người thứ hai duyệt lại ~10% (khoảng 22 box) để báo cáo tỷ lệ đồng thuận.

## 3. Freeze

```bash
.venv/bin/python tools/fn_v5_review/freeze_v5_reviewed.py --review artifacts/fn_v5_review/review.csv
```

- Tạo `benchmakring_transmission_line_defect_model/data/test_main_v5r_human_verified/`: nhãn, ảnh (hard link
  từ v1), `dataset.yaml`, `manifest.csv` (SHA-256 ảnh + nhãn), `review_decisions.csv`, `freeze.json`.
- Từ chối freeze nếu còn box chưa quyết, thiếu reviewer, thiếu ghi chú, box adjust không hợp lệ, hoặc còn
  `unsure` (trừ khi chạy với `--unsure-as-reject`, được ghi vào `freeze.json`).
- Không bao giờ ghi đè: các thư mục `test_main_v5_*` cũ giữ nguyên để truy vết; thư mục v5r đã có thì dừng.
- Đã kiểm thử: accept cả 215 → đúng 2.769 box, phân bố khớp V5 gốc.
