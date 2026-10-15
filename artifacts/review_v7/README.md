# Review v7: chỉ những gì cần người quyết

Tự động (đã làm, `datasets/v6p5_v7`, chi tiết `changes_v7.csv`): chỉnh 331 box lệch rõ theo teacher (conf ≥ 0.8, IoU 0.5–0.75),
gộp 5 nhóm vết nứt thành 1 box/tán, thêm 301 insulator thiếu nhãn. Val **không** bị sửa tự động (để điểm không bị thổi phồng).

## A. Quyết định quy ước (ảnh hưởng lớn nhất tới gap mAP50 → mAP50-95)
1. insulator_broken: **1 box / tán hỏng** (đề xuất) hay 1 box / vết nứt?
2. broken_strand: **box sát chỗ tưa** (đề xuất) hay cả đoạn dây? (nguồn 640×640 đang vẽ cả đoạn, 640×480 vẽ sát)
3. insulator_flashover: **cả tán bị cháy** (đề xuất) hay chỉ vùng vết cháy?
4. Drone/UAV bay trong ảnh có tính là foreign_object không?

## B. Rò rỉ train → val (`val_train_near_dups.json`)
148 ảnh val_clean_v6 (6.5%) gần trùng ảnh train (đã kiểm tra mắt 8/8 là trùng thật, có bản là ảnh augment của Roboflow).
Đề xuất: **xoá các ảnh train trùng** ở v8 (giữ val nguyên). 409 ảnh val có bản gần trùng ngay trong val → CI bootstrap đang hẹp hơn thực tế.

## C. Hàng đợi `review_queue.csv` (283 dòng, cột `suggest` = gợi ý của mình sau khi xem mẫu, cột `decision` để bạn điền)
Sheet ảnh: `*_NN.jpg` (số `#id` trên ảnh = cột `id`). Trắng = box đang xét, màu = nhãn hiện có.

| Nhóm | Số | Gợi ý | Việc của bạn |
|---|---|---|---|
| val bird_nest missing_label | 13 | **add**: đã xem, ~11/13 là tổ thật chưa gán | xác nhận nhanh 1 sheet |
| val FO missed_gt | 12 | **ok**: nhãn đúng, model yếu (diều/túi/bóng) | không cần |
| val bird_nest missed_gt | 42 | **ok**: đã xem, gần hết là tổ thật bị che / lẫn thiết bị | không cần |
| val insulator_broken missed_gt | 82 | nhiều box là chói sáng / mẻ cực nhỏ / đặt trên kẹp-dây buộc (#64, #77) | **cần người có chuyên môn**: ok / drop |
| val broken_strand missed_gt | 27 | phụ thuộc quy ước A2 | làm sau khi chốt A2 |
| val flashover missed_gt | 11 | có ảnh có **vòng tròn đỏ vẽ sẵn** (#24, #25): model có thể học "vòng đỏ = lỗi" | ok / drop, đánh dấu ảnh vòng đỏ |
| train flashover / FO | 90 | ưu tiên thấp | để sau |

Thứ tự đề xuất: A (5 phút) → bird_nest add (2 phút) → insulator_broken 82 ô (4 sheet, ~20 phút). Phần còn lại không bắt buộc.
