# NHẬT KÝ TUẦN 18/10/2026

**Thời gian:** 12/10/2026 – 18/10/2026  
**Đề tài:** Phát hiện lỗi hạ tầng điện từ ảnh UAV; nghiên cứu tính nhất quán cấu trúc và mô hình phát hiện hai tầng Raspberry Pi – server.

## 1. Mục tiêu trong tuần

- Rà soát và cải thiện chất lượng nhãn của bộ dữ liệu V6.5.
- Huấn luyện, so sánh mô hình teacher và student phục vụ hệ thống hai tầng.
- Kiểm tra ảnh hưởng của fine-tuning, độ phân giải và các kỹ thuật suy luận đến độ chính xác.
- Đánh giá khả năng sử dụng tính nhất quán cấu trúc để phát hiện nhãn sai và dự đoán không đáng tin cậy.
- Chuẩn bị mô hình và công cụ đo hiệu năng trên Raspberry Pi.

## 2. Nội dung công việc đã thực hiện

| Thời gian/mốc ghi nhận | Công việc | Kết quả, sản phẩm |
|---|---|---|
| 12/10 | Xây dựng quy trình kiểm tra nhãn; bổ sung các notebook huấn luyện teacher và distillation student trên các phiên bản dữ liệu v2–v4. | Có quy trình hỗ trợ rà soát nhãn và đối chiếu các phiên bản mô hình. |
| 13/10 | Xây dựng dữ liệu v5 dùng chung với 7 lớp, bổ sung nhãn insulator và person; hoàn thiện luồng huấn luyện tự động, đóng gói dữ liệu và notebook chạy trên GPU. | Bổ sung công cụ huấn luyện student YOLO26n/YOLO26s và đánh giá hệ thống hai tầng. |
| 15/10 | Rà soát foreign_object, tạo dữ liệu v6; so sánh teacher v5/v6 và student; thử fine-tuning nhẹ, đóng băng một phần mạng, trộn trọng số và các kỹ thuật suy luận. | Xác định teacher hiện tại đã chững lại; các biến thể thử nghiệm chưa vượt baseline một cách thuyết phục. |
| 15/10 | Kiểm tra chất lượng box, nhãn thiếu, ảnh gần trùng và khác biệt miền ảnh; thử sửa nhãn v7 và sinh ảnh bằng diffusion. | Phát hiện rò rỉ train–validation, thiếu nhãn và sự khác biệt giữa ảnh drone với ảnh mặt đất. Dữ liệu diffusion thử nghiệm chưa được đưa vào huấn luyện. |
| 15/10 | Thực hiện nghiên cứu structural consistency; xác minh mù 205 box và so sánh với điểm confidence, loss, disagreement. | Chưa tìm thấy lợi ích bổ sung rõ ràng từ các đặc trưng cấu trúc thủ công đang sử dụng. |
| 15–18/10 | Xây dựng v8, sửa một phần nhãn validation và đánh giá lại trên tập không rò; bổ sung bootstrap theo cụm ảnh gần trùng. | Có báo cáo theo lớp, khoảng tin cậy và phân tích riêng broken_strand theo miền ảnh. |
| 16–18/10 | So sánh student ở độ phân giải 640/800; xuất mô hình NCNN ở 640×640 và 640×480; chuẩn bị script benchmark Pi. | Có gói mô hình và công cụ đo thời gian suy luận, nhiệt độ, hạ xung; chưa có kết quả tốc độ thực tế trên Pi trong các nguồn đã tổng hợp. |
| Đến 18/10 | Chuẩn bị dữ liệu v9 và tập đánh giá foreign_object; tích hợp kết quả vào repository. | Log xây dựng ghi nhận tập test_fo gồm 145 ảnh, 150 box foreign_object; PR #1 được merge ngày 18/10. |

*Các mốc dựa trên lịch sử commit, tên notebook và tài liệu phiên làm việc; không suy diễn công việc riêng cho những ngày không có ghi nhận.*

## 3. Kết quả nổi bật

### 3.1. Chất lượng dữ liệu và đánh giá

- Phát hiện 148 ảnh validation gần trùng ảnh train, tương đương khoảng 6,5% validation; đồng thời ghi nhận 409 ảnh validation có bản gần trùng ngay trong validation.
- Xây dựng v8 từ v6: loại 170 ảnh train liên quan đến 148 ảnh validation bị rò; bổ sung 11 nhãn bird_nest bị thiếu và loại 2 nhãn insulator_broken không đúng đối tượng.
- Đánh giá mô hình cũ trên 2.123 ảnh validation không rò, trong tổng số 2.271 ảnh val_clean_v8.
- Chuẩn hóa chỉ số MEAN_5: trung bình AP50 của 5 lớp lỗi insulator_broken, insulator_flashover, bird_nest, foreign_object và broken_strand; không phải mAP của toàn bộ 7 lớp.

| Cấu hình | MEAN_5 trên validation không rò |
|---|---:|
| Student n_v5 chạy độc lập | 0,761 |
| Teacher v5_orig chạy độc lập | 0,818 |
| Hệ hai tầng n_v5 → v5_orig | 0,817 |

Hệ hai tầng đạt AP50 theo lớp: insulator_broken 0,847; insulator_flashover 0,841; bird_nest 0,927; foreign_object 0,688; broken_strand 0,784. Báo cáo bootstrap theo ảnh cho MEAN_5 khoảng tin cậy 95% là 0,79–0,85. Chưa đủ cơ sở khẳng định hệ thống vượt mục tiêu 0,82.

### 3.2. Thử nghiệm mô hình

- Fine-tuning thông thường làm giảm điểm ngay cả khi giữ dữ liệu v5; kết quả cho thấy không thể quy việc giảm điểm chỉ cho dữ liệu v6.
- Fine-tuning nhẹ và trộn trọng số giúp giảm mức suy giảm nhưng chưa vượt teacher v5_orig rõ ràng.
- Tăng độ phân giải foreign_object lên 1024, TTA và WBF trong các cấu hình đã thử đều kém baseline; tiếp tục giữ teacher v5_orig với foreign_object ở độ phân giải 800.
- Student huấn luyện mới ở 640 thấp hơn n_v5 khoảng 0,0186 MEAN_5; bản 800 chưa cho thấy cải thiện rõ ràng so với n_v5. Tiếp tục giữ n_v5 làm student tham chiếu.

### 3.3. Nghiên cứu tính nhất quán cấu trúc

- Với bài toán phân biệt dự đoán đáng tin cậy, thêm đặc trưng cấu trúc không cải thiện AUROC tổng thể: khoảng 0,945 ở cả hai cấu hình.
- Đợt xác minh mù 205 box ghi nhận 26 lỗi, khoảng 12,7%.
- Precision@50: loss 0,24; confidence 0,20; structural consistency 0,14; disagreement 0,12. Khoảng tin cậy còn rộng do số lỗi được xác minh ít.
- Kết luận trong phạm vi thử nghiệm: chưa có bằng chứng cho thấy đặc trưng cấu trúc thủ công mang lại lợi ích bổ sung so với các tín hiệu hiện có; cần xem lại đặc trưng và thiết kế nghiên cứu trước khi mở rộng.

### 3.4. Chuẩn bị triển khai trên Raspberry Pi

- Xuất student n_v5 sang NCNN cho đầu vào 640×640 và 640×480.
- Bản 640×480 giảm 25% số pixel; kiểm tra được ghi nhận trên 40 ảnh drone tỉ lệ 4:3 cho kết quả giống bản .pt. Kết quả này chỉ áp dụng trong phạm vi mẫu đã kiểm tra.
- Chuẩn bị benchmark chạy liên tục 300 giây để đo độ trễ, nhiệt độ và tình trạng hạ xung; cần chạy trên thiết bị thực tế để kết luận hiệu năng.

## 4. Khó khăn và tồn tại

- Nhãn insulator_broken còn khác nhau về quy ước vùng bao; vẫn còn các box nhỏ đáng ngờ cần người có chuyên môn xác minh.
- Lớp foreign_object có ít mẫu trên tập đánh giá không rò nên độ bất định còn lớn.
- broken_strand lệch miền dữ liệu: AP50 trên ảnh vuông/ảnh mặt đất khoảng 0,827, trong khi nhóm ảnh được báo cáo là drone/không vuông khoảng 0,570. Cần bổ sung ảnh UAV thực tế và kiểm chứng miền ảnh bằng nguồn dữ liệu.
- Sinh ảnh diffusion cho dây tưa chưa đạt chất lượng; chưa sử dụng các mẫu thử nghiệm để train.
- Các kỹ thuật tăng độ phức tạp huấn luyện hoặc suy luận chưa giải quyết được giới hạn chất lượng dữ liệu.

## 5. Kế hoạch tuần tiếp theo

1. Tiếp tục rà soát các nhãn insulator_broken đáng ngờ, nhãn thiếu và thống nhất quy ước box.
2. Bổ sung ảnh drone thật cho broken_strand và foreign_object; kiểm soát ảnh gần trùng giữa các tập.
3. Sau khi ổn định nhãn, huấn luyện lại đầy đủ và so sánh các phiên bản dữ liệu với cùng công thức huấn luyện.
4. Chạy benchmark NCNN trên Raspberry Pi, ghi nhận độ trễ, nhiệt độ và hạ xung để chọn độ phân giải phù hợp.
5. Sử dụng tập test_fo đã chuẩn bị theo một giao thức đánh giá cố định; tiếp tục báo cáo kết quả theo lớp, miền ảnh và khoảng tin cậy.
6. Xem xét đặc trưng cấu trúc học được hoặc tăng quy mô xác minh nếu tiếp tục hướng nghiên cứu structural consistency.

## 6. Nguồn đối chiếu trong repository

- Lịch sử commit từ 12–18/10/2026; commit `4d3d1f72` và merge `1b06592c` ngày 18/10.
- [Tổng kết phiên 15/10](SESSION_2026-10-15_v6p5_plateau.md).
- [Báo cáo đánh giá v8](../artifacts/eval_v8/report.md) và [bootstrap theo cụm](../artifacts/eval_v8/report_cluster.md).
- [So sánh student 640/800](../artifacts/exp_student_res/final_compare.csv).
- [Gói mô hình và hướng dẫn benchmark Pi](../artifacts/pi_v8/README.md).
- [Log xây dựng v9](../artifacts/build_v9.log).
