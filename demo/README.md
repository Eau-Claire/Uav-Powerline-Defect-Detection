# Streamlit model test app

Ứng dụng thử nghiệm checkpoint YOLO26n đã train:

`benchmakring_transmission_line_defect_model/runs/curated_yolo26n/weights/best.pt`

Có hai nguồn ảnh:

- **Tải ảnh**: chọn một hoặc nhiều ảnh từ máy tính.
- **Camera IP**: xem luồng MJPEG/HTTP hoặc RTSP và nhận diện liên tục từng frame. Dùng nút bắt đầu/dừng stream để điều khiển; log thời gian, số frame, số đối tượng, loại lỗi và FPS hiển thị trong ứng dụng.

## Cài đặt và chạy

Mở terminal tại thư mục `test`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Với IP Camera Lite, đặt `CAMERA_STREAM_URL` thành địa chỉ **MJPEG** mà ứng dụng camera hiển thị (đường dẫn thường kết thúc bằng `/video`). Ưu tiên URL **LAN** nếu máy tính và điện thoại đang cùng Wi-Fi. Nếu dùng một camera RTSP khác, có thể đặt `CAMERA_RTSP_URL` hoặc cấu hình `CAMERA_IP`, `CAMERA_RTSP_PORT` và `CAMERA_RTSP_PATH`. `CAMERA_STREAM_URL` được ưu tiên nếu có giá trị.

Nếu URL không kết nối được, mở URL đó trong trình duyệt trên chính máy tính đang chạy Streamlit để kiểm tra. Điện thoại cần bật stream và máy tính phải truy cập được cùng mạng; URL WAN có thể không truy cập được do nhà mạng, firewall hoặc NAT. Không chia sẻ file `.env` vì file có thể chứa địa chỉ mạng hoặc thông tin đăng nhập. Khởi động lại Streamlit sau khi sửa `.env`.

Chạy ứng dụng:

```powershell
streamlit run app.py
```

Có thể đặt `MODEL_PATH` trong `.env` để dùng checkpoint khác. Đường dẫn tương đối được tìm từ thư mục gốc project trước, rồi mới thử các vị trí liên quan khác. Cần cho phép máy chạy ứng dụng truy cập được camera qua mạng; endpoint RTSP và đường dẫn phụ thuộc model camera.
