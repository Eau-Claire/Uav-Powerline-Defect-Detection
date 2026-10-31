# Powerline AI trên Raspberry Pi 3

Model dự kiến: **curated YOLO11n**, checkpoint đã đặt tại `models/best.pt`.
Nguồn và SHA-256 nằm trong `checkpoint_source.json`. Đã load và xác minh YOLO11n, đúng 6 class; đã export ONNX và kiểm tra CPU parity. Chưa export NCNN và chưa đo trên Pi 3.

Pi chạy NCNN CPU; không cài PyTorch và không export trên Pi. Thư mục này độc lập với `../modal` (YOLO26m).

## 1. Người bàn giao: export trên máy tính

Dùng Python 3.11 trên máy tính có kết nối Internet (export có thể tải công cụ phụ thuộc).
Chạy từ thư mục `deployment/pi3`:

```bash
python3.11 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -r requirements-export.txt
python prepare_model.py
python package_runtime.py
```

Windows PowerShell: tạo venv bằng `py -3.11 -m venv .venv-export`, kích hoạt bằng `.venv-export\Scripts\Activate.ps1`, các lệnh Python tiếp theo giống trên.

`prepare_model.py` mặc định đọc `models/best.pt`, kiểm tra đúng 6 class rồi xuất `models/pi_ncnn_model` ở input 640, FP32, batch 1. Chỉ dùng checkpoint tin cậy.
`package_runtime.py` tạo `dist/powerline-pi3.tar.gz` và file SHA-256; dừng nếu chưa có export. Gói chỉ chứa runtime và model NCNN, không chứa checkpoint PT, môi trường Python hay credential.

Gửi hai file trong `dist/` cho người deploy. Có thể export `python prepare_model.py --size 320` để thử giảm tải, rồi đóng gói lại; phải đánh giá lại độ chính xác, không dùng số benchmark 640 cho cấu hình này.

## 2. Bạn deploy: cài trên Pi

Chuẩn bị Raspberry Pi OS **64-bit**, Internet, đủ dung lượng thẻ nhớ và nguồn ổn định. Kiểm tra `uname -m` trả về `aarch64`.

```bash
sha256sum -c powerline-pi3.tar.gz.sha256
tar -xzf powerline-pi3.tar.gz
cd powerline-pi3
bash setup.sh
bash run.sh --image /home/pi/sample.jpg --warmup 1
```

Script cài NumPy/OpenCV bằng apt, tạo `.venv` và cài NCNN wheel. Không có wheel tương thích thì dừng, không tự build trên Pi. Khi lỗi, gửi lại output `uname -m`, `python3 --version` và lỗi pip để chọn wheel/build phù hợp. Phiên bản cài thực tế được lưu trong `runtime-versions.txt`.

Chạy camera USB (dừng bằng Ctrl+C):

```bash
bash run.sh --camera 0 --warmup 1
```

Đo thời gian xử lý ảnh sau warmup:

```bash
bash run.sh --image /home/pi/sample.jpg --warmup 3 --repeat 20
```

Mỗi dòng stdout là JSON chứa `detections` và `local_ms`. Mỗi detection có `class_id`, `class_name`, `confidence`, `bbox_xyxy` theo pixel của ảnh gốc. `local_ms` tính preprocessing + inference + postprocessing, không gồm đọc ảnh, gọi cloud hoặc ghi output; không phải FPS toàn pipeline. Không mở cửa sổ GUI. Camera CSI/Picamera2 chưa có adapter.

Các tùy chọn: `--conf 0.25 --iou 0.45 --threads 3`. Không cam kết realtime trên Pi 3; cần kiểm tra RAM, nhiệt độ và latency trên thiết bị thật.

## 3. Cloud tùy chọn

Mặc định chạy local độc lập. Nếu đặt `POWERLINE_CLOUD_URL`, `MODAL_KEY`, `MODAL_SECRET` trong môi trường, runtime gửi ảnh JPEG lên Modal khi local phát hiện class khác `insulator`. Đây là gọi HTTPS đồng bộ, không phải consumer RabbitMQ hoặc tích hợp BE. Những lỗi local bỏ sót sẽ không được tự động chuyển lên cloud. Không lưu credential trong gói bàn giao.

## Trạng thái kiểm chứng

Code setup, export và đóng gói đã chuẩn bị; export thực tế, đối chiếu output PT/NCNN và smoke test ảnh/camera trên Pi vẫn cần thực hiện. Không coi có `best.pt` là đã có runtime NCNN sẵn chạy.

Tham khảo [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/): tài liệu hiện tập trung Pi 4/5; không dùng benchmark đó làm kết quả Pi 3.

## 4. Cảnh báo từ Pi về BE

Luồng: inference → SQLite outbox → worker HTTPS → BE VisionBridge → NotificationPushEvent → notification frontend. Pi chỉ cần truy cập HTTPS của BE; không cần expose RabbitMQ ra Internet.

BE đã có `POST /api/v1/vision/detections/json` (kiểm tra prefix gateway thực tế). Endpoint yêu cầu Bearer token có role hợp lệ; UAV phải đăng ký và có mission `Executing`. BE tự tìm mission từ UAV.

```bash
cp alerts.example.json alerts.json
# Sửa drone_id, category_code và các rule trước khi chạy.
bash run.sh --camera 0 --alerts-config alerts.json --outbox alerts.sqlite3
```

Mở terminal thứ hai trong cùng thư mục:

```bash
export POWERLINE_BE_AUTH_MODE=jwt # Legacy endpoint; device-key is the new default
export POWERLINE_BE_ALERT_URL='https://uavpms.duckdns.org/api/v1/vision/detections/json'
export POWERLINE_BE_TOKEN_FILE='/home/pi/.config/powerline/be.token'
# Ghi JWT hợp lệ vào file trên, chmod 600; không commit token.
.venv/bin/python send_alerts.py --outbox alerts.sqlite3
```

Mẫu bật `insulator_broken`, `insulator_flashover`, `broken_strand` ở confidence 0.8, cooldown 30 giây/class. Đây là cấu hình khởi đầu cần nhóm xác nhận, không phải phân loại mức nguy hiểm đã được thẩm định. `category_code` phải đổi thành đúng `DefectCategory.CategoryCode` trong DB; không giả định tên class YOLO chính là mã DB. Đặt `IsEmergencyClass=true` cho các category cần BE phát `CriticalAlert`; cấu hình Pi chỉ quyết định gửi gì, BE quyết định độ khẩn cấp. BE hiện có fallback category ID=1 cho mã không tồn tại, vì vậy cần đối chiếu mapping trước khi bật thật.

Worker giữ bản ghi khi lỗi mạng, token hết hạn hoặc BE trả `Success=false`, thử lại sau 10 giây. Có thể cập nhật file token khi đang chạy. Cooldown giữ qua restart; một class chỉ gửi detection confidence cao nhất mỗi khoảng cooldown, có thể gộp nhiều đối tượng khác nhau. Outbox không có giới hạn dung lượng tự động; cần theo dõi ổ đĩa khi mất mạng lâu. BE hiện chưa có idempotency cho endpoint này: timeout sau khi server đã lưu có thể tạo bản ghi trùng khi retry. Không bảo đảm exactly-once hoặc thời hạn cảnh báo.

Payload hiện chỉ chứa bbox/class/confidence/thời gian, không gửi ảnh bằng endpoint JSON; GPS=0,0 được BE coi là chưa có vị trí. Không bịa GPS hoặc track ID. Chưa có refresh token, telemetry GPS, ảnh bằng chứng và kiểm thử end-to-end với BE thật.

## 5. Export ONNX trên máy tính

```bash
python -m pip install -r requirements-export.txt onnx onnxruntime
python export_onnx.py
```

Output `models/best.onnx`: batch 1, RGB NCHW 640×640, FP32, opset 17, không tích hợp NMS. Input phải letterbox và chia 255 giống `detect.py`; output YOLO11 cần decode và NMS. Script kiểm tra ONNX và so sánh tensor với PyTorch, ghi `models/best.onnx.json`. Runtime `run.sh` vẫn dùng NCNN; có ONNX không có nghĩa Pi đã chuyển sang ONNX Runtime.

Swagger triển khai đã kiểm tra: `/swagger/services/ai-inspection/v1/swagger.json` có hai POST vision. GET `/api/v1/vision/health` trả 401 khi không có token: domain truy cập được, chưa xác minh nghiệp vụ có auth. Chưa gửi detection vào môi trường thật.

ONNX đã xuất thành công: `models/best.onnx` (~10.1 MB), output `[1,10,8400]`. ONNX checker và đối chiếu PyTorch/ONNX Runtime trên tensor ngẫu nhiên seed 42 đã pass (`rtol=1e-3`, `atol=1e-3`, sai lệch tuyệt đối lớn nhất 0.00090027). Đây chưa phải đánh giá độ chính xác bằng dataset hoặc benchmark trên Pi. POST JSON rỗng tới BE thật không JWT cũng trả HTTP 401; chưa kiểm thử ghi cảnh báo có auth.

## Device key + idempotency (chuẩn bị cho BE mới)

BE phải triển khai contract `../BE_HANDOFF.md` trước. Endpoint mới `/api/v1/edge/detections` chưa deploy.

```bash
python3 generate_device_key.py --out /home/pi/.config/powerline/device.key
# Đăng ký SHA-256 được in ra vào BE, liên kết UAV.
cp device.env.example device.env
set -a
source device.env
set +a
.venv/bin/python send_alerts.py --outbox alerts.sqlite3
```

Chạy detect với alerts-config trước để tạo/migrate outbox. Mỗi bản ghi có event_id UUID được lưu bền vững, giữ qua retry/restart; sender truyền Idempotency-Key. Sender mặc định device-key, bỏ DroneId khỏi payload và BE lấy UAV từ key. Bật POWERLINE_BE_AUTH_MODE=jwt chỉ để dùng endpoint cũ. File key không được commit hoặc gửi chat. BE phải implement receipt transaction/unique constraint; chỉ gửi header chưa đủ chống trùng.
