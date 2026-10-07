# YOLO11s Baseline V5 trên Modal

API multipart JPEG/PNG cho capstone: `POST /detect`, `GET /health`.
Giao diện tiếng Việt tại `GET /`: upload ảnh, xem bounding box/class/confidence
và thời gian xử lý. Không tự gọi inference khi mở trang hoặc chọn ảnh.
Giữ cố định `imgsz=640`, `conf=0.10`, không train, TTA, tile hay đổi model.
Thứ tự class: `blq_jyjc`, `czjyz_lw`, `czjyz_ps`, `czjyz_sl`, `dx_dg_sg`,
`dx_jypps`, `nw`. Startup sẽ báo lỗi nếu checkpoint có class khác thứ tự này.

## Setup EndeavourOS / Arch Linux

Chạy tại thư mục `drone-modal` (các lệnh bên dưới dùng bash):

```bash
sudo pacman -S python python-pip python-virtualenv
sudo pacman -S --needed curl tar
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup
```

Python local chỉ cần Modal/API/client, không cài PyTorch/CUDA local. Container
Modal dùng Python 3.12, PyTorch 2.6.0+cpu, torchvision 0.21.0+cpu,
Ultralytics 8.3.199 và OpenCV headless. Torch/torchvision được cài từ
[CPU wheel index chính thức](https://docs.pytorch.org/get-started/previous-versions/),
không cài bản CUDA. Nếu checkpoint
được train với custom layer hoặc bản Ultralytics không tương thích, startup sẽ
báo lỗi; cần đối chiếu môi trường training, không tự thay checkpoint.

## 1. Lấy checkpoint, không lấy dataset

Nếu đã có đúng checkpoint, đặt tại `models/baseline_v5_seed42_best.pt`.
Nếu chưa có, nhập key trong terminal (không ghi key vào source/history):

```bash
read -rsp 'Camber API key: ' CAMBER_API_KEY; echo
export CAMBER_API_KEY
bash scripts/fetch_model.sh
unset CAMBER_API_KEY
```

Script kiểm tra `camber` trong PATH, `~/.camber/bin/camber`, rồi `.tools/camber`.
Nếu chưa có, tải CLI từ release chính thức vào `.tools/`, không sửa shell config.
Script dùng `CAMBER_API_KEY` có sẵn, hoặc credentials từ `camber login`.
Nếu cần đăng nhập sau khi script cài CLI: `.tools/camber login`.
Các stash path được thử đúng thứ tự:

1. `stash://bunpmc/projects/yolov11sdi/checkpoints/v4_7class/stage12_v5_clean/baseline_v5_seed42_best.pt`
2. `stash://bunpmc/projects/yolov11sdi/checkpoints/v4_7class/stage12/baseline_v5_seed42_best.pt`
3. `stash://bunpmc/projects/yolov11sdi/checkpoints/v4_7class/baseline_v5_seed42_best.pt`

Nguồn tải thành công được lưu trong `models/baseline_v5_seed42_best.pt.source`,
SHA256 được in ra. File tải dở không được dùng làm checkpoint. `.pt`, models,
`.env`, venv và CLI đều nằm trong `.gitignore`. Chỉ dùng checkpoint tin cậy.

## 2–3. Tạo Volume và upload

```bash
modal volume create drone-yolo11s-models
modal volume put drone-yolo11s-models models/baseline_v5_seed42_best.pt /baseline_v5_seed42_best.pt
modal volume ls drone-yolo11s-models
```

Nếu Volume đã tồn tại, bỏ qua lệnh create. Không ghi đè checkpoint khi deployment
đang chạy: container cũ giữ model trong RAM. Nếu chủ động cập nhật checkpoint,
cần redeploy để load lại. Code không tự tạo Volume và không tự download model.

## Optional API token

Public demo là mặc định khi không set token. Muốn bảo vệ `/detect`, set biến này
**trước `modal serve` hoặc `modal deploy`**:

```bash
read -rsp 'Detect API token: ' DETECT_API_TOKEN; echo
export DETECT_API_TOKEN
```

Code chuyển biến này thành Modal Secret cho CPU gateway; không nhúng vào image.
Client đọc cùng biến và gửi `Authorization: Bearer ...`. Thay token cần redeploy.
`.env` không tự được load. `/health` public và không gọi inference. Public endpoint có
thể bị người khác gọi và phát sinh phí; token phù hợp khi chia sẻ ngoài buổi demo.

## 4. Dev (có thể phát sinh phí)

```bash
modal serve app.py
```

Modal in URL dev dạng `https://<workspace>--drone-yolo11s-baseline-v5-web-dev.modal.run`.
Giữ terminal dev mở khi test, Ctrl+C để dừng. `/health` chỉ kiểm tra gateway sống,
không xác nhận checkpoint sẵn sàng. Request `/detect` hợp lệ mới gọi CPU inference.

## 5. Deploy (chỉ chạy khi sẵn sàng phát sinh phí)

```bash
modal deploy app.py
```

Dùng URL **thực tế CLI in ra**, thông thường:

```text
POST https://<workspace>--drone-yolo11s-baseline-v5-web.modal.run/detect
GET  https://<workspace>--drone-yolo11s-baseline-v5-web.modal.run/health
```

## 6. Test endpoint

```bash
export MODAL_DETECT_URL='https://<workspace>--drone-yolo11s-baseline-v5-web.modal.run/detect'
python client_test.py test.jpg
# Hoặc:
python client_test.py test.jpg --url "$MODAL_DETECT_URL"
curl "${MODAL_DETECT_URL%/detect}/health"
curl --fail-with-body -X POST "$MODAL_DETECT_URL" \
  -H "Authorization: Bearer ${DETECT_API_TOKEN:-}" \
  -F 'file=@test.jpg'
```

## Dùng giao diện web và camera iPhone

### Một lệnh chạy cả Modal dev và web local

Sau khi đã `modal setup` và upload checkpoint vào Volume (mục 1–3), chạy:

```bash
cd /home/minhchau/RiderProjects/SEP490/yolov11/drone-modal
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
```

Script chạy `modal serve app.py`, lấy URL dev tự động và truyền `MODAL_DETECT_URL`
sang web local. Chờ dòng **Web: http://127.0.0.1:8000**, mở địa chỉ đó.
Không cần hai terminal hoặc tự chép URL. Ctrl+C dừng cả hai; URL dev hết hiệu lực
sau khi dừng. Lần build image đầu có thể lâu. Nếu lỗi xác thực/Volume/model,
đọc log Modal ngay trong terminal này. CPU Modal có thể phát sinh phí khi nhận diện.
Nếu API có token, export `DETECT_API_TOKEN` trước khi chạy script; cả hai dùng chung.

**Test detect bằng ảnh:** chọn Upload ảnh → chọn JPEG/PNG → Phân tích khung hình.
Kiểm tra bounding box, class, confidence và latency. Model chỉ load khi inference
đầu tiên cần chạy; request đầu thường chậm hơn. Không thấy đối tượng có thể là
kết quả hợp lệ (`count=0`), không đồng nghĩa API lỗi.

**Test camera liên tục:** mở Camera Server trên iPhone, cùng Wi-Fi với máy tính;
chọn Camera IP → nhập URL MJPEG đầy đủ → **Bắt đầu stream + nhận diện**.
Video LIVE và bounding box AI hiển thị trong cùng một canvas. Local web chỉ mở
một kết nối MJPEG đến camera; frame mới nhất được chia sẻ cho preview và AI.
Preview tối đa 25 fps, AI chạy theo tốc độ CPU. Bbox gần nhất có ghi độ trễ;
nét đứt khi frame gốc cũ hơn 2 giây, không phải tracking chính xác theo chuyển động.
Trạng thái AI hiển thị số giây đang xử lý, kết quả (kể cả 0 đối tượng) hoặc lỗi rõ ràng.
Mặc định nghỉ 0 giây SAU mỗi kết quả rồi gửi frame mới nhất, chỉnh 0–30 giây.
AI giới hạn tối đa 5 request/giây và không gửi chồng nhau. Video và AI có FPS
đo riêng trên giao diện; 25 FPS preview không đồng nghĩa model xử lý 25 FPS.
Preview bỏ frame cũ khi nghẽn; kết quả AI được ưu tiên gửi trước frame video.
Web local tái sử dụng kết nối HTTP tới Modal thay vì tạo client mới mỗi frame.
Không gửi request nhận diện chồng nhau; tốc độ thực tế phụ thuộc CPU/cold start.
Chuyển tab trình duyệt không tự dừng stream hoặc AI. Khi quay lại, web vẽ frame
mới nhất thay vì phát lại hàng đợi cũ. Nút **Dừng stream** hoặc đóng trang mới
dừng gửi frame mới. Bấm **Kết nối lại** nếu camera/websocket bị ngắt.
Request đã gửi vẫn có thể chạy tới khi xong; local web dùng khóa chung để phiên
kết nối lại không gửi inference chồng lên request cũ. Lỗi AI giữ video chạy,
hiển thị lý do và ngừng gửi inference mới cho tới khi bạn kết nối lại.

Preview và AI dùng chung một kết nối MJPEG lâu dài trong LAN. Web nhận video nhị
phân và kết quả JSON qua một WebSocket local; không mở lại camera mỗi lần inference.
Không chuyển luồng video đầy đủ lên Modal; chỉ gửi ảnh lấy mẫu tới API CPU.

**Upload ảnh:** mở URL gốc Modal (bỏ `/detect`), chọn JPEG/PNG hoặc kéo thả,
điền API token nếu deployment có bảo vệ, rồi bấm **Phân tích khung hình**.
Ảnh HEIC cần đổi sang JPEG/PNG trước. Token chỉ giữ trong bộ nhớ trang hiện tại.

**Camera IP Camera Lite trên iPhone:** máy tính và iPhone phải cùng Wi-Fi.
Modal không truy cập trực tiếp được IP nội bộ như `192.168.x.x`; dùng web local
trên EndeavourOS để đọc một frame rồi gửi ảnh lên API CPU:

```bash
cd /home/minhchau/RiderProjects/SEP490/yolov11/drone-modal
source .venv/bin/activate
pip install -r requirements.txt
export MODAL_DETECT_URL='https://<workspace>--drone-yolo11s-baseline-v5-web.modal.run/detect'
# Nếu API có token, đặt DETECT_API_TOKEN trong terminal này như hướng dẫn trên.
python local_web.py
```

Mở **http://127.0.0.1:8000 trên máy tính**, chọn **Camera IP**, dán URL
snapshot JPEG/PNG hoặc MJPEG do app cung cấp (đầy đủ IP, port, đường dẫn).
Bật Camera Server trong app iPhone. Bấm **Lấy một khung hình**, kiểm tra ảnh,
rồi **Phân tích khung hình**. Không tự đoán đường dẫn `/video` hay `/snapshot`;
dùng đúng URL app cung cấp. Kết nối RTSP, camera có đăng nhập, hostname `.local`
và IPv6 chưa được hỗ trợ; URL hiện tại cần HTTP(S) với IPv4 nội bộ.

Local bridge chỉ nghe `127.0.0.1`, không mở ra LAN/Internet, không bật CORS.
Nó chặn request từ trang khác, redirect và IP ngoài dải LAN; không chuyển token
Modal tới camera. Không đổi host sang `0.0.0.0`. Token API được đọc từ biến môi
trường của local bridge; không cần nhập token vào giao diện local.
Camera URL không được gửi tới Modal; chỉ ảnh được gửi khi bấm phân tích hoặc bật
nhận diện liên tục. Lấy frame có timeout 12 giây; preview MJPEG mở tới khi dừng.
Không background worker trên cloud. Chưa xác nhận tương thích camera iPhone thật
nếu chưa có URL/app thực tế; các tests dùng snapshot/MJPEG giả lập.

Trang Modal hiển thị hướng dẫn mở web local ở tab Camera; chức năng camera LAN
chỉ hoạt động trên web local. Hai chế độ đều giữ nguyên model CPU và threshold.

## Response API

```json
{
  "detections": [
    {"class_id": 5, "class_name": "dx_jypps", "confidence": 0.8342,
     "bbox": [121.4, 80.2, 330.8, 281.1]}
  ],
  "count": 1,
  "inference_ms": 42.3
}
```

Đây là response minh họa, không phải benchmark. `bbox` là pixel `[x1,y1,x2,y2]`
theo ảnh gốc được encode, không tự xoay EXIF. Không có detection trả mảng rỗng.
`inference_ms` đo lời gọi predict trên CPU (preprocess, model, postprocess);
không bao gồm upload, queue, cold start, load model hay decode ảnh. Client in thêm
HTTP latency tổng; lần đầu thường lâu hơn do cold start và lazy predictor setup.

Validation chạy tại gateway trước inference: tối đa 10 MiB/file, thêm 64 KiB cho multipart,
tối đa 20 megapixel, một file, không thêm field. Kiểm tra nội dung ảnh, decode lỗi,
JPEG/PNG thực tế. MIME cho phép image/jpeg, image/png hoặc application/octet-stream.
Lỗi: 401 token; 400 multipart hỏng; 413 quá lớn; 415 sai format/ảnh hỏng;
422 thiếu `file`; 503 inference lỗi; 504 timeout. Chi tiết lỗi model chỉ ở logs.

## Chi phí và lifecycle

CPU gateway chuyển ảnh đã kiểm tra qua Modal RPC tới CPU inference. Model được load một lần
trong `@modal.enter()` mỗi container, tái sử dụng cho các request tiếp theo.
Inference dùng `cpu=4.0`, `memory=4096` (MiB), `device="cpu"`,
xử lý tuần tự, tối đa một container; gateway tối đa bốn request đồng thời.
`min_containers=0`, `buffer_containers=0`, `scaledown_window=60` cho inference.
Gateway cũng scale-to-zero sau 60 giây rảnh. Không khai báo tài nguyên GPU.
Không background worker, lịch chạy hay keepalive cloud. Video live chỉ qua LAN;
Modal nhận một frame mỗi request tuần tự, không kỳ vọng CPU inference 30 FPS.
Khi bật nhận diện liên tục, CPU có thể hoạt động liên tục và phát sinh phí;
dừng stream để deployment trở về idle rồi scale-to-zero.

CPU được cấp khi inference cần chạy, nhưng **thời gian startup và tối đa 60 giây
rảnh vẫn có thể bị tính phí**. Scale-to-zero không đồng nghĩa tổng phí bằng 0.
Giới hạn một container không phải giới hạn ngân sách; tải liên tục vẫn tốn tiền.

Deployment chỉ dùng CPU/RAM/storage. Kiểm tra lại
[giá Modal](https://modal.com/pricing) trước khi chạy.
Để dừng deployment, lấy APP_ID từ `modal app list`, rồi `modal app stop APP_ID`.

## Kiểm tra local, không gọi Modal

```bash
python -m compileall -q app.py client_test.py local_web.py camera_session.py run_demo.py test_app.py test_local_web.py test_run_demo.py
bash -n scripts/fetch_model.sh
python -c 'import app, client_test; print("Imports OK")'
python -m unittest -v test_app.py test_local_web.py test_run_demo.py
node --check static/app.js  # Nếu đã cài Node.js
node --test test_stream.cjs
```

Tests dùng predictor giả để kiểm tra HTTP/auth/validation, không chứng minh
checkpoint tương thích hay CPU inference thành công. Kiểm tra thật cần checkpoint,
Modal credentials và một request có tính phí do người dùng chạy.

Tham khảo API chính thức đã kiểm tra trước khi viết code:
[lifecycle](https://modal.com/docs/guide/lifecycle-functions),
[ASGI](https://modal.com/docs/guide/webhooks),
[scale-to-zero](https://modal.com/docs/guide/cold-start),
[Volumes](https://modal.com/docs/guide/volumes),
[Camber installation](https://docs.cambercloud.com/docs/camber-cli/installation/).
