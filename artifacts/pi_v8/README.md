# Pi 3: n_v5 (Pi hiện tại) — 640 vuông vs 640×480 (cùng độ chính xác, ít pixel hơn 25%)

Cùng model `n_v5.pt` (MEAN_5 Pi một mình 0.761 trên val_clean_v8 không rò). Bản 640×480 cho kết quả giống hệt bản .pt
trên 40/40 ảnh drone 4:3 (đã kiểm tra), 4.05 GFLOPs thay vì ~5.4. Chỉ dùng 640×480 nếu camera drone là 4:3.

Trên Pi (Python 3):
```bash
pip install ncnn numpy opencv-python-headless
unzip pi_v8.zip && cd pi_v8
python3 pi_bench.py --model n_v5_640_ncnn_model     --imgsz 640     --seconds 300
python3 pi_bench.py --model n_v5_640x480_ncnn_model --imgsz 640x480 --seconds 300
```
Gửi lại dòng `TONG: ...` của mỗi lệnh (ms/ảnh, nhiệt độ, throttled). Chạy liền 300 s để thấy cả lúc Pi nóng và hạ xung.
Con số này quyết định: Pi 3 có kham nổi student ở độ phân giải cao hơn (800) để bắt vết lỗi nhỏ hay không.
