# Power-Line AI on Raspberry Pi 3

Target model: **curated YOLO11n**, stored locally as `models/best.pt`. Source provenance and SHA-256 are recorded in `checkpoint_source.json`. The checkpoint has been loaded and verified as a six-class YOLO11n model; ONNX export and CPU parity checks passed. NCNN export and physical Pi 3 benchmarks are still pending.

The Pi runtime uses NCNN on CPU. Do not install PyTorch or export models on the Pi. This directory is independent from the YOLO26m service in `../modal`.

## 1. Export and package on a workstation

Use Python 3.11 on an Internet-connected workstation. Run from `deployment/pi3`:

```bash
python3.11 -m venv .venv-export
source .venv-export/bin/activate
python -m pip install -r requirements-export.txt
python prepare_model.py
python package_runtime.py
```

On Windows PowerShell, create the environment with `py -3.11 -m venv .venv-export`, activate it with `.venv-export\Scripts\Activate.ps1`, and run the same Python commands.

`prepare_model.py` reads `models/best.pt`, verifies the six-class contract, and exports `models/pi_ncnn_model` at 640 px, FP32, batch 1. Use only a trusted checkpoint. `package_runtime.py` creates `dist/powerline-pi3.tar.gz` and its SHA-256 file; it stops when an export is missing. The package contains only the runtime and NCNN model—not the PT checkpoint, Python environment, or credentials.

Send both files from `dist/` to the deployer. A 320 px experiment can be exported with `python prepare_model.py --size 320`, but its accuracy must be evaluated independently; do not reuse 640 px benchmark values.

## 2. Install and run on the Pi

Use 64-bit Raspberry Pi OS, a stable power supply, sufficient storage, and Internet access. Confirm that `uname -m` returns `aarch64`.

```bash
sha256sum -c powerline-pi3.tar.gz.sha256
tar -xzf powerline-pi3.tar.gz
cd powerline-pi3
bash setup.sh
bash run.sh --image /home/pi/sample.jpg --warmup 1
```

`setup.sh` installs NumPy and OpenCV through APT, creates `.venv`, and installs the NCNN wheel. It stops when no compatible wheel exists; it does not attempt a local NCNN build. On failure, report `uname -m`, `python3 --version`, and the pip error so an appropriate wheel or build can be selected. Installed versions are recorded in `runtime-versions.txt`.

USB camera mode:

```bash
bash run.sh --camera 0 --warmup 1
```

Post-warmup image timing:

```bash
bash run.sh --image /home/pi/sample.jpg --warmup 3 --repeat 20
```

Each stdout line is JSON containing `detections` and `local_ms`. A detection includes `class_id`, `class_name`, `confidence`, and `bbox_xyxy` in source-image pixels. `local_ms` covers preprocessing, inference, and postprocessing; it excludes image reads, cloud calls, and output writes and is not end-to-end FPS. No GUI window is opened. CSI/Picamera2 cameras currently require an adapter.

Useful options include `--conf 0.25 --iou 0.45 --threads 3`. Real-time performance on a Pi 3 is not guaranteed; measure latency, RAM, temperature, and throttling on the physical device.

## 3. Optional cloud fallback

Local-only operation is the default. When `POWERLINE_CLOUD_URL`, `MODAL_KEY`, and `MODAL_SECRET` are present, the runtime sends JPEG frames to Modal when local inference detects a class other than `insulator`. This is a synchronous HTTPS call, not a RabbitMQ consumer or backend integration. Local false negatives are not automatically escalated. Never include credentials in the deployment package.

## Verification status

Setup, export, and packaging code is present. Physical NCNN export parity, image/camera smoke tests on the Pi, and device benchmarks remain required. The presence of `best.pt` does not mean an NCNN runtime is ready.

The [Ultralytics Raspberry Pi guide](https://docs.ultralytics.com/guides/raspberry-pi/) focuses primarily on Pi 4 and Pi 5. Do not report its benchmarks as Pi 3 results.

## 4. Send alerts from the Pi to the backend

Flow:

```text
inference → SQLite outbox → HTTPS worker → BE VisionBridge
          → NotificationPushEvent → frontend notification
```

The Pi only needs HTTPS access to the backend; RabbitMQ must not be exposed to the Internet.

The legacy backend endpoint is `POST /api/v1/vision/detections/json` subject to the deployed gateway prefix. It requires a valid bearer token and UAV role. The UAV must be registered and have an `Executing` mission; the backend resolves the mission from the UAV.

```bash
cp alerts.example.json alerts.json
# Review drone_id, category_code, and every rule before use.
bash run.sh --camera 0 --alerts-config alerts.json --outbox alerts.sqlite3
```

Start the sender in another terminal:

```bash
export POWERLINE_BE_AUTH_MODE=jwt  # legacy endpoint only; device-key is preferred
export POWERLINE_BE_ALERT_URL='https://uavpms.duckdns.org/api/v1/vision/detections/json'
export POWERLINE_BE_TOKEN_FILE='/run/secrets/powerline-be.jwt'
# Write a valid JWT to that file and chmod 600. Never commit the token.
.venv/bin/python send_alerts.py --outbox alerts.sqlite3
```

The example enables `insulator_broken`, `insulator_flashover`, and `broken_strand` at confidence 0.8 with a 30-second per-class cooldown. This is an initial configuration requiring team approval, not a validated hazard classification. `category_code` must match `DefectCategory.CategoryCode` in the database; a YOLO class name is not automatically a database code. Set `IsEmergencyClass=true` in the backend category when a defect should generate `CriticalAlert`. The Pi decides what to send; the backend decides severity. The legacy backend falls back to category ID 1 for unknown codes, so validate mappings before production use.

The worker retains records after network failures, expired tokens, or backend responses with `Success=false`, and retries after 10 seconds. The token file can be replaced while the worker is running. Cooldowns survive restarts. Only the highest-confidence detection for a class is sent during a cooldown window, which can collapse distinct objects. The outbox has no automatic size cap and must be monitored during long outages.

The legacy endpoint has no idempotency guarantee: a timeout after the server commits can create duplicates on retry. The JSON payload contains bounding box, class, confidence, and time, but no image. Coordinates `0,0` are treated as unknown GPS; never invent GPS or track IDs. Refresh tokens, GPS telemetry, evidence images, and authenticated end-to-end production tests are still pending.

## 5. ONNX export on a workstation

```bash
python -m pip install -r requirements-export.txt onnx onnxruntime
python export_onnx.py
```

The output `models/best.onnx` uses batch 1, RGB NCHW 640×640, FP32, opset 17, and does not include NMS. Input must be letterboxed and divided by 255 exactly as in `detect.py`; YOLO11 output requires decoding and NMS. The exporter checks the ONNX model, compares it against PyTorch, and writes `models/best.onnx.json`. `run.sh` still uses NCNN; an ONNX file does not switch the Pi runtime to ONNX Runtime.

The current export is approximately 10.1 MB with output shape `[1,10,8400]`. ONNX checker and PyTorch/ONNX Runtime comparison passed on a seed-42 random tensor with `rtol=1e-3`, `atol=1e-3`, and maximum absolute difference `0.00090027`. This is not a dataset accuracy evaluation or Pi benchmark. An unauthenticated empty request to the live backend returned HTTP 401, confirming reachability but not authenticated alert creation.

## Device key and idempotency

The backend must first implement the contract in `../BE_HANDOFF.md`; the new `/api/v1/edge/detections` endpoint is not yet deployed.

```bash
python generate_device_key.py --out device.env
# Register the printed SHA-256 with the backend and associate it with the UAV.
set -a
source device.env
set +a
.venv/bin/python send_alerts.py --outbox alerts.sqlite3
```

Run detection once with `--alerts-config` to create or migrate the outbox. Every record has a durable UUID `event_id` retained across retries and restarts; the sender transmits it as `Idempotency-Key`. Device-key mode is the default, omits `DroneId` from the payload, and lets the backend resolve the UAV from the key. Set `POWERLINE_BE_AUTH_MODE=jwt` only for the legacy endpoint.

Never commit or share the key file in chat. The backend must implement a transactional receipt table or unique constraint; sending the header alone does not prevent duplicates.
