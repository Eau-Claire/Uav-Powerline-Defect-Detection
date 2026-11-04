# YOLO11s Baseline V5 on Modal

For deployment on a persistent Docker server while preserving the same backend API contract, see [`../docker_api/README.md`](../docker_api/README.md).

This deployment exposes a multipart JPEG/PNG API for the capstone project:

- `POST /detect`
- `GET /health`
- `GET /` for the upload interface

The interface displays bounding boxes, class names, confidence, and processing time. Opening the page or selecting an image never triggers inference. The runtime is fixed to `imgsz=640` and `conf=0.10`; it does not train, use TTA or tiling, or change the model. Expected class order:

```text
blq_jyjc, czjyz_lw, czjyz_ps, czjyz_sl, dx_dg_sg, dx_jypps, nw
```

Startup fails if the checkpoint uses another class order.

## Local setup

Run commands from `deployment/modal`:

```bash
sudo pacman -S python python-pip python-virtualenv
sudo pacman -S --needed curl tar
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
modal setup
```

The local environment only needs Modal, API, and client dependencies. PyTorch and CUDA are not installed locally. The Modal container uses Python 3.12, CPU PyTorch 2.6.0, torchvision 0.21.0, Ultralytics 8.3.199, and headless OpenCV. Torch packages come from the official CPU wheel index. Startup fails when a checkpoint requires incompatible custom layers or an incompatible Ultralytics version; compare against the training environment instead of silently replacing the checkpoint.

## Fetch the checkpoint

If the verified checkpoint is already available, place it at `models/baseline_v5_seed42_best.pt`. Otherwise, enter the Camber key in the terminal; never place it in source code or shell history:

```bash
read -rsp 'Camber API key: ' CAMBER_API_KEY; echo
export CAMBER_API_KEY
bash scripts/fetch_model.sh
unset CAMBER_API_KEY
```

The script searches for `camber` in `PATH`, `~/.camber/bin/camber`, and `.tools/camber`. If missing, it downloads the official CLI to `.tools/` without changing shell configuration. It uses `CAMBER_API_KEY` or existing `camber login` credentials and checks these stash paths in order:

1. `stash://bunpmc/projects/yolov11sdi/checkpoints/v4_7class/stage12_v5_clean/baseline_v5_seed42_best.pt`
2. `stash://bunpmc/projects/yolov11sdi/checkpoints/v4_7class/stage12/baseline_v5_seed42_best.pt`
3. `stash://bunpmc/projects/yolov11sdi/checkpoints/v4_7class/baseline_v5_seed42_best.pt`

The successful source is written to `models/baseline_v5_seed42_best.pt.source`, and the script prints SHA-256. Partial downloads are never used as checkpoints. Model files, `.env`, virtual environments, and the local CLI are ignored by Git. Use only trusted checkpoints.

## Create the Modal Volume

```bash
modal volume create drone-yolo11s-models
modal volume put drone-yolo11s-models models/baseline_v5_seed42_best.pt /baseline_v5_seed42_best.pt
modal volume ls drone-yolo11s-models
```

Skip `volume create` when the volume already exists. Do not overwrite a checkpoint while deployment containers are active: existing containers retain the old model in memory. Redeploy after an intentional checkpoint update. The application never creates the volume or downloads the model automatically.

## Optional API token

Without a token, the demo is public. Set a token before `modal serve` or `modal deploy` to protect `/detect`:

```bash
read -rsp 'Detect API token: ' DETECT_API_TOKEN; echo
export DETECT_API_TOKEN
```

The application passes this value to the CPU gateway as a Modal Secret; it is not embedded in the image. Clients send `Authorization: Bearer ...`. Redeploy after changing the token. `.env` is not loaded automatically. `/health` remains public and never invokes inference. A public endpoint can be called by other users and can incur charges.

## Development and deployment

Development, which may incur charges:

```bash
modal serve app.py
```

Keep the terminal open; Ctrl+C stops the development application. `/health` verifies that the gateway is reachable but does not prove that the checkpoint is loaded. A valid `/detect` request invokes CPU inference.

Production deployment, only when ready to incur charges:

```bash
modal deploy app.py
```

Use the exact URL printed by the CLI, typically:

```text
POST https://<workspace>--drone-yolo11s-baseline-v5-web.modal.run/detect
GET  https://<workspace>--drone-yolo11s-baseline-v5-web.modal.run/health
```

Test the endpoint:

```bash
export MODAL_DETECT_URL='https://<workspace>--drone-yolo11s-baseline-v5-web.modal.run/detect'
python client_test.py test.jpg
python client_test.py test.jpg --url "$MODAL_DETECT_URL"
curl "${MODAL_DETECT_URL%/detect}/health"
curl --fail-with-body -X POST "$MODAL_DETECT_URL" \
  -H "Authorization: Bearer ${DETECT_API_TOKEN:-}" \
  -F 'file=@test.jpg'
```

## Upload interface and iPhone camera

After Modal setup and model upload, start both Modal development mode and the local web bridge with one command:

```bash
cd deployment/modal
source .venv/bin/activate
pip install -r requirements.txt
python run_demo.py
```

The script runs `modal serve app.py`, extracts its development URL, and passes `MODAL_DETECT_URL` to the local bridge. Open the displayed `http://127.0.0.1:8000` URL. Ctrl+C stops both processes and invalidates the development URL. The first image build and first inference may be slow. Export `DETECT_API_TOKEN` before running the script when token protection is enabled.

For image testing, select a JPEG or PNG and run analysis. A valid no-object result has `count=0`; it is not an API failure.

For continuous camera testing:

1. Start a camera-server application on an iPhone connected to the same Wi-Fi network.
2. Select **IP Camera**, enter the complete MJPEG URL, and start streaming and detection.
3. The local page uses one persistent MJPEG connection. It shares the latest frame between preview and inference and sends sampled images—not the full video stream—to Modal.

Preview is capped at 25 FPS, independently of inference FPS. AI requests are sequential and capped at five per second. The most recent bounding box is drawn with its latency; a dashed box means the source frame is more than two seconds old and is not motion tracking. Old preview frames are dropped under backpressure, while inference results take priority. Switching browser tabs does not stop the stream. Use **Stop stream** or close the page to stop new requests.

To run only the local bridge:

```bash
cd deployment/modal
source .venv/bin/activate
export MODAL_DETECT_URL='https://<workspace>--drone-yolo11s-baseline-v5-web.modal.run/detect'
python local_web.py
```

Open `http://127.0.0.1:8000`. Use the exact snapshot or MJPEG URL provided by the phone application; do not guess `/video` or `/snapshot`. RTSP, authenticated cameras, `.local` hostnames, and IPv6 are not supported. The current URL must use HTTP(S) and a private IPv4 address.

The bridge listens only on `127.0.0.1`, rejects cross-site requests, redirects, and non-LAN targets, and never forwards the Modal token to the camera. Do not bind it to `0.0.0.0`. The camera URL remains local; only image bytes are sent to Modal. Snapshot reads time out after 12 seconds. Camera compatibility has been tested with synthetic snapshot/MJPEG streams but still requires validation against the actual iPhone application.

## API response

```json
{
  "detections": [
    {
      "class_id": 5,
      "class_name": "dx_jypps",
      "confidence": 0.8342,
      "bbox": [121.4, 80.2, 330.8, 281.1]
    }
  ],
  "count": 1,
  "inference_ms": 42.3
}
```

This is an example response, not a benchmark. `bbox` is `[x1,y1,x2,y2]` in encoded source-image pixels; EXIF orientation is not applied automatically. No detections produce an empty array. `inference_ms` covers CPU prediction preprocessing, model execution, and postprocessing, but excludes upload, queueing, cold start, model loading, and image decoding. The client also reports total HTTP latency.

Gateway validation occurs before inference: one file, at most 10 MiB plus 64 KiB multipart overhead, at most 20 megapixels, and no extra fields. The decoder verifies actual JPEG or PNG content. Accepted MIME types are `image/jpeg`, `image/png`, and `application/octet-stream`.

| Status | Meaning |
|---:|---|
| 400 | malformed multipart request |
| 401 | invalid or missing token |
| 413 | file, body, or decoded image is too large |
| 415 | unsupported or invalid image |
| 422 | missing `file` field |
| 503 | inference failure |
| 504 | inference timeout |

Detailed model errors are logged server-side and are not returned to clients.

## Cost and lifecycle

The gateway validates the image and invokes a CPU inference function through Modal RPC. Each container loads the model once in `@modal.enter()` and reuses it. Inference uses four CPUs, 4096 MiB RAM, sequential processing, and at most one inference container; the gateway allows at most four concurrent requests. Both services scale to zero after 60 idle seconds. No GPU, scheduled task, keepalive, or cloud background video worker is configured.

Startup and up to 60 idle seconds may still be billable. Scale-to-zero does not guarantee zero cost, and a one-container limit is not a budget limit. Continuous detection can keep the CPU active. Check [Modal pricing](https://modal.com/pricing) before use. To stop a deployment, obtain its application ID with `modal app list`, then run `modal app stop APP_ID`.

## Offline verification

These checks do not contact Modal:

```bash
python -m compileall -q app.py client_test.py local_web.py camera_session.py run_demo.py test_app.py test_local_web.py test_run_demo.py
bash -n scripts/fetch_model.sh
python -c 'import app, client_test; print("Imports OK")'
python -m unittest -v test_app.py test_local_web.py test_run_demo.py
node --check static/app.js
node --test test_stream.cjs
```
