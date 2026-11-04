"""YOLO11s Baseline V5: CPU HTTP gateway -> on-demand CPU inference."""

import asyncio
import hmac
import hashlib
import io
import logging
import os
import time
import warnings
from pathlib import Path

import modal
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile

CLASSES = ("blq_jyjc", "czjyz_lw", "czjyz_ps", "czjyz_sl", "dx_dg_sg", "dx_jypps", "nw")
MODEL_NAME = "baseline_v5_seed42_best.pt"
MODEL_PATH = Path("/models") / MODEL_NAME
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_BODY_BYTES = MAX_FILE_BYTES + 64 * 1024
MAX_PIXELS = 20_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
logger = logging.getLogger(__name__)
LOCAL_STATIC = Path(__file__).parent / "static"
STATIC_DIR = LOCAL_STATIC if LOCAL_STATIC.is_dir() else Path("/web")


class FreshStaticFiles(StaticFiles):
    """Avoid serving old control logic after a local restart or Modal redeploy."""
    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-store"
        return response

app = modal.App("drone-yolo11s-baseline-v5")
volume = modal.Volume.from_name("drone-yolo11s-models")
cpu_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi==0.141.1", "python-multipart==0.0.32", "pillow==12.3.0"
).add_local_dir(LOCAL_STATIC, remote_path="/web", copy=True)
inference_image = cpu_image.pip_install(
    "torch==2.6.0+cpu", "torchvision==0.21.0+cpu",
    index_url="https://download.pytorch.org/whl/cpu",
).pip_install("ultralytics==8.3.199").run_commands(
    "python -m pip uninstall -y opencv-python opencv-python-headless",
    "python -m pip install opencv-python-headless==4.12.0.88",
)

# Pass the optional local token through a Modal Secret, never an image env layer.
api_secrets = (
    [modal.Secret.from_dict({"DETECT_API_TOKEN": os.environ["DETECT_API_TOKEN"]})]
    if os.environ.get("DETECT_API_TOKEN") else []
)


def decode_image(data: bytes) -> Image.Image:
    """Verify actual format and decoded size, not just the supplied MIME type."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise HTTPException(415, "Only JPEG and PNG images are supported")
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(413, "Image exceeds 20 megapixels")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                # Preserve encoded pixel coordinates; no EXIF rotation of bbox space.
                return image.convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(413, "Image exceeds safe decoded size") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(415, "Invalid or corrupt JPEG/PNG") from exc


@app.cls(
    image=inference_image,
    cpu=4.0,
    memory=4096,
    volumes={"/models": volume},
    scaledown_window=60,
    min_containers=0, max_containers=1, buffer_containers=0,
    timeout=120,
)
class Detector:
    @modal.enter()
    def load_model(self):
        from ultralytics import YOLO

        if not MODEL_PATH.is_file():
            raise RuntimeError(f"Upload the Baseline V5 checkpoint to {MODEL_PATH}")
        self.model = YOLO(str(MODEL_PATH), task="detect")
        names = self.model.names
        actual = tuple(names[i] for i in range(len(names)))
        if actual != CLASSES:
            raise RuntimeError(f"Checkpoint class order mismatch: {actual!r}")
        self.model.to("cpu")

    @modal.method()
    def predict(self, data: bytes) -> dict:
        image = decode_image(data)
        start = time.perf_counter()
        result = self.model.predict(
            image, imgsz=640, conf=0.10, device="cpu",
            augment=False, half=False, save=False, verbose=False,
        )[0]
        elapsed = (time.perf_counter() - start) * 1000
        detections = []
        if result.boxes is not None:
            for xyxy, confidence, class_id in zip(
                result.boxes.xyxy.cpu().tolist(),
                result.boxes.conf.cpu().tolist(),
                result.boxes.cls.cpu().tolist(),
            ):
                class_id = int(class_id)
                detections.append({
                    "class_id": class_id, "class_name": CLASSES[class_id],
                    "confidence": round(float(confidence), 4),
                    "bbox": [round(float(v), 2) for v in xyxy],
                })
        return {"detections": detections, "count": len(detections),
                "inference_ms": round(elapsed, 2)}


async def remote_predict(data: bytes) -> dict:
    return await Detector().predict.remote.aio(data)


def create_api(predictor=None, *, camera_bridge=False) -> FastAPI:
    """Inject a predictor for CPU-only validation tests, without invoking Modal."""
    predictor = predictor or remote_predict
    api = FastAPI(title="YOLO11s Baseline V5", version="1.0.0")
    api.mount("/static", FreshStaticFiles(directory=STATIC_DIR), name="static")

    @api.get("/", include_in_schema=False)
    async def home():
        html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
        # A changed URL also invalidates files cached BEFORE no-store was added.
        for name in ("app.js", "style.css", "stream.css"):
            digest = hashlib.sha256((STATIC_DIR / name).read_bytes()).hexdigest()[:12]
            html = html.replace(f'/static/{name}"', f'/static/{name}?v={digest}"')
        return HTMLResponse(html, headers={"Cache-Control": "no-store"})

    @api.get("/ui-config", include_in_schema=False)
    async def ui_config():
        return {"camera_bridge": camera_bridge, "token_managed": camera_bridge}

    @api.get("/health")
    async def health():
        return {"status": "ok", "model": "YOLO11s Baseline V5"}

    @api.post("/detect")
    async def detect(request: Request):
        # The loopback-only bridge supplies the cloud token server-side.
        expected = os.environ.get("DETECT_API_TOKEN") if not camera_bridge else None
        if expected:
            scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                supplied.encode(), expected.encode()
            ):
                raise HTTPException(401, "Invalid bearer token", headers={"WWW-Authenticate": "Bearer"})
        if request.headers.get("content-type", "").split(";", 1)[0].strip().lower() != "multipart/form-data":
            raise HTTPException(415, "Use multipart/form-data with file=<image>")
        # Bound the complete body BEFORE multipart parsing, including chunked uploads.
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                raise HTTPException(413, "Upload exceeds 10 MiB plus multipart overhead")
            body.extend(chunk)

        async def receive():
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        bounded_request = Request(request.scope, receive)
        async with bounded_request.form(max_files=1, max_fields=0) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(422, "Missing multipart file field")
            if upload.content_type not in {"image/jpeg", "image/png", "application/octet-stream"}:
                raise HTTPException(415, "Only JPEG and PNG images are supported")
            data = await upload.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, "File exceeds 10 MiB")
        decoded = await asyncio.to_thread(decode_image, data)
        decoded.close()
        try:
            return await asyncio.wait_for(predictor(data), timeout=150)
        except HTTPException:
            raise
        except (TimeoutError, modal.exception.FunctionTimeoutError) as exc:
            raise HTTPException(504, "Inference timed out; retry later") from exc
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(503, "Inference unavailable; check deployment logs") from exc

    return api


@app.function(
    image=cpu_image, secrets=api_secrets, min_containers=0,
    max_containers=1, scaledown_window=60, timeout=180,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    return create_api()
