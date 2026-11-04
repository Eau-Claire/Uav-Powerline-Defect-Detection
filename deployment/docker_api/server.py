"""Docker/VM inference API compatible with the existing Modal endpoint."""

import asyncio
import hmac
import io
import logging
import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from starlette.datastructures import UploadFile

CLASSES = ("blq_jyjc", "czjyz_lw", "czjyz_ps", "czjyz_sl", "dx_dg_sg", "dx_jypps", "nw")
MODEL_PATH = Path(os.getenv("MODEL_PATH", "/models/baseline_v5_seed42_best.pt"))
DEVICE = os.getenv("DEVICE", "cpu")
IMGSZ = int(os.getenv("IMGSZ", "640"))
CONFIDENCE = float(os.getenv("CONFIDENCE", "0.10"))
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(10 * 1024 * 1024)))
MAX_BODY_BYTES = MAX_FILE_BYTES + 64 * 1024
MAX_PIXELS = int(os.getenv("MAX_PIXELS", "20000000"))
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
logger = logging.getLogger(__name__)


def decode_image(data: bytes) -> Image.Image:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as image:
                if image.format not in {"JPEG", "PNG"}:
                    raise HTTPException(415, "Only JPEG and PNG images are supported")
                if image.width * image.height > MAX_PIXELS:
                    raise HTTPException(413, f"Image exceeds {MAX_PIXELS} pixels")
                image.verify()
            with Image.open(io.BytesIO(data)) as image:
                return image.convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise HTTPException(413, "Image exceeds safe decoded size") from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise HTTPException(415, "Invalid or corrupt JPEG/PNG") from exc


class Detector:
    def __init__(self, model_path: Path = MODEL_PATH):
        from ultralytics import YOLO

        if not model_path.is_file():
            raise RuntimeError(f"Checkpoint not found: {model_path}")
        self.model = YOLO(str(model_path), task="detect")
        names = self.model.names
        actual = tuple(names[i] for i in range(len(names)))
        if actual != CLASSES:
            raise RuntimeError(f"Checkpoint class order mismatch: {actual!r}")
        self.model.to(DEVICE)
        self.lock = asyncio.Lock()

    async def predict(self, data: bytes) -> dict:
        image = decode_image(data)
        try:
            async with self.lock:
                return await asyncio.to_thread(self._predict_sync, image)
        finally:
            image.close()

    def _predict_sync(self, image: Image.Image) -> dict:
        start = time.perf_counter()
        result = self.model.predict(
            image, imgsz=IMGSZ, conf=CONFIDENCE, device=DEVICE,
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
                    "class_id": class_id,
                    "class_name": CLASSES[class_id],
                    "confidence": round(float(confidence), 4),
                    "bbox": [round(float(value), 2) for value in xyxy],
                })
        return {"detections": detections, "count": len(detections),
                "inference_ms": round(elapsed, 2)}


def create_api(predictor=None) -> FastAPI:
    state = {"predictor": predictor}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["predictor"] is None:
            detector = await asyncio.to_thread(Detector)
            state["predictor"] = detector.predict
        yield

    api = FastAPI(title="Power-line Defect Detection API", version="1.0.0", lifespan=lifespan)

    @api.get("/health")
    async def health():
        return {"status": "ok", "model": MODEL_PATH.name, "device": DEVICE}

    @api.get("/ready")
    async def ready():
        if state["predictor"] is None:
            raise HTTPException(503, "Model is not loaded")
        return {"status": "ready"}

    @api.post("/detect")
    async def detect(request: Request):
        expected = os.getenv("DETECT_API_TOKEN")
        if expected:
            scheme, _, supplied = request.headers.get("authorization", "").partition(" ")
            if scheme.lower() != "bearer" or not hmac.compare_digest(
                supplied.encode(), expected.encode()
            ):
                raise HTTPException(401, "Invalid bearer token",
                                    headers={"WWW-Authenticate": "Bearer"})
        content_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type != "multipart/form-data":
            raise HTTPException(415, "Use multipart/form-data with file=<image>")
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY_BYTES:
                raise HTTPException(413, "Upload exceeds configured size limit")
            body.extend(chunk)

        sent = False

        async def receive():
            nonlocal sent
            if sent:
                return {"type": "http.disconnect"}
            sent = True
            return {"type": "http.request", "body": bytes(body), "more_body": False}

        bounded = Request(request.scope, receive)
        async with bounded.form(max_files=1, max_fields=0) as form:
            upload = form.get("file")
            if not isinstance(upload, UploadFile):
                raise HTTPException(422, "Missing multipart file field")
            if upload.content_type not in {"image/jpeg", "image/png", "application/octet-stream"}:
                raise HTTPException(415, "Only JPEG and PNG images are supported")
            data = await upload.read(MAX_FILE_BYTES + 1)
            if len(data) > MAX_FILE_BYTES:
                raise HTTPException(413, "File exceeds configured size limit")
        image = await asyncio.to_thread(decode_image, data)
        image.close()
        try:
            return await asyncio.wait_for(state["predictor"](data), timeout=150)
        except HTTPException:
            raise
        except TimeoutError as exc:
            raise HTTPException(504, "Inference timed out; retry later") from exc
        except Exception as exc:
            logger.exception("Inference failed")
            raise HTTPException(503, "Inference unavailable; check server logs") from exc

    return api


app = create_api()
