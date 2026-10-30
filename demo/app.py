from __future__ import annotations

import os
import logging
import time
from datetime import datetime
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import cv2
import streamlit as st
from dotenv import load_dotenv
from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO


APP_DIR = Path(__file__).resolve().parent
ROOT_DIR = APP_DIR.parent
DEFAULT_MODEL_PATH = (
    ROOT_DIR
    / "benchmakring_transmission_line_defect_model"
    / "runs"
    / "curated_yolo26n"
    / "weights"
    / "best.pt"
)
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]

load_dotenv(APP_DIR / ".env")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("powerline_camera")

st.set_page_config(page_title="Power-line Defect Detector", layout="wide")


def resolve_model_path() -> Path:
    configured_path = os.getenv("MODEL_PATH", "").strip()
    if not configured_path:
        return DEFAULT_MODEL_PATH

    model_path = Path(configured_path).expanduser()
    if model_path.is_absolute():
        return model_path

    candidates = (
        ROOT_DIR / model_path,
        APP_DIR / model_path,
        ROOT_DIR.parent / model_path,
    )
    return next((candidate for candidate in candidates if candidate.is_file()), candidates[0])


@st.cache_resource(show_spinner="Đang tải mô hình...")
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def get_camera_url() -> tuple[str | None, str | None]:
    configured_url = os.getenv("CAMERA_STREAM_URL", "").strip()
    if configured_url:
        if not configured_url.lower().startswith(("http://", "https://", "rtsp://")):
            return None, "`CAMERA_STREAM_URL` phải bắt đầu bằng http://, https:// hoặc rtsp://."
        return configured_url, None

    configured_url = os.getenv("CAMERA_RTSP_URL", "").strip()
    if configured_url:
        return configured_url, None

    camera_ip = os.getenv("CAMERA_IP", "").strip()
    if not camera_ip:
        return None, (
            "Hãy đặt `CAMERA_STREAM_URL` trong `.env` bằng URL MJPEG hiển thị "
            "trong IP Camera Lite (thường có dạng `http://IP:8081/video`)."
        )

    try:
        port = int(os.getenv("CAMERA_RTSP_PORT", "554"))
    except ValueError:
        return None, "`CAMERA_RTSP_PORT` phải là số nguyên."
    if not 1 <= port <= 65535:
        return None, "`CAMERA_RTSP_PORT` phải nằm trong khoảng 1-65535."

    username = os.getenv("CAMERA_USERNAME", "").strip()
    password = os.getenv("CAMERA_PASSWORD", "")
    credentials = ""
    if username:
        credentials = f"{quote(username, safe='')}:{quote(password, safe='')}@"

    stream_path = os.getenv("CAMERA_RTSP_PATH", "/stream1").strip()
    if not stream_path.startswith("/"):
        stream_path = f"/{stream_path}"

    return f"rtsp://{credentials}{camera_ip}:{port}{stream_path}", None


def add_camera_log(message: str, **details: str | int | float) -> None:
    entry: dict[str, str | int | float] = {
        "Thời gian": datetime.now().astimezone().strftime("%H:%M:%S"),
        "Sự kiện": message,
        **details,
    }
    logs = st.session_state.setdefault("camera_logs", [])
    logs.append(entry)
    del logs[:-100]
    logger.info("%s | %s", message, details)


def render_camera_logs() -> None:
    with st.expander("Log nhận diện trực tiếp", expanded=True):
        logs = st.session_state.get("camera_logs", [])
        if logs:
            st.dataframe(logs, hide_index=True, use_container_width=True)
        else:
            st.caption("Log sẽ xuất hiện tại đây khi bắt đầu stream.")


def run_detection(
    model: YOLO,
    image: Image.Image,
    confidence: float,
    image_size: int,
) -> tuple[Image.Image, list[dict[str, float | int | str]]]:
    result = model.predict(
        source=image,
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )[0]
    annotated = Image.fromarray(
        cv2.cvtColor(result.plot(conf=False), cv2.COLOR_BGR2RGB)
    )
    detections: list[dict[str, float | int | str]] = []

    if result.boxes is not None:
        boxes = result.boxes
        coordinates = boxes.xyxy.cpu().tolist()
        confidences = boxes.conf.cpu().tolist()
        class_ids = boxes.cls.cpu().tolist()
        for coordinate, score, class_id in zip(
            coordinates, confidences, class_ids
        ):
            detections.append(
                {
                    "Loại": str(result.names[int(class_id)]),
                    "Độ tin cậy": round(float(score), 4),
                    "x1": round(float(coordinate[0]), 1),
                    "y1": round(float(coordinate[1]), 1),
                    "x2": round(float(coordinate[2]), 1),
                    "y2": round(float(coordinate[3]), 1),
                }
            )

    return annotated, detections


def show_detection(
    model: YOLO,
    image: Image.Image,
    confidence: float,
    image_size: int,
) -> None:
    annotated, detections = run_detection(model, image, confidence, image_size)
    st.image(annotated, caption="Kết quả nhận diện", use_container_width=True)
    st.metric("Số đối tượng phát hiện", len(detections))
    if detections:
        st.dataframe(detections, hide_index=True, use_container_width=True)
    else:
        st.info("Không phát hiện đối tượng nào vượt ngưỡng tin cậy đã chọn.")


st.title("Nhận diện khiếm khuyết đường dây điện")
st.caption("YOLO26n — mô hình benchmark curated")

model_path = resolve_model_path()
with st.sidebar:
    st.header("Cấu hình mô hình")
    st.code(str(model_path), language=None)
    confidence = st.slider("Ngưỡng tin cậy", 0.05, 0.95, 0.25, 0.05)
    image_size = st.select_slider(
        "Kích thước ảnh",
        options=[320, 480, 640, 800, 960, 1280],
        value=640,
    )

if not model_path.is_file():
    st.error(f"Không tìm thấy file mô hình: `{model_path}`")
    st.stop()

upload_tab, camera_tab = st.tabs(["Tải ảnh", "Camera IP"])

with upload_tab:
    uploaded_files = st.file_uploader(
        "Chọn một hoặc nhiều ảnh từ máy tính",
        type=SUPPORTED_IMAGE_TYPES,
        accept_multiple_files=True,
    )
    if uploaded_files and st.button(
        "Nhận diện ảnh", type="primary", key="run_upload_detection"
    ):
        model = load_model(str(model_path))
        for uploaded_file in uploaded_files:
            st.subheader(uploaded_file.name)
            try:
                image = Image.open(BytesIO(uploaded_file.getvalue())).convert("RGB")
            except (UnidentifiedImageError, OSError) as error:
                st.error(f"{uploaded_file.name}: không đọc được ảnh ({error}).")
                continue
            show_detection(model, image, confidence, image_size)

with camera_tab:
    camera_url, camera_error = get_camera_url()
    if camera_error:
        st.warning(camera_error)
    else:
        st.info("Stream camera MJPEG/HTTP hoặc RTSP, nhận diện liên tục từng frame.")
        start_column, stop_column = st.columns(2)

        with start_column:
            start_stream = st.button(
                "Bắt đầu stream", type="primary", disabled=st.session_state.get(
                    "camera_active", False
                )
            )
        with stop_column:
            stop_stream = st.button(
                "Dừng stream", disabled=not st.session_state.get("camera_active", False)
            )

        if start_stream:
            st.session_state["camera_logs"] = []
            st.session_state["camera_frame_count"] = 0
            add_camera_log("Đang kết nối camera")
            capture = cv2.VideoCapture(
                camera_url,
                cv2.CAP_FFMPEG,
                [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,
                    5000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC,
                    5000,
                ],
            )
            if capture.isOpened():
                st.session_state["camera_capture"] = capture
                st.session_state["camera_active"] = True
                add_camera_log("Đã kết nối camera; bắt đầu nhận diện")
            else:
                capture.release()
                add_camera_log("Kết nối camera thất bại")
                st.error(
                    "Không kết nối được camera. Hãy kiểm tra URL stream, điện thoại "
                    "đang phát stream và kết nối mạng."
                )

        if stop_stream:
            capture = st.session_state.pop("camera_capture", None)
            if capture is not None:
                capture.release()
            st.session_state["camera_active"] = False
            add_camera_log("Đã dừng stream")

        @st.fragment(run_every=0.2)
        def render_camera_stream() -> None:
            if not st.session_state.get("camera_active", False):
                st.caption("Stream đang dừng.")
                render_camera_logs()
                return

            capture = st.session_state.get("camera_capture")
            if capture is None:
                st.session_state["camera_active"] = False
                add_camera_log("Lỗi: không tìm thấy kết nối camera")
                st.error("Không tìm thấy kết nối camera đang hoạt động.")
                render_camera_logs()
                return

            success, frame = capture.read()
            if not success:
                capture.release()
                st.session_state.pop("camera_capture", None)
                st.session_state["camera_active"] = False
                add_camera_log("Lỗi: mất luồng hình ảnh từ camera")
                st.error(
                    "Mất luồng hình ảnh từ camera. Hãy kiểm tra stream và kết nối mạng, "
                    "sau đó bắt đầu lại."
                )
                render_camera_logs()
                return

            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            try:
                model = load_model(str(model_path))
                started_at = time.perf_counter()
                annotated, detections = run_detection(
                    model, image, confidence, image_size
                )
                inference_fps = 1 / max(time.perf_counter() - started_at, 1e-9)
            except Exception as error:
                add_camera_log(f"Lỗi nhận diện: {error}")
                logger.exception("Camera frame inference failed")
                st.error(f"Nhận diện thất bại: {error}")
                st.session_state["camera_active"] = False
                capture.release()
                st.session_state.pop("camera_capture", None)
                render_camera_logs()
                return

            class_summary: dict[str, int] = {}
            for detection in detections:
                class_name = str(detection["Loại"])
                class_summary[class_name] = class_summary.get(class_name, 0) + 1
            add_camera_log(
                "Đã nhận diện frame",
                **{
                    "Frame": st.session_state.get("camera_frame_count", 0) + 1,
                    "Số đối tượng": len(detections),
                    "FPS xử lý": round(inference_fps, 1),
                    "Phân loại": ", ".join(
                        f"{name}: {count}" for name, count in class_summary.items()
                    )
                    or "Không phát hiện",
                },
            )
            st.session_state["camera_frame_count"] = (
                st.session_state.get("camera_frame_count", 0) + 1
            )

            st.image(annotated, caption="Stream trực tiếp — kết quả nhận diện")
            st.caption(
                f"Đối tượng hiện tại: {len(detections)} · "
                f"Tốc độ xử lý: {inference_fps:.1f} FPS"
            )
            if detections:
                st.dataframe(detections, hide_index=True, use_container_width=True)
            render_camera_logs()

        render_camera_stream()
