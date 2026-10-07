"""Loopback web UI: read one LAN camera frame and send images to Modal CPU API."""

import asyncio
import ipaddress
import os
from contextlib import asynccontextmanager
from urllib.parse import urlsplit

import httpx
from fastapi import HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, Response, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app import MAX_FILE_BYTES, create_api, decode_image
from camera_session import camera_frames, run_camera_session

LAN_NETWORKS = tuple(ipaddress.ip_network(value) for value in (
    "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16",
))


def validate_camera_url(url: str) -> str:
    """Explicit LAN IPv4 literals only; no redirects, DNS, loopback or cloud metadata."""
    try:
        parts = urlsplit(url)
        address = ipaddress.ip_address(parts.hostname or "")
        if (parts.scheme not in {"http", "https"} or parts.username or parts.password
                or parts.fragment or not any(address in network for network in LAN_NETWORKS)):
            raise ValueError()
        if parts.port is not None and not 1 <= parts.port <= 65535:
            raise ValueError()
    except ValueError as exc:
        raise HTTPException(422, "Dùng URL HTTP(S) với IP nội bộ của iPhone, gồm port/path; không dùng RTSP, hostname hoặc mật khẩu trong URL.") from exc
    return url


async def read_camera_frame(url: str, *, transport=None) -> tuple[bytes, str]:
    url = validate_camera_url(url)
    async with httpx.AsyncClient(timeout=10, follow_redirects=False, trust_env=False, transport=transport) as client:
        async with client.stream("GET", url) as response:
            if response.status_code in {401, 403}:
                raise HTTPException(502, "Camera yêu cầu đăng nhập. Kết nối hiện tại chưa hỗ trợ camera có mật khẩu.")
            if response.status_code != 200:
                raise HTTPException(502, f"Camera trả HTTP {response.status_code}. Kiểm tra URL snapshot/MJPEG.")
            content_type = response.headers.get("content-type", "").lower()
            mjpeg = content_type.startswith("multipart/x-mixed-replace")
            if not (mjpeg or content_type.split(";", 1)[0] in {
                "image/jpeg", "image/png", "application/octet-stream",
            }):
                raise HTTPException(415, "URL không trả ảnh JPEG/PNG hoặc MJPEG. Dùng URL ảnh/video, không dùng trang quản lý camera.")
            buffer = bytearray()
            async for chunk in response.aiter_bytes(chunk_size=4096):
                buffer.extend(chunk)
                if mjpeg:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if end >= 0:
                        frame = bytes(buffer[start:end + 2])
                        if len(frame) > MAX_FILE_BYTES:
                            raise HTTPException(413, "Frame camera vượt quá 10 MiB.")
                        decoded = await asyncio.to_thread(decode_image, frame)
                        decoded.close()
                        return frame, "image/jpeg"
                if len(buffer) > MAX_FILE_BYTES + (65536 if mjpeg else 0):
                    raise HTTPException(413, "Frame camera vượt quá 10 MiB.")
            if mjpeg:
                raise HTTPException(502, "Chưa nhận được frame JPEG hoàn chỉnh từ camera.")
            frame = bytes(buffer)
            decoded = await asyncio.to_thread(decode_image, frame)
            decoded.close()
            return frame, "image/png" if frame.startswith(b"\x89PNG") else "image/jpeg"


def modal_detect_url() -> str:
    url = os.environ.get("MODAL_DETECT_URL", "").strip().rstrip("/")
    parts = urlsplit(url)
    if (parts.scheme != "https" or not (parts.hostname or "").endswith(".modal.run")
            or parts.username or parts.password or parts.query or parts.fragment):
        raise HTTPException(503, "Đặt MODAL_DETECT_URL thành URL HTTPS của API Modal trước khi chạy web local.")
    return url if parts.path.endswith("/detect") else url + "/detect"


async def forward_to_modal(data: bytes, *, client=None) -> dict:
    if client is None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(145, connect=15), follow_redirects=False) as owned:
            return await forward_to_modal(data, client=owned)
    url = modal_detect_url()
    token = os.environ.get("DETECT_API_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    mime = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
    try:
        response = await client.post(url, headers=headers, files={"file": ("frame", data, mime)})
        if response.status_code != 200:
            if response.status_code == 401:
                raise HTTPException(401, "Kiểm tra DETECT_API_TOKEN trong terminal chạy web local.")
            raise HTTPException(502, f"API Modal trả HTTP {response.status_code}. Kiểm tra deployment/logs.")
        return response.json()
    except httpx.TimeoutException as exc:
        raise HTTPException(504, "API Modal hết thời gian chờ.") from exc
    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(502, "Không kết nối được API Modal hoặc response không hợp lệ.") from exc


class CameraInput(BaseModel):
    url: str = Field(min_length=8, max_length=2048)


def create_local_api(predictor=None, frame_reader=None, *, stream_transport=None):
    inference_lock = asyncio.Lock()
    inflight = set()
    cloud_client = None

    @asynccontextmanager
    async def lifespan(_api):
        nonlocal cloud_client
        async with httpx.AsyncClient(timeout=httpx.Timeout(145, connect=15), follow_redirects=False) as client:
            cloud_client = client
            try:
                yield
            finally:
                cloud_client = None

    async def serialized_predict(data):
        started = False
        async def finish_request():
            nonlocal started
            async with inference_lock:
                started = True
                return await predictor(data) if predictor else await forward_to_modal(data, client=cloud_client)
        task = asyncio.create_task(finish_request())
        inflight.add(task)

        def finished(completed):
            inflight.discard(completed)
            if not completed.cancelled():
                completed.exception()  # Consume errors if the browser disconnected meanwhile.
        task.add_done_callback(finished)
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            # An already-started request finishes under the lock; reconnect cannot overlap it.
            # A queued request is cancelled by the session before it needs to be submitted.
            if not started:
                task.cancel()
            raise

    api = create_api(serialized_predict, camera_bridge=True)
    api.router.lifespan_context = lifespan
    reader = frame_reader or read_camera_frame
    api.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @api.websocket("/camera/live")
    async def camera_live(socket: WebSocket):
        expected_origin = f"{('https' if socket.url.scheme == 'wss' else 'http')}://{socket.headers.get('host', '')}"
        if socket.headers.get("origin") != expected_origin:
            await socket.close(code=1008)
            return
        await socket.accept()
        try:
            config = await asyncio.wait_for(socket.receive_json(), timeout=10)
            url = validate_camera_url(config.get("url", ""))
            delay = max(0, min(30, float(config.get("delay", 0))))
            await run_camera_session(socket, camera_frames(url, transport=stream_transport), serialized_predict, delay)
        except (HTTPException, ValueError, TypeError, AttributeError) as exc:
            await socket.send_json({"type": "camera_error", "message": str(exc.detail) if isinstance(exc, HTTPException) else "URL hoặc thời gian nghỉ không hợp lệ."})
        except (WebSocketDisconnect, TimeoutError):
            pass
        finally:
            try:
                await socket.close()
            except (RuntimeError, WebSocketDisconnect):
                pass

    @api.middleware("http")
    async def same_origin(request: Request, call_next):
        origin = request.headers.get("origin")
        if (request.method not in {"GET", "HEAD"} or request.url.path == "/camera/stream") and (
            (origin is not None and origin != str(request.base_url).rstrip("/"))
            or request.headers.get("sec-fetch-site") == "cross-site"
        ):
            return JSONResponse({"detail": "Chỉ chấp nhận thao tác từ web local."}, status_code=403)
        return await call_next(request)

    @api.get("/camera/stream")
    async def camera_stream(url: str):
        url = validate_camera_url(url)
        client = httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5), follow_redirects=False,
            trust_env=False, transport=stream_transport)
        upstream = None

        async def close():
            if upstream is not None:
                await upstream.aclose()
            await client.aclose()

        try:
            upstream = await client.send(client.build_request("GET", url), stream=True)
            content_type = upstream.headers.get("content-type", "")
            if upstream.status_code != 200 or not content_type.lower().startswith("multipart/x-mixed-replace"):
                raise HTTPException(502, "Stream cần URL MJPEG (multipart/x-mixed-replace), không phải snapshot hoặc RTSP.")
        except httpx.RequestError as exc:
            await close()
            raise HTTPException(502, "Không kết nối được MJPEG camera. Kiểm tra URL và Wi-Fi.") from exc
        except Exception:
            await close()
            raise

        async def chunks():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await close()

        return StreamingResponse(chunks(), media_type=content_type,
            headers={"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff"},
            background=BackgroundTask(close))

    @api.post("/camera/frame")
    async def camera_frame(payload: CameraInput):
        try:
            data, mime = await asyncio.wait_for(reader(payload.url), timeout=12)
            return Response(data, media_type=mime, headers={"Cache-Control": "no-store"})
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise HTTPException(504, "Camera không phản hồi. Kiểm tra cùng Wi-Fi và Camera Server đang bật.") from exc
        except httpx.RequestError as exc:
            raise HTTPException(502, "Không kết nối được camera. Kiểm tra IP, port và Wi-Fi.") from exc

    return api


if __name__ == "__main__":
    import uvicorn

    print("Web local: http://127.0.0.1:8000 — Ctrl+C để dừng.")
    # Never expose the unauthenticated LAN bridge on 0.0.0.0.
    uvicorn.run(create_local_api(), host="127.0.0.1", port=8000)
