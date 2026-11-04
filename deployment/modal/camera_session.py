"""One MJPEG connection feeds both live preview and sequential AI inference."""
import asyncio
import struct
import time

import httpx
from fastapi import HTTPException, WebSocketDisconnect

from app import MAX_FILE_BYTES, decode_image


async def camera_frames(url, *, transport=None):
    async with httpx.AsyncClient(timeout=httpx.Timeout(15, connect=5),
                                 follow_redirects=False, trust_env=False, transport=transport) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise HTTPException(502, f"Camera trả HTTP {response.status_code}. Kiểm tra URL hoặc đăng nhập camera.")
            if not response.headers.get("content-type", "").lower().startswith("multipart/x-mixed-replace"):
                raise HTTPException(415, "Cần URL MJPEG /video, không phải trang HTML, snapshot hoặc RTSP.")
            buffer = bytearray()
            async for chunk in response.aiter_bytes():
                buffer.extend(chunk)
                while True:
                    start = buffer.find(b"\xff\xd8")
                    end = buffer.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                    if end < 0:
                        break
                    if end + 2 - start > MAX_FILE_BYTES:
                        raise HTTPException(413, "Frame camera vượt quá 10 MiB.")
                    frame = bytes(buffer[start:end + 2])
                    del buffer[:end + 2]
                    yield frame
                if len(buffer) > MAX_FILE_BYTES + 65536:
                    raise HTTPException(413, "Frame MJPEG quá lớn hoặc không hoàn chỉnh.")
    raise HTTPException(502, "Camera đã ngắt stream. Bấm Kết nối lại.")


async def run_camera_session(socket, frames, predict, delay):
    latest = None
    fresh = asyncio.Event()
    outgoing = asyncio.Queue(maxsize=8)
    preview = None
    available = asyncio.Event()
    send_lock = asyncio.Lock()

    async def send(message):
        async with send_lock:
            if isinstance(message, bytes):
                await socket.send_bytes(message)
            else:
                await socket.send_json(message)

    async def reader():
        nonlocal latest, preview
        number, last_preview = 0, 0
        async for data in frames:
            number += 1
            latest = (number, data, time.time())
            fresh.set()
            now = time.monotonic()
            if now - last_preview >= 1 / 25:
                # Timestamp prefix; no base64 expansion and no camera URL sent to Modal.
                preview = struct.pack("!d", latest[2] * 1000) + data
                available.set()
                last_preview = now
        raise HTTPException(502, "Camera đã ngắt stream. Bấm Kết nối lại.")

    async def writer():
        nonlocal preview
        while True:
            await available.wait()
            if not outgoing.empty():
                await send(outgoing.get_nowait())
            elif preview is not None:
                newest, preview = preview, None
                await send(newest)
            else:
                available.clear()

    async def publish(message):
        await outgoing.put(message)
        available.set()

    async def inference():
        previous = 0
        while True:
            await fresh.wait()
            fresh.clear()
            number, data, captured_at = latest
            if number == previous:
                continue
            previous = number
            await publish({"type": "ai_status", "message": "AI đang phân tích… Lần đầu có thể cần khởi động model."})
            started = time.perf_counter()
            try:
                decoded = await asyncio.to_thread(decode_image, data)
                width, height = decoded.size
                decoded.close()
                result = await predict(data)
                await publish({"type": "result", **result, "width": width, "height": height,
                    "captured_at": captured_at * 1000, "total_ms": (time.perf_counter() - started) * 1000})
            except HTTPException as exc:
                await publish({"type": "ai_error", "message": str(exc.detail)})
                # Keep video alive, but require explicit reconnect before another billable attempt.
                await asyncio.Future()
            except Exception:
                await publish({"type": "ai_error", "message": "AI không phản hồi. Kiểm tra log Modal, rồi bấm Kết nối lại."})
                await asyncio.Future()
            # No artificial idle gap by default; retain an upper bound of 5 calls/s.
            await asyncio.sleep(max(delay, 0.2 - (time.perf_counter() - started), 0))

    async def disconnected():
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                return

    tasks = [asyncio.create_task(job()) for job in (reader, writer, inference, disconnected)]
    try:
        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            task.result()
    except (WebSocketDisconnect, OSError):
        pass
    except Exception as exc:
        message = str(exc.detail) if isinstance(exc, HTTPException) else "Mất kết nối camera. Kiểm tra Wi-Fi và bấm Kết nối lại."
        try:
            await send({"type": "camera_error", "message": message})
        except (WebSocketDisconnect, RuntimeError, OSError):
            pass
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await frames.aclose()
