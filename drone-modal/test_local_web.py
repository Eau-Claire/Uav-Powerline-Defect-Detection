"""Local bridge tests with synthetic camera streams; no network or real inference."""
import io
import asyncio
import os
import unittest
from unittest.mock import patch

import httpx
from fastapi import HTTPException
from fastapi.testclient import TestClient
from PIL import Image

from app import create_api
from local_web import create_local_api, modal_detect_url, read_camera_frame, validate_camera_url


def encoded(format="JPEG"):
    output = io.BytesIO()
    Image.new("RGB", (32, 24), "green").save(output, format=format)
    return output.getvalue()


class CameraTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_png_and_jpeg(self):
        for format, mime in [("PNG", "image/png"), ("JPEG", "image/jpeg")]:
            frame = encoded(format)
            transport = httpx.MockTransport(lambda request: httpx.Response(200, content=frame, headers={"content-type": mime}))
            self.assertEqual(await read_camera_frame("http://192.168.1.20/shot", transport=transport), (frame, mime))

    async def test_mjpeg_first_frame_only(self):
        frame = encoded()
        body = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n--frame\r\n" + encoded()
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=body, headers={"content-type": "multipart/x-mixed-replace; boundary=frame"}))
        self.assertEqual(await read_camera_frame("http://192.168.1.20/video", transport=transport), (frame, "image/jpeg"))

    async def test_redirect_and_html_rejected(self):
        for code, mime in [(302, "image/jpeg"), (200, "text/html")]:
            transport = httpx.MockTransport(lambda request: httpx.Response(code, content=b"bad", headers={"content-type": mime, "location": "http://127.0.0.1"}))
            with self.assertRaises(HTTPException):
                await read_camera_frame("http://192.168.1.20/", transport=transport)

    async def test_camera_size_limit(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=encoded(), headers={"content-type": "image/jpeg"}))
        with patch("local_web.MAX_FILE_BYTES", 16), self.assertRaises(HTTPException) as exc:
            await read_camera_frame("http://192.168.1.20/shot", transport=transport)
        self.assertEqual(exc.exception.status_code, 413)


class BridgeTests(unittest.TestCase):
    def test_single_connection_video_and_ai(self):
        requests = []
        predicted = []
        frame = encoded()

        class LiveStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                while True:
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                    await asyncio.sleep(0.02)

        def upstream(request):
            requests.append(request)
            return httpx.Response(200, stream=LiveStream(), headers={"content-type": "multipart/x-mixed-replace; boundary=frame"})

        async def predict(data):
            predicted.append(data)
            await asyncio.sleep(0.05)
            return {"detections": [{"class_id": 5, "class_name": "dx_jypps", "confidence": .8, "bbox": [1,2,20,22]}], "count": 1, "inference_ms": 50}

        api = create_local_api(predict, stream_transport=httpx.MockTransport(upstream))
        with TestClient(api, base_url="http://127.0.0.1:8000") as client:
            with client.websocket_connect("ws://127.0.0.1:8000/camera/live", headers={"origin": "http://127.0.0.1:8000"}) as ws:
                ws.send_json({"url": "http://192.168.1.20/video", "delay": 1})
                got_frame = False
                for _ in range(20):
                    event = ws.receive()
                    if event.get("bytes"):
                        self.assertEqual(event["bytes"][8:], frame)
                        got_frame = True
                    elif event.get("text"):
                        import json
                        result = json.loads(event["text"])
                        if result["type"] == "result":
                            self.assertEqual(result["count"], 1)
                            self.assertEqual(result["width"], 32)
                            break
                else:
                    self.fail("No AI result received")
                self.assertTrue(got_frame)
                self.assertEqual(len(requests), 1)
                self.assertEqual(predicted, [frame])

    def test_live_mjpeg_proxy_and_origin_check(self):
        frame = encoded()
        body = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n--frame--\r\n"
        calls = []

        def upstream(request):
            calls.append(request)
            return httpx.Response(200, content=body, headers={"content-type": "multipart/x-mixed-replace; boundary=frame"})

        with TestClient(create_local_api(stream_transport=httpx.MockTransport(upstream)), base_url="http://127.0.0.1:8000") as client:
            response = client.get("/camera/stream", params={"url": "http://192.168.1.20/video"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.content, body)
            self.assertIn("boundary=frame", response.headers["content-type"])
            self.assertNotIn("authorization", calls[0].headers)
            response = client.get("/camera/stream", params={"url": "http://192.168.1.20/video"}, headers={"Sec-Fetch-Site": "cross-site"})
            self.assertEqual(response.status_code, 403)
            self.assertEqual(len(calls), 1)

    def test_live_requires_mjpeg(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, content=encoded(), headers={"content-type": "image/jpeg"}))
        with TestClient(create_local_api(stream_transport=transport), base_url="http://127.0.0.1:8000") as client:
            response = client.get("/camera/stream", params={"url": "http://192.168.1.20/snapshot"})
            self.assertEqual(response.status_code, 502)

    def test_allowed_and_blocked_camera_addresses(self):
        for url in ["http://192.168.1.20:8080/video", "http://10.0.0.2/photo", "https://172.16.0.1/photo"]:
            self.assertEqual(validate_camera_url(url), url)
        for url in ["http://127.0.0.1", "http://169.254.169.254", "http://example.com", "http://8.8.8.8", "rtsp://192.168.1.20", "http://user:pass@192.168.1.20", "http://[::1]", "http://192.168.1.20:99999"]:
            with self.assertRaises(HTTPException):
                validate_camera_url(url)

    def test_cloud_ui_has_no_camera_proxy(self):
        with TestClient(create_api()) as client:
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn('/static/app.js?v=', page.text)
            self.assertEqual(page.headers["cache-control"], "no-store")
            script = client.get("/static/app.js")
            self.assertEqual(script.status_code, 200)
            self.assertEqual(script.headers["cache-control"], "no-store")
            self.assertFalse(client.get("/ui-config").json()["camera_bridge"])
            self.assertEqual(client.post("/camera/frame", json={"url": "http://192.168.1.20"}).status_code, 404)

    def test_local_frame_and_detection(self):
        calls = []

        async def predict(data):
            calls.append(data)
            return {"detections": [], "count": 0, "inference_ms": 1}

        async def camera(url):
            return encoded(), "image/jpeg"

        with TestClient(create_local_api(predict, camera), base_url="http://127.0.0.1:8000") as client:
            self.assertTrue(client.get("/ui-config").json()["camera_bridge"])
            frame = client.post("/camera/frame", json={"url": "http://192.168.1.20/video"})
            self.assertEqual(frame.status_code, 200)
            self.assertEqual(calls, [])
            self.assertEqual(client.post("/detect", files={"file": ("frame.jpg", frame.content, "image/jpeg")}).status_code, 200)
            self.assertEqual(len(calls), 1)
            self.assertEqual(client.post("/camera/frame", json={"url": "http://192.168.1.20"}, headers={"Origin": "https://evil.example"}).status_code, 403)
            self.assertEqual(client.get("/", headers={"Host": "evil.example"}).status_code, 400)

    def test_modal_url_validation(self):
        with patch.dict(os.environ, {"MODAL_DETECT_URL": "https://demo.modal.run"}):
            self.assertEqual(modal_detect_url(), "https://demo.modal.run/detect")
        with patch.dict(os.environ, {"MODAL_DETECT_URL": "http://127.0.0.1"}):
            with self.assertRaises(HTTPException):
                modal_detect_url()


if __name__ == "__main__":
    unittest.main()
