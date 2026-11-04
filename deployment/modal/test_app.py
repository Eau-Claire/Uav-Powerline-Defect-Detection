"""CPU-only contract tests; never contact Modal or load a checkpoint."""
import io
import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

import app


class APITests(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"DETECT_API_TOKEN": ""})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.calls = []

        async def predict(data):
            self.calls.append(data)
            return {"detections": [], "count": 0, "inference_ms": 1.25}

        self.client = TestClient(app.create_api(predict))
        self.addCleanup(self.client.close)

    def image(self, format="PNG"):
        buffer = io.BytesIO()
        Image.new("RGB", (32, 24)).save(buffer, format=format)
        return buffer.getvalue()

    def post(self, data, mime="image/png"):
        return self.client.post("/detect", files={"file": ("image", data, mime)})

    def test_health_never_calls_inference(self):
        self.assertEqual(self.client.get("/health").status_code, 200)
        self.assertEqual(self.calls, [])

    def test_jpeg_png_and_serializable_response(self):
        for format, mime in [("PNG", "image/png"), ("JPEG", "image/jpeg")]:
            response = self.post(self.image(format), mime)
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), {"detections": [], "count": 0, "inference_ms": 1.25})
        self.assertEqual(len(self.calls), 2)

    def test_auth_before_inference(self):
        os.environ["DETECT_API_TOKEN"] = "local-test-token"
        self.assertEqual(self.post(self.image()).status_code, 401)
        self.assertEqual(self.calls, [])
        response = self.client.post("/detect", headers={"Authorization": "Bearer local-test-token"},
                                    files={"file": ("x.png", self.image(), "image/png")})
        self.assertEqual(response.status_code, 200)

    def test_reject_invalid_images_before_inference(self):
        for data, mime in [(b"not image", "image/png"), (self.image("GIF"), "image/png"),
                           (self.image(), "text/plain"), (b"", "image/png")]:
            self.assertEqual(self.post(data, mime).status_code, 415)
        self.assertEqual(self.calls, [])

    def test_limits_before_inference(self):
        with patch.object(app, "MAX_FILE_BYTES", 32):
            self.assertEqual(self.post(self.image()).status_code, 413)
        with patch.object(app, "MAX_BODY_BYTES", 32):
            self.assertEqual(self.post(self.image()).status_code, 413)
        with patch.object(app, "MAX_PIXELS", 10):
            self.assertEqual(self.post(self.image()).status_code, 413)
        self.assertEqual(self.calls, [])

    def test_missing_field_and_wrong_body(self):
        response = self.client.post("/detect", files={"wrong": ("x.png", self.image(), "image/png")})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(self.client.post("/detect", json={}).status_code, 415)

    def test_malformed_multipart_and_extra_files(self):
        response = self.client.post("/detect", content=b"broken",
                                    headers={"Content-Type": "multipart/form-data"})
        self.assertEqual(response.status_code, 400)
        response = self.client.post("/detect", files=[
            ("file", ("a.png", self.image(), "image/png")),
            ("file", ("b.png", self.image(), "image/png")),
        ])
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.calls, [])

    def test_chunked_upload_limit(self):
        with patch.object(app, "MAX_BODY_BYTES", 16):
            response = self.client.post("/detect", content=iter([b"x" * 10, b"y" * 10]),
                headers={"Content-Type": "multipart/form-data; boundary=test"})
        self.assertEqual(response.status_code, 413)
        self.assertEqual(self.calls, [])

    def test_inference_errors_are_sanitized(self):
        async def broken(data):
            raise RuntimeError("private internal detail")

        with TestClient(app.create_api(broken)) as client:
            with self.assertLogs(app.logger, level="ERROR"):
                response = client.post("/detect", files={"file": ("x.png", self.image(), "image/png")})
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private internal detail", response.text)

    def test_timeout(self):
        async def timeout(data):
            raise TimeoutError()

        with TestClient(app.create_api(timeout)) as client:
            response = client.post("/detect", files={"file": ("x.png", self.image(), "image/png")})
        self.assertEqual(response.status_code, 504)


if __name__ == "__main__":
    unittest.main()
