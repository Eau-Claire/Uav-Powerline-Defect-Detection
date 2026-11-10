import io
import os
import unittest

import httpx
from PIL import Image

from server import create_api


async def fake_predictor(_: bytes) -> dict:
    return {"detections": [], "count": 0, "inference_ms": 1.25}


def jpeg() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 8), "white").save(output, "JPEG")
    return output.getvalue()


class ApiTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        os.environ["DETECT_API_TOKEN"] = "test-secret"
        self.app = create_api(fake_predictor)

    def tearDown(self):
        os.environ.pop("DETECT_API_TOKEN", None)

    def client(self):
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url="http://test"
        )

    async def test_health_and_ready(self):
        async with self.client() as client:
            self.assertEqual((await client.get("/health")).status_code, 200)
            self.assertEqual((await client.get("/ready")).json(), {"status": "ready"})

    async def test_detect_contract(self):
        async with self.client() as client:
            response = await client.post(
                "/detect",
                headers={"Authorization": "Bearer test-secret"},
                files={"file": ("sample.jpg", jpeg(), "image/jpeg")},
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"detections": [], "count": 0, "inference_ms": 1.25})

    async def test_rejects_missing_token(self):
        async with self.client() as client:
            response = await client.post(
                "/detect", files={"file": ("sample.jpg", jpeg(), "image/jpeg")}
            )
        self.assertEqual(response.status_code, 401)


if __name__ == "__main__":
    unittest.main()
