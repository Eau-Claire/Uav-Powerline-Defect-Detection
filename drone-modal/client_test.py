"""python client_test.py test.jpg --url https://...modal.run/detect"""

import argparse
import mimetypes
import os
import sys
import time
from pathlib import Path

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Test YOLO11s Baseline V5 multipart API")
    parser.add_argument("image", type=Path)
    parser.add_argument("--url", default=os.environ.get("MODAL_DETECT_URL"))
    args = parser.parse_args()
    if not args.url:
        parser.error("Set MODAL_DETECT_URL or provide --url (full /detect URL)")
    url = args.url.rstrip("/")
    if not url.endswith("/detect"):
        url += "/detect"
    token = os.environ.get("DETECT_API_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        if args.image.stat().st_size > 10 * 1024 * 1024:
            raise ValueError("Image exceeds 10 MiB")
        start = time.perf_counter()
        with args.image.open("rb") as file:
            response = httpx.post(
                url, files={"file": (args.image.name, file,
                    mimetypes.guess_type(args.image.name)[0] or "application/octet-stream")},
                headers=headers, timeout=httpx.Timeout(240, connect=20),
            )
        latency = (time.perf_counter() - start) * 1000
        response.raise_for_status()
        result = response.json()
        print(f"Detections: {result['count']}")
        for detection in result["detections"]:
            bbox = ", ".join(f"{v:.2f}" for v in detection["bbox"])
            print(f"  {detection['class_name']} (id={detection['class_id']})  "
                  f"confidence={detection['confidence']:.4f}  bbox=[{bbox}]")
        print(f"Inference: {result['inference_ms']:.2f} ms")
        print(f"HTTP latency (includes upload/queue/cold start): {latency:.2f} ms")
        return 0
    except httpx.HTTPStatusError as exc:
        print(f"HTTP {exc.response.status_code}: {exc.response.text[:1000]}", file=sys.stderr)
    except (OSError, ValueError, KeyError, TypeError, httpx.RequestError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
