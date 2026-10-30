"""Pull the public Roboflow GPU server image with visible layer progress."""

from __future__ import annotations

import os
from pathlib import Path

os.environ["DOCKER_CONFIG"] = str(Path(__file__).resolve().parents[2] / ".docker-anon")

import docker


IMAGE = os.getenv(
    "ROBOFLOW_SERVER_IMAGE",
    "roboflow/roboflow-inference-server-gpu:latest",
)


def main() -> None:
    client = docker.from_env()
    last_by_layer: dict[str, str] = {}
    print(f"Pulling {IMAGE}", flush=True)
    for event in client.api.pull(IMAGE, stream=True, decode=True):
        if "error" in event:
            raise RuntimeError(event["error"])
        layer = event.get("id", "image")
        status = event.get("status", "")
        progress = event.get("progress", "")
        rendered = f"{status} {progress}".strip()
        if rendered and last_by_layer.get(layer) != rendered:
            last_by_layer[layer] = rendered
            print(f"{layer}: {rendered}", flush=True)
    image = client.images.get(IMAGE)
    print(f"Ready: {IMAGE} ({image.short_id})", flush=True)


if __name__ == "__main__":
    main()
