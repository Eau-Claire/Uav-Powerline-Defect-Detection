# Docker inference API

This package runs the same seven-class checkpoint and preserves the Modal API contract for a backend service:

- `POST /detect` — multipart field `file`, JPEG or PNG
- `GET /health` — process health and runtime configuration
- `GET /ready` — model readiness, used by Docker health checks
- `GET /docs` — OpenAPI/Swagger documentation

The response fields remain `detections`, `count`, and `inference_ms`; each detection contains `class_id`, `class_name`, `confidence`, and an `xyxy` source-pixel `bbox`.

## Server deployment

Install Docker Engine with the Compose plugin, clone the repository, and make sure this verified checkpoint exists:

```text
deployment/modal/models/baseline_v5_seed42_best.pt
```

Create the server-only environment file. Never commit it:

```bash
cd deployment/docker_api
cp .env.example .env
openssl rand -hex 32
# Paste the generated value into DETECT_API_TOKEN in .env.
docker compose build
docker compose up -d
docker compose ps
curl http://127.0.0.1:8000/ready
```

The default binding is loopback-only. Put Nginx or Caddy with HTTPS in front of it on an Internet-facing server. To expose the port directly on a trusted private network, set `API_BIND_ADDRESS=0.0.0.0`; do not expose unencrypted HTTP publicly.

Upgrade without embedding the model or token in the image:

```bash
git pull --ff-only
docker compose build --pull
docker compose up -d
```

View logs or stop the service:

```bash
docker compose logs -f --tail=200
docker compose down
```

## Backend request

```bash
curl --fail-with-body http://127.0.0.1:8000/detect \
  -H "Authorization: Bearer $DETECT_API_TOKEN" \
  -F 'file=@sample.jpg'
```

Example response:

```json
{
  "detections": [
    {
      "class_id": 5,
      "class_name": "dx_jypps",
      "confidence": 0.8342,
      "bbox": [121.4, 80.2, 330.8, 281.1]
    }
  ],
  "count": 1,
  "inference_ms": 42.3
}
```

Use one Uvicorn worker: each worker loads another model copy. The API serializes model access inside the process because the Ultralytics model object is not treated as thread-safe. Horizontal scaling should use multiple containers behind a reverse proxy.

## Verification

From this directory, with the Modal development environment active:

```bash
python -m unittest -v test_server.py
docker compose config
docker compose build
```

The supplied image is CPU-only and matches the Modal runtime. A CUDA server requires a separate NVIDIA base image, NVIDIA Container Toolkit, and `DEVICE=0`; do not switch only the environment variable on this CPU image.
