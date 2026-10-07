# SCI demo

CPU-only Gradio inference demo for the audited custom YOLO11s + SCI checkpoint. Camber is used only to download the artifact; inference runs locally from `models/sci_best.pt`.

## Deploy on Ubuntu

```bash
git clone ...
cd sci-demo

curl -sL https://cli.cambercloud.com/install-v2.sh | bash
export PATH="$HOME/.camber/bin:$PATH"

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

export CAMBER_API_KEY="..."
python download_model.py
python app.py
```

Open `http://SERVER_IP:7860`. The downloader is idempotent and skips an existing non-empty checkpoint. Never hardcode the Camber token.

The import alias `stage3_sci_defs.py` is intentionally present before `YOLO(...)` loads the checkpoint, because pickle may refer to that historical module path. This demo does not export ONNX/NCNN and does not run the official test dataset.
