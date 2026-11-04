"""Build a Pi handoff archive after desktop export; never include PT or secrets."""
import hashlib
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def main():
    model = ROOT / "models/pi_ncnn_model"
    files = [ROOT / name for name in ("README.md", "detect.py", "setup.sh", "run.sh", "requirements.txt", "alerts.py", "send_alerts.py", "alerts.example.json", "device.env.example", "generate_device_key.py")]
    files += [model / name for name in ("deployment.json", "model.ncnn.param", "model.ncnn.bin")]
    for path in files:
        if not path.is_file() or not path.stat().st_size:
            raise SystemExit(f"Missing {path}. Run prepare_model.py on desktop first.")
    cfg = json.loads((model / "deployment.json").read_text())
    if cfg["family"] not in ("yolo11n", "yolo26n"):
        raise SystemExit("Unsupported model family")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    target = dist / "powerline-pi3.tar.gz"
    with tarfile.open(target, "w:gz") as archive:
        for path in files:
            archive.add(path, arcname=str(Path("powerline-pi3") / path.relative_to(ROOT)))
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    target.with_suffix(target.suffix + ".sha256").write_text(f"{digest}  {target.name}\n")
    print(target)

if __name__ == "__main__":
    main()
