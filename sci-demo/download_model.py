"""Download the audited SCI checkpoint once from Camber stash."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CHECKPOINT = "stash://bunpmc/projects/yolov11sdi/checkpoints/v4_7class/stage3/sci_seed3407_best.pt"
TARGET = Path(__file__).resolve().parent / "models" / "sci_best.pt"


def main() -> None:
    if TARGET.is_file() and TARGET.stat().st_size > 0:
        print(f"Checkpoint already exists: {TARGET}")
        return
    if not os.environ.get("CAMBER_API_KEY"):
        raise SystemExit("CAMBER_API_KEY is required; export it before downloading.")
    if shutil.which("camber") is None:
        raise SystemExit("Camber CLI not found. Install it and ensure it is on PATH.")
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["camber", "stash", "cp", CHECKPOINT, str(TARGET)], check=True)
    print(f"Downloaded checkpoint to {TARGET}")


if __name__ == "__main__":
    main()
