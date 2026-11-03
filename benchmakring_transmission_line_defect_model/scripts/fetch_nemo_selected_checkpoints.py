#!/usr/bin/env python
"""Download the six user-selected Finding Nemo checkpoints from Camber stash."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEST = REPO / "benchmakring_transmission_line_defect_model/artifacts/finding_nemo_selected_checkpoints"
CAMBER = "/home/intern/.camber/bin/camber"
FILES = {
    "yolo26n_epoch45.pt": "stash://bunpmc/projects/finding_nemo/yolo26n/weights/epoch45.pt",
    "yolo26m_epoch45.pt": "stash://bunpmc/projects/finding_nemo/yolo26m/weights/epoch45.pt",
    "yolo11n_last.pt": "stash://bunpmc/projects/finding_nemo/yolo11n/weights/last.pt",
    "yolo11m_epoch45.pt": "stash://bunpmc/projects/finding_nemo/yolo11m/weights/epoch45.pt",
    "rtdetr50_epoch15.pt": "stash://bunpmc/projects/finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/r50vd/checkpoints/epoch15.pt",
    "rtdetr101_epoch15.pt": "stash://bunpmc/projects/finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/r101vd/checkpoints/epoch15.pt",
}


def load_key() -> str:
    for line in (REPO / ".env").read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and key.strip() == "CAMBER_API_KEY":
            return value.strip().strip("'\"")
    raise RuntimeError("CAMBER_API_KEY is missing")


def wsl_path(path: Path) -> str:
    path = path.resolve()
    return f"/mnt/{path.drive[0].lower()}{path.as_posix().split(':', 1)[1]}"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CAMBER_API_KEY"] = load_key()
    entries = [item for item in env.get("WSLENV", "").split(":") if item]
    if "CAMBER_API_KEY" not in entries:
        entries.append("CAMBER_API_KEY")
    env["WSLENV"] = ":".join(entries)
    for filename, source in FILES.items():
        target = DEST / filename
        if target.is_file() and target.stat().st_size:
            print(f"[SKIP] {filename} ({target.stat().st_size:,} bytes)", flush=True)
            continue
        print(f"[GET] {source}", flush=True)
        result = subprocess.run(
            ["wsl.exe", "-d", "Ubuntu", "--", CAMBER, "stash", "cp", source, wsl_path(target)],
            env=env,
            check=False,
        )
        if result.returncode:
            raise SystemExit(result.returncode)
        print(f"[OK] {filename} ({target.stat().st_size:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
