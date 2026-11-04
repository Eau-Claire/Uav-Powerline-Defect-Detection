#!/usr/bin/env python
"""Download and extract the human-reviewed Finding Nemo dataset from Camber stash."""
from __future__ import annotations

import os
import subprocess
import tarfile
from pathlib import Path
import yaml

REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "benchmakring_transmission_line_defect_model"
DATA_DIR = BENCHMARK / "data"
DEST_DIR = DATA_DIR / "finding_nemo_human_reviewed_6cls"
ARCHIVE_PATH = DATA_DIR / "finding_nemo_human_reviewed_6cls_20261101.tar.gz"
STASH_SOURCE = "stash://bunpmc/projects/finding_nemo/finding_nemo_human_reviewed_6cls_20261101.tar.gz"
CAMBER = "/home/intern/.camber/bin/camber"


def load_key() -> str:
    env_file = REPO / ".env"
    if not env_file.is_file():
        raise RuntimeError(".env file missing")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        k, sep, v = line.partition("=")
        if sep and k.strip() == "CAMBER_API_KEY":
            return v.strip().strip("'\"")
    raise RuntimeError("CAMBER_API_KEY missing from .env")


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive[0].lower()
    posix_path = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{posix_path}"


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CAMBER_API_KEY"] = load_key()
    wsl_entries = [item for item in env.get("WSLENV", "").split(":") if item]
    if "CAMBER_API_KEY" not in wsl_entries:
        wsl_entries.append("CAMBER_API_KEY")
    env["WSLENV"] = ":".join(wsl_entries)

    # 1. Download if archive doesn't exist or is empty
    if not ARCHIVE_PATH.is_file() or ARCHIVE_PATH.stat().st_size == 0:
        print(f"[DOWNLOAD] Downloading {STASH_SOURCE} -> {ARCHIVE_PATH}...", flush=True)
        cmd = ["wsl.exe", "-d", "Ubuntu", "--", CAMBER, "stash", "cp", STASH_SOURCE, wsl_path(ARCHIVE_PATH)]
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"Camber stash cp failed with return code {res.returncode}")
        print(f"[OK] Downloaded archive: {ARCHIVE_PATH.stat().st_size:,} bytes", flush=True)
    else:
        print(f"[SKIP] Archive already exists: {ARCHIVE_PATH.stat().st_size:,} bytes", flush=True)

    # 2. Extract archive
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[EXTRACT] Extracting {ARCHIVE_PATH.name} to {DEST_DIR}...", flush=True)
    
    # Use WSL tar if available for fast multi-core decompression, fallback to python tarfile
    tar_cmd = ["wsl.exe", "-d", "Ubuntu", "--", "tar", "-xzf", wsl_path(ARCHIVE_PATH), "-C", wsl_path(DEST_DIR)]
    res = subprocess.run(tar_cmd)
    if res.returncode != 0:
        print("[WARN] WSL tar failed, falling back to python tarfile...", flush=True)
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(DEST_DIR)

    print(f"[OK] Extracted to {DEST_DIR}", flush=True)

    # Check for nested directory if tar packed with root folder
    nested_dir = DEST_DIR / "finding_nemo_human_reviewed_6cls"
    if nested_dir.is_dir() and (nested_dir / "data.yaml").is_file():
        print(f"[INFO] Flattening nested directory {nested_dir}...", flush=True)
        # We can point data.yaml or keep
        target_root = nested_dir
    else:
        target_root = DEST_DIR

    # 3. Fix data.yaml paths
    data_yaml = target_root / "data.yaml"
    if not data_yaml.is_file():
        # Check subdirectories
        found = list(target_root.glob("**/data.yaml"))
        if found:
            data_yaml = found[0]
            target_root = data_yaml.parent

    if data_yaml.is_file():
        cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        cfg["path"] = target_root.resolve().as_posix()
        data_yaml.write_text(yaml.dump(cfg, sort_keys=False), encoding="utf-8")
        print(f"[OK] Configured {data_yaml} with root: {cfg['path']}", flush=True)

    print("\nDataset is ready for evaluation!")


if __name__ == "__main__":
    main()

