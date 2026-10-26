from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_ROOT = SCRIPT_DIR.parent
PROJECT_ROOT = BENCHMARK_ROOT.parent / "Uav-Powerline-Defect-Detection"

REMOTE_RAW_TAR = "stash://bunpmc/projects/yolov11sdi/datasets/v6p5_12_transfer/20261010/v6p5_roboflow_ablation.tar"
RAW_TAR_LOCAL = BENCHMARK_ROOT / "data" / "v6p5_roboflow_ablation.tar"
RAW_DIR = BENCHMARK_ROOT / "data" / "raw_combined"
CURATED_DIR = BENCHMARK_ROOT / "data" / "curated_combined"

def log(msg: str):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)

def get_camber_api_key() -> str:
    for p in [PROJECT_ROOT / ".env", BENCHMARK_ROOT / ".env", BENCHMARK_ROOT.parent / ".env"]:
        if p.exists():
            for line in p.read_text().splitlines():
                if line.strip().startswith("CAMBER_API_KEY="):
                    return line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    return os.environ.get("CAMBER_API_KEY", "")

def download_raw_tar():
    if RAW_DIR.exists() and (RAW_DIR / "data.yaml").exists():
        log(f"Raw dataset already extracted at: {RAW_DIR}")
        return True

    RAW_TAR_LOCAL.parent.mkdir(parents=True, exist_ok=True)
    if not RAW_TAR_LOCAL.exists() or RAW_TAR_LOCAL.stat().st_size < 1024**2:
        log(f"Starting download of {REMOTE_RAW_TAR} via Camber CLI...")
        drive = RAW_TAR_LOCAL.drive[0].lower()
        rel_path = RAW_TAR_LOCAL.as_posix().split(":", 1)[1]
        wsl_dst = f"/mnt/{drive}{rel_path}"
        key = get_camber_api_key()
        if not key:
            log("ERROR: CAMBER_API_KEY not found in .env files.")
            return False
        
        cmd = f"export CAMBER_API_KEY='{key}'; ~/.camber/bin/camber stash cp '{REMOTE_RAW_TAR}' '{wsl_dst}'"
        proc = subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "bash", "-c", cmd],
                              capture_output=False)
        if proc.returncode != 0 or not RAW_TAR_LOCAL.exists():
            log(f"ERROR: Failed to download raw archive from Camber (code {proc.returncode})")
            return False
        log(f"Download completed! File size: {RAW_TAR_LOCAL.stat().st_size / (1024**3):.2f} GB")

    log(f"Extracting {RAW_TAR_LOCAL.name} into {RAW_DIR}...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    # Use tar in WSL or python tarfile for maximum speed
    drive = RAW_DIR.drive[0].lower()
    rel_path = RAW_DIR.as_posix().split(":", 1)[1]
    wsl_raw_dir = f"/mnt/{drive}{rel_path}"
    
    tar_drive = RAW_TAR_LOCAL.drive[0].lower()
    tar_rel = RAW_TAR_LOCAL.as_posix().split(":", 1)[1]
    wsl_tar = f"/mnt/{tar_drive}{tar_rel}"
    
    cmd_extract = f"tar -xf '{wsl_tar}' -C '{wsl_raw_dir}' --strip-components=1"
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "bash", "-c", cmd_extract])
    if proc.returncode != 0:
        # Fallback if strip-components fails
        cmd_extract2 = f"tar -xf '{wsl_tar}' -C '{wsl_raw_dir}'"
        subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "bash", "-c", cmd_extract2], check=True)
        
    log("Extraction completed!")
    # Normalize data.yaml inside raw_dir
    data_yaml = RAW_DIR / "data.yaml"
    if not data_yaml.exists():
        sub_yamls = list(RAW_DIR.glob("**/data.yaml"))
        if sub_yamls:
            data_yaml = sub_yamls[0]
            
    if data_yaml.exists():
        cfg = yaml.safe_load(data_yaml.read_text())
        cfg["path"] = str(RAW_DIR.resolve()).replace("\\", "/")
        (BENCHMARK_ROOT / "configs" / "dataset_raw.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        log(f"Updated configs/dataset_raw.yaml with path: {cfg['path']}")

    return True

def build_curated_dataset():
    if CURATED_DIR.exists() and (CURATED_DIR / "data.yaml").exists():
        log(f"Curated dataset already exists at: {CURATED_DIR}")
        return True

    log("Building Curated Combined dataset via 8-rule curation pipeline...")
    build_script = PROJECT_ROOT / "scripts" / "label_audit_build.py"
    raw_yaml = BENCHMARK_ROOT / "configs" / "dataset_raw.yaml"
    audit_dir = PROJECT_ROOT / "data" / "label_audit" / "yolo11n_adamw"
    person_dir = PROJECT_ROOT / "data" / "label_audit" / "coco_person"
    review_csv = PROJECT_ROOT / "data" / "label_audit" / "review_decisions.csv"
    tighten_csv = PROJECT_ROOT / "data" / "label_audit" / "sam_tighten" / "sam_boxes.csv"

    cmd = [
        sys.executable,
        str(build_script),
        "--src", str(raw_yaml),
        "--audit", str(audit_dir),
        "--person", str(person_dir),
        "--review", str(review_csv),
        "--tighten", str(tighten_csv),
        "--keep-insulator",
        "--out", str(CURATED_DIR)
    ]
    
    log(f"Running: {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=False)
    if proc.returncode != 0:
        log(f"ERROR: label_audit_build.py failed with return code {proc.returncode}")
        return False

    # Create dataset_curated.yaml
    curated_yaml = BENCHMARK_ROOT / "configs" / "dataset_curated.yaml"
    curated_cfg = {
        "path": str(CURATED_DIR.resolve()).replace("\\", "/"),
        "train": "images/train",
        "val": "images/val_clean",
        "test": "images/test",
        "names": {
            0: "insulator",
            1: "insulator_broken",
            2: "insulator_flashover",
            3: "bird_nest",
            4: "foreign_object",
            5: "broken_strand"
        }
    }
    curated_yaml.write_text(yaml.safe_dump(curated_cfg, sort_keys=False))
    log(f"Created configs/dataset_curated.yaml: {curated_yaml}")
    return True

def main():
    log("=== DATASET PREPARATION PIPELINE (RAW & CURATED) ===")
    if not download_raw_tar():
        sys.exit(1)
    if not build_curated_dataset():
        sys.exit(1)
    log("=== BOTH DATASETS READY FOR BENCHMARK TRAINING ===")

if __name__ == "__main__":
    main()
