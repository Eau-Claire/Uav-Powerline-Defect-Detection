from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple
import yaml

from .config import ROOT_DIR, get_benchmark_config

def setup_dataset_yaml(
    dataset_type: str,
    data_dir: Path,
    val_split_name: str = "val"
) -> Path:
    """Creates a YOLO data.yaml pointing to the specified directory."""
    cfg = get_benchmark_config()
    classes = {int(k): v for k, v in cfg["taxonomy"]["names"].items()}
    
    yaml_dict = {
        "path": str(data_dir.resolve()),
        "train": "images/train",
        "val": f"images/{val_split_name}",
        "test": "images/test",
        "names": classes
    }
    
    out_yaml = ROOT_DIR / "configs" / f"dataset_{dataset_type}.yaml"
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    with open(out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(yaml_dict, f, sort_keys=False)
        
    return out_yaml

def verify_dataset_structure(data_dir: Path, val_split_name: str = "val") -> Dict[str, Any]:
    """Verifies image and label counts across train, val, test splits."""
    stats = {}
    splits = [("train", "images/train", "labels/train"),
              ("val", f"images/{val_split_name}", f"labels/{val_split_name}"),
              ("test", "images/test", "labels/test")]
    
    for split_key, img_rel, lbl_rel in splits:
        img_p = data_dir / img_rel
        lbl_p = data_dir / lbl_rel
        
        n_images = len(list(img_p.glob("*.jpg"))) + len(list(img_p.glob("*.png"))) if img_p.exists() else 0
        n_labels = len(list(lbl_p.glob("*.txt"))) if lbl_p.exists() else 0
        
        stats[split_key] = {
            "images": n_images,
            "labels": n_labels,
            "exists": img_p.exists() and lbl_p.exists()
        }
    
    return stats

def download_camber_archive(remote_stash_path: str, local_dst_path: Path) -> bool:
    """Helper to download dataset archive from Camber Stash using WSL bridge."""
    local_dst_path = Path(local_dst_path).resolve()
    local_dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert Windows path (D:\...) to WSL path (/mnt/d/...)
    drive = local_dst_path.drive[0].lower()
    rel_path = local_dst_path.as_posix().split(":", 1)[1]
    wsl_dst = f"/mnt/{drive}{rel_path}"
    
    env_file = ROOT_DIR.parent / "Uav-Powerline-Defect-Detection" / ".env"
    if not env_file.exists():
        env_file = ROOT_DIR / ".env"
        
    cmd = (
        f"export CAMBER_API_KEY=$(grep CAMBER_API_KEY '{env_file.as_posix()}' | cut -d= -f2 | tr -d ' \\r\\n'); "
        f"~/.camber/bin/camber stash cp '{remote_stash_path}' '{wsl_dst}'"
    )
    
    proc = subprocess.run(["wsl.exe", "-d", "Ubuntu", "-e", "bash", "-c", cmd],
                          capture_output=True, text=True)
    return proc.returncode == 0 and local_dst_path.exists()

