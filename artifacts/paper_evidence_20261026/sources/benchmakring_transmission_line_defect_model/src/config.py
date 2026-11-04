from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent

def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT_DIR / path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def get_benchmark_config(config_path: str | Path = "configs/benchmark_config.yaml") -> Dict[str, Any]:
    return load_yaml(config_path)

def get_class_names() -> Dict[int, str]:
    cfg = get_benchmark_config()
    return {int(k): v for k, v in cfg["taxonomy"]["names"].items()}

