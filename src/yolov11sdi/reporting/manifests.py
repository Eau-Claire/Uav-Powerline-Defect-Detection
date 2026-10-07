"""Experiment manifest + local freeze registry helpers."""
from __future__ import annotations

import shutil
from pathlib import Path

from .. import __version__
from ..io_utils import atomic_write_json, read_json


def register_freeze(freeze_path: Path, freezes_dir: Path) -> Path:
    """Copy a freeze JSON into artifacts/freezes (immutable: never overwritten if different)."""
    dst = freezes_dir / freeze_path.name
    if dst.exists():
        if dst.read_bytes() != freeze_path.read_bytes():
            raise RuntimeError(f"Refusing to overwrite immutable freeze with different content: {dst}")
        return dst
    freezes_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(freeze_path, dst)
    return dst


def experiment_manifest(cfg, paths, dataset, compose_tag: str, run_name: str, train_hash: str,
                        remote_root: str, commit: str | None) -> dict:
    t = cfg.get("training")
    return {
        "experiment": t["experiment"],
        "run_name": run_name,
        "purpose": "V6.5 data ablation (NOT the final V6.5 vs V6.4 benchmark)",
        "model": t["model"], "imgsz": t["imgsz"], "epochs": t["epochs"], "batch": t["batch"],
        "seed": t["seed"], "ultralytics_version": t["ultralytics_version"],
        "parent_dataset_tag": cfg.get("v6p2e.tag"),
        "compose_tag": compose_tag,
        "dataset_tag": dataset.tag,
        "dataset_counts": dataset.counts,
        "val_policy": "V6.2E VAL frozen and unmodified",
        "test_policy": "TEST not materialized, not evaluated",
        "direct_filter_by_review": t["direct_filter_by_review"],
        "train_hash": train_hash,
        "config_hash": cfg.config_hash(),
        "code_commit": commit,
        "code_version": __version__,
        "remote_root": remote_root,
        "dataset_manifest": paths.rel(dataset.root / "dataset_manifest.csv"),
    }


def write_once_json(path: Path, obj: dict) -> Path:
    prev = read_json(path)
    if prev is not None and prev != obj:
        backup = path.with_suffix(".prev.json")
        atomic_write_json(backup, prev)
    return atomic_write_json(path, obj)
