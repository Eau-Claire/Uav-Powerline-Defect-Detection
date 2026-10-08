#!/usr/bin/env python
"""Train the local YOLO11n edge baseline without loading the dataset into RAM."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8, choices=[8, 16])
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2, choices=[2, 4])
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--project", type=Path, default=Path("runs/yolo11n_640"))
    p.add_argument("--name", default="local_edge")
    return p.parse_args()


def validate_data(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {path}")
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    expected = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest",
                "foreign_object", "broken_strand"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    if names != expected:
        raise ValueError(f"Expected canonical six-class taxonomy, got {names!r}")
    for split in ("train", "val"):
        if not cfg.get(split):
            raise ValueError(f"data.yaml is missing {split!r}")
    return cfg


def main() -> int:
    args = parse_args()
    data = args.data.resolve()
    validate_data(data)
    os.environ.setdefault("PYTHONHASHSEED", str(args.seed))
    import torch
    from ultralytics import YOLO

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM GiB: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}")
    print(f"PyTorch CUDA: {torch.version.cuda}")
    print(f"data: {data}")
    print(f"batch={args.batch}, workers={args.workers}, estimated train iterations/epoch: dataset-dependent")

    model = YOLO(args.model)
    model.train(
        data=str(data), model=args.model, imgsz=args.imgsz, epochs=args.epochs,
        batch=args.batch, device=args.device, workers=args.workers, cache=False,
        amp=True, optimizer="AdamW", lr0=5e-4, lrf=0.05, warmup_epochs=1.0,
        weight_decay=5e-4, cos_lr=True, patience=args.patience, seed=args.seed,
        deterministic=True, pretrained=True, project=str(args.project), name=args.name,
        exist_ok=True, plots=True, verbose=True,
    )
    run = args.project / args.name
    print(f"Training complete. Expected outputs: {run / 'weights/best.pt'}, {run / 'weights/last.pt'}, {run / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
