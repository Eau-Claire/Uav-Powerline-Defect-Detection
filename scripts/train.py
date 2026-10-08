#!/usr/bin/env python
"""Generic local trainer for Ultralytics YOLO/RT-DETR models.

Use the same command for YOLO11n, YOLO11m, RT-DETR, or another Ultralytics
checkpoint; model-specific choices are supplied through CLI arguments.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import yaml


NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest",
         "foreign_object", "broken_strand"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--model", default="yolo11n.pt")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=20261008)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--project", type=Path, default=Path("runs"))
    p.add_argument("--name", default=None)
    p.add_argument("--optimizer", default="AdamW")
    p.add_argument("--lr0", type=float, default=5e-4)
    p.add_argument("--lrf", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--cache", action="store_true", help="Not recommended for 16 GB RAM")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()

    data = args.data.resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    cfg = yaml.safe_load(data.read_text(encoding="utf-8")) or {}
    names = cfg.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    if names != NAMES:
        raise ValueError(f"Expected canonical six-class taxonomy, got {names!r}")
    for split in ("train", "val"):
        if not cfg.get(split):
            raise ValueError(f"data.yaml missing {split}")

    import torch
    from ultralytics import YOLO

    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM GiB: {torch.cuda.get_device_properties(0).total_memory / 2**30:.2f}")
    print(f"PyTorch CUDA: {torch.version.cuda}")

    name = args.name or f"{Path(args.model).stem}_{args.imgsz}"
    project = args.project.resolve()
    run_dir = project / name
    model = YOLO(args.model)
    model.train(
        data=str(data), imgsz=args.imgsz, epochs=args.epochs, batch=args.batch,
        device=args.device, workers=args.workers, cache=args.cache, amp=args.amp,
        optimizer=args.optimizer, lr0=args.lr0, lrf=args.lrf,
        warmup_epochs=args.warmup_epochs, weight_decay=args.weight_decay,
        cos_lr=True, patience=args.patience, seed=args.seed, deterministic=True,
        pretrained=True, project=str(project), name=name, exist_ok=True,
        plots=True, verbose=True,
    )
    for required in (run_dir / "weights/best.pt", run_dir / "weights/last.pt", run_dir / "results.csv"):
        if not required.exists():
            raise RuntimeError(f"Expected training output missing: {required}")
    print(f"best.pt: {run_dir / 'weights/best.pt'}")
    print(f"last.pt: {run_dir / 'weights/last.pt'}")
    print(f"results.csv: {run_dir / 'results.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
