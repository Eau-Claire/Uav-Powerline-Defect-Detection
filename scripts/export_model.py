#!/usr/bin/env python
"""Export a trained Ultralytics checkpoint for Raspberry Pi benchmarking."""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--format", required=True, choices=["onnx", "ncnn"])
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()
    from ultralytics import YOLO
    YOLO(str(args.weights)).export(format=args.format, imgsz=args.imgsz, half=False, int8=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
