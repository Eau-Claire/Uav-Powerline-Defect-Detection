"""Fine-tune seed85 on one V7.2 dataset variant.

Hyperparameters are identical to train_v7p1_yolo26m_clean_nodup.py so that the
dataset is the only thing that changes between V7.1 and the V7.2 variants.

Example:
  python benchmakring_transmission_line_defect_model/scripts/train_v7p2_variant.py --variant b_fo_large
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "benchmakring_transmission_line_defect_model/data"
WEIGHT = ROOT / "weights/v7p1_seed85_curated_yolo26m_best.pt"
PROJECT = ROOT / "runs/finding_v7p2"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--variant", required=True, choices=["a_base_only", "b_fo_large", "c_fo_large_rep"])
    p.add_argument("--prefix", default="finding_nemo_6cls_v7p2")
    p.add_argument("--weights", type=Path, default=WEIGHT)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="0")
    a = p.parse_args()

    data = DATA_ROOT / f"{a.prefix}_{a.variant}" / "data_local.yaml"
    name = f"yolo26m_v7p2_{a.variant}"
    for path in (data, a.weights):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (PROJECT / name).exists():
        raise FileExistsError(f"Refusing to mix with an existing run: {PROJECT / name}")

    torch.cuda.empty_cache()
    started = time.time()
    model = YOLO(str(a.weights))

    def progress(trainer) -> None:
        m = trainer.metrics or {}
        print(f"[V7.2 {a.variant}] epoch {trainer.epoch + 1}/{trainer.epochs} | "
              f"mAP50={float(m.get('metrics/mAP50(B)', float('nan'))):.5f} | "
              f"mAP50-95={float(m.get('metrics/mAP50-95(B)', float('nan'))):.5f} | "
              f"elapsed={(time.time() - started) / 60:.1f} min", flush=True)

    model.add_callback("on_fit_epoch_end", progress)
    model.train(
        data=str(data), epochs=a.epochs, imgsz=640, batch=a.batch, device=a.device,
        optimizer="AdamW", lr0=1e-4, lrf=0.01, cos_lr=True, warmup_epochs=1.0,
        momentum=0.937, weight_decay=5e-4,
        mosaic=1.0, close_mosaic=10, fliplr=0.5, flipud=0.0, translate=0.1, scale=0.5,
        degrees=0.0, shear=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, erasing=0.4,
        amp=True, seed=85, deterministic=True, workers=2, cache=False,
        project=str(PROJECT), name=name, exist_ok=False,
        save=True, save_period=5, plots=True, patience=12, val=True, verbose=True,
    )


if __name__ == "__main__":
    main()
