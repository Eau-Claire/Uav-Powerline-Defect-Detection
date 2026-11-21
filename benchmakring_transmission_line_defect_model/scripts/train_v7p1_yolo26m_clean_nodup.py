from pathlib import Path
import time

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls_v2_171126/data_local.yaml"
WEIGHT = ROOT / "weights/v7p1_seed85_curated_yolo26m_best.pt"
PROJECT = ROOT / "runs/finding_v7p1_171126"
NAME = "yolo26m_finetune_clean_nodup_20261117"
RUN_DIR = PROJECT / NAME


def main() -> None:
    if not DATA.is_file():
        raise FileNotFoundError(DATA)
    if not WEIGHT.is_file():
        raise FileNotFoundError(WEIGHT)
    if RUN_DIR.exists():
        raise FileExistsError(f"Refusing to mix with an existing run: {RUN_DIR}")

    torch.cuda.empty_cache()
    started = time.time()
    model = YOLO(str(WEIGHT))

    def progress(trainer) -> None:
        metrics = trainer.metrics or {}
        print(
            f"[V7.1 CLEAN] epoch {trainer.epoch + 1}/{trainer.epochs} | "
            f"mAP50={float(metrics.get('metrics/mAP50(B)', float('nan'))):.5f} | "
            f"mAP50-95={float(metrics.get('metrics/mAP50-95(B)', float('nan'))):.5f} | "
            f"elapsed={(time.time() - started) / 60:.1f} min",
            flush=True,
        )

    model.add_callback("on_fit_epoch_end", progress)
    model.train(
        data=str(DATA),
        epochs=30,
        imgsz=640,
        batch=8,
        device=0,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.01,
        cos_lr=True,
        warmup_epochs=1.0,
        momentum=0.937,
        weight_decay=5e-4,
        mosaic=1.0,
        close_mosaic=10,
        fliplr=0.5,
        flipud=0.0,
        translate=0.1,
        scale=0.5,
        degrees=0.0,
        shear=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        erasing=0.4,
        amp=True,
        seed=85,
        deterministic=True,
        workers=2,
        cache=False,
        project=str(PROJECT),
        name=NAME,
        exist_ok=False,
        save=True,
        save_period=5,
        plots=True,
        patience=12,
        val=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()
