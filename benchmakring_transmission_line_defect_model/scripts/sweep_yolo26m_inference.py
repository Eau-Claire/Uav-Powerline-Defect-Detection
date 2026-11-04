#!/usr/bin/env python
"""Select YOLO26m inference resolution/TTA on Val, then evaluate the winner on raw Test."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/finding_nemo_selected_checkpoints/yolo26m_epoch45.pt"
VAL_DATA = ROOT / "data/finding_nemo_human_reviewed_6cls/data.yaml"
TEST_DATA = ROOT / "data/test_main_v1/dataset.yaml"
OUT = ROOT / "results/finding_nemo/yolo26m_inference_sweep"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]


def evaluate(model: YOLO, data: Path, split: str, imgsz: int, augment: bool, device: str, batch: int):
    result = model.val(data=str(data.resolve()), split=split, imgsz=imgsz, augment=augment,
                       device=device, batch=batch, conf=0.001, iou=0.7,
                       plots=False, verbose=False, save_json=False)
    row = {
        "split": split,
        "imgsz": imgsz,
        "tta": augment,
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
    }
    for index, name in enumerate(NAMES):
        row[f"{name}_AP50"] = float(result.box.ap50[index])
        row[f"{name}_AP50_95"] = float(result.box.ap[index])
    row["MEAN_5"] = sum(row[f"{name}_AP50"] for name in NAMES[1:]) / 5
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--sizes", type=int, nargs="+", default=[640, 800, 960])
    parser.add_argument("--include-tta", action="store_true")
    parser.add_argument("--val-only", action="store_true")
    args = parser.parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Activate the CUDA environment or pass --device cpu.")
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    rows = []
    modes = [False, True] if args.include_tta else [False]
    for size in args.sizes:
        for tta in modes:
            print(f"[VAL] imgsz={size} TTA={tta}", flush=True)
            row = evaluate(model, VAL_DATA, "val", size, tta, args.device, args.batch)
            rows.append(row)
            print(f"  mAP50={row['mAP50']:.4f} insulator={row['insulator_AP50']:.4f} MEAN_5={row['MEAN_5']:.4f}", flush=True)
    # Primary selection is six-class Val mAP50; latest smaller size wins an exact tie.
    winner = max(rows, key=lambda r: (r["mAP50"], -r["imgsz"], not r["tta"]))
    print(f"[WINNER] imgsz={winner['imgsz']} TTA={winner['tta']} Val mAP50={winner['mAP50']:.4f}", flush=True)
    if not args.val_only:
        print("[TEST] Running the frozen Val-selected configuration once", flush=True)
        test = evaluate(model, TEST_DATA, "test", winner["imgsz"], winner["tta"], args.device, args.batch)
        test["selected_by_val"] = True
        rows.append(test)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (OUT / "sweep_results.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (OUT / "winner.json").write_text(json.dumps({"winner": winner, "rows": rows}, indent=2), encoding="utf-8")
    print(f"Saved: {(OUT / 'sweep_results.csv').resolve()}")


if __name__ == "__main__":
    main()
