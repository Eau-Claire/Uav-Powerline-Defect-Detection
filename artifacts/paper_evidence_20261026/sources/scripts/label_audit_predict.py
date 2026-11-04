#!/usr/bin/env python
"""Run a trained detector over a YOLO dataset and dump raw predictions (JSONL) for label auditing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    p.add_argument("--conf", type=float, default=0.05)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch", type=int, default=16)
    a = p.parse_args()

    from ultralytics import YOLO

    cfg = yaml.safe_load(a.data.read_text())
    root = Path(cfg["path"])
    model = YOLO(str(a.weights))
    a.out.mkdir(parents=True, exist_ok=True)
    for split in a.splits:
        imgs = sorted((root / cfg[split]).glob("*.*"))
        out = a.out / f"pred_{split}.jsonl"
        done = set()
        if out.exists():  # resumable
            done = {json.loads(l)["image"] for l in out.open()}
        todo = [i for i in imgs if i.name not in done]
        print(f"{split}: {len(imgs)} images, {len(todo)} to predict", flush=True)
        with out.open("a") as fh:
            for s in range(0, len(todo), a.batch):
                chunk = todo[s:s + a.batch]
                for path, r in zip(chunk, model.predict([str(c) for c in chunk], conf=a.conf, imgsz=a.imgsz,
                                                        device=a.device, verbose=False, max_det=100)):
                    b = r.boxes
                    fh.write(json.dumps({
                        "image": path.name,
                        "wh": [int(r.orig_shape[1]), int(r.orig_shape[0])],
                        "cls": b.cls.int().tolist(),
                        "conf": [round(x, 4) for x in b.conf.tolist()],
                        "xywhn": [[round(v, 5) for v in row] for row in b.xywhn.tolist()],
                    }) + "\n")
                fh.flush()
                if (s // a.batch) % 50 == 0:
                    print(f"{split}: {s + len(chunk)}/{len(todo)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
