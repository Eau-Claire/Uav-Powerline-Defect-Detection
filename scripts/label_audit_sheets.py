#!/usr/bin/env python
"""Render contact sheets (GT green, prediction red, flagged box yellow) for sampled audit issues."""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import cv2
import numpy as np
import yaml


def draw(img, xywhn, color, label="", t=2):
    H, W = img.shape[:2]
    x, y, w, h = xywhn
    p1 = (int((x - w / 2) * W), int((y - h / 2) * H))
    p2 = (int((x + w / 2) * W), int((y + h / 2) * H))
    cv2.rectangle(img, p1, p2, color, t)
    if label:
        cv2.putText(img, label, (p1[0], max(p1[1] - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--pred-dir", required=True, type=Path)
    p.add_argument("--type", required=True)
    p.add_argument("--cls", default=None)
    p.add_argument("--split", default="val")
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--min-conf", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tile", type=int, default=420)
    p.add_argument("--offset", type=int, default=0, help="skip this many images (paging)")
    p.add_argument("--sort", choices=["random", "conf"], default="random")
    a = p.parse_args()

    cfg = yaml.safe_load(a.data.read_text())
    names = cfg["names"] if isinstance(cfg["names"], list) else [cfg["names"][i] for i in sorted(cfg["names"])]
    short = {n: n.replace("insulator_", "ins_").replace("insulator", "ins").replace("broken_strand", "strand")
             .replace("foreign_object", "fo").replace("bird_nest", "nest") for n in names}
    root = Path(cfg["path"])
    rows = [r for r in csv.DictReader((a.pred_dir / "audit/issues.csv").open())
            if r["split"] == a.split and r["type"] == a.type and (a.cls is None or r["class"] == a.cls)
            and (not r["conf"] or float(r["conf"]) >= a.min_conf)]
    if a.sort == "conf":
        rows.sort(key=lambda r: -float(r["conf"] or 0))
    else:
        random.Random(a.seed).shuffle(rows)
    seen, pick = set(), []
    for r in rows:
        if r["image"] not in seen:
            seen.add(r["image"]); pick.append(r)
        if len(pick) == a.offset + a.n:
            break
    pick = pick[a.offset:]
    seen = {r["image"] for r in pick}
    preds = {}
    for line in (a.pred_dir / f"pred_{a.split}.jsonl").open():
        d = json.loads(line)
        if d["image"] in seen:
            preds[d["image"]] = d
    tiles = []
    for k, r in enumerate(pick):
        img = cv2.imread(str(root / cfg[a.split] / r["image"]))
        lbl = root / cfg[a.split].replace("images", "labels", 1) / (Path(r["image"]).stem + ".txt")
        for l in (lbl.read_text().splitlines() if lbl.exists() else []):
            v = l.split()
            if v:
                draw(img, list(map(float, v[1:5])), (0, 200, 0), "GT " + short[names[int(v[0])]])
        d = preds[r["image"]]
        for c, f, b in zip(d["cls"], d["conf"], d["xywhn"]):
            if f >= 0.25:
                draw(img, b, (0, 0, 255), f"{short[names[c]]} {f:.2f}", 1)
        draw(img, [float(r[k2]) for k2 in "xywh"], (0, 255, 255), "", 3)
        s = a.tile / max(img.shape[:2])
        img = cv2.resize(img, None, fx=s, fy=s)
        canvas = np.full((a.tile, a.tile, 3), 30, np.uint8)
        canvas[:img.shape[0], :img.shape[1]] = img
        cv2.putText(canvas, f"#{k} {r['image'][:22]}", (4, a.tile - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(canvas)
    while len(tiles) % 4:
        tiles.append(np.full((a.tile, a.tile, 3), 30, np.uint8))
    sheet = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
    out = a.pred_dir / "sheets" / f"{a.split}_{a.type}_{a.cls or 'all'}_{a.offset}.jpg"
    out.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(out, f"({len(rows)} matching issues, showing {len(pick)})")
    for k, r in enumerate(pick):
        print(k, r["image"], r["class"], r["conf"], r["extra"], "gt:", r["gt_classes"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
