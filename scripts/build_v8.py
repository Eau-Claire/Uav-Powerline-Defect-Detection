#!/usr/bin/env python
"""Build dataset v6p5_v8 from v6p5_v6 (7 classes, ids unchanged): remove the train->val leakage and apply the val fixes
that need no labelling convention (artifacts/review_v7/README.md, reviewed 2026-10-15).

Rule L  leak     train images that near-duplicate a val_clean_v6 image (pHash <= 6, artifacts/review_v7/
                 val_train_near_dups.json, 8/8 checked by eye) are removed, so models trained on v8 can use all of val.
Rule M  val_fix  val_clean_v8 = val_clean_v6 + 11 bird_nest boxes that were missing (review_queue ids 140-145, 147-151,
                 nests visible on towers / cross-arms) - 2 insulator_broken boxes not on an insulator (id 64 on the
                 binding wire, id 77 on a metal clamp). Drone = not FO: no val label to change (id 2 is unlabelled).
`val`, `val_clean`, `val_clean_v6` are copied unchanged. leaked_val.txt lists the 148 val images whose near-duplicate
was in the v5/v6 train set: evaluate models trained on v5/v6 on val_clean_v8 WITHOUT them (fair), v8 models on all.
Not changed here (need a convention / expert): insulator_broken 1 box per shed vs per crack, broken_strand tight vs
whole span, flashover whole shed vs burn mark, the 80 dubious small insulator_broken boxes (ids 57-138).
Usage: python scripts/build_v8.py --v6 datasets/v6p5_v6 --out datasets/v6p5_v8
"""
from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
REVIEW = REPO / "artifacts/review_v7"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand", "person"]
ADD_NEST = [140, 141, 142, 143, 144, 145, 147, 148, 149, 150, 151]
DROP_BROKEN = [64, 77]


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def iou(a, b):
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); h = max(0.0, min(a[3], b[3]) - max(a[1], b[1])); i = w * h
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v6", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    dups = json.loads((REVIEW / "val_train_near_dups.json").read_text())
    leaked_val = sorted({v for v, _ in dups}); leak_train = sorted({t for _, ts in dups for t, _ in ts})
    q = pd.read_csv(REVIEW / "review_queue.csv").set_index("id")
    add, drop = defaultdict(list), defaultdict(list)
    for i in ADD_NEST:
        r = q.loc[i]; assert r.cls == "bird_nest" and r.flag == "missing_label" and r.split == "val_clean_v6", i
        add[r.image].append((3, ast.literal_eval(r.box)))
    for i in DROP_BROKEN:
        r = q.loc[i]; assert r.cls == "insulator_broken" and r.split == "val_clean_v6", i
        drop[r.image].append((1, ast.literal_eval(r.box), i))

    changes, stats = [], defaultdict(Counter)
    def log(split, image, rule, detail=""):
        changes.append({"split": split, "image": image, "rule": rule, "detail": detail}); stats[split][rule] += 1

    a.out.mkdir(parents=True, exist_ok=False)
    for split, src in (("train", "train"), ("val", "val"), ("val_clean", "val_clean"), ("val_clean_v6", "val_clean_v6"),
                       ("val_clean_v8", "val_clean_v6")):
        (a.out / "images" / split).mkdir(parents=True); (a.out / "labels" / split).mkdir(parents=True)
        for img in sorted((a.v6 / "images" / src).glob("*.jpg")):
            if split == "train" and img.name in set(leak_train):
                log(split, img.name, "L_leak", "near-duplicate of a val image"); continue
            lines = [l for l in (a.v6 / "labels" / src / (img.stem + ".txt")).read_text().splitlines() if l.strip()]
            if split == "val_clean_v8":
                for c, b, i in drop.get(img.name, []):
                    xy = [(k, [float(v) for v in l.split()[1:]]) for k, l in enumerate(lines) if int(l.split()[0]) == c]
                    m = [(iou(b, [x - w / 2, y - h / 2, x + w / 2, y + h / 2]), k) for k, (x, y, w, h) in xy]
                    best = max(m) if m else (0, -1); assert best[0] > 0.9, (i, img.name, best)
                    lines.pop(best[1]); log(split, img.name, "M_val_fix", f"drop insulator_broken (review id {i})")
                for c, b in add.get(img.name, []):
                    lines.append(f"{c} {(b[0]+b[2])/2:.6f} {(b[1]+b[3])/2:.6f} {b[2]-b[0]:.6f} {b[3]-b[1]:.6f}")
                    log(split, img.name, "M_val_fix", "add bird_nest")
            (a.out / "labels" / split / (img.stem + ".txt")).write_bytes(("\n".join(lines) + ("\n" if lines else "")).encode())
            link(img, a.out / "images" / split / img.name)
            stats[split]["images"] += 1
            for l in lines: stats[split][f"box:{NAMES[int(l.split()[0])]}"] += 1
    assert stats["val_clean_v8"]["M_val_fix"] == len(ADD_NEST) + len(DROP_BROKEN)
    (a.out / "leaked_val.txt").write_text("\n".join(leaked_val) + "\n")
    for f, v in (("data.yaml", "val"), ("data_val_clean_v6.yaml", "val_clean_v6"), ("data_val_clean_v8.yaml", "val_clean_v8")):
        (a.out / f).write_text(yaml.safe_dump({"path": str(a.out.resolve()), "train": "images/train", "val": f"images/{v}",
                                               "names": dict(enumerate(NAMES))}, sort_keys=False))
    with (a.out / "changes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "image", "rule", "detail"]); w.writeheader(); w.writerows(changes)
    h = hashlib.sha256()
    for f in sorted((a.out / "labels").rglob("*.txt")):
        h.update(f.relative_to(a.out).as_posix().encode() + f.read_bytes().replace(b"\r\n", b"\n"))
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "source": "v6p5_v6",
                "source_labels_sha256": json.loads((a.v6 / "manifest.json").read_text())["labels_sha256"],
                "labels_sha256": h.hexdigest(), "leaked_val": len(leaked_val), "leak_train_removed": len(leak_train),
                "val_policy": "val_clean_v8 = main metric; legacy (v5/v6-trained) models: evaluate without leaked_val.txt",
                "stats": {k: dict(sorted(v.items())) for k, v in stats.items()}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("leaked_val", "leak_train_removed", "labels_sha256")}, indent=1))
    print(json.dumps(manifest["stats"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
