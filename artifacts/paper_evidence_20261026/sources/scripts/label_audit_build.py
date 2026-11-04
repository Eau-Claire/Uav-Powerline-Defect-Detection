#!/usr/bin/env python
"""Build a cleaned 5-class dataset from the 6-class ablation dataset using the label-audit outputs.

Rules (see data/label_audit/README.md):
  A  hflip    mirror GT x for images detected by label_audit_flip.py           (train + val_clean)
  B  add_nest add bird_nest predictions conf>=--nest-conf with IoU<0.3 to GT   (train only)
  C  person   drop foreign_object GT overlapping a COCO person (IoU>=0.5);
              drop synthetic person pastes                                     (train + val_clean)
  D  syn_fo   drop synthetic foreign_object images whose pasted object was never fitted (train)
  E  quarant. exclude train images holding an unseen broken_strand / foreign_object GT
  F  tiny     drop GT boxes with a side < 4 px                                 (train)
  G  review   with --review: apply human/visual review decisions (review_decisions.csv):
              drop_box removes a box, drop_image removes an image, and rule E only quarantines
              images reviewed as `quarantine` (other E images are restored)    (train + val_clean)
  H  tighten  with --tighten: replace insulator GT by the SAM box (bird_nest: SAM cuts off the twigs at the
              nest edge; foreign_object: SAM cuts off strings/tails -> mixed box convention; both excluded)
              (sam_boxes.csv) when it lies inside the GT (shrink only) and 0.5<=IoU<0.85 (train + val_clean)
  P  person   with --person-all: add a `person` class from COCO person detections conf>=--person-label-conf
              that cover <=20% of any GT box (COCO calls close-up insulators / cable joints "person")
              (all splits; person is a negative for foreign_object, never an alert)
  I  ins_add  with --add-insulator: add insulator boxes proposed for images without any insulator label
              (insulator_ov/fuse.py: Grounding DINO + OWLv2 agreement)            (train + val_clean)
review_decisions.csv may also hold `add_box` rows (out space, with w,h) for objects the labels missed.
The insulator class is dropped and ids are remapped 1..5 -> 0..4 (same as v6p5_roboflow_5cls),
unless --keep-insulator (6 classes, ids unchanged; insulator labels are source-specific/incomplete).
Images are symlinked; the original dataset is never modified. VAL is copied unchanged as `val`.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from label_audit_analyze import iou, read_gt, xywh2xyxy  # noqa: E402

NAMES6 = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
NEST, FO, STRAND, PERSON = 3, 4, 5, 6  # PERSON only with --person-all


def load_jsonl(path: Path) -> dict:
    return {r["image"]: r for r in map(json.loads, path.open())} if path.exists() else {}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", required=True, type=Path, help="6-class data.yaml")
    p.add_argument("--audit", required=True, type=Path, help="label_audit/<model> dir")
    p.add_argument("--person", required=True, type=Path, help="label_audit/coco_person dir")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--nest-conf", type=float, default=0.6)
    p.add_argument("--person-conf", type=float, default=0.6)
    p.add_argument("--review", type=Path, help="review_decisions.csv (rule G)")
    p.add_argument("--tighten", type=Path, help="sam_boxes.csv (rule H)")
    p.add_argument("--add-insulator", type=Path, help="insulator_add.csv: split,image,x,y,w,h (rule I)")
    p.add_argument("--person-all", type=Path, help="label_audit/person_all dir (rule P)")
    p.add_argument("--person-label-conf", type=float, default=0.8)
    p.add_argument("--keep-insulator", action="store_true", help="write 6 classes (no drop/remap)")
    a = p.parse_args()

    cfg = yaml.safe_load(a.src.read_text())
    root = Path(cfg["path"])
    flips = {(r["split"], r["image"]) for r in csv.DictReader((a.audit / "audit/flips.csv").open())
             if r["transform"] == "hflip"}
    issues = list(csv.DictReader((a.audit / "audit/issues.csv").open()))
    unseen = defaultdict(list)
    for r in issues:
        if r["type"] == "unseen_gt":
            unseen[(r["split"], r["image"])].append((r["class"], float(r["x"]), float(r["y"])))

    # review: src-space rows match the source labels, out-space rows match the written (5-class) labels
    src_drop, out_drop, drop_image, quarantine = defaultdict(list), defaultdict(list), set(), set()
    add_box = defaultdict(list)
    for r in csv.DictReader(a.review.open()) if a.review else []:
        key = (r["split"], r["image"])
        box = (int(r["class6"]), float(r["x"]), float(r["y"]))
        if r["decision"] == "drop_box":
            (src_drop if r["space"] == "src" else out_drop)[key].append(box)
        elif r["decision"] == "drop_image":
            drop_image.add(key)
        elif r["decision"] == "quarantine":
            quarantine.add(key)
        elif r["decision"] == "add_box":
            add_box[key].append((box[0], [box[1], box[2], float(r["w"]), float(r["h"])]))

    # rule H: SAM box must sit inside the GT (2% slack per side) -> only shrinks, never grows
    tight = defaultdict(list)
    for r in csv.DictReader(a.tighten.open()) if a.tighten else []:
        x, y, w, h, sx, sy, sw, sh = (float(r[k]) for k in ("x", "y", "w", "h", "sx", "sy", "sw", "sh"))
        inside = (sx - sw / 2 >= x - w / 2 - 0.02 * w and sx + sw / 2 <= x + w / 2 + 0.02 * w
                  and sy - sh / 2 >= y - h / 2 - 0.02 * h and sy + sh / 2 <= y + h / 2 + 0.02 * h)
        if inside and 0.5 <= float(r["iou"]) < 0.85 and int(r["class6"]) == 0:
            tight[(r["split"], r["image"])].append((int(r["class6"]), x, y, (sx, sy, sw, sh)))

    ins_add = defaultdict(list)
    for r in csv.DictReader(a.add_insulator.open()) if a.add_insulator else []:
        ins_add[(r["split"], r["image"])].append([float(r[k]) for k in ("x", "y", "w", "h")])

    def drop_matching(gc, gb, boxes, tol=1e-3):
        hit = np.zeros(len(gc), bool)
        for c, x, y in boxes:
            hit |= (gc == c) & (np.abs(gb[:, 0] - x) < tol) & (np.abs(gb[:, 1] - y) < tol)
        return gc[~hit], gb[~hit], int(hit.sum())

    changes, stats = [], defaultdict(Counter)

    def log(split, image, rule, detail=""):
        changes.append({"split": split, "image": image, "rule": rule, "detail": detail})
        stats[split][rule] += 1

    a.out.mkdir(parents=True, exist_ok=False)
    for split, out_split, full in (("train", "train", True), ("val", "val", False), ("val", "val_clean", None)):
        # full=True: all train rules; None: objective fixes only (A, C); False: verbatim copy
        preds = load_jsonl(a.audit / f"pred_{split}.jsonl")
        persons = load_jsonl(a.person / f"pred_{split}.jsonl")
        persons_all = load_jsonl(a.person_all / f"pred_{split}.jsonl") if a.person_all else {}
        img_dir, lbl_dir = root / cfg[split], root / cfg[split].replace("images", "labels", 1)
        (a.out / "images" / out_split).mkdir(parents=True)
        (a.out / "labels" / out_split).mkdir(parents=True)
        for img in sorted(p for p in img_dir.iterdir() if p.suffix != ".npy"):  # skip ultralytics cache
            name, key = img.name, (split, img.name)
            gc, gb = read_gt(lbl_dir / (img.stem + ".txt"))
            gb = gb.copy()
            synthetic = img.stem.startswith("v65_syn")
            out_key = (out_split, name)
            if full is not False and a.review:
                if out_key in drop_image:
                    log(out_split, name, "G_review", "drop image")
                    continue
                gc, gb, n = drop_matching(gc, gb, src_drop.get(out_key, []))
                if n:
                    log(out_split, name, "G_review", f"drop {n} boxes")
            if full is not False:
                if full and synthetic and "_syn_person_" in name:
                    log(out_split, name, "C_person", "synthetic person paste")
                    continue
                if full and synthetic and any(c == "foreign_object" for c, *_ in unseen.get(key, [])):
                    log(out_split, name, "D_syn_fo", "pasted object never fitted")
                    continue
                if full and any(c in ("broken_strand", "foreign_object") for c, *_ in unseen.get(key, [])) \
                        and not synthetic and (not a.review or out_key in quarantine):
                    log(out_split, name, "E_quarantine",
                        ";".join(f"{c}@{x:.3f},{y:.3f}" for c, x, y in unseen[key]))
                    continue
                if key in flips:
                    gb[:, 0] = 1 - gb[:, 0]
                    log(out_split, name, "A_hflip", f"{len(gc)} boxes")
                pr = persons.get(name)
                if pr and len(gc) and (gc == FO).any():
                    pb = np.array([b for b, f in zip(pr["xywhn"], pr["conf"]) if f >= a.person_conf]).reshape(-1, 4)
                    if len(pb):
                        m = iou(xywh2xyxy(gb), xywh2xyxy(pb)).max(1)
                        drop = (gc == FO) & (m >= 0.5)
                        if drop.any():
                            log(out_split, name, "C_person", f"{int(drop.sum())} fo boxes")
                            gc, gb = gc[~drop], gb[~drop]
                if full and name in preds:
                    W, H = preds[name]["wh"]
                    small = (gb[:, 2] * W < 4) | (gb[:, 3] * H < 4)
                    if small.any():
                        log(out_split, name, "F_tiny", f"{int(small.sum())} boxes")
                        gc, gb = gc[~small], gb[~small]
                    r = preds[name]
                    nb = np.array([b for c, f, b in zip(r["cls"], r["conf"], r["xywhn"])
                                   if c == NEST and f >= a.nest_conf]).reshape(-1, 4)
                    if len(nb):
                        if len(gc):
                            nb = nb[iou(xywh2xyxy(nb), xywh2xyxy(gb)).max(1) < 0.3]
                        if len(nb):
                            log(out_split, name, "B_add_nest", f"{len(nb)} boxes")
                            gc = np.r_[gc, np.full(len(nb), NEST)]
                            gb = np.r_[gb, nb]
            if full is not False and a.review:
                gc, gb, n = drop_matching(gc, gb, out_drop.get(out_key, []))
                if n:
                    log(out_split, name, "G_review", f"drop {n} boxes")
            if full is not False and out_key in add_box:
                for c, b in add_box[out_key]:
                    gc, gb = np.r_[gc, c], np.r_[gb, [b]]
                log(out_split, name, "G_review", f"add {len(add_box[out_key])} boxes")
            if full is not False and out_key in tight:
                n = 0
                for c, x, y, sb in tight[out_key]:
                    hit = np.flatnonzero((gc == c) & (np.abs(gb[:, 0] - x) < 1e-5) & (np.abs(gb[:, 1] - y) < 1e-5))
                    if len(hit):
                        gb[hit[0]] = sb
                        n += 1
                if n:
                    log(out_split, name, "H_tighten", f"{n} boxes")
            if full is not False and out_key in ins_add and not (gc == 0).any():  # rule I
                gc, gb = np.r_[gc, np.zeros(len(ins_add[out_key]), int)], np.r_[gb, ins_add[out_key]]
                log(out_split, name, "I_ins_add", f"{len(ins_add[out_key])} boxes")
            if a.person_all:  # rule P, also on the frozen val: person boxes do not touch the defect classes
                pr = persons_all.get(name)
                pb = np.array([b for b, f in zip(pr["xywhn"], pr["conf"]) if f >= a.person_label_conf]).reshape(-1, 4) \
                    if pr else np.zeros((0, 4))
                if len(pb) and len(gb):  # COCO calls close-up insulators / cable joints "person": skip boxes on any GT
                    p1, p2 = pb[:, None, :2] - pb[:, None, 2:] / 2, pb[:, None, :2] + pb[:, None, 2:] / 2
                    g1, g2 = gb[None, :, :2] - gb[None, :, 2:] / 2, gb[None, :, :2] + gb[None, :, 2:] / 2
                    wh = np.clip(np.minimum(p2, g2) - np.maximum(p1, g1), 0, None)
                    pb = pb[(wh[..., 0] * wh[..., 1] / (pb[:, None, 2] * pb[:, None, 3])).max(1) <= 0.2]
                if len(pb):
                    gc, gb = np.r_[gc, np.full(len(pb), PERSON)], np.r_[gb, pb]
                    stats[out_split]["P_person"] += 1
            keep = np.ones(len(gc), bool) if a.keep_insulator else gc != 0  # drop insulator, remap 1..5 -> 0..4
            shift = 0 if a.keep_insulator else 1
            lines = [f"{c - shift} " + " ".join(f"{v:.6f}" for v in np.clip(b, 0, 1)) for c, b in zip(gc[keep], gb[keep])]
            (a.out / "labels" / out_split / (img.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            try:
                os.link(img.resolve(), a.out / "images" / out_split / name)
            except OSError:
                try:
                    os.symlink(img.resolve(), a.out / "images" / out_split / name)
                except OSError:
                    shutil.copy2(img.resolve(), a.out / "images" / out_split / name)
            stats[out_split]["images"] += 1
            for c in gc[keep]:
                stats[out_split][f"box:{(NAMES6 + ['person'])[c]}"] += 1

    names_all = NAMES6 + (["person"] if a.person_all else [])
    names5 = {i: n for i, n in enumerate(names_all if a.keep_insulator else names_all[1:])}
    (a.out / "data.yaml").write_text(yaml.safe_dump(
        {"path": str(a.out.resolve()), "train": "images/train", "val": "images/val", "names": names5}, sort_keys=False))
    (a.out / "data_val_clean.yaml").write_text(yaml.safe_dump(
        {"path": str(a.out.resolve()), "train": "images/train", "val": "images/val_clean", "names": names5},
        sort_keys=False))
    with (a.out / "changes.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["split", "image", "rule", "detail"])
        w.writeheader()
        w.writerows(changes)
    # COCO view of the same labels (RT-DETR / DETR-family repos); category ids = YOLO ids, bbox in pixels
    (a.out / "annotations").mkdir()
    for out_split in ("train", "val", "val_clean"):
        images, anns = [], []
        files = sorted(p for p in (a.out / "images" / out_split).iterdir() if p.suffix != ".npy")
        with ThreadPoolExecutor(32) as ex:  # header reads only; threads hide network-storage latency
            sizes = list(ex.map(lambda f: Image.open(f).size, files))
        for i, (img, (W, H)) in enumerate(zip(files, sizes)):
            images.append({"id": i, "file_name": f"images/{out_split}/{img.name}", "width": W, "height": H})
            for line in (a.out / "labels" / out_split / (img.stem + ".txt")).read_text().splitlines():
                c, x, y, w, h = line.split()
                x, y, w, h = float(x) * W, float(y) * H, float(w) * W, float(h) * H
                anns.append({"id": len(anns), "image_id": i, "category_id": int(c), "iscrowd": 0, "area": w * h,
                             "bbox": [round(x - w / 2, 2), round(y - h / 2, 2), round(w, 2), round(h, 2)]})
        (a.out / "annotations" / f"instances_{out_split}.json").write_text(json.dumps(
            {"images": images, "annotations": anns,
             "categories": [{"id": k, "name": v} for k, v in names5.items()]}))
    label_hash = hashlib.sha256()
    for f in sorted((a.out / "labels").rglob("*.txt")):
        label_hash.update(f.relative_to(a.out).as_posix().encode() + f.read_bytes())
    manifest = {
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": str(a.src.resolve()),
        "source_data_yaml_sha256": hashlib.sha256(a.src.read_bytes()).hexdigest(),
        "audit_dir": str(a.audit.resolve()),
        "labels_sha256": label_hash.hexdigest(),
        "params": {"nest_conf": a.nest_conf, "person_conf": a.person_conf,
                   "review": str(a.review.resolve()) if a.review else None,
                   "keep_insulator": a.keep_insulator,
                   "tighten": str(a.tighten.resolve()) if a.tighten else None,
                   "add_insulator": str(a.add_insulator.resolve()) if a.add_insulator else None,
                   "person_all": str(a.person_all.resolve()) if a.person_all else None,
                   "person_label_conf": a.person_label_conf if a.person_all else None},
        "dropped_classes": [] if a.keep_insulator else ["insulator"],
        "val_policy": "val = untouched source VAL (frozen, comparable); val_clean = A_hflip + C_person"
                      + (" + G_review" if a.review else "") + (" + H_tighten" if a.tighten else "")
                      + (" + I_ins_add" if a.add_insulator else ""),
        "stats": {k: dict(sorted(v.items())) for k, v in stats.items()},
    }
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["stats"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
