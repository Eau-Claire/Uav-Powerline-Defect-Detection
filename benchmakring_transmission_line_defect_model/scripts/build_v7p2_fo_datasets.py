"""Build V7.2 foreign_object ablation datasets from the reviewed V7.1 dataset.

Why: V7.1 added ~3,000 imported foreign_object (FO) boxes whose median area is
0.025 of the image, while val/test FO median area is 0.09-0.12 and train-base FO
is 0.037. The model drifted toward small FO and lost FO AP50-95 on val and test.

Variants (val/test are copied unchanged in every variant):

  a_base_only    curated-base train images only (control: effect of the label review)
  b_fo_large     a + imported images that contain FO, where every FO box has
                 area >= --min-fo-area and the image holds only FO/bird_nest boxes
                 (no imported insulator convention), capped at --max-import-fo boxes
  c_fo_large_rep b + extra copies of train images holding a FO box with area
                 >= --large-fo-area (oversampling the val/test-like FO size)

Images are never kept with some of their objects removed: an imported image
with any small FO box is dropped whole, so no visible object is left unlabeled.

Example (Windows):
  python benchmakring_transmission_line_defect_model/scripts/build_v7p2_fo_datasets.py ^
    --source benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls_v2_171126 ^
    --inventory artifacts/dataset_audit/v7p1_171126_imported_annotation_audit/image_inventory.csv ^
    --output-root benchmakring_transmission_line_defect_model/data
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
FO, BIRD = 4, 3
ALLOWED_IMPORT_CLASSES = {FO, BIRD}
VARIANTS = ("a_base_only", "b_fo_large", "c_fo_large_rep")
REP_RE = re.compile(r"__rep\d+$")
AREA_BINS = ((0.0, 0.01, "small_lt1pct"), (0.01, 0.05, "medium_1to5pct"), (0.05, 1.01, "large_ge5pct"))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", type=Path, required=True, help="reviewed V7.1 dataset (images/, labels/)")
    p.add_argument("--inventory", type=Path, required=True, help="image_inventory.csv with image,group,source")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--prefix", default="finding_nemo_6cls_v7p2")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--min-fo-area", type=float, default=0.03, help="min normalized area for every FO box of an imported image")
    p.add_argument("--max-import-fo", type=int, default=1200, help="cap on imported FO boxes")
    p.add_argument("--large-fo-area", type=float, default=0.05, help="FO area that triggers oversampling in variant c")
    p.add_argument("--repeat", type=int, default=1, help="extra copies per large-FO image in variant c")
    p.add_argument("--seed", type=int, default=85)
    return p.parse_args()


def read_boxes(path: Path):
    boxes = []
    if not path.is_file():
        return boxes
    for n, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{n}: expected 5 fields, got {len(parts)}")
        boxes.append((int(float(parts[0])), *map(float, parts[1:])))
    return boxes


def link_or_copy(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def area_bin(area: float) -> str:
    return next(name for lo, hi, name in AREA_BINS if lo <= area < hi)


def list_images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def load_inventory(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        return {r["image"]: (r["group"], r["source"]) for r in csv.DictReader(f)}


def select_imports(train_images, labels_dir, inventory, a):
    """Pick imported FO images that match the val/test FO size and carry no insulator classes."""
    reasons = Counter()
    eligible = []
    for img in train_images:
        group, source = inventory[img.name]
        if group != "imported":
            continue
        boxes = read_boxes(labels_dir / f"{img.stem}.txt")
        fo = [b for b in boxes if b[0] == FO]
        if not fo:
            reasons["no_fo"] += 1
        elif any(b[0] not in ALLOWED_IMPORT_CLASSES for b in boxes):
            reasons["has_insulator_or_other_class"] += 1
        elif any(b[3] * b[4] < a.min_fo_area for b in fo):
            reasons["has_small_fo"] += 1
        else:
            eligible.append((img, source, len(fo)))
    random.Random(a.seed).shuffle(eligible)
    chosen, fo_total = [], 0
    for img, source, n_fo in eligible:
        if fo_total + n_fo > a.max_import_fo:
            reasons["over_cap"] += 1
            continue
        chosen.append((img, source, n_fo))
        fo_total += n_fo
    reasons["selected"] = len(chosen)
    return chosen, reasons


def validate(out: Path):
    """Return a list of problems; empty means the dataset is consistent."""
    problems = []
    stems = {}
    for split in ("train", "val", "test"):
        imgs = list_images(out / "images" / split)
        labels = {p.stem for p in (out / "labels" / split).glob("*.txt")}
        img_stems = {p.stem for p in imgs}
        if img_stems - labels:
            problems.append(f"{split}: {len(img_stems - labels)} images without label")
        if labels - img_stems:
            problems.append(f"{split}: {len(labels - img_stems)} labels without image")
        stems[split] = {REP_RE.sub("", s) for s in img_stems}
        for lp in (out / "labels" / split).glob("*.txt"):
            try:
                for c, x, y, w, h in read_boxes(lp):
                    if not 0 <= c < len(NAMES):
                        problems.append(f"{lp.name}: class {c}")
                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                        problems.append(f"{lp.name}: box out of range {x} {y} {w} {h}")
            except ValueError as e:
                problems.append(str(e))
    for s1, s2 in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = stems[s1] & stems[s2]
        if overlap:
            problems.append(f"{s1}/{s2} share {len(overlap)} image names, e.g. {sorted(overlap)[:3]}")
    return problems


def stats(out: Path):
    result = {}
    for split in ("train", "val", "test"):
        cls, fo_bins, n_img = Counter(), Counter(), 0
        for lp in (out / "labels" / split).glob("*.txt"):
            n_img += 1
            for c, _, _, w, h in read_boxes(lp):
                cls[NAMES[c]] += 1
                if c == FO:
                    fo_bins[area_bin(w * h)] += 1
        result[split] = {"images": n_img, "boxes": {n: cls[n] for n in NAMES},
                         "fo_area_bins": {name: fo_bins[name] for *_, name in AREA_BINS}}
    return result


def build_variant(variant, a, src, inventory, chosen, out):
    for split in ("train", "val", "test"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    chosen_names = {img.name for img, *_ in chosen}
    rows = []

    for split in ("val", "test"):
        for img in list_images(src / "images" / split):
            link_or_copy(img, out / "images" / split / img.name)
            shutil.copy2(src / "labels" / split / f"{img.stem}.txt", out / "labels" / split / f"{img.stem}.txt")

    train_kept = []
    for img in list_images(src / "images" / "train"):
        group, source = inventory[img.name]
        keep = group == "base" or (variant != "a_base_only" and img.name in chosen_names)
        if not keep:
            continue
        label = src / "labels" / "train" / f"{img.stem}.txt"
        link_or_copy(img, out / "images" / "train" / img.name)
        if label.is_file():
            shutil.copy2(label, out / "labels" / "train" / label.name)
        else:
            (out / "labels" / "train" / label.name).write_text("", encoding="utf-8")
        train_kept.append((img, label))
        if group == "imported":
            rows.append({"image": img.name, "source": source, "action": "imported_fo_large"})

    if variant == "c_fo_large_rep" and a.repeat > 0:
        for img, label in train_kept:
            if not any(b[0] == FO and b[3] * b[4] >= a.large_fo_area for b in read_boxes(label)):
                continue
            for k in range(1, a.repeat + 1):
                stem = f"{img.stem}__rep{k}"
                link_or_copy(img, out / "images" / "train" / f"{stem}{img.suffix}")
                shutil.copy2(label, out / "labels" / "train" / f"{stem}.txt")
                rows.append({"image": f"{stem}{img.suffix}", "source": inventory[img.name][1],
                             "action": "oversample_large_fo"})

    yaml_text = (f"path: {out.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    (out / "data.yaml").write_text(yaml_text, encoding="utf-8")
    (out / "data_local.yaml").write_text(yaml_text, encoding="utf-8")
    with (out / "dataset_diff.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["image", "source", "action"])
        w.writeheader()
        w.writerows(rows)
    return Counter(r["action"] for r in rows)


def main():
    a = parse_args()
    src, root = a.source.resolve(), a.output_root.resolve()
    inventory = load_inventory(a.inventory)
    train_images = list_images(src / "images" / "train")
    unknown = [p.name for p in train_images if p.name not in inventory]
    if unknown:
        raise RuntimeError(f"{len(unknown)} train images missing from inventory, e.g. {unknown[:5]}")
    targets = {v: root / f"{a.prefix}_{v}" for v in a.variants}
    for out in targets.values():
        if out.exists():
            raise FileExistsError(f"Refusing to overwrite: {out}")

    chosen, reasons = select_imports(train_images, src / "labels" / "train", inventory, a)
    print("Import selection:", dict(reasons), "| FO boxes:", sum(n for *_, n in chosen))

    report = {}
    for variant, out in targets.items():
        actions = build_variant(variant, a, src, inventory, chosen, out)
        problems = validate(out)
        summary = {
            "variant": variant, "source_dataset": str(src), "inventory": str(a.inventory.resolve()),
            "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
            "import_selection": dict(reasons), "train_additions": dict(actions),
            "stats": stats(out), "validation_problems": problems,
            "policy": "val/test unchanged; imported FO images kept whole only when every FO box >= min area "
                      "and the image has only FO/bird_nest boxes",
        }
        (out / "BUILD_V7P2.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        report[variant] = summary
        tr = summary["stats"]["train"]
        print(f"\n[{variant}] {out}\n  train images={tr['images']} FO={tr['boxes']['foreign_object']} "
              f"FO bins={tr['fo_area_bins']}\n  additions={dict(actions)}\n  validation: "
              + ("OK" if not problems else f"{len(problems)} problems, first: {problems[:3]}"))
    if any(r["validation_problems"] for r in report.values()):
        raise SystemExit("Validation failed; see BUILD_V7P2.json")


if __name__ == "__main__":
    main()
