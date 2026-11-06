"""Build cumulative data-curation ablations from ``data/raw_combined``.

The protocol is intentionally train-only. Validation (``val``) and held-out
evaluation (``val_clean``, exposed as ``test``) are linked byte-for-byte from
the raw dataset for every stage.

Stages:
  m0_raw       no curation (YAML points directly at raw_combined)
  m1_a_b_f     Module I: A horizontal flip, B reviewed missing boxes, F tiny boxes
  m2_c_d       + Module II: C reviewed person/object fixes, D synthetic purging
  m3_e_g       + Module III: E reviewed quarantine, G human review decisions
  m4_h         + Module IV: H SAM box tightening

Images are hard-linked where possible, so the ablation datasets do not duplicate
the multi-gigabyte image payload. Labels are independently materialized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image


NAMES = {
    0: "insulator",
    1: "insulator_broken",
    2: "insulator_flashover",
    3: "bird_nest",
    4: "foreign_object",
    5: "broken_strand",
}
CLASS_BY_NAME = {v: k for k, v in NAMES.items()}
STAGES = (
    ("m1_a_b_f", 1),
    ("m2_c_d", 2),
    ("m3_e_g", 3),
    ("m4_h", 4),
)


def read_labels(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                values = [float(v) for v in line.split()]
                rows.append([int(values[0]), *values[1:]])
    return rows


def write_labels(path: Path, rows: list[list[float]]) -> None:
    lines = [f"{int(r[0])} " + " ".join(f"{v:.6f}" for v in r[1:]) for r in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def link_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def drop_matching(rows: list[list[float]], targets: list[tuple[int, float, float]], tol: float = 1e-3):
    kept: list[list[float]] = []
    removed = 0
    for row in rows:
        hit = any(int(row[0]) == c and abs(row[1] - x) < tol and abs(row[2] - y) < tol
                  for c, x, y in targets)
        if hit:
            removed += 1
        else:
            kept.append(row)
    return kept, removed


def load_protocol_inputs(repo_root: Path):
    audit = repo_root / "data/label_audit/yolo11n_adamw/audit"
    review_path = repo_root / "data/label_audit/review_decisions.csv"
    sam_path = repo_root / "data/label_audit/sam_tighten/sam_boxes.csv"

    flips = {
        r["image"]
        for r in csv.DictReader((audit / "flips.csv").open(encoding="utf-8"))
        if r["split"] == "train" and r["transform"] == "hflip"
    }
    issues = list(csv.DictReader((audit / "issues.csv").open(encoding="utf-8")))
    missing = [r for r in issues if r["split"] == "train" and r["type"] == "missing"]
    unseen = defaultdict(list)
    tiny = defaultdict(list)
    for r in issues:
        if r["split"] != "train":
            continue
        target = (CLASS_BY_NAME[r["class"]], float(r["x"]), float(r["y"]))
        if r["type"] == "unseen_gt":
            unseen[r["image"]].append(target)
        elif r["type"] == "tiny_gt":
            tiny[r["image"]].append(target)

    review = list(csv.DictReader(review_path.open(encoding="utf-8")))
    train_review = [r for r in review if r["split"] == "train"]
    drops = defaultdict(list)
    reviewed_adds = defaultdict(list)
    quarantine = set()
    for r in train_review:
        if r["decision"] == "drop_box":
            drops[r["image"]].append((int(r["class6"]), float(r["x"]), float(r["y"]), r["reason"]))
        elif r["decision"] == "add_box":
            reviewed_adds[r["image"]].append(
                [int(r["class6"]), float(r["x"]), float(r["y"]), float(r["w"]), float(r["h"])]
            )
        elif r["decision"] == "quarantine":
            quarantine.add(r["image"])

    sam = defaultdict(list)
    for r in csv.DictReader(sam_path.open(encoding="utf-8")):
        if r["split"] != "train" or int(r["class6"]) not in (0, 4):
            continue
        x, y, w, h, sx, sy, sw, sh = (float(r[k]) for k in ("x", "y", "w", "h", "sx", "sy", "sw", "sh"))
        inside = (
            sx - sw / 2 >= x - w / 2 - 0.02 * w
            and sx + sw / 2 <= x + w / 2 + 0.02 * w
            and sy - sh / 2 >= y - h / 2 - 0.02 * h
            and sy + sh / 2 <= y + h / 2 + 0.02 * h
        )
        if inside and 0.50 <= float(r["iou"]) < 0.85:
            sam[r["image"]].append((int(r["class6"]), x, y, [sx, sy, sw, sh]))

    return {
        "flips": flips,
        "missing_candidates": missing,
        "tiny": tiny,
        "unseen": unseen,
        "drops": drops,
        "reviewed_adds": reviewed_adds,
        "quarantine": quarantine,
        "sam": sam,
    }


def apply_stage(image: Path, label: Path, stage: int, inputs: dict, stats: Counter):
    rows = read_labels(label)
    name = image.name
    stem = image.stem

    # Module I: A, B, F.
    if name in inputs["flips"]:
        for row in rows:
            row[1] = 1.0 - row[1]
        stats["A_images"] += 1
    # B only accepts explicit human-reviewed additions. Unreviewed judge output
    # remains evidence and is never promoted to ground truth automatically.
    additions = inputs["reviewed_adds"].get(name, [])
    if additions:
        rows.extend([r.copy() for r in additions])
        stats["B_boxes_added"] += len(additions)
    rows, removed = drop_matching(rows, inputs["tiny"].get(name, []))
    stats["F_boxes_removed"] += removed

    if stage >= 2:
        # Module II: C removes only human-confirmed person/foreign-object errors.
        c_targets = [(c, x, y) for c, x, y, reason in inputs["drops"].get(name, []) if reason == "person"]
        rows, removed = drop_matching(rows, c_targets)
        stats["C_boxes_removed"] += removed

        # D follows the audited synthetic-artifact criterion used by the source pipeline.
        if stem.startswith("v65_syn") and any(c == 4 for c, _, _ in inputs["unseen"].get(name, [])):
            stats["D_images_removed"] += 1
            return None

    if stage >= 3:
        # Module III: quarantine only cases explicitly confirmed during review.
        if name in inputs["quarantine"]:
            stats["E_images_quarantined"] += 1
            return None
        g_targets = [(c, x, y) for c, x, y, reason in inputs["drops"].get(name, []) if reason != "person"]
        rows, removed = drop_matching(rows, g_targets)
        stats["G_boxes_removed"] += removed

    if stage >= 4:
        # Module IV: tighten insulator and foreign_object only, never grow a box.
        tightened = 0
        for cls, x, y, proposal in inputs["sam"].get(name, []):
            for row in rows:
                if int(row[0]) == cls and abs(row[1] - x) < 1e-5 and abs(row[2] - y) < 1e-5:
                    row[1:] = proposal
                    tightened += 1
                    break
        stats["H_boxes_tightened"] += tightened

    return rows


def dataset_yaml(root: Path) -> dict:
    return {
        "path": root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/val_clean",
        "names": NAMES,
    }


def hash_labels(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((root / "labels").rglob("*.txt")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build_stage(raw: Path, out: Path, stage: int, inputs: dict, force: bool) -> dict:
    if out.exists():
        if not force:
            raise FileExistsError(f"Output exists: {out}; pass --force to replace it")
        shutil.rmtree(out)
    stats = Counter()
    changes = []

    for split in ("val", "val_clean"):
        for kind in ("images", "labels"):
            for src in sorted((raw / kind / split).iterdir()):
                if src.is_file() and src.suffix != ".cache":
                    link_file(src, out / kind / split / src.name)

    image_dir = raw / "images/train"
    label_dir = raw / "labels/train"
    for image in sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix != ".cache"):
        before = stats.copy()
        rows = apply_stage(image, label_dir / f"{image.stem}.txt", stage, inputs, stats)
        if rows is None:
            changes.append({"image": image.name, "action": "drop_image"})
            continue
        link_file(image, out / "images/train" / image.name)
        target = out / "labels/train" / f"{image.stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        write_labels(target, rows)
        stats["train_images"] += 1
        delta = {k: stats[k] - before[k] for k in stats if stats[k] != before[k] and k != "train_images"}
        if delta:
            changes.append({"image": image.name, "action": delta})

    (out / "data.yaml").write_text(yaml.safe_dump(dataset_yaml(out), sort_keys=False), encoding="utf-8")
    manifest = {
        "source": raw.resolve().as_posix(),
        "stage": stage,
        "modules": list(range(1, stage + 1)),
        "protocol": "train-only; val and val_clean are byte-identical hard links to raw_combined",
        "split_policy": {"train": "curated", "val": "frozen raw val", "test": "frozen raw val_clean"},
        "judge": {"model": "YOLO11n", "reported_mAP50": 0.7564},
        "thresholds": {"missing_conf": 0.60, "missing_max_iou": 0.30, "tiny_px": 4,
                       "person_conf": 0.60, "person_iou": 0.50, "sam_iou": [0.50, 0.85]},
        "unreviewed_missing_candidates": len(inputs["missing_candidates"]),
        "stats": dict(sorted(stats.items())),
    }
    manifest["labels_sha256"] = hash_labels(out)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (out / "changes.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in changes), encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    script = Path(__file__).resolve()
    benchmark_root = script.parent.parent
    repo_root = benchmark_root.parent
    parser.add_argument("--raw", type=Path, default=benchmark_root / "data/raw_combined")
    parser.add_argument("--out", type=Path, default=benchmark_root / "data/curation_module_ablation")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    raw = args.raw.resolve()
    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    inputs = load_protocol_inputs(repo_root)

    raw_yaml = out_root / "m0_raw.yaml"
    raw_yaml.write_text(yaml.safe_dump(dataset_yaml(raw), sort_keys=False), encoding="utf-8")
    summary = {"m0_raw": {"data_yaml": raw_yaml.as_posix(), "modules": []}}
    for name, stage in STAGES:
        manifest = build_stage(raw, out_root / name, stage, inputs, args.force)
        summary[name] = {
            "data_yaml": (out_root / name / "data.yaml").as_posix(),
            "modules": manifest["modules"],
            "stats": manifest["stats"],
            "labels_sha256": manifest["labels_sha256"],
        }
        print(name, json.dumps(manifest["stats"], sort_keys=True), flush=True)
    (out_root / "ablation_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
