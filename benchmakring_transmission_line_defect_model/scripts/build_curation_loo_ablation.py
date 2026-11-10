"""Build Leave-One-Out (LOO) data-curation ablations: m_124, m_134, m_234.

Combinations:
  m_124: Modules 1, 2, 4 (Excludes Module 3: Quarantine E, Non-person drop G)
  m_134: Modules 1, 3, 4 (Excludes Module 2: Person drop C, Synthetic purge D)
  m_234: Modules 2, 3, 4 (Excludes Module 1: Flip fix A, Missing adds B, Tiny drop F)
  m_1234: Modules 1, 2, 3, 4 (Full curation baseline - optional)

The protocol is intentionally train-only. Validation (``val``) and held-out
evaluation (``val_clean``, exposed as ``test``) are linked byte-for-byte for
every stage.
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

NAMES = {
    0: "insulator",
    1: "insulator_broken",
    2: "insulator_flashover",
    3: "bird_nest",
    4: "foreign_object",
    5: "broken_strand",
}
CLASS_BY_NAME = {v: k for k, v in NAMES.items()}

LOO_STAGES = {
    "m_124": [1, 2, 4],
    "m_134": [1, 3, 4],
    "m_234": [2, 3, 4],
}


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


def apply_modules(image: Path, label: Path, active_modules: set[int], inputs: dict, stats: Counter):
    rows = read_labels(label)
    name = image.name
    stem = image.stem

    # --- Module 1: A (flips), B (reviewed missing additions), F (tiny box removals) ---
    if 1 in active_modules:
        if name in inputs["flips"]:
            for row in rows:
                row[1] = 1.0 - row[1]
            stats["A_images"] += 1

        additions = inputs["reviewed_adds"].get(name, [])
        if additions:
            rows.extend([r.copy() for r in additions])
            stats["B_boxes_added"] += len(additions)

        rows, removed = drop_matching(rows, inputs["tiny"].get(name, []))
        stats["F_boxes_removed"] += removed

    # --- Module 2: C (person-as-FO removal), D (bad synthetic image purging) ---
    if 2 in active_modules:
        c_targets = [(c, x, y) for c, x, y, reason in inputs["drops"].get(name, []) if reason == "person"]
        rows, removed = drop_matching(rows, c_targets)
        stats["C_boxes_removed"] += removed

        # D: Synthetic foreign-object image purge
        if stem.startswith("v65_syn") and any(c == 4 for c, _, _ in inputs["unseen"].get(name, [])):
            stats["D_images_removed"] += 1
            return None

    # --- Module 3: E (reviewed quarantine), G (human review non-person box drops) ---
    if 3 in active_modules:
        if name in inputs["quarantine"]:
            stats["E_images_quarantined"] += 1
            return None
        g_targets = [(c, x, y) for c, x, y, reason in inputs["drops"].get(name, []) if reason != "person"]
        rows, removed = drop_matching(rows, g_targets)
        stats["G_boxes_removed"] += removed

    # --- Module 4: H (SAM box tightening for insulator and foreign_object) ---
    if 4 in active_modules:
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


def resolve_dataset_sources(raw: Path, benchmark_root: Path):
    """Auto-detect source directories for images/train, labels/train, images/val, labels/val."""
    candidates = [
        raw,
        benchmark_root / "data/Curated",
        raw / "curated_combined",
        benchmark_root / "data/raw_combined",
    ]
    train_img, train_lbl, val_img, val_lbl = None, None, None, None
    for cand in candidates:
        if (cand / "images/train").is_dir() and train_img is None:
            train_img = cand / "images/train"
        if (cand / "labels/train").is_dir() and train_lbl is None:
            train_lbl = cand / "labels/train"
        if (cand / "images/val").is_dir() and val_img is None:
            val_img = cand / "images/val"
        if (cand / "labels/val").is_dir() and val_lbl is None:
            val_lbl = cand / "labels/val"

    if not all([train_img, train_lbl, val_img, val_lbl]):
        raise FileNotFoundError(
            f"Could not locate dataset splits. Found: train_img={train_img}, train_lbl={train_lbl}, "
            f"val_img={val_img}, val_lbl={val_lbl}. Please check data path."
        )

    # Find instances_val_clean.json for held-out evaluation
    val_clean_json_candidates = [
        benchmark_root / "data/raw_combined/curated_combined/annotations/instances_val_clean.json",
        raw / "annotations/instances_val_clean.json",
        raw / "curated_combined/annotations/instances_val_clean.json",
    ]
    val_clean_json = next((p for p in val_clean_json_candidates if p.is_file()), None)

    return train_img, train_lbl, val_img, val_lbl, val_clean_json


def materialize_val_clean(val_img_dir: Path, val_clean_json: Path, out_root: Path):
    """Materialize val_clean split (2,271 images, 2,863 boxes) from instances_val_clean.json."""
    data = json.loads(val_clean_json.read_text(encoding="utf-8"))
    ann_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        ann_by_image[ann["image_id"]].append(ann)

    img_out = out_root / "images/val_clean"
    lbl_out = out_root / "labels/val_clean"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    for img in data.get("images", []):
        file_name = Path(img["file_name"]).name
        src_img = val_img_dir / file_name
        if src_img.is_file():
            link_file(src_img, img_out / file_name)

        w, h = float(img["width"]), float(img["height"])
        rows = []
        for ann in ann_by_image.get(img["id"], []):
            bx, by, bw, bh = ann["bbox"]
            xc = (bx + bw / 2.0) / w
            yc = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            rows.append([int(ann["category_id"]), xc, yc, nw, nh])

        target_lbl = lbl_out / f"{src_img.stem}.txt"
        write_labels(target_lbl, rows)


def build_loo_stage(
    train_img_dir: Path,
    train_lbl_dir: Path,
    val_img_dir: Path,
    val_lbl_dir: Path,
    val_clean_json: Path | None,
    out: Path,
    stage_name: str,
    modules: list[int],
    inputs: dict,
    force: bool,
) -> dict:
    if out.exists():
        if not force:
            raise FileExistsError(f"Output exists: {out}; pass --force to replace it")
        shutil.rmtree(out)

    stats = Counter()
    changes = []
    active_modules = set(modules)

    # 1. Hard-link frozen val
    for src in sorted(val_img_dir.iterdir()):
        if src.is_file() and src.suffix != ".cache":
            link_file(src, out / "images/val" / src.name)

    for src in sorted(val_lbl_dir.iterdir()):
        if src.is_file() and src.suffix == ".txt":
            link_file(src, out / "labels/val" / src.name)

    # 2. Materialize val_clean (frozen test set)
    if val_clean_json is not None:
        materialize_val_clean(val_img_dir, val_clean_json, out)
    else:
        # Fallback if val_clean already exists
        for split in ("val_clean",):
            for kind, d in (("images", val_img_dir.parent / split), ("labels", val_lbl_dir.parent / split)):
                if d.is_dir():
                    for src in sorted(d.iterdir()):
                        if src.is_file() and src.suffix != ".cache":
                            link_file(src, out / kind / split / src.name)

    # 3. Apply active curation modules on train images & labels
    for image in sorted(p for p in train_img_dir.iterdir() if p.is_file() and p.suffix != ".cache"):
        before = stats.copy()
        lbl_file = train_lbl_dir / f"{image.stem}.txt"
        rows = apply_modules(image, lbl_file, active_modules, inputs, stats)
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
        "source": train_img_dir.parent.resolve().as_posix(),
        "stage": stage_name,
        "modules": modules,
        "protocol": "train-only; val and val_clean are byte-identical hard links to raw_combined",
        "split_policy": {"train": "curated", "val": "frozen raw val", "test": "frozen raw val_clean"},
        "judge": {"model": "YOLO11n", "reported_mAP50": 0.7564},
        "thresholds": {
            "missing_conf": 0.60,
            "missing_max_iou": 0.30,
            "tiny_px": 4,
            "person_conf": 0.60,
            "person_iou": 0.50,
            "sam_iou": [0.50, 0.85],
        },
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
    parser.add_argument(
        "--raw",
        type=Path,
        default=benchmark_root / "data/Curated",
        help="Path to source dataset (data/Curated or data/raw_combined)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=benchmark_root / "data/curation_module_ablation",
        help="Output directory for ablation datasets",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["m_124", "m_134", "m_234"],
        choices=list(LOO_STAGES.keys()) + ["m_1234"],
        help="Which stages to build",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing output directories")
    args = parser.parse_args()

    out_root = args.out.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    inputs = load_protocol_inputs(repo_root)

    train_img, train_lbl, val_img, val_lbl, val_clean_json = resolve_dataset_sources(
        args.raw.resolve(), benchmark_root
    )
    print(f"Dataset sources resolved successfully:")
    print(f"  Train images: {train_img}")
    print(f"  Train labels: {train_lbl}")
    print(f"  Val images:   {val_img}")
    print(f"  Val labels:   {val_lbl}")
    print(f"  Val clean:    {val_clean_json}")

    stages_to_build = {k: LOO_STAGES[k] for k in args.stages if k in LOO_STAGES}
    if "m_1234" in args.stages:
        stages_to_build["m_1234"] = [1, 2, 3, 4]

    summary = {}
    for name, modules in stages_to_build.items():
        print(f"\nBuilding {name} with modules {modules}...", flush=True)
        manifest = build_loo_stage(
            train_img,
            train_lbl,
            val_img,
            val_lbl,
            val_clean_json,
            out_root / name,
            name,
            modules,
            inputs,
            args.force,
        )
        summary[name] = {
            "data_yaml": (out_root / name / "data.yaml").as_posix(),
            "modules": manifest["modules"],
            "stats": manifest["stats"],
            "labels_sha256": manifest["labels_sha256"],
        }
        print(f"Done {name}: {manifest['stats']['train_images']} train images created", flush=True)

    summary_file = out_root / "loo_ablation_manifest.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nAll LOO ablation datasets built successfully! Summary at: {summary_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
