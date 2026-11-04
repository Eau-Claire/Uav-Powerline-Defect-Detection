from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFECT_IDS = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path
    relative_image: str
    relative_label: str
    width: int
    height: int
    class_counts: tuple[int, ...]
    min_box_area: float
    max_box_area: float
    stress_score: float
    selection_reasons: tuple[str, ...]
    sha256: str

    @property
    def defect_ids(self) -> tuple[int, ...]:
        return tuple(i for i in DEFECT_IDS if self.class_counts[i] > 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_split(dataset_yaml: Path, split: str) -> tuple[Path, Path, dict[int, str]]:
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    split_value = cfg.get(split)
    if not split_value or isinstance(split_value, list):
        raise ValueError(f"Expected one path for split={split!r}, got {split_value!r}")
    images = (root / split_value).resolve()
    parts = list(Path(split_value).parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as exc:
        raise ValueError(f"Split path must contain an 'images' component: {split_value}") from exc
    labels = (root / Path(*parts)).resolve()
    names_raw = cfg.get("names", {})
    names = {int(k): str(v) for k, v in names_raw.items()} if isinstance(names_raw, dict) else {
        i: str(v) for i, v in enumerate(names_raw)
    }
    return images, labels, names


def read_label(path: Path, n_classes: int) -> tuple[tuple[int, ...], float, float]:
    counts = [0] * n_classes
    areas: list[float] = []
    if not path.exists():
        return tuple(counts), 0.0, 0.0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) < 5:
            raise ValueError(f"Invalid YOLO row at {path}:{line_number}")
        class_id = int(float(fields[0]))
        if class_id < 0 or class_id >= n_classes:
            raise ValueError(f"Class {class_id} outside 0..{n_classes - 1} at {path}:{line_number}")
        width, height = float(fields[3]), float(fields[4])
        counts[class_id] += 1
        areas.append(max(0.0, width * height))
    return tuple(counts), min(areas, default=0.0), max(areas, default=0.0)


def build_records(images_dir: Path, labels_dir: Path, n_classes: int) -> list[Record]:
    image_paths = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_dir}")
    records: list[Record] = []
    for image_path in image_paths:
        relative_image = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_image.with_suffix(".txt")
        counts, min_area, max_area = read_label(label_path, n_classes)
        defect_boxes = sum(counts[i] for i in DEFECT_IDS if i < n_classes)
        defect_classes = sum(counts[i] > 0 for i in DEFECT_IDS if i < n_classes)
        if defect_boxes == 0:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        # Ground-truth-only difficulty score. No model prediction may enter selection.
        reasons: list[str] = []
        tiny_bonus = 3.0 if 0 < min_area < 0.0025 else 0.0
        if tiny_bonus:
            reasons.append("tiny_box")
        small_bonus = 2.0 if 0.0025 <= min_area < 0.01 else 0.0
        if small_bonus:
            reasons.append("small_box")
        multi_class_bonus = 1.5 * max(0, defect_classes - 1)
        if multi_class_bonus:
            reasons.append("multi_defect_class")
        density_bonus = min(2.0, 0.25 * max(0, defect_boxes - 1))
        if density_bonus:
            reasons.append("dense_annotations")
        aspect_bonus = 0.5 if max(width / height, height / width) >= 1.5 else 0.0
        if aspect_bonus:
            reasons.append("non_square_image")
        oversized_bonus = 1.0 if max_area > 0.20 else 0.0
        if oversized_bonus:
            reasons.append("oversized_box_review")
        # These two classes are explicitly targeted by the curation rules:
        # missing bird nests and false/invalid foreign-object annotations.
        curation_class_bonus = (1.5 if counts[3] else 0.0) + (2.0 if counts[4] else 0.0)
        if counts[3]:
            reasons.append("missing_nest_rule_target")
        if counts[4]:
            reasons.append("foreign_object_cleanup_rule_target")
        score = (
            tiny_bonus + small_bonus + multi_class_bonus + density_bonus + aspect_bonus
            + oversized_bonus + curation_class_bonus
        )
        records.append(Record(
            image=image_path,
            label=label_path,
            relative_image=relative_image.as_posix(),
            relative_label=relative_image.with_suffix(".txt").as_posix(),
            width=width,
            height=height,
            class_counts=counts,
            min_box_area=min_area,
            max_box_area=max_area,
            stress_score=score,
            selection_reasons=tuple(reasons),
            sha256=sha256_file(image_path),
        ))
    return records


def select_records(records: list[Record], target_images: int, min_images_per_class: int, seed: int) -> list[Record]:
    if target_images <= 0:
        raise ValueError("target-images must be positive")
    rng = random.Random(seed)
    tie_break = {record.sha256: rng.random() for record in records}
    ordered = sorted(records, key=lambda r: (-r.stress_score, tie_break[r.sha256], r.relative_image))
    selected: dict[str, Record] = {}
    class_image_counts: Counter[int] = Counter()

    # Cover every defect class first using only labels and image geometry.
    for class_id in DEFECT_IDS:
        candidates = [record for record in ordered if class_id in record.defect_ids]
        for record in candidates:
            if class_image_counts[class_id] >= min_images_per_class:
                break
            if record.sha256 not in selected:
                selected[record.sha256] = record
                for present_id in record.defect_ids:
                    class_image_counts[present_id] += 1

    if len(selected) > target_images:
        raise ValueError(
            f"Class quotas require {len(selected)} images, exceeding target-images={target_images}. "
            "Raise --target-images or lower --min-images-per-class."
        )

    # Fill the remaining capacity with high-stress examples, capped so one class cannot dominate.
    soft_cap = max(min_images_per_class, int(target_images * 0.45))
    for record in ordered:
        if len(selected) >= target_images:
            break
        if record.sha256 in selected:
            continue
        if all(class_image_counts[c] >= soft_cap for c in record.defect_ids):
            continue
        selected[record.sha256] = record
        for class_id in record.defect_ids:
            class_image_counts[class_id] += 1

    if len(selected) < target_images:
        for record in ordered:
            if len(selected) >= target_images:
                break
            selected.setdefault(record.sha256, record)

    result = sorted(selected.values(), key=lambda r: r.relative_image)
    missing = [class_id for class_id in DEFECT_IDS if sum(class_id in r.defect_ids for r in result) < min_images_per_class]
    if missing:
        raise ValueError(f"Frozen test cannot satisfy quota for class IDs: {missing}")
    return result


def link_or_copy(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    if mode == "copy":
        shutil.copy2(source, destination)
    else:
        try:
            os.link(source, destination)
        except OSError:
            # Windows hardlinks cannot cross volumes; preserve the requested build by copying.
            shutil.copy2(source, destination)


def write_outputs(
    selected: list[Record], out_dir: Path, names: dict[int, str], seed: int,
    target_images: int, min_images_per_class: int, materialize: str,
    source_candidate_records: int, source_unique_sha256: int,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    manifest_fields = [
        "image", "label", "sha256", "width", "height", "classes", "defect_box_count",
        "min_box_area", "max_box_area", "stress_score", "selection_reasons",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest_fields)
        writer.writeheader()
        for record in selected:
            writer.writerow({
                "image": record.relative_image,
                "label": record.relative_label,
                "sha256": record.sha256,
                "width": record.width,
                "height": record.height,
                "classes": "|".join(names[i] for i in record.defect_ids),
                "defect_box_count": sum(record.class_counts[i] for i in DEFECT_IDS),
                "min_box_area": f"{record.min_box_area:.8f}",
                "max_box_area": f"{record.max_box_area:.8f}",
                "stress_score": f"{record.stress_score:.3f}",
                "selection_reasons": "|".join(record.selection_reasons) or "class_quota_baseline",
            })

    distribution_path = out_dir / "distribution.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["class_id", "class", "images", "boxes"])
        writer.writeheader()
        for class_id, class_name in sorted(names.items()):
            writer.writerow({
                "class_id": class_id,
                "class": class_name,
                "images": sum(record.class_counts[class_id] > 0 for record in selected),
                "boxes": sum(record.class_counts[class_id] for record in selected),
            })

    if materialize != "none":
        for record in selected:
            link_or_copy(record.image, out_dir / "images" / "test" / record.relative_image, materialize)
            if record.label.exists():
                link_or_copy(record.label, out_dir / "labels" / "test" / record.relative_label, materialize)
        dataset_yaml = {
            "path": str(out_dir.resolve()).replace("\\", "/"),
            "train": "images/test",
            "val": "images/test",
            "test": "images/test",
            "names": names,
        }
        (out_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")

    manifest_hash = sha256_file(manifest_path)
    freeze = {
        "name": "curation_stress_test_v1",
        "selection_policy": "ground_truth_and_image_geometry_only",
        "model_outputs_used": False,
        "seed": seed,
        "target_images": target_images,
        "min_images_per_defect_class": min_images_per_class,
        "source_candidate_records": source_candidate_records,
        "source_unique_sha256": source_unique_sha256,
        "exact_duplicate_candidate_rows_excluded": source_candidate_records - source_unique_sha256,
        "selected_images": len(selected),
        "manifest_sha256": manifest_hash,
    }
    (out_dir / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a disclosed curation stress test without model-based selection.")
    parser.add_argument("--dataset-yaml", type=Path, required=True, help="YOLO YAML containing the frozen source split.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/curation_stress_test_v1"))
    parser.add_argument("--target-images", type=int, default=600)
    parser.add_argument("--min-images-per-class", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20261027)
    parser.add_argument("--materialize", choices=("none", "hardlink", "copy"), default="hardlink")
    args = parser.parse_args()

    images_dir, labels_dir, names = resolve_split(args.dataset_yaml.resolve(), args.split)
    missing_classes = [class_id for class_id in DEFECT_IDS if class_id not in names]
    if missing_classes:
        raise ValueError(f"Dataset taxonomy is missing defect class IDs: {missing_classes}")
    records = build_records(images_dir, labels_dir, len(names))
    selected = select_records(records, args.target_images, args.min_images_per_class, args.seed)
    write_outputs(
        selected, args.out_dir.resolve(), names, args.seed,
        args.target_images, args.min_images_per_class, args.materialize,
        len(records), len({record.sha256 for record in records}),
    )


if __name__ == "__main__":
    main()
