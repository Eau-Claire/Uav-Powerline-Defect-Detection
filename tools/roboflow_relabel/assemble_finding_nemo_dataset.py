"""Assemble final human-reviewed train/val/test dataset with relative YOLO paths."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil

import yaml


ROOT = Path(__file__).resolve().parents[2]
DESTINATION = ROOT / "benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls"
EXPECTED_NAMES = ["insulator", "insulator_broken", "insulator_flashover",
                  "bird_nest", "foreign_object", "broken_strand"]
SOURCES = {
    "train": ROOT / "data/working/curated_combined_cross_labeled",
    "val": ROOT / "data/working/curated_combined_val_clean_cross_labeled",
    "test": ROOT / "data/working/test_cross_labeled",
}
SOURCE_SPLITS = {"train": "train", "val": "val_clean", "test": "test"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def link_or_copy(source: Path, destination: Path) -> None:
    if destination.is_file():
        source_stat = source.stat()
        destination_stat = destination.stat()
        if source_stat.st_size == destination_stat.st_size:
            return
        raise RuntimeError(f"Existing destination differs: {destination}")
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_label(path: Path) -> int:
    boxes = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number}: expected 5 fields")
        class_id = int(fields[0])
        coordinates = [float(value) for value in fields[1:]]
        if class_id not in range(6) or any(value < 0 or value > 1 for value in coordinates):
            raise ValueError(f"{path}:{line_number}: invalid YOLO label")
        boxes += 1
    return boxes


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "human_review_applied", "created_at": datetime.now(timezone.utc).isoformat(),
                "taxonomy": EXPECTED_NAMES, "splits": {}}
    for output_split, source_root in SOURCES.items():
        source_status_path = source_root / "HUMAN_REVIEW_STATUS.json"
        source_status = json.loads(source_status_path.read_text(encoding="utf-8"))
        if source_status.get("status") != "human_review_applied":
            raise RuntimeError(f"Source is not finalized: {source_root}")
        source_split = SOURCE_SPLITS[output_split]
        source_images = source_root / "images" / source_split
        source_labels = source_root / "labels" / source_split
        destination_images = DESTINATION / "images" / output_split
        destination_labels = DESTINATION / "labels" / output_split
        destination_images.mkdir(parents=True, exist_ok=True)
        destination_labels.mkdir(parents=True, exist_ok=True)
        images = {path.stem: path for path in source_images.iterdir()
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
        labels = {path.stem: path for path in source_labels.glob("*.txt")}
        if not images or set(images) != set(labels):
            raise RuntimeError(f"{output_split} image/label mismatch: {len(images)}/{len(labels)}")
        box_count = 0
        for stem in sorted(images):
            link_or_copy(images[stem], destination_images / images[stem].name)
            destination_label = destination_labels / labels[stem].name
            shutil.copy2(labels[stem], destination_label)
            box_count += validate_label(destination_label)
        manifest["splits"][output_split] = {
            "images": len(images), "labels": len(labels), "boxes": box_count,
            "source": str(source_root), "source_split": source_split,
            "review_status_sha256": sha256(source_status_path),
        }

    data = {
        "train": "images/train", "val": "images/val", "test": "images/test",
        "names": dict(enumerate(EXPECTED_NAMES)),
    }
    data_path = DESTINATION / "data.yaml"
    data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    manifest["data_yaml_sha256"] = sha256(data_path)
    (DESTINATION / "DATASET_STATUS.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
