from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import zipfile
from collections import Counter
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_label(text: str, origin: str) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {origin}:{line_number}: {line!r}")
        class_id = int(float(fields[0]))
        if class_id not in NAMES:
            raise ValueError(f"Invalid class {class_id} at {origin}:{line_number}")
        coords = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in coords):
            raise ValueError(f"Coordinate outside [0,1] at {origin}:{line_number}")
        if coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"Non-positive box at {origin}:{line_number}")
        counts[class_id] += 1
    return counts


def link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct the immutable V6.2E TEST split and verify it.")
    parser.add_argument("--records-dir", type=Path, required=True)
    parser.add_argument("--sources-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    records_dir = args.records_dir.resolve()
    sources_dir = args.sources_dir.resolve()
    out_dir = args.out_dir.resolve()
    image_out = out_dir / "images" / "test"
    label_out = out_dir / "labels" / "test"
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    source_roots = {
        "base": sources_dir / "base" / "curated_v6",
        "fotl": sources_dir / "overlay" / "v6p1_fotl_overlay",
    }
    for name, root in source_roots.items():
        if not (root / "images" / "test").is_dir() or not (root / "labels" / "test").is_dir():
            raise FileNotFoundError(f"Missing extracted {name} TEST source at {root}")

    manifest_path = records_dir / "v6p2e_split_manifest.csv"
    rows = [row for row in csv.DictReader(manifest_path.open(encoding="utf-8")) if row["split"] == "test"]
    if len(rows) != 2012:
        raise ValueError(f"Expected 2012 TEST manifest rows, found {len(rows)}")
    unique_sha_count = len({row["sha256"] for row in rows})

    replacements: dict[str, str] = {}
    labels_zip = records_dir / "v6p2e_reconciled_labels.zip"
    with zipfile.ZipFile(labels_zip) as archive:
        for member in archive.namelist():
            if member.lower().endswith(".txt"):
                replacements[Path(member).stem] = archive.read(member).decode("utf-8")

    class_images: Counter[int] = Counter()
    class_boxes: Counter[int] = Counter()
    rebuilt_rows: list[dict[str, str]] = []
    replacement_count = 0
    sha_occurrences: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        source = row["source"]
        if source not in source_roots:
            raise ValueError(f"Unknown source {source!r} in TEST manifest")
        image_name = Path(row["image_path_source"]).name
        label_name = Path(row["label_path_source"]).name
        source_image = source_roots[source] / "images" / "test" / image_name
        source_label = source_roots[source] / "labels" / "test" / label_name
        if not source_image.is_file() or not source_label.is_file():
            raise FileNotFoundError(f"Missing source pair for manifest row {index}: {source_image}, {source_label}")
        actual_sha = file_sha256(source_image)
        expected_sha = row["sha256"].lower()
        if actual_sha != expected_sha:
            raise ValueError(f"Image hash mismatch at manifest row {index}: {image_name}")

        if expected_sha in replacements:
            label_text = replacements[expected_sha]
            label_origin = f"{labels_zip}!/{expected_sha}.txt"
            replacement_count += 1
        else:
            label_text = source_label.read_text(encoding="utf-8")
            label_origin = str(source_label)
        counts = parse_label(label_text, label_origin)
        for class_id, count in counts.items():
            class_images[class_id] += 1
            class_boxes[class_id] += count

        sha_occurrences[expected_sha] += 1
        occurrence = sha_occurrences[expected_sha]
        output_stem = expected_sha if occurrence == 1 else f"{expected_sha}__dup{occurrence}"
        output_image = image_out / f"{output_stem}{source_image.suffix.lower()}"
        output_label = label_out / f"{output_stem}.txt"
        link_or_copy(source_image, output_image)
        output_label.write_text(label_text.rstrip() + ("\n" if label_text.strip() else ""), encoding="utf-8")
        rebuilt_rows.append({
            "sha256": expected_sha,
            "source": source,
            "original_image": image_name,
            "output_image": output_image.name,
            "output_label": output_label.name,
            "label_origin": "reconciled" if expected_sha in replacements else "source",
        })

    expected_distribution = {
        row["class"]: (int(row["test_images"]), int(row["test_boxes"]))
        for row in csv.DictReader((records_dir / "v6p2e_final_distribution.csv").open(encoding="utf-8"))
    }
    actual_distribution = {
        NAMES[class_id]: (class_images[class_id], class_boxes[class_id]) for class_id in NAMES
    }
    if actual_distribution != expected_distribution:
        raise ValueError(
            "Reconstructed TEST distribution does not match the authoritative report:\n"
            f"expected={expected_distribution}\nactual={actual_distribution}"
        )

    rebuilt_manifest = out_dir / "frozen_test_manifest.csv"
    with rebuilt_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rebuilt_rows[0]))
        writer.writeheader()
        writer.writerows(rebuilt_rows)

    distribution_path = out_dir / "distribution.csv"
    with distribution_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["class_id", "class", "images", "boxes"])
        for class_id, class_name in NAMES.items():
            writer.writerow([class_id, class_name, class_images[class_id], class_boxes[class_id]])

    dataset_yaml = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": NAMES,
    }
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    source_freeze = json.loads((records_dir / "v6p2e.freeze.json").read_text(encoding="utf-8"))
    freeze = {
        "dataset": source_freeze["final_tag"],
        "test_policy": source_freeze["test_policy"],
        "images": len(rebuilt_rows),
        "unique_image_sha256": unique_sha_count,
        "within_test_duplicate_rows_preserved": len(rebuilt_rows) - unique_sha_count,
        "labels": len(rebuilt_rows),
        "reconciled_label_files_applied": replacement_count,
        "manifest_sha256": file_sha256(rebuilt_manifest),
        "source_manifest_sha256": file_sha256(manifest_path),
        "distribution_verified": True,
    }
    (out_dir / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
