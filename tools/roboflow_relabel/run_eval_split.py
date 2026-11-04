"""Cross-label val_clean or test without modifying the source dataset.

Phases are resumable. ``export`` creates an unreviewed working copy plus
``changes.json``/``conflicts.json``; human decisions must be applied separately.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

from PIL import Image
import yaml

from core import Box, ROBOFLOW_CLASS_MAP, iou, load_jsonl, read_yolo_boxes


ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ["insulator", "insulator_broken", "insulator_flashover",
            "bird_nest", "foreign_object", "broken_strand"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PRESETS = {
    "val_clean": {
        "images": ROOT / "benchmakring_transmission_line_defect_model/data/curated_combined/images/val_clean",
        "labels": ROOT / "benchmakring_transmission_line_defect_model/data/curated_combined/labels/val_clean",
        "metadata": ROOT / "benchmakring_transmission_line_defect_model/data/curated_combined/data.yaml",
        "output": ROOT / "artifacts/roboflow_relabel/val_clean_serverless",
        "destination": ROOT / "data/working/curated_combined_val_clean_cross_labeled",
    },
    "test": {
        "images": ROOT / "data/working/test/images/test",
        "labels": ROOT / "data/working/test/labels/test",
        "metadata": ROOT / "data/working/test/dataset.yaml",
        "output": ROOT / "artifacts/roboflow_relabel/test_serverless",
        "destination": ROOT / "data/working/test_cross_labeled",
    },
}


def load_env_file(path: Path, override: bool) -> None:
    """Load simple KEY=VALUE entries without requiring python-dotenv."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if separator and key and (override or key not in os.environ):
            value = value.strip()
            if value[:1] in {"'", '"'} and value[-1:] == value[:1]:
                value = value[1:-1]
            else:
                value = value.split(" #", 1)[0].rstrip()
            os.environ[key] = value


def records_by_stem(path: Path, image_field: str) -> tuple[list[dict], dict[str, dict]]:
    rows = load_jsonl(path)
    indexed = {Path(row[image_field].replace("\\", "/")).stem: row for row in rows}
    if len(indexed) != len(rows):
        raise RuntimeError(f"Duplicate records in {path}")
    return rows, indexed


def validate_taxonomy(metadata_path: Path) -> None:
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
    names = metadata["names"]
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if names != EXPECTED:
        raise RuntimeError(f"Wrong dataset taxonomy: {names}")


def collect_pairs(images_dir: Path, labels_dir: Path) -> list[tuple[Path, Path]]:
    images = {path.stem: path for path in images_dir.iterdir()
              if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    labels = {path.stem: path for path in labels_dir.glob("*.txt")}
    if not images or set(images) != set(labels):
        raise RuntimeError(f"Image/label mismatch: images={len(images)}, labels={len(labels)}")
    return [(images[stem], labels[stem]) for stem in sorted(images)]


def write_manifest(path: Path, identity: dict) -> None:
    if path.is_file() and json.loads(path.read_text(encoding="utf-8")) != identity:
        raise RuntimeError(f"Run inputs changed; choose a fresh output directory: {path.parent}")
    path.write_text(json.dumps(identity, indent=2), encoding="utf-8")


def yolo_phase(pairs: list[tuple[Path, Path]], output: Path, model_path: Path, device: str) -> None:
    from ultralytics import YOLO

    cache = output / "yolo.jsonl"
    rows = load_jsonl(cache)
    done = {Path(row["image"].replace("\\", "/")).stem for row in rows}
    print(f"YOLO resume: {len(done)}/{len(pairs)} complete", flush=True)
    model = YOLO(str(model_path))
    actual = [model.names[index] for index in range(len(model.names))]
    if actual != EXPECTED:
        raise RuntimeError(f"Wrong teacher taxonomy: {actual}")
    with cache.open("a", encoding="utf-8") as handle:
        for index, (image, _) in enumerate(pairs, 1):
            if image.stem in done:
                continue
            result = model.predict(str(image), imgsz=640, conf=0.6, device=device, verbose=False)[0]
            predictions = [
                Box(*xy, int(class_id), float(score), model.names[int(class_id)]).__dict__
                for xy, class_id, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(),
                                                result.boxes.conf.tolist())
            ]
            handle.write(json.dumps({"image": str(image), "predictions": predictions}) + "\n")
            handle.flush()
            if index % 25 == 0 or index == len(pairs):
                print(f"YOLO {index}/{len(pairs)}", flush=True)


def roboflow_phase(pairs: list[tuple[Path, Path]], output: Path) -> None:
    from core import environment_defaults, run_workflow_batch

    defaults = environment_defaults()
    candidates = [{"image_path": str(image), "label_path": str(label), "box_count": 0}
                  for image, label in pairs]
    run_workflow_batch(
        candidates, output / "roboflow", defaults["api_url"], os.environ["ROBOFLOW_API_KEY"],
        defaults["workspace_name"], defaults["workflow_id"], {"insulator"}, 0.6, 1.1, 20,
        resume=True, workers=4, max_upload_dimension=1600,
    )


def hardlink_or_copy(source: Path, destination: Path) -> None:
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def export_phase(pairs: list[tuple[Path, Path]], output: Path, destination: Path,
                 split_name: str) -> None:
    yolo_rows, yolo = records_by_stem(output / "yolo.jsonl", "image")
    rf_rows, rf = records_by_stem(output / "roboflow/proposals.jsonl", "image_path")
    expected_stems = {image.stem for image, _ in pairs}
    if set(yolo) != expected_stems or set(rf) != expected_stems:
        raise RuntimeError(
            f"Inference incomplete: expected={len(expected_stems)}, yolo={len(yolo)}, roboflow={len(rf)}"
        )

    image_out = destination / "images" / split_name
    label_out = destination / "labels" / split_name
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)
    counts: Counter[str] = Counter()
    changes: list[dict] = []
    conflicts: list[dict] = []
    review_items: list[dict] = []
    for image, label in pairs:
        with Image.open(image) as opened:
            width, height = opened.size
        old = read_yolo_boxes(label, (width, height))
        accepted: list[Box] = []
        accepted_meta: list[tuple[Box, str, str]] = []
        candidates = [(Box(**box), "YOLO26m", box.get("class_name") or "")
                      for box in yolo[image.stem]["predictions"]]
        for raw in rf[image.stem]["predictions"]:
            target = ROBOFLOW_CLASS_MAP.get((raw.get("class_name") or "").casefold())
            if target is not None and (raw.get("confidence") or 0) >= 0.6:
                candidates.append((Box(**dict(raw, class_id=target, class_name=EXPECTED[target])),
                                   "Roboflow", raw.get("class_name") or ""))
        image_conflict_index = 0
        for box, source, source_class in candidates:
            box = Box(max(0, box.x1), max(0, box.y1), min(width, box.x2), min(height, box.y2),
                      box.class_id, box.confidence, box.class_name)
            if box.area <= 0:
                continue
            if any(other.class_id == box.class_id and iou(box, other) >= 0.3
                   for other in old + accepted):
                continue
            conflict = any(other.class_id != box.class_id
                           and {other.class_id, box.class_id} != {0, 1}
                           and iou(box, other) >= 0.5 for other in old + accepted)
            if conflict:
                conflicts.append({"image": str(image), "box": box.__dict__, "source": source})
                review_items.append({
                    "key": f"{image.stem}:conflict:{image_conflict_index}", "image": str(image),
                    "kind": "conflict", "source": source, "source_class": source_class,
                    "box": box.__dict__,
                })
                image_conflict_index += 1
                continue
            accepted.append(box)
            accepted_meta.append((box, source, source_class))
            counts[box.class_name or str(box.class_id)] += 1
        original = label.read_text(encoding="utf-8").rstrip("\n")
        additions = [
            f"{box.class_id} {(box.x1 + box.x2) / 2 / width:.6f} "
            f"{(box.y1 + box.y2) / 2 / height:.6f} {box.width / width:.6f} {box.height / height:.6f}"
            for box in accepted
        ]
        (label_out / label.name).write_text(
            original + "\n" + ("\n".join(additions) + "\n" if additions else ""), encoding="utf-8"
        )
        hardlink_or_copy(image, image_out / image.name)
        changes.append({"image": str(image), "additions": [box.__dict__ for box in accepted]})
        review_items.extend({
            "key": f"{image.stem}:addition:{index}", "image": str(image), "kind": "addition",
            "source": source, "source_class": source_class, "box": box.__dict__,
        } for index, (box, source, source_class) in enumerate(accepted_meta))

    data = {
        "path": str(destination.resolve()), "train": f"images/{split_name}",
        "val": f"images/{split_name}", "test": f"images/{split_name}",
        "names": dict(enumerate(EXPECTED)),
    }
    (destination / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (output / "changes.json").write_text(json.dumps(changes), encoding="utf-8")
    (output / "conflicts.json").write_text(json.dumps(conflicts), encoding="utf-8")
    (output / "review_items.json").write_text(json.dumps({
        "target": split_name, "status": "pending_human_review", "items": review_items,
    }, indent=2), encoding="utf-8")
    summary = {
        "images": len(pairs), "split": split_name, "added_boxes": dict(counts),
        "conflicts": len(conflicts), "dataset": str(destination),
        "status": "pseudo_labels_unreviewed",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=PRESETS, required=True)
    parser.add_argument("--phase", choices=("yolo", "roboflow", "export", "all"), default="all")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    os.chdir(ROOT)
    load_env_file(ROOT / "tools/roboflow_relabel/.env", override=True)
    load_env_file(ROOT / ".env", override=False)
    preset = PRESETS[args.target]
    validate_taxonomy(preset["metadata"])
    pairs = collect_pairs(preset["images"], preset["labels"])
    model_path = Path(os.environ["CROSS_MODEL"])
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"Missing teacher checkpoint: {model_path}")
    output = preset["output"]
    output.mkdir(parents=True, exist_ok=True)
    identity = {
        "target": args.target, "images": str(preset["images"].resolve()),
        "labels": str(preset["labels"].resolve()), "count": len(pairs),
        "stems": [image.stem for image, _ in pairs], "model": str(model_path.resolve()),
        "model_size": model_path.stat().st_size, "model_mtime_ns": model_path.stat().st_mtime_ns,
        "workflow": os.environ.get("ROBOFLOW_WORKFLOW_ID"),
    }
    write_manifest(output / "run_manifest.json", identity)
    print(f"{args.target}: {len(pairs)} images | phase={args.phase}", flush=True)
    if args.phase in {"yolo", "all"}:
        yolo_phase(pairs, output, model_path, args.device)
    if args.phase in {"roboflow", "all"}:
        roboflow_phase(pairs, output)
    if args.phase in {"export", "all"}:
        export_phase(pairs, output, preset["destination"], args.target)


if __name__ == "__main__":
    main()
