"""Rebuild an eval split from original labels plus human-accepted proposals."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from PIL import Image

from core import Box
from run_eval_split import PRESETS


ROOT = Path(__file__).resolve().parents[2]
CONFIG = {
    "val_clean": {
        "review": ROOT / "artifacts/roboflow_relabel/val_clean_serverless/review_items.json",
        "decisions": ROOT / "artifacts/roboflow_relabel/val_clean_review_decisions.json",
        "source_labels": ROOT / "benchmakring_transmission_line_defect_model/data/curated_combined/labels/val_clean",
    },
    "test": {
        "review": ROOT / "artifacts/roboflow_relabel/test_serverless/review_items.json",
        "decisions": ROOT / "artifacts/roboflow_relabel/test_review_decisions.json",
        "source_labels": ROOT / "data/working/test/labels/test",
    },
}


def same_box(left: dict, right: dict, tolerance: float = 1e-4) -> bool:
    return all(abs(float(left[name]) - float(right[name])) <= tolerance
               for name in ("x1", "y1", "x2", "y2"))


def yolo_line(box: Box, size: tuple[int, int]) -> str:
    width, height = size
    values = (((box.x1 + box.x2) / 2) / width, ((box.y1 + box.y2) / 2) / height,
              box.width / width, box.height / height)
    if width <= 0 or height <= 0 or box.area <= 0 or any(v < -1e-6 or v > 1 + 1e-6 for v in values):
        raise ValueError(f"Invalid box geometry: size={size}, box={box}")
    clipped = [max(0.0, min(1.0, value)) for value in values]
    return f"{box.class_id} {clipped[0]:.6f} {clipped[1]:.6f} {clipped[2]:.6f} {clipped[3]:.6f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", choices=CONFIG, required=True)
    args = parser.parse_args()
    config = CONFIG[args.target]
    package = json.loads(config["review"].read_text(encoding="utf-8"))
    items = {item["key"]: item for item in package["items"]}
    review = json.loads(config["decisions"].read_text(encoding="utf-8"))
    decisions = review["decisions"]
    if review.get("total_items") != len(items):
        raise RuntimeError(f"Package mismatch: decisions header={review.get('total_items')}, items={len(items)}")
    unknown = set(decisions) - set(items)
    if unknown:
        raise RuntimeError(f"Unknown decision keys: {sorted(unknown)[:5]}")

    accepted: defaultdict[str, list[Box]] = defaultdict(list)
    accepted_classes: Counter[str] = Counter()
    accepted_sources: Counter[str] = Counter()
    decision_counts: Counter[str] = Counter()
    for key, decision in decisions.items():
        item = items[key]
        if decision["decision"] not in {"accept", "reject", "unsure"}:
            raise RuntimeError(f"Invalid decision: {key}={decision['decision']}")
        if decision["image"] != Path(item["image"]).name or decision["kind"] != item["kind"]:
            raise RuntimeError(f"Decision metadata mismatch: {key}")
        if decision["class_name"] != item["box"].get("class_name") or not same_box(decision["box"], item["box"]):
            raise RuntimeError(f"Decision box mismatch: {key}")
        decision_counts[decision["decision"]] += 1
        if decision["decision"] == "accept":
            box = Box(**item["box"])
            accepted[Path(item["image"]).stem].append(box)
            accepted_classes[box.class_name or str(box.class_id)] += 1
            accepted_sources[item["source"]] += 1

    destination = PRESETS[args.target]["destination"]
    split_name = args.target
    destination_labels = destination / "labels" / split_name
    source_labels = sorted(config["source_labels"].glob("*.txt"))
    rendered: dict[str, str] = {}
    for source_label in source_labels:
        image_candidates = list((destination / "images" / split_name).glob(f"{source_label.stem}.*"))
        if len(image_candidates) != 1:
            raise RuntimeError(f"Expected one image for {source_label.stem}, found {len(image_candidates)}")
        with Image.open(image_candidates[0]) as image:
            size = image.size
        original = source_label.read_text(encoding="utf-8").rstrip("\n")
        additions = [yolo_line(box, size) for box in accepted.get(source_label.stem, [])]
        rendered[source_label.name] = original + "\n" + ("\n".join(additions) + "\n" if additions else "")

    if len(rendered) != len(source_labels) or not destination_labels.is_dir():
        raise RuntimeError("Unexpected destination structure")
    for name, text in rendered.items():
        (destination_labels / name).write_text(text, encoding="utf-8")

    summary = {
        "status": "human_review_applied", "target": args.target,
        "total_review_items": len(items), "reviewed_items": len(decisions),
        "accepted": decision_counts["accept"], "rejected": decision_counts["reject"],
        "unsure_excluded": decision_counts["unsure"], "unreviewed_excluded": len(items) - len(decisions),
        "accepted_boxes_by_class": dict(accepted_classes),
        "accepted_boxes_by_source": dict(accepted_sources),
        "images": len(source_labels), "dataset": str(destination),
        "decision_exported_at": review.get("exported_at"),
    }
    text = json.dumps(summary, indent=2)
    artifact_dir = config["review"].parent
    (artifact_dir / "human_review_summary.json").write_text(text, encoding="utf-8")
    (destination / "HUMAN_REVIEW_STATUS.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
