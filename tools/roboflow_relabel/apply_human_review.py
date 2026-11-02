"""Rebuild TRAIN labels from originals using only human-accepted proposals."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from core import Box, load_jsonl


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmakring_transmission_line_defect_model/data/curated_combined"
RUN = ROOT / "artifacts/roboflow_relabel/curated_combined_serverless"
DECISIONS = ROOT / "artifacts/roboflow_relabel/human_review_decisions.json"
DESTINATION = ROOT / "data/working/curated_combined_cross_labeled"


def item_key(image_name: str, kind: str, index: int) -> str:
    return f"{Path(image_name).stem}:{kind}:{index}"


def same_box(left: dict, right: dict, tolerance: float = 1e-4) -> bool:
    return all(abs(float(left[name]) - float(right[name])) <= tolerance
               for name in ("x1", "y1", "x2", "y2"))


def yolo_line(box: Box, image_size: tuple[int, int]) -> str:
    width, height = image_size
    values = (
        ((box.x1 + box.x2) / 2) / width,
        ((box.y1 + box.y2) / 2) / height,
        box.width / width,
        box.height / height,
    )
    if width <= 0 or height <= 0 or box.area <= 0:
        raise ValueError(f"Invalid image or box geometry: image={image_size}, box={box}")
    if any(value < -1e-6 or value > 1 + 1e-6 for value in values):
        raise ValueError(f"Box outside image: image={image_size}, box={box}")
    clipped = [max(0.0, min(1.0, value)) for value in values]
    return f"{box.class_id} {clipped[0]:.6f} {clipped[1]:.6f} {clipped[2]:.6f} {clipped[3]:.6f}"


def build_items() -> tuple[dict[str, dict], dict[str, tuple[int, int]]]:
    items: dict[str, dict] = {}
    changes = json.loads((RUN / "changes.json").read_text(encoding="utf-8"))
    for record in changes:
        image_name = Path(record["image"].replace("\\", "/")).name
        for index, box in enumerate(record["additions"]):
            key = item_key(image_name, "addition", index)
            items[key] = {"image": image_name, "kind": "addition", "box": box}

    conflict_indexes: defaultdict[str, int] = defaultdict(int)
    conflicts = json.loads((RUN / "conflicts.json").read_text(encoding="utf-8"))
    for record in conflicts:
        image_name = Path(record["image"].replace("\\", "/")).name
        index = conflict_indexes[image_name]
        conflict_indexes[image_name] += 1
        key = item_key(image_name, "conflict", index)
        items[key] = {"image": image_name, "kind": "conflict", "box": record["box"]}

    sizes = {
        Path(record["image_path"].replace("\\", "/")).stem: tuple(record["image_size"])
        for record in load_jsonl(RUN / "roboflow/proposals.jsonl")
    }
    return items, sizes


def main() -> None:
    review = json.loads(DECISIONS.read_text(encoding="utf-8"))
    decisions = review["decisions"]
    items, sizes = build_items()
    if review["total_items"] != len(items):
        raise RuntimeError(f"Review/package mismatch: header={review['total_items']}, items={len(items)}")
    unknown = set(decisions) - set(items)
    if unknown:
        raise RuntimeError(f"Unknown decision keys: {sorted(unknown)[:5]}")
    for key, decision in decisions.items():
        item = items[key]
        box = item["box"]
        if decision["image"] != item["image"] or decision["kind"] != item["kind"]:
            raise RuntimeError(f"Decision metadata mismatch: {key}")
        if decision["class_name"] != box["class_name"] or not same_box(decision["box"], box):
            raise RuntimeError(f"Decision box mismatch: {key}")
        if decision["decision"] not in {"accept", "reject", "unsure"}:
            raise RuntimeError(f"Unknown decision value for {key}: {decision['decision']}")

    accepted_by_image: defaultdict[str, list[Box]] = defaultdict(list)
    accepted_counts: Counter[str] = Counter()
    accepted_kinds: Counter[str] = Counter()
    for key, decision in decisions.items():
        if decision["decision"] != "accept":
            continue
        item = items[key]
        box = Box(**item["box"])
        accepted_by_image[Path(item["image"]).stem].append(box)
        accepted_counts[box.class_name or str(box.class_id)] += 1
        accepted_kinds[item["kind"]] += 1

    source_labels = sorted((SOURCE / "labels/train").glob("*.txt"))
    destination_labels = DESTINATION / "labels/train"
    if len(source_labels) != 17903 or not destination_labels.is_dir():
        raise RuntimeError("Unexpected source/destination TRAIN structure")
    rendered: dict[str, str] = {}
    for source_label in source_labels:
        stem = source_label.stem
        if stem not in sizes:
            raise RuntimeError(f"Missing image size for {stem}")
        original = source_label.read_text(encoding="utf-8").rstrip("\n")
        added = [yolo_line(box, sizes[stem]) for box in accepted_by_image.get(stem, [])]
        rendered[source_label.name] = original + "\n" + ("\n".join(added) + "\n" if added else "")

    # All inputs and output text are validated before replacing any label file.
    for name, text in rendered.items():
        (destination_labels / name).write_text(text, encoding="utf-8")

    decision_counts = Counter(value["decision"] for value in decisions.values())
    summary = {
        "status": "human_review_applied",
        "package": review["package"],
        "decision_exported_at": review["exported_at"],
        "total_review_items": len(items),
        "reviewed_items": len(decisions),
        "accepted": decision_counts["accept"],
        "rejected": decision_counts["reject"],
        "unsure_excluded": decision_counts["unsure"],
        "unreviewed_excluded": len(items) - len(decisions),
        "accepted_by_kind": dict(accepted_kinds),
        "accepted_boxes_by_class": dict(accepted_counts),
        "train_images": len(source_labels),
        "dataset": str(DESTINATION),
    }
    output = json.dumps(summary, indent=2)
    (RUN / "human_review_summary.json").write_text(output, encoding="utf-8")
    (DESTINATION / "HUMAN_REVIEW_STATUS.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
