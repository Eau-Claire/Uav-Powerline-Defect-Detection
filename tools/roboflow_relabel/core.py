from __future__ import annotations

import csv
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import local
from typing import Any, Iterable

from PIL import Image, ImageDraw


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

# Dataset IDs confirmed by the dataset owner; power-pole is deliberately unmapped.
ROBOFLOW_CLASS_MAP = {
    'insulator': 0,
    'damage_insulator': 1,
    'critical': 1,
    'centerdamage': 1,
    'edgesmalldamage': 1,
    'edgebigdamage': 1,
    'cautious': 1,
    'low': 1,
}


@dataclass(frozen=True)
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    confidence: float | None = None
    class_name: str | None = None

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2, (self.y1 + self.y2) / 2)


def find_image(images_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate
        candidate = images_dir / f"{stem}{suffix.upper()}"
        if candidate.is_file():
            return candidate
    return None


def read_yolo_boxes(label_path: Path, image_size: tuple[int, int]) -> list[Box]:
    image_width, image_height = image_size
    boxes: list[Box] = []
    if not label_path.is_file():
        return boxes
    for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
        parts = raw_line.split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
        class_id = int(float(parts[0]))
        x_center, y_center, width, height = map(float, parts[1:])
        boxes.append(
            Box(
                (x_center - width / 2) * image_width,
                (y_center - height / 2) * image_height,
                (x_center + width / 2) * image_width,
                (y_center + height / 2) * image_height,
                class_id,
            )
        )
    return boxes


def scan_candidates(dataset_root: Path, split: str, class_id: int = 1) -> list[dict[str, Any]]:
    images_dir = dataset_root / "images" / split
    labels_dir = dataset_root / "labels" / split
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected images/{split} and labels/{split} under {dataset_root}")
    candidates: list[dict[str, Any]] = []
    for label_path in sorted(labels_dir.glob("*.txt")):
        lines = label_path.read_text(encoding="utf-8").splitlines()
        count = sum(1 for line in lines if line.split() and int(float(line.split()[0])) == class_id)
        if not count:
            continue
        image_path = find_image(images_dir, label_path.stem)
        if image_path:
            candidates.append({"image_path": str(image_path), "label_path": str(label_path), "box_count": count})
    return candidates


def _prediction_dicts(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        keys = set(value)
        if {"x", "y", "width", "height"}.issubset(keys):
            yield value
        for child in value.values():
            yield from _prediction_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _prediction_dicts(child)


def extract_predictions(result: Any) -> list[Box]:
    predictions: list[Box] = []
    for item in _prediction_dicts(result):
        try:
            x = float(item["x"])
            y = float(item["y"])
            width = float(item["width"])
            height = float(item["height"])
        except (TypeError, ValueError):
            continue
        class_name = item.get("class") or item.get("class_name") or item.get("label")
        class_value = item.get("class_id", item.get("class_index", -1))
        try:
            class_value = int(class_value)
        except (TypeError, ValueError):
            class_value = -1
        confidence = item.get("confidence", item.get("score"))
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        predictions.append(
            Box(
                x - width / 2,
                y - height / 2,
                x + width / 2,
                y + height / 2,
                class_value,
                confidence,
                str(class_name) if class_name is not None else None,
            )
        )
    return predictions


def contains_center(container: Box, target: Box) -> bool:
    x, y = target.center
    return container.x1 <= x <= container.x2 and container.y1 <= y <= container.y2


def iou(left: Box, right: Box) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    return intersection / union if union else 0.0


def propose_matches(
    old_boxes: list[Box],
    predictions: list[Box],
    allowed_classes: set[str],
    min_confidence: float,
    min_area_ratio: float,
    max_area_ratio: float,
) -> list[dict[str, Any]]:
    allowed = {name.strip().casefold() for name in allowed_classes if name.strip()}
    filtered = [
        box for box in predictions
        if (box.confidence is None or box.confidence >= min_confidence)
        and (not allowed or (box.class_name or "").casefold() in allowed)
    ]
    proposals: list[dict[str, Any]] = []
    for old_index, old in enumerate(old_boxes):
        if old.class_id != 1:
            continue
        matches: list[tuple[float, Box, float]] = []
        for predicted in filtered:
            area_ratio = predicted.area / old.area if old.area else float("inf")
            if not contains_center(predicted, old):
                continue
            if not min_area_ratio <= area_ratio <= max_area_ratio:
                continue
            # Prefer the tightest full-insulator box; confidence breaks ties.
            score = area_ratio - 0.1 * (predicted.confidence or 0.0)
            matches.append((score, predicted, area_ratio))
        matches.sort(key=lambda item: item[0])
        best = matches[0] if matches else None
        proposals.append(
            {
                "old_index": old_index,
                "old_box": asdict(old),
                "new_box": asdict(best[1]) if best else None,
                "area_ratio": best[2] if best else None,
                "iou": iou(old, best[1]) if best else None,
                "match_count": len(matches),
                "status": "proposed" if len(matches) == 1 else ("ambiguous" if matches else "no_match"),
            }
        )
    return proposals


def propose_insulator_additions(
    current_boxes: list[Box],
    predictions: list[Box],
    allowed_classes: set[str],
    min_confidence: float,
    duplicate_iou: float = 0.5,
) -> list[dict[str, Any]]:
    """Map workflow insulator detections to dataset class 0 without touching class 1."""
    allowed = {name.strip().casefold() for name in allowed_classes if name.strip()}
    existing_insulators = [box for box in current_boxes if box.class_id == 0]
    additions: list[dict[str, Any]] = []
    accepted: list[Box] = []
    for predicted in predictions:
        if allowed and (predicted.class_name or "").casefold() not in allowed:
            continue
        if predicted.confidence is not None and predicted.confidence < min_confidence:
            continue
        mapped = Box(
            predicted.x1,
            predicted.y1,
            predicted.x2,
            predicted.y2,
            class_id=0,
            confidence=predicted.confidence,
            class_name="insulator",
        )
        overlap = max((iou(mapped, box) for box in [*existing_insulators, *accepted]), default=0.0)
        status = "duplicate" if overlap >= duplicate_iou else "add"
        additions.append(
            {
                "box": asdict(mapped),
                "source_class": predicted.class_name,
                "source_confidence": predicted.confidence,
                "max_existing_iou": overlap,
                "status": status,
            }
        )
        if status == "add":
            accepted.append(mapped)
    return additions


def draw_review(
    image_path: Path,
    old_boxes: list[Box],
    proposals: list[dict[str, Any]] | None = None,
    additions: list[dict[str, Any]] | None = None,
) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for box in old_boxes:
        if box.class_id == 1:
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="red", width=4)
        elif box.class_id == 0:
            draw.rectangle((box.x1, box.y1, box.x2, box.y2), outline="deepskyblue", width=3)
    for proposal in proposals or []:
        new_box = proposal.get("new_box")
        if new_box:
            draw.rectangle(
                (new_box["x1"], new_box["y1"], new_box["x2"], new_box["y2"]),
                outline="lime",
                width=4,
            )
    for addition in additions or []:
        if addition.get("status") != "add":
            continue
        box = addition["box"]
        draw.rectangle((box["x1"], box["y1"], box["x2"], box["y2"]), outline="lime", width=4)
    return image


def run_workflow_batch(
    candidates: list[dict[str, Any]],
    output_dir: Path,
    api_url: str,
    api_key: str,
    workspace_name: str,
    workflow_id: str,
    allowed_classes: set[str],
    min_confidence: float,
    min_area_ratio: float,
    max_area_ratio: float,
    resume: bool = False,
    workers: int = 1,
) -> Path:
    from inference_sdk import InferenceConfiguration, InferenceHTTPClient

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "proposals.jsonl"
    completed: set[str] = set()
    if resume and output_path.is_file():
        completed = {
            record["image_path"]
            for record in load_jsonl(output_path)
            if record.get("image_path")
        }
    thread_state = local()

    def process(candidate: dict[str, Any]) -> tuple[dict[str, Any], int, str]:
        if not hasattr(thread_state, "client"):
            thread_state.client = InferenceHTTPClient(api_url=api_url, api_key=api_key).configure(
                InferenceConfiguration(api_key_transport="header")
            )
        client = thread_state.client
        image_path = Path(candidate["image_path"])
        with Image.open(image_path) as image:
            image_size = image.size
        old_boxes = read_yolo_boxes(Path(candidate["label_path"]), image_size)
        result = client.run_workflow(
            workspace_name=workspace_name,
            workflow_id=workflow_id,
            images={"image": str(image_path)},
            use_cache=True,
        )
        predictions = extract_predictions(result)
        additions = propose_insulator_additions(
            old_boxes, predictions, allowed_classes, min_confidence
        )
        record = {
            **candidate,
            "image_size": image_size,
            "predictions": [asdict(box) for box in predictions],
            "additions": additions,
            "raw_result": result,
        }
        add_count = sum(1 for item in additions if item["status"] == "add")
        return record, add_count, image_path.name

    mode = "a" if resume and output_path.is_file() else "w"
    pending = [candidate for candidate in candidates if candidate["image_path"] not in completed]
    print(
        f"Resume: {len(completed):,} complete, {len(pending):,} pending, workers={workers}",
        flush=True,
    )
    with output_path.open(mode, encoding="utf-8") as handle:
        executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        results = executor.map(process, pending) if executor else map(process, pending)
        for index, (record, add_count, image_name) in enumerate(results, len(completed) + 1):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(
                f"[{index}/{len(candidates)}] {image_name} class0_additions={add_count}",
                flush=True,
            )
        if executor:
            executor.shutdown()
    return output_path


def box_to_yolo(box: Box, image_size: tuple[int, int]) -> str:
    image_width, image_height = image_size
    x_center = ((box.x1 + box.x2) / 2) / image_width
    y_center = ((box.y1 + box.y2) / 2) / image_height
    width = box.width / image_width
    height = box.height / image_height
    values = [max(0.0, min(1.0, value)) for value in (x_center, y_center, width, height)]
    return f"0 {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"


def export_staged_labels(records: list[dict[str, Any]], output_dir: Path) -> dict[str, int]:
    """Copy touched labels and append accepted class-0 proposals; class-1 lines stay byte-identical."""
    output_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    addition_count = 0
    for record in records:
        source_label = Path(record["label_path"])
        original = source_label.read_text(encoding="utf-8").rstrip("\n")
        added_lines = []
        for addition in record.get("additions", []):
            if addition.get("status") != "add":
                continue
            added_lines.append(box_to_yolo(Box(**addition["box"]), tuple(record["image_size"])))
        if not added_lines:
            continue
        destination = output_dir / source_label.name
        destination.write_text(original + "\n" + "\n".join(added_lines) + "\n", encoding="utf-8")
        image_count += 1
        addition_count += len(added_lines)
    return {"changed_images": image_count, "added_insulator_boxes": addition_count}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    records = []
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if line_number != len(lines):
                raise
            print(f'Ignoring truncated final checkpoint line in {path}', flush=True)
    return records


def save_decision(decisions_path: Path, image_path: str, decision: str, note: str) -> None:
    decisions_path.parent.mkdir(parents=True, exist_ok=True)
    rows: dict[str, dict[str, str]] = {}
    if decisions_path.is_file():
        with decisions_path.open("r", encoding="utf-8", newline="") as handle:
            rows = {row["image_path"]: row for row in csv.DictReader(handle)}
    rows[image_path] = {
        "image_path": image_path,
        "decision": decision,
        "note": note,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with decisions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["image_path", "decision", "note", "updated_at"])
        writer.writeheader()
        writer.writerows(rows.values())


def default_output_dir() -> Path:
    return Path("artifacts") / "roboflow_relabel" / time.strftime("%Y%m%d_%H%M%S")


def environment_defaults() -> dict[str, str]:
    return {
        "api_url": os.getenv("ROBOFLOW_API_URL", "http://localhost:9001"),
        "workspace_name": os.getenv("ROBOFLOW_WORKSPACE", "my-workspace-j75em"),
        "workflow_id": os.getenv("ROBOFLOW_WORKFLOW_ID", "evn-object-detection"),
        "allowed_classes": os.getenv("ROBOFLOW_INSULATOR_CLASSES", "insulator"),
    }
