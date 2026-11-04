"""Render samples with accepted additions from both YOLO26m and Roboflow."""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from core import Box, ROBOFLOW_CLASS_MAP, iou, load_jsonl, read_yolo_boxes

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "benchmakring_transmission_line_defect_model/data/curated_combined"
RUN = ROOT / "artifacts/roboflow_relabel/curated_combined_serverless"
OUT = RUN / "both_sources_preview"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]


def accepted_for(record, rf_record):
    image = SOURCE / "images/train" / Path(record["image"].replace("\\", "/")).name
    width, height = rf_record["image_size"]
    old = read_yolo_boxes(SOURCE / "labels/train" / f"{image.stem}.txt", (width, height))
    candidates = [(Box(**box), "YOLO26m", box.get("class_name")) for box in record["predictions"]]
    for raw in rf_record["predictions"]:
        source_class = raw.get("class_name") or ""
        target = ROBOFLOW_CLASS_MAP.get(source_class.casefold())
        if target is not None and (raw.get("confidence") or 0) >= 0.6:
            candidates.append((Box(**dict(raw, class_id=target, class_name=NAMES[target])), "Roboflow", source_class))
    accepted = []
    for box, source, source_class in candidates:
        box = Box(max(0, box.x1), max(0, box.y1), min(width, box.x2), min(height, box.y2),
                  box.class_id, box.confidence, box.class_name)
        if box.area <= 0:
            continue
        if any(other.class_id == box.class_id and iou(box, other) >= 0.3 for other, _, _ in accepted) or \
                any(other.class_id == box.class_id and iou(box, other) >= 0.3 for other in old):
            continue
        conflict = any(other.class_id != box.class_id and {other.class_id, box.class_id} != {0, 1}
                       and iou(box, other) >= 0.5 for other in old)
        conflict |= any(other.class_id != box.class_id and {other.class_id, box.class_id} != {0, 1}
                        and iou(box, other) >= 0.5 for other, _, _ in accepted)
        if not conflict:
            accepted.append((box, source, source_class))
    return image, old, accepted


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rf = {Path(row["image_path"].replace("\\", "/")).stem: row
          for row in load_jsonl(RUN / "roboflow/proposals.jsonl")}
    matches = []
    for record in load_jsonl(RUN / "yolo.jsonl"):
        stem = Path(record["image"].replace("\\", "/")).stem
        image, old, accepted = accepted_for(record, rf[stem])
        sources = {source for _, source, _ in accepted}
        if sources == {"YOLO26m", "Roboflow"}:
            score = sum(box.confidence or 0 for box, _, _ in accepted)
            matches.append((score, image, old, accepted))
    matches.sort(reverse=True, key=lambda item: item[0])
    selected = matches[:6]
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 18)
    small = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    canvas = Image.new("RGB", (1600, 1530), "#111827")
    ImageDraw.Draw(canvas).text((20, 12), "CYAN = existing | GREEN = YOLO26m | MAGENTA = Roboflow", font=font, fill="white")
    manifest = []
    for index, (_, path, old, accepted) in enumerate(selected):
        with Image.open(path) as source:
            source = source.convert("RGB")
            tile = ImageOps.contain(source, (780, 420))
            scale_x, scale_y = tile.width / source.width, tile.height / source.height
        draw = ImageDraw.Draw(tile)
        for box in old:
            draw.rectangle((box.x1 * scale_x, box.y1 * scale_y, box.x2 * scale_x, box.y2 * scale_y), outline="#00cfff", width=3)
        additions = []
        for box, origin, source_class in accepted:
            color = "#54ff53" if origin == "YOLO26m" else "#ff4fd8"
            coords = (box.x1 * scale_x, box.y1 * scale_y, box.x2 * scale_x, box.y2 * scale_y)
            draw.rectangle(coords, outline=color, width=4)
            label = f"{origin}: {source_class} -> {box.class_name} {box.confidence:.2f}"
            draw.text((max(0, coords[0]), max(0, coords[1] - 18)), label, font=small,
                      fill=color, stroke_width=2, stroke_fill="black")
            additions.append({"source": origin, "source_class": source_class, "box": box.__dict__})
        panel = Image.new("RGB", (800, 490), "#111827")
        panel_draw = ImageDraw.Draw(panel)
        panel_draw.text((10, 5), f"{index + 1}. {path.name}", font=font, fill="white")
        panel.paste(tile, ((800 - tile.width) // 2, 42))
        panel.save(OUT / f"sample_{index + 1}.jpg", quality=95)
        canvas.paste(panel, ((index % 2) * 800, 55 + (index // 2) * 490))
        manifest.append({"image": str(path), "additions": additions})
    canvas.save(OUT / "contact_sheet.jpg", quality=95)
    (OUT / "samples.json").write_text(json.dumps({"matching_images": len(matches), "samples": manifest}, indent=2), encoding="utf-8")
    print(json.dumps({"matching_images": len(matches), "rendered": len(selected), "preview": str(OUT / "contact_sheet.jpg")}, indent=2))


if __name__ == "__main__":
    main()
