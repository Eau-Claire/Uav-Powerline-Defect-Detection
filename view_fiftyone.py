"""View the Transmission Line Object Detection subset in FiftyOne."""

import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import fiftyone as fo
from dotenv import load_dotenv


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def get_dataset_root() -> Path:
    default = "dataset/Power-equipment-image-dataset/transmission line object detection"
    return Path(os.getenv("RAW_ROOT", default))


def find_image(xml_path: Path, filename: str | None) -> Path | None:
    if not filename:
        return None

    candidates = [
        xml_path.parent.parent / "JPEGImages" / filename,
        xml_path.parent / filename,
    ]
    return next((path for path in candidates if path.exists()), None)


def parse_annotation(xml_path: Path, image_path: Path) -> list[fo.Detection]:
    root = ET.parse(xml_path).getroot()
    width = int(float(root.findtext("size/width", "0")))
    height = int(float(root.findtext("size/height", "0")))

    if width <= 0 or height <= 0:
        from PIL import Image

        with Image.open(image_path) as image:
            width, height = image.size

    detections = []
    for obj in root.findall("object"):
        name = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if not name or box is None:
            continue

        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            continue

        xmin, xmax = sorted((max(0, xmin), min(width, xmax)))
        ymin, ymax = sorted((max(0, ymin), min(height, ymax)))
        if xmax <= xmin or ymax <= ymin:
            continue

        detections.append(
            fo.Detection(
                label=name,
                bounding_box=[
                    xmin / width,
                    ymin / height,
                    (xmax - xmin) / width,
                    (ymax - ymin) / height,
                ],
            )
        )
    return detections


def load_dataset(root: Path, name: str, limit: int | None, recreate: bool) -> fo.Dataset:
    if fo.dataset_exists(name):
        if recreate:
            fo.delete_dataset(name)
        else:
            return fo.load_dataset(name)

    dataset = fo.Dataset(name)
    xml_files = sorted(root.rglob("*.xml"))
    added = 0
    skipped = 0

    for xml_path in xml_files:
        annotation_root = ET.parse(xml_path).getroot()
        image_path = find_image(xml_path, annotation_root.findtext("filename"))
        if image_path is None:
            skipped += 1
            continue

        sample = fo.Sample(filepath=str(image_path.resolve()))
        sample["ground_truth"] = fo.Detections(
            detections=parse_annotation(xml_path, image_path)
        )
        dataset.add_sample(sample)
        added += 1
        if limit and added >= limit:
            break

    dataset.persistent = True
    print(f"Loaded {added:,} samples from {root.resolve()}")
    if skipped:
        print(f"Skipped {skipped:,} XML files without a matching image")
    return dataset


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="power-equipment-transmission-lines")
    parser.add_argument("--limit", type=int, help="Load only the first N samples")
    parser.add_argument("--recreate", action="store_true", help="Rebuild an existing FiftyOne dataset")
    args = parser.parse_args()

    dataset = load_dataset(get_dataset_root(), args.name, args.limit, args.recreate)
    session = fo.launch_app(dataset)
    session.wait()


if __name__ == "__main__":
    main()
