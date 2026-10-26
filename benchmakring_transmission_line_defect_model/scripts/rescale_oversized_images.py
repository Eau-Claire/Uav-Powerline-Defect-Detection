"""
Rescale Oversized UAV Images Script
Downscales ultra-high resolution drone images (e.g. 8688x5792, ~50MP)
to a maximum dimension of 2048px while strictly preserving the aspect ratio.

Because YOLO labels use normalized bounding box coordinates [x_center, y_center, w, h] in [0, 1],
rescaling the image with preserved aspect ratio leaves bounding box annotations 100% identical!

Training resolution is imgsz=640, so 2048px preserves 3.2x oversampling with zero loss of defect detail,
while reducing uncompressed decoding RAM from ~150 MB down to ~8 MB per image.
"""

from __future__ import annotations

import os
from pathlib import Path
from PIL import Image

MAX_DIM = 2048
JPEG_QUALITY = 95

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASETS = [
    ROOT_DIR / "data" / "raw_combined",
    ROOT_DIR / "data" / "curated_combined",
]

def rescale_oversized_images():
    print(f"=== Scanning for images with max dimension > {MAX_DIM}px ===")
    total_scanned = 0
    rescaled_count = 0
    
    for dataset_path in DATASETS:
        if not dataset_path.exists():
            continue
        
        images_dir = dataset_path / "images"
        if not images_dir.exists():
            continue
            
        print(f"\nScanning: {dataset_path.name}")
        for img_path in images_dir.rglob("*.*"):
            if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            
            total_scanned += 1
            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    if max(w, h) > MAX_DIM:
                        # Calculate new size maintaining exact aspect ratio
                        scale = MAX_DIM / max(w, h)
                        new_w = int(round(w * scale))
                        new_h = int(round(h * scale))
                        
                        # Resample
                        resized_img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        
                        # Save in-place
                        orig_size_kb = img_path.stat().st_size / 1024
                        resized_img.save(img_path, quality=JPEG_QUALITY, optimize=True)
                        new_size_kb = img_path.stat().st_size / 1024
                        
                        rescaled_count += 1
                        print(f"  [{rescaled_count}] {img_path.name}: {w}x{h} ({orig_size_kb:.0f}KB) -> {new_w}x{new_h} ({new_size_kb:.0f}KB)")
            except Exception as e:
                print(f"  [ERROR] {img_path.name}: {e}")

    print(f"\nTotal images scanned: {total_scanned}")
    print(f"Total oversized images rescaled: {rescaled_count}")

    # Invalidate old Ultralytics .cache files so dimensions are reloaded
    print("\nInvalidating stale label .cache files...")
    for dataset_path in DATASETS:
        labels_dir = dataset_path / "labels"
        if labels_dir.exists():
            for cache_file in labels_dir.glob("*.cache"):
                try:
                    cache_file.unlink()
                    print(f"  Removed: {cache_file.relative_to(ROOT_DIR)}")
                except Exception as e:
                    print(f"  Failed to remove {cache_file}: {e}")

    print("\n=== Done! Dataset is now ultra-lightweight and high-speed! ===")

if __name__ == "__main__":
    rescale_oversized_images()

