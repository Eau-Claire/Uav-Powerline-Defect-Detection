"""
Fast ZIP Packager for Kaggle Dataset Upload
Uses ZIP_STORED because images are already compressed JPEGs.
Runs in ~15-25 seconds and generates a ready-to-upload raw_combined.zip.
"""

from __future__ import annotations

import os
import sys
import time
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT_DIR / "data" / "raw_combined"
OUTPUT_ZIP = ROOT_DIR / "raw_combined.zip"

def compress_raw_dataset():
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"Source dir not found: {SOURCE_DIR}")
    
    print(f"=== Compressing '{SOURCE_DIR.name}' for Kaggle Upload ===")
    print(f"Source: {SOURCE_DIR}")
    print(f"Target: {OUTPUT_ZIP}")
    
    start_time = time.time()
    file_count = 0
    total_bytes = 0
    
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, dirs, files in os.walk(SOURCE_DIR):
            for file in files:
                # Skip .cache files
                if file.endswith(".cache"):
                    continue
                file_path = Path(root) / file
                arcname = file_path.relative_to(SOURCE_DIR)
                zf.write(file_path, arcname)
                file_count += 1
                total_bytes += file_path.stat().st_size
                if file_count % 5000 == 0:
                    elapsed = time.time() - start_time
                    print(f"  Processed {file_count:,} files ({total_bytes / (1024**3):.2f} GB) in {elapsed:.1f}s...")

    elapsed = time.time() - start_time
    zip_size_gb = OUTPUT_ZIP.stat().st_size / (1024**3)
    print(f"\n[SUCCESS] Packaged {file_count:,} files into '{OUTPUT_ZIP.name}'")
    print(f"Output Size: {zip_size_gb:.2f} GB")
    print(f"Total Time : {elapsed:.1f} seconds")
    print(f"Ready for upload to Kaggle Datasets!")

if __name__ == "__main__":
    compress_raw_dataset()

