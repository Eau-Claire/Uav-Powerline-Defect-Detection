"""Create the corrected Kaggle RT-DETR notebook for the final 6-class dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat


DATA_CELL = r'''# Cell 2: Download final human-reviewed train/val/test dataset from Camber
import hashlib
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

DATASET_STASH = "stash://bunpmc/projects/finding_nemo/finding_nemo_human_reviewed_6cls_20261101.tar.gz"
ARCHIVE = Path("/kaggle/working/finding_nemo_human_reviewed_6cls_20261101.tar.gz")
ARCHIVE_SHA256 = "5555fd783a52551225cec93b981e4362dcba99637fb54f463480df9e9eaf1643"
EXTRACT_ROOT = Path("/tmp/camber_finding_nemo_6cls_20261101_extract")
dest_dataset_dir = Path("/tmp/finding_nemo_human_reviewed_6cls_20261101")
EXPECTED = {
    "train": {"images": 17903, "boxes": 25811},
    "val": {"images": 2271, "boxes": 3204},
    "test": {"images": 1931, "boxes": 2850},
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def file_sha256(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# CAMBER and CAMBER_API_KEY are configured in the previous setup cell.
if not ARCHIVE.is_file() or file_sha256(ARCHIVE) != ARCHIVE_SHA256:
    if ARCHIVE.exists():
        print("Removing stale/incomplete archive:", ARCHIVE)
        ARCHIVE.unlink()
    print(f"Downloading dataset from {DATASET_STASH}")
    result = subprocess.run([CAMBER, "stash", "cp", DATASET_STASH, str(ARCHIVE)], check=False)
    if result.returncode != 0:
        raise RuntimeError("Camber download failed. Check CAMBER_API_KEY and stash access.")

actual_sha256 = file_sha256(ARCHIVE)
if actual_sha256 != ARCHIVE_SHA256:
    raise RuntimeError(f"Dataset checksum mismatch: expected {ARCHIVE_SHA256}, got {actual_sha256}")
print(f"Archive verified: {ARCHIVE} ({ARCHIVE.stat().st_size / (1024 ** 3):.2f} GiB)")

with ARCHIVE.open("rb") as handle:
    magic = handle.read(4)
if magic[:2] != b"\x1f\x8b":
    raise ValueError(f"Expected gzip archive, got header {magic.hex()}")

EXTRACT_ROOT.mkdir(parents=True, exist_ok=True)
marker = EXTRACT_ROOT / ".extraction_complete"
if not marker.exists() or marker.read_text() != ARCHIVE_SHA256:
    if any(EXTRACT_ROOT.iterdir()):
        shutil.rmtree(EXTRACT_ROOT)
        EXTRACT_ROOT.mkdir(parents=True)
    print("Extracting verified .tar.gz ...")
    with tarfile.open(ARCHIVE, mode="r|gz") as archive:
        safe_root = EXTRACT_ROOT.resolve()
        for member in archive:
            target = (safe_root / member.name).resolve()
            if (not target.is_relative_to(safe_root) or member.issym() or member.islnk()
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f"Unsafe archive member: {member.name}")
            archive.extract(member, path=safe_root, filter="data")
    marker.write_text(ARCHIVE_SHA256)
else:
    print("Reusing checksum-verified extraction")


def is_dataset_root(path):
    return all((path / "images" / split).is_dir() and (path / "labels" / split).is_dir()
               for split in EXPECTED)


roots = [path for path in [EXTRACT_ROOT, *EXTRACT_ROOT.rglob("*")]
         if path.is_dir() and is_dataset_root(path)]
if len(roots) != 1:
    raise RuntimeError(f"Expected exactly one complete train/val/test dataset root, found: {roots}")
source_root = roots[0]
print("Dataset root:", source_root)


def validate_split(root, split):
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    images = sorted(path for path in image_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    labels = {path.stem: path for path in label_dir.glob("*.txt")}
    image_stems = {path.stem for path in images}
    if image_stems != set(labels):
        raise RuntimeError(
            f"{split} image/label mismatch: images={len(image_stems)}, labels={len(labels)}"
        )
    boxes = sum(sum(1 for line in labels[image.stem].read_text().splitlines() if line.strip())
                for image in images)
    expected = EXPECTED[split]
    if len(images) != expected["images"] or boxes != expected["boxes"]:
        raise RuntimeError(
            f"{split} mismatch: got {len(images)} images/{boxes} boxes; "
            f"expected {expected['images']}/{expected['boxes']}"
        )
    print(f"Verified {split}: {len(images):,} images, {boxes:,} boxes")
    return {path.name for path in images}


names = {split: validate_split(source_root, split) for split in EXPECTED}
for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
    overlap = names[left] & names[right]
    if overlap:
        raise RuntimeError(f"{left}/{right} share {len(overlap)} filenames: {sorted(overlap)[:3]}")

# Create a unified stable path used by the remaining notebook cells.
dest_dataset_dir.mkdir(parents=True, exist_ok=True)
for kind in ("images", "labels"):
    (dest_dataset_dir / kind).mkdir(exist_ok=True)
    for split in EXPECTED:
        target = dest_dataset_dir / kind / split
        expected_source = (source_root / kind / split).resolve()
        if target.is_symlink():
            if target.resolve() == expected_source:
                continue
            target.unlink()
        if target.exists():
            raise RuntimeError(f"Refusing to overwrite {target}")
        target.symlink_to(expected_source, target_is_directory=True)

print(f"Dataset ready: {dest_dataset_dir}")
for split in EXPECTED:
    print(split, "images:", len(list((dest_dataset_dir / "images" / split).iterdir())))
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    notebook = nbformat.read(args.source, as_version=4)
    matching = [index for index, cell in enumerate(notebook.cells)
                if cell.cell_type == "code" and "VALIDATION_EXPECTED_IMAGES" in cell.source]
    if matching != [3]:
        raise RuntimeError(f"Expected dataset setup at cell 3, found {matching}")
    notebook.cells[3].source = DATA_CELL
    notebook.cells[3].outputs = []
    notebook.cells[3].execution_count = None
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.destination)
    print(args.destination)


if __name__ == "__main__":
    main()
