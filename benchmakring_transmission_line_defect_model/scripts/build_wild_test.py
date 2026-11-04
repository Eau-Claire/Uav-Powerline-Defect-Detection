"""Freeze an out-of-distribution "in the wild" test set (ground-level photos, other cameras, web images).

It is reported separately from test_main_v1: test_main_v1 shares sources with TRAIN, so it measures
in-distribution accuracy only. This set measures how far the models carry to images they were not built for.

Inputs:
  --images-dir / --labels-dir   YOLO images and labels with the six-class taxonomy (empty/missing label = background).
  --sources-csv                 One row per image: image,origin,capture,license,notes.
                                capture is one of CAPTURE_TYPES; origin is the URL, device or person it came from.

Checks, all before anything is written:
  * every label is valid YOLO with class ids 0..5, every image decodes, every image has a sources row;
  * exact duplicates (SHA-256) are collapsed;
  * pHash (imagehash.phash, hash_size 8, same as V6.2E) is compared with every V6.2E TRAIN/VAL/TEST image and
    with the other images of this set; pairs within --max-hamming are written to near_duplicates.csv and the
    build stops unless --drop-near-duplicates is given. Web images are often copies of public datasets that are
    already in TRAIN, so this check is the point of the script.
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import imagehash
import yaml
from PIL import Image

from build_clean_test import IMAGE_SUFFIXES, copy_file, distribution_rows, parse_label_text, sha256_file, sha256_text, write_csv


NAMES = {0: "insulator", 1: "insulator_broken", 2: "insulator_flashover", 3: "bird_nest", 4: "foreign_object", 5: "broken_strand"}
CAPTURE_TYPES = {"ground", "drone", "ipcam", "web", "other"}
SOURCE_FIELDS = ["image", "origin", "capture", "license", "notes"]
DEFAULT_PHASH = Path(__file__).resolve().parents[1] / "data/v6p2e_cache/records/perceptual_hash_manifest.csv"


def read_sources(path: Path) -> dict[str, dict[str, str]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    missing = [f for f in SOURCE_FIELDS if rows and f not in rows[0]]
    if not rows or missing:
        raise ValueError(f"{path} needs columns {SOURCE_FIELDS}; missing {missing or 'all rows'}")
    sources: dict[str, dict[str, str]] = {}
    for row in rows:
        name = row["image"].strip()
        if name in sources:
            raise ValueError(f"Duplicate sources row for {name}")
        if row["capture"].strip() not in CAPTURE_TYPES:
            raise ValueError(f"capture for {name} must be one of {sorted(CAPTURE_TYPES)}, got {row['capture']!r}")
        if not row["origin"].strip():
            raise ValueError(f"origin is empty for {name}")
        sources[name] = {k: (row.get(k) or "").strip() for k in SOURCE_FIELDS}
    return sources


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze an in-the-wild OOD test set with leakage checks against V6.2E.")
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--labels-dir", type=Path, required=True)
    parser.add_argument("--sources-csv", type=Path, required=True)
    parser.add_argument("--phash-manifest", type=Path, default=DEFAULT_PHASH,
                        help="V6.2E perceptual_hash_manifest.csv (all splits); pulled by pull_v6p2e_test_local.sh.")
    parser.add_argument("--max-hamming", type=int, default=4, help="Same threshold as configs/v6p5_external.yaml.")
    parser.add_argument("--drop-near-duplicates", action="store_true",
                        help="Drop flagged images instead of stopping (within-set: the later image of each pair).")
    parser.add_argument("--out-dir", type=Path, default=Path("data/test_wild_v1"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if (out_dir / "freeze.json").exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir / 'freeze.json'} exists; pass --overwrite to rebuild.")
    if not args.phash_manifest.is_file():
        raise FileNotFoundError(f"{args.phash_manifest} missing; it is v6p2e/<tag>/perceptual_hash_manifest.csv on Camber.")
    sources = read_sources(args.sources_csv)

    images = sorted(p for p in args.images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No images under {args.images_dir}")
    no_source = [p.name for p in images if p.name not in sources]
    if no_source:
        raise ValueError(f"{len(no_source)} images have no sources row, e.g. {no_source[:3]}")

    records: dict[str, dict] = {}
    duplicate_rows = 0
    for path in images:
        sha = sha256_file(path)
        if sha in records:
            duplicate_rows += 1
            continue
        label = args.labels_dir / path.relative_to(args.images_dir).with_suffix(".txt")
        text = label.read_text(encoding="utf-8") if label.exists() else ""
        with Image.open(path) as im:
            im.load()
            width, height = im.size
            phash = imagehash.phash(im.convert("RGB"), hash_size=8)
        records[sha] = dict(path=path, label_text=text, label_sha256=sha256_text(text), width=width, height=height,
                            phash=int(str(phash), 16), counts=parse_label_text(text, len(NAMES), str(label)), source=sources[path.name])

    reference = [(row["sha256"], row["old_split"], int(row["phash"], 16))
                 for row in csv.DictReader(args.phash_manifest.open(encoding="utf-8"))]
    flagged: list[dict] = []
    order = list(records)
    for i, sha in enumerate(order):
        rec = records[sha]
        for ref_sha, split, ref_hash in reference:
            distance = (rec["phash"] ^ ref_hash).bit_count()
            if distance <= args.max_hamming:
                flagged.append(dict(image=rec["path"].name, sha256=sha, match=ref_sha, match_in=f"v6p2e_{split}", hamming=distance))
        for other in order[:i]:
            distance = (rec["phash"] ^ records[other]["phash"]).bit_count()
            if distance <= args.max_hamming:
                flagged.append(dict(image=rec["path"].name, sha256=sha, match=other, match_in="this_set", hamming=distance))

    report_dir = out_dir if args.drop_near_duplicates else args.sources_csv.resolve().parent
    if flagged:
        report_dir.mkdir(parents=True, exist_ok=True)
        write_csv(report_dir / "near_duplicates.csv", ["image", "sha256", "match", "match_in", "hamming"], flagged)
        if not args.drop_near_duplicates:
            by_where = Counter(row["match_in"] for row in flagged)
            sys.exit(f"{len({r['sha256'] for r in flagged})} images are near-duplicates {dict(by_where)}; "
                     f"see {report_dir / 'near_duplicates.csv'} or rerun with --drop-near-duplicates.")
    dropped = {row["sha256"] for row in flagged}
    kept = [records[sha] | {"sha256": sha} for sha in order if sha not in dropped]
    if not kept:
        raise ValueError("Every image was flagged; nothing to freeze.")

    report = out_dir / "near_duplicates.csv"
    keep_report = report.read_bytes() if report.exists() and flagged else None
    if out_dir.exists():
        shutil.rmtree(out_dir)
    for rec in kept:
        copy_file(rec["path"], out_dir / "images" / "test" / f"{rec['sha256']}{rec['path'].suffix.lower()}")
        label_target = out_dir / "labels" / "test" / f"{rec['sha256']}.txt"
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text(rec["label_text"], encoding="utf-8")
    if keep_report is not None:
        report.write_bytes(keep_report)

    manifest_path = out_dir / "manifest.csv"
    write_csv(manifest_path, ["image", "sha256", "label_sha256", "classes", "box_count", "width", "height", "phash",
                              "origin", "capture", "license", "notes"], [{
        "image": rec["path"].name, "sha256": rec["sha256"], "label_sha256": rec["label_sha256"],
        "classes": "|".join(NAMES[i] for i, n in enumerate(rec["counts"]) if n), "box_count": sum(rec["counts"]),
        "width": rec["width"], "height": rec["height"], "phash": f"{rec['phash']:016x}",
        **{k: rec["source"][k] for k in ("origin", "capture", "license", "notes")},
    } for rec in kept])
    write_csv(out_dir / "distribution.csv", ["class_id", "class", "images", "boxes"],
              distribution_rows([rec["counts"] for rec in kept], NAMES))
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump({
        "path": str(out_dir).replace("\\", "/"), "train": "images/test", "val": "images/test", "test": "images/test",
        "names": NAMES}, sort_keys=False), encoding="utf-8")
    freeze = {
        "name": out_dir.name,
        "role": "out-of-distribution test; report separately from test_main_v1, never train or tune on it",
        "label_edits_after_freeze": False,
        "source_rows": len(images),
        "unique_sha256": len(records),
        "duplicate_rows_collapsed": duplicate_rows,
        "leakage_reference": str(args.phash_manifest.resolve()).replace("\\", "/"),
        "leakage_reference_sha256": sha256_file(args.phash_manifest),
        "leakage_rule": f"imagehash.phash hash_size=8, hamming <= {args.max_hamming} vs V6.2E all splits and within set",
        "near_duplicates_dropped": len(dropped),
        "images": len(kept),
        "capture": dict(Counter(rec["source"]["capture"] for rec in kept)),
        "sources_csv_sha256": sha256_file(args.sources_csv),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (out_dir / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
