"""Build a stratified random clean-test subset of the frozen V6.2E TEST and a blind label-audit package.

Two subcommands:

  sample    Select images by stratified random sampling (no class bonus, no model output),
            freeze the manifest, and export an anonymized audit package.
  finalize  After the blind audit, apply the auditors' verdicts and build the clean test dataset.

Strata are assigned by the rarest class present in each image, where rarity is measured on the
source pool itself. Every stratum gets the same floor; the rest is allocated proportionally.
Each manifest row records its inclusion probability so population-weighted estimates remain possible.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
BACKGROUND = "background"
AUDIT_STATUSES = {"ok", "fixed", "drop"}


@dataclass(frozen=True)
class Record:
    image: Path
    label: Path
    relative_image: str
    sha256: str
    label_sha256: str
    class_counts: tuple[int, ...]

    @property
    def class_ids(self) -> tuple[int, ...]:
        return tuple(i for i, n in enumerate(self.class_counts) if n > 0)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def resolve_split(dataset_yaml: Path, split: str) -> tuple[Path, Path, dict[int, str]]:
    cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    root = Path(cfg.get("path", dataset_yaml.parent))
    if not root.is_absolute():
        root = (dataset_yaml.parent / root).resolve()
    split_value = cfg.get(split)
    if not split_value or isinstance(split_value, list):
        raise ValueError(f"Expected one path for split={split!r}, got {split_value!r}")
    images = (root / split_value).resolve()
    parts = list(Path(split_value).parts)
    try:
        parts[parts.index("images")] = "labels"
    except ValueError as exc:
        raise ValueError(f"Split path must contain an 'images' component: {split_value}") from exc
    labels = (root / Path(*parts)).resolve()
    names_raw = cfg.get("names", {})
    names = {int(k): str(v) for k, v in names_raw.items()} if isinstance(names_raw, dict) else {
        i: str(v) for i, v in enumerate(names_raw)
    }
    return images, labels, names


def parse_label_text(text: str, n_classes: int, origin: str) -> tuple[int, ...]:
    counts = [0] * n_classes
    for line_number, line in enumerate(text.splitlines(), start=1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO row at {origin}:{line_number}: {line!r}")
        class_id = int(float(fields[0]))
        if class_id < 0 or class_id >= n_classes:
            raise ValueError(f"Class {class_id} outside 0..{n_classes - 1} at {origin}:{line_number}")
        coords = [float(value) for value in fields[1:]]
        if not all(0.0 <= value <= 1.0 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
            raise ValueError(f"Invalid box geometry at {origin}:{line_number}: {line!r}")
        counts[class_id] += 1
    return tuple(counts)


def build_records(images_dir: Path, labels_dir: Path, n_classes: int) -> tuple[list[Record], dict]:
    image_paths = sorted(p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    if not image_paths:
        raise FileNotFoundError(f"No images found under {images_dir}")
    by_sha: dict[str, Record] = {}
    duplicate_rows = 0
    conflicting_label_shas: list[str] = []
    for image_path in image_paths:
        relative_image = image_path.relative_to(images_dir)
        label_path = labels_dir / relative_image.with_suffix(".txt")
        text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
        record = Record(
            image=image_path,
            label=label_path,
            relative_image=relative_image.as_posix(),
            sha256=sha256_file(image_path),
            label_sha256=sha256_text(text),
            class_counts=parse_label_text(text, n_classes, str(label_path)),
        )
        # Exact duplicate rows are collapsed so one image cannot be sampled or audited twice.
        kept = by_sha.get(record.sha256)
        if kept is None:
            by_sha[record.sha256] = record
            continue
        duplicate_rows += 1
        if kept.label_sha256 != record.label_sha256:
            conflicting_label_shas.append(record.sha256)
    stats = {
        "source_rows": len(image_paths),
        "unique_sha256": len(by_sha),
        "duplicate_rows_collapsed": duplicate_rows,
        "duplicate_rows_with_conflicting_labels": sorted(set(conflicting_label_shas)),
    }
    return sorted(by_sha.values(), key=lambda r: r.relative_image), stats


def assign_strata(records: list[Record], names: dict[int, str]) -> tuple[dict[str, str], list[str]]:
    # Rarity comes from image frequency in the pool itself, not from any model or curation result.
    image_freq = Counter(c for record in records for c in record.class_ids)
    rarity_order = sorted(image_freq, key=lambda c: (image_freq[c], c))
    strata: dict[str, str] = {}
    for record in records:
        present = set(record.class_ids)
        rarest = next((c for c in rarity_order if c in present), None)
        strata[record.sha256] = BACKGROUND if rarest is None else names[rarest]
    order = [names[c] for c in rarity_order]
    if BACKGROUND in strata.values():
        order.append(BACKGROUND)
    return strata, order


def allocate(pool_sizes: dict[str, int], target: int, floor: int) -> dict[str, int]:
    total = sum(pool_sizes.values())
    if target > total:
        raise ValueError(f"target-images={target} exceeds the {total} unique source images")
    alloc = {s: min(floor, n) for s, n in pool_sizes.items()}
    if sum(alloc.values()) > target:
        raise ValueError(f"Floors need {sum(alloc.values())} images, exceeding target-images={target}")
    # Largest-remainder proportional allocation of what is left, capped by stratum size.
    while sum(alloc.values()) < target:
        remaining = target - sum(alloc.values())
        open_strata = {s: n for s, n in pool_sizes.items() if alloc[s] < n}
        open_total = sum(open_strata.values())
        quotas = {s: remaining * n / open_total for s, n in open_strata.items()}
        extra = {s: min(int(q), pool_sizes[s] - alloc[s]) for s, q in quotas.items()}
        leftover = remaining - sum(extra.values())
        for s in sorted(quotas, key=lambda s: (-(quotas[s] - int(quotas[s])), s)):
            if leftover <= 0:
                break
            if alloc[s] + extra[s] < pool_sizes[s]:
                extra[s] += 1
                leftover -= 1
        for s, n in extra.items():
            alloc[s] += n
    return alloc


def copy_file(source: Path, destination: Path) -> None:
    # Always copy: hardlinks would let audit edits silently rewrite the frozen TEST labels.
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def distribution_rows(counts_list: list[tuple[int, ...]], names: dict[int, str]) -> list[dict]:
    return [{
        "class_id": class_id,
        "class": class_name,
        "images": sum(counts[class_id] > 0 for counts in counts_list),
        "boxes": sum(counts[class_id] for counts in counts_list),
    } for class_id, class_name in sorted(names.items())]


def cmd_sample(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    if (out_dir / "freeze.json").exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir / 'freeze.json'} exists; the subset is frozen. Pass --overwrite to rebuild.")
    images_dir, labels_dir, names = resolve_split(args.dataset_yaml.resolve(), args.split)
    # Conflicting duplicates (re-annotated identical video frames) keep the first path in sort order,
    # i.e. the canonical file rather than a __dupN copy; the blind audit re-checks that label anyway.
    records, source_stats = build_records(images_dir, labels_dir, len(names))
    strata, stratum_order = assign_strata(records, names)

    pools: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        pools[strata[record.sha256]].append(record)
    pool_sizes = {s: len(pools[s]) for s in stratum_order}
    alloc = allocate(pool_sizes, args.target_images, args.min_per_stratum)

    rng = random.Random(args.seed)
    selected: list[tuple[Record, str, float]] = []
    for stratum in stratum_order:
        # Uniform simple random sample inside each stratum; pools are pre-sorted by path for determinism.
        picks = rng.sample(pools[stratum], alloc[stratum])
        selected.extend((record, stratum, alloc[stratum] / pool_sizes[stratum]) for record in picks)
    selected.sort(key=lambda item: item[0].relative_image)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"
    write_csv(manifest_path, [
        "image", "sha256", "label_sha256", "stratum", "inclusion_probability", "classes", "box_count",
    ], [{
        "image": record.relative_image,
        "sha256": record.sha256,
        "label_sha256": record.label_sha256,
        "stratum": stratum,
        "inclusion_probability": f"{prob:.6f}",
        "classes": "|".join(names[i] for i in record.class_ids),
        "box_count": sum(record.class_counts),
    } for record, stratum, prob in selected])
    write_csv(out_dir / "strata.csv", ["stratum", "pool_images", "sampled_images", "inclusion_probability"], [{
        "stratum": s,
        "pool_images": pool_sizes[s],
        "sampled_images": alloc[s],
        "inclusion_probability": f"{alloc[s] / pool_sizes[s]:.6f}",
    } for s in stratum_order])
    write_csv(out_dir / "distribution_pre_audit.csv", ["class_id", "class", "images", "boxes"],
              distribution_rows([record.class_counts for record, _, _ in selected], names))

    # Blind audit package: random order, anonymous ids, no stratum, source name or label provenance.
    audit_dir = out_dir / "audit"
    if audit_dir.exists():
        shutil.rmtree(audit_dir)
    audit_order = [record for record, _, _ in selected]
    random.Random(args.seed + 1).shuffle(audit_order)
    key_rows, sheet_rows = [], []
    for index, record in enumerate(audit_order, start=1):
        audit_id = f"A{index:04d}"
        copy_file(record.image, audit_dir / "images" / f"{audit_id}{record.image.suffix.lower()}")
        label_target = audit_dir / "labels" / f"{audit_id}.txt"
        if record.label.exists():
            copy_file(record.label, label_target)
        else:
            label_target.parent.mkdir(parents=True, exist_ok=True)
            label_target.write_text("", encoding="utf-8")
        key_rows.append({"audit_id": audit_id, "sha256": record.sha256, "image": record.relative_image,
                         "image_suffix": record.image.suffix.lower(), "label_sha256": record.label_sha256})
        sheet_rows.append({"audit_id": audit_id, "status": "", "auditor": "", "notes": ""})
    (audit_dir / "classes.txt").write_text("".join(f"{names[i]}\n" for i in sorted(names)), encoding="utf-8")
    write_csv(audit_dir / "audit_sheet.csv", ["audit_id", "status", "auditor", "notes"], sheet_rows)
    # The key lives outside audit/ so the package can be handed over without it.
    write_csv(out_dir / "audit_key.csv", ["audit_id", "sha256", "image", "image_suffix", "label_sha256"], key_rows)

    freeze = {
        "name": args.out_dir.name,
        "selection_policy": "stratified_simple_random_by_rarest_class_present",
        "class_bonus": False,
        "model_outputs_used": False,
        "source_dataset_yaml": str(args.dataset_yaml.resolve()).replace("\\", "/"),
        "split": args.split,
        "seed": args.seed,
        "target_images": args.target_images,
        "min_per_stratum": args.min_per_stratum,
        "stratum_order": stratum_order,
        **source_stats,
        "selected_images": len(selected),
        "manifest_sha256": sha256_file(manifest_path),
        "audit_key_sha256": sha256_file(out_dir / "audit_key.csv"),
    }
    (out_dir / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(freeze, indent=2))


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def cmd_technical(args: argparse.Namespace) -> None:
    # No-human fallback: fix only annotation-tool artifacts. Deliberately applies none of the curation
    # rules (no flips, no tiny-box removal, no SAM tightening, no added or dropped objects).
    audit_dir = args.out_dir.resolve() / "audit"
    sheet_path = audit_dir / "audit_sheet.csv"
    rows = list(csv.DictReader(sheet_path.open(encoding="utf-8")))
    if any(row["status"].strip() for row in rows) and not args.overwrite:
        raise ValueError("audit_sheet.csv already has statuses; pass --overwrite to replace them")
    removed_total = 0
    for row in rows:
        label_path = audit_dir / "labels" / f"{row['audit_id']}.txt"
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.split()]
        kept: list[tuple[int, list[float], str]] = []
        removed = 0
        for line in lines:
            fields = line.split()
            class_id, box = int(float(fields[0])), [float(v) for v in fields[1:]]
            if any(k_cls == class_id and box_iou(k_box, box) >= args.duplicate_iou for k_cls, k_box, _ in kept):
                removed += 1
                continue
            kept.append((class_id, box, line))
        if removed:
            label_path.write_text("".join(f"{line}\n" for _, _, line in kept), encoding="utf-8")
        removed_total += removed
        row["status"] = "fixed" if removed else "ok"
        row["auditor"] = "script:technical"
        row["notes"] = f"removed {removed} duplicate box(es) IoU>={args.duplicate_iou}" if removed else ""
    write_csv(sheet_path, ["audit_id", "status", "auditor", "notes"], rows)
    print(json.dumps({"images": len(rows), "images_fixed": sum(r["status"] == "fixed" for r in rows),
                      "duplicate_boxes_removed": removed_total}, indent=2))


def cmd_finalize(args: argparse.Namespace) -> None:
    out_dir = args.out_dir.resolve()
    freeze = json.loads((out_dir / "freeze.json").read_text(encoding="utf-8"))
    if sha256_file(out_dir / "manifest.csv") != freeze["manifest_sha256"]:
        raise ValueError("manifest.csv no longer matches freeze.json")
    if sha256_file(out_dir / "audit_key.csv") != freeze["audit_key_sha256"]:
        raise ValueError("audit_key.csv no longer matches freeze.json")
    names_yaml = yaml.safe_load(Path(freeze["source_dataset_yaml"]).read_text(encoding="utf-8"))["names"]
    names = {int(k): str(v) for k, v in names_yaml.items()} if isinstance(names_yaml, dict) else {
        i: str(v) for i, v in enumerate(names_yaml)
    }
    audit_dir = out_dir / "audit"
    key = {row["audit_id"]: row for row in csv.DictReader((out_dir / "audit_key.csv").open(encoding="utf-8"))}
    sheet = {row["audit_id"]: row for row in csv.DictReader((audit_dir / "audit_sheet.csv").open(encoding="utf-8"))}
    if set(sheet) != set(key):
        raise ValueError("audit_sheet.csv ids do not match audit_key.csv")
    pending = [i for i, row in sheet.items() if row["status"].strip().lower() not in AUDIT_STATUSES]
    if pending:
        raise ValueError(f"{len(pending)} rows lack a status in {sorted(AUDIT_STATUSES)}, e.g. {pending[:5]}")

    final_dir = out_dir / "final"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    status_counts: Counter[str] = Counter()
    kept_counts: list[tuple[int, ...]] = []
    rows = []
    for audit_id in sorted(key):
        entry, status = key[audit_id], sheet[audit_id]["status"].strip().lower()
        status_counts[status] += 1
        label_path = audit_dir / "labels" / f"{audit_id}.txt"
        text = label_path.read_text(encoding="utf-8")
        changed = sha256_text(text) != entry["label_sha256"]
        if status == "ok" and changed:
            raise ValueError(f"{audit_id} is marked ok but its label file was edited; mark it fixed")
        rows.append({"audit_id": audit_id, "sha256": entry["sha256"], "status": status,
                     "label_changed": int(changed), "notes": sheet[audit_id]["notes"]})
        if status == "drop":
            continue
        counts = parse_label_text(text, len(names), str(label_path))
        kept_counts.append(counts)
        stem = entry["sha256"]
        copy_file(audit_dir / "images" / f"{audit_id}{entry['image_suffix']}",
                  final_dir / "images" / "test" / f"{stem}{entry['image_suffix']}")
        copy_file(label_path, final_dir / "labels" / "test" / f"{stem}.txt")

    write_csv(final_dir / "audit_outcome.csv", ["audit_id", "sha256", "status", "label_changed", "notes"], rows)
    write_csv(final_dir / "distribution.csv", ["class_id", "class", "images", "boxes"],
              distribution_rows(kept_counts, names))
    dataset_yaml = {
        "path": str(final_dir).replace("\\", "/"),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": names,
    }
    (final_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    final_freeze = {
        "name": f"{freeze['name']}_final",
        "parent_manifest_sha256": freeze["manifest_sha256"],
        "status_counts": dict(sorted(status_counts.items())),
        "images": len(kept_counts),
        "audit_outcome_sha256": sha256_file(final_dir / "audit_outcome.csv"),
    }
    (final_dir / "freeze.json").write_text(json.dumps(final_freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(final_freeze, indent=2))


def cmd_full(args: argparse.Namespace) -> None:
    # Whole source split, deduplicated by SHA-256, minus images listed in an exclusion review.
    # No sampling and no label edits.
    out_dir = args.out_dir.resolve()
    if (out_dir / "freeze.json").exists() and not args.overwrite:
        raise FileExistsError(f"{out_dir / 'freeze.json'} exists; pass --overwrite to rebuild.")
    images_dir, labels_dir, names = resolve_split(args.dataset_yaml.resolve(), args.split)
    records, source_stats = build_records(images_dir, labels_dir, len(names))
    review = list(csv.DictReader(args.exclude_csv.open(encoding="utf-8")))
    excluded_shas = {Path(row["image"]).stem for row in review if row["verdict"] != "no_person"}
    unknown = excluded_shas - {record.sha256 for record in records}
    if unknown:
        raise ValueError(f"{len(unknown)} excluded images are not in the source split, e.g. {sorted(unknown)[:3]}")
    kept = [record for record in records if record.sha256 not in excluded_shas]

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for record in kept:
        copy_file(record.image, out_dir / "images" / "test" / f"{record.sha256}{record.image.suffix.lower()}")
        label_target = out_dir / "labels" / "test" / f"{record.sha256}.txt"
        if record.label.exists():
            copy_file(record.label, label_target)
        else:
            label_target.write_text("", encoding="utf-8")
    manifest_path = out_dir / "manifest.csv"
    write_csv(manifest_path, ["image", "sha256", "label_sha256", "classes", "box_count"], [{
        "image": record.relative_image,
        "sha256": record.sha256,
        "label_sha256": record.label_sha256,
        "classes": "|".join(names[i] for i in record.class_ids),
        "box_count": sum(record.class_counts),
    } for record in kept])
    write_csv(out_dir / "distribution.csv", ["class_id", "class", "images", "boxes"],
              distribution_rows([record.class_counts for record in kept], names))
    dataset_yaml = {
        "path": str(out_dir).replace("\\", "/"),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": names,
    }
    (out_dir / "dataset.yaml").write_text(yaml.safe_dump(dataset_yaml, sort_keys=False), encoding="utf-8")
    freeze = {
        "name": out_dir.name,
        "selection_policy": "full_source_split_dedup_minus_exclusions",
        "label_edits": False,
        "human_audit": False,
        "source_dataset_yaml": str(args.dataset_yaml.resolve()).replace("\\", "/"),
        "split": args.split,
        **source_stats,
        "exclusion_review": str(args.exclude_csv.resolve()).replace("\\", "/"),
        "exclusion_review_sha256": sha256_file(args.exclude_csv),
        "exclusion_rule": "foreign_object box whose main content is a person (verdict person_only or mixed)",
        "excluded_images": len(excluded_shas),
        "images": len(kept),
        "manifest_sha256": sha256_file(manifest_path),
    }
    (out_dir / "freeze.json").write_text(json.dumps(freeze, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in freeze.items() if k != "duplicate_rows_with_conflicting_labels"}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample", help="Stratified random selection and blind audit export.")
    sample.add_argument("--dataset-yaml", type=Path, required=True, help="YOLO YAML of the frozen source split.")
    sample.add_argument("--split", default="test", choices=("train", "val", "test"))
    sample.add_argument("--out-dir", type=Path, default=Path("data/clean_test_v2"))
    sample.add_argument("--target-images", type=int, default=600)
    sample.add_argument("--min-per-stratum", type=int, default=60)
    sample.add_argument("--seed", type=int, default=20261027)
    sample.add_argument("--overwrite", action="store_true", help="Rebuild even if freeze.json already exists.")
    sample.set_defaults(func=cmd_sample)

    full = sub.add_parser("full", help="Whole split, deduplicated, minus reviewed exclusions; no sampling.")
    full.add_argument("--dataset-yaml", type=Path, required=True, help="YOLO YAML of the frozen source split.")
    full.add_argument("--split", default="test", choices=("train", "val", "test"))
    full.add_argument("--exclude-csv", type=Path, required=True, help="Review CSV with image and verdict columns.")
    full.add_argument("--out-dir", type=Path, default=Path("data/test_main_v1"))
    full.add_argument("--overwrite", action="store_true", help="Rebuild even if freeze.json already exists.")
    full.set_defaults(func=cmd_full)

    technical = sub.add_parser("technical", help="No-human fallback: remove duplicate same-class boxes only.")
    technical.add_argument("--out-dir", type=Path, default=Path("data/clean_test_v2"))
    technical.add_argument("--duplicate-iou", type=float, default=0.95)
    technical.add_argument("--overwrite", action="store_true", help="Replace statuses already in audit_sheet.csv.")
    technical.set_defaults(func=cmd_technical)

    finalize = sub.add_parser("finalize", help="Apply audit verdicts and build the clean test dataset.")
    finalize.add_argument("--out-dir", type=Path, default=Path("data/clean_test_v2"))
    finalize.set_defaults(func=cmd_finalize)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
