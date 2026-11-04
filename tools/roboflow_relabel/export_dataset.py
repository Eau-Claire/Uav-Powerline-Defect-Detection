"""Build a complete YOLO dataset with reviewed Roboflow class-0 additions."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import yaml

from core import export_staged_labels, load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)
    return destination


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {output}")
    if not (source / "images").is_dir() or not (source / "labels").is_dir():
        raise SystemExit(f"Invalid YOLO dataset root: {source}")

    records = load_jsonl(args.proposals)
    if not records:
        raise SystemExit(f"No proposal records found: {args.proposals}")

    output.mkdir(parents=True)
    shutil.copytree(source / "images", output / "images", copy_function=hardlink_or_copy)
    shutil.copytree(source / "labels", output / "labels", copy_function=shutil.copy2)
    summary = export_staged_labels(records, output / "labels" / "train")

    data = yaml.safe_load((source / "data.yaml").read_text(encoding="utf-8"))
    data["path"] = str(output)
    (output / "data.yaml").write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    manifest = {
        "source_dataset": str(source),
        "proposal_file": str(args.proposals.resolve()),
        "taxonomy_contract": {"0": "insulator", "1": "insulator_broken"},
        **summary,
    }
    (output / "ROBOFLOW_RELABEL_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
