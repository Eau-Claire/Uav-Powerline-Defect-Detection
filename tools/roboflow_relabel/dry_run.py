"""Run a small, non-destructive Roboflow proposal batch against localhost."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from core import default_output_dir, environment_defaults, run_workflow_batch, scan_candidates


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cache/base_extract/curated_v6"))
    parser.add_argument("--split", choices=("train", "val"), default="train")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--min-confidence", type=float, default=0.5)
    parser.add_argument("--min-area-ratio", type=float, default=1.1)
    parser.add_argument("--max-area-ratio", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ROBOFLOW_API_KEY is missing from .env")
    defaults = environment_defaults()
    candidates = scan_candidates(args.dataset_root, args.split, class_id=1)
    selected = candidates[: args.limit] if args.limit > 0 else candidates
    print(f"Found {len(candidates):,} candidates; running {len(selected)} images", flush=True)
    output_path = run_workflow_batch(
        selected,
        args.output_dir or default_output_dir(),
        defaults["api_url"],
        api_key,
        defaults["workspace_name"],
        defaults["workflow_id"],
        {item.strip() for item in defaults["allowed_classes"].split(",")},
        args.min_confidence,
        args.min_area_ratio,
        args.max_area_ratio,
        resume=args.resume,
        workers=args.workers,
    )
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
