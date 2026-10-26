#!/usr/bin/env python
"""Upload benchmark YOLO run folders to Camber Stash.

Examples:
    python scripts/upload_yolo_runs_to_camber.py --dry-run
    python scripts/upload_yolo_runs_to_camber.py
    python scripts/upload_yolo_runs_to_camber.py --run raw_yolo11n curated_yolo11n
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from yolov11sdi.camber import CamberClient
from yolov11sdi.environment import load_camber_api_key
from yolov11sdi.paths import ProjectPaths


RUN_NAME_PATTERN = re.compile(r"^(?:raw|curated)_yolo.+$")
DEFAULT_REMOTE_ROOT = "stash://bunpmc/projects"


def load_api_key(env_file: Path) -> None:
    if os.environ.get("CAMBER_API_KEY"):
        return
    if not env_file.is_file():
        raise FileNotFoundError(f"Camber API key environment file not found: {env_file}")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip().removeprefix("export ").strip() == "CAMBER_API_KEY":
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if value:
                os.environ["CAMBER_API_KEY"] = value
                return
    raise ValueError(f"CAMBER_API_KEY was not found in {env_file}")


def collect_run_files(run_dir: Path) -> list[Path]:
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlinks are not supported in run folders: {path}")
        if path.is_file():
            resolved = path.resolve()
            if not resolved.is_relative_to(run_dir.resolve()):
                raise ValueError(f"Run file resolves outside its folder: {path}")
            files.append(path)
    if not files:
        raise FileNotFoundError(f"No files found in run folder: {run_dir}")
    return files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "runs",
        help="Directory containing raw_yolo*/curated_yolo* folders",
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help=f"Camber Stash destination root (default: {DEFAULT_REMOTE_ROOT})",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=PROJECT_ROOT / ".env",
        help="File containing CAMBER_API_KEY (default: project .env)",
    )
    parser.add_argument(
        "--run",
        nargs="+",
        help="Specific run folder name(s); by default upload all raw_yolo*/curated_yolo* folders",
    )
    parser.add_argument("--dry-run", action="store_true", help="List upload paths without transferring files")
    args = parser.parse_args()

    runs_dir = args.runs_dir.resolve()
    if not runs_dir.is_dir():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")
    if not args.remote_root.startswith("stash://"):
        raise ValueError(f"Remote root must start with 'stash://': {args.remote_root}")
    remote_root = args.remote_root.rstrip("/")

    if args.run:
        run_names = args.run
        invalid_names = [name for name in run_names if not RUN_NAME_PATTERN.fullmatch(name)]
        if invalid_names:
            raise ValueError(f"Unsupported run folder name(s): {', '.join(invalid_names)}")
    else:
        run_names = sorted(
            path.name for path in runs_dir.iterdir()
            if path.is_dir() and RUN_NAME_PATTERN.fullmatch(path.name)
        )
    if not run_names:
        raise FileNotFoundError(f"No raw_yolo*/curated_yolo* run folders found in {runs_dir}")

    pairs: list[tuple[Path, str]] = []
    for run_name in run_names:
        run_dir = (runs_dir / run_name).resolve()
        if run_dir.parent != runs_dir or not run_dir.is_dir():
            raise FileNotFoundError(f"Run folder not found directly under {runs_dir}: {run_name}")
        for local_path in collect_run_files(run_dir):
            relative_path = local_path.relative_to(run_dir).as_posix()
            remote_path = f"{remote_root}/{run_name}/{relative_path}"
            pairs.append((local_path, remote_path))

    total_bytes = sum(path.stat().st_size for path, _ in pairs)
    print(f"Runs directory: {runs_dir}")
    print(f"Destination:    {remote_root}")
    print(f"Files:          {len(pairs)}")
    print(f"Total size:     {total_bytes / (1024 ** 3):.2f} GiB")
    for local_path, remote_path in pairs:
        print(f"{local_path} -> {remote_path}")

    paths = ProjectPaths.detect(project_root=PROJECT_ROOT)
    load_api_key(args.env_file.resolve())
    load_camber_api_key(paths.env)
    camber = CamberClient(paths, "stash://bunpmc", dry_run=args.dry_run)
    if not args.dry_run and not camber.available():
        print(
            "Camber is unavailable. Install the Camber CLI and set CAMBER_API_KEY "
            "or authenticate with `camber login`.",
            file=sys.stderr,
        )
        return 1

    for index, (local_path, remote_path) in enumerate(pairs, start=1):
        print(f"[{index}/{len(pairs)}] Uploading {local_path.name} -> {remote_path}")
        camber.put(local_path, remote_path)

    action = "Would upload" if args.dry_run else "Uploaded"
    print(f"{action} {len(pairs)} files ({total_bytes / (1024 ** 3):.2f} GiB).")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
