#!/usr/bin/env python
"""Print pipeline status from state/*.json (+ optional read-only Camber listing).

    python scripts/status.py
    python scripts/status.py --remote
    python scripts/status.py --json
"""
from __future__ import annotations

import argparse
import json

import _bootstrap  # noqa: F401

from yolov11sdi.camber import CamberClient
from yolov11sdi.config import load_config
from yolov11sdi.environment import load_camber_api_key
from yolov11sdi.paths import ProjectPaths
from yolov11sdi.reporting import status


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="yolo11n_ablation")
    ap.add_argument("--remote", action="store_true", help="also list v6p5 tags on Camber (read-only)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = ProjectPaths.detect()
    table = status.stage_table(cfg, paths)
    info = status.overview(cfg, paths)
    if args.remote:
        load_camber_api_key(paths.env)
        info["camber"] = status.remote_tags(cfg, CamberClient(paths, cfg.get("camber.stash_prefix"),
                                                               cfg.runtime("camber_bin")))
    if args.json:
        print(json.dumps({"stages": table.to_dict("records"), **info}, indent=2, default=str))
        return
    print(table.to_string(index=False))
    print()
    d = info.pop("disk")
    print(f"disk ({d['path']}): free {d['free_gb']:.1f} / {d['total_gb']:.1f} GB")
    for k, v in info.items():
        print(f"{k:22s} {v}")


if __name__ == "__main__":
    main()
