#!/usr/bin/env python
"""Sync local artifacts with Camber Stash (the local mirror uses the same hierarchy).

    python scripts/sync_camber.py push --stage external_freeze     # upload a stage's outputs
    python scripts/sync_camber.py push --run runs/yolo11n_640/<run> --remote-root stash://...
    python scripts/sync_camber.py pull stash://bunpmc/projects/yolov11sdi/datasets/v6p2e/v6p2e_23562a45b343/v6p2e.freeze.json
    add --dry-run to print without transferring.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401

from yolov11sdi.camber import CamberClient
from yolov11sdi.config import load_config
from yolov11sdi.environment import load_camber_api_key
from yolov11sdi.paths import ProjectPaths
from yolov11sdi.state import StateStore
from yolov11sdi.training.ultralytics_runner import SYNC_FILES


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--stage", help="upload every output of state/<stage>.json that has a remote path")
    p.add_argument("--run", help="run directory to sync (checkpoints, run_state)")
    p.add_argument("--remote-root", help="remote root for --run (default: run_state.json remote_root)")
    q = sub.add_parser("pull")
    q.add_argument("remote", nargs="+")
    q.add_argument("--bulk", action="store_true", help="store under data/cache/remote instead of artifacts mirror")
    for s in (p, q):
        s.add_argument("--dry-run", action="store_true")
        s.add_argument("--config", default="yolo11n_ablation")
    args = ap.parse_args()

    cfg = load_config(args.config)
    paths = ProjectPaths.detect()
    load_camber_api_key(paths.env)
    cam = CamberClient(paths, cfg.get("camber.stash_prefix"), cfg.runtime("camber_bin"), dry_run=args.dry_run)
    if not args.dry_run and not cam.available():
        print("Camber unavailable: install the CLI and export CAMBER_API_KEY (or `camber login`).", file=sys.stderr)
        return 1

    if args.cmd == "pull":
        for r in args.remote:
            print(cam.fetch(r, bulk=args.bulk))
        return 0

    pairs = []
    if args.stage:
        st = StateStore(paths.state).load(args.stage)
        if st is None:
            print(f"no state for {args.stage}", file=sys.stderr)
            return 1
        pairs += [(paths.resolve(o["path"]), o["remote"]) for o in st.outputs if o.get("remote")]
    if args.run:
        import json

        run_dir = Path(args.run)
        root = args.remote_root or json.loads((run_dir / "run_state.json").read_text()).get("remote_root")
        if not root:
            print("no --remote-root and run_state.json has none", file=sys.stderr)
            return 1
        pairs += [(run_dir / n, f"{root}/{Path(n).name}") for n in SYNC_FILES if (run_dir / n).exists()]
    for local, remote in pairs:
        print(f"{paths.rel(local)} -> {remote}")
    done = cam.put_many(pairs, required=False)
    print(f"uploaded {len(done)}/{len(pairs)}")
    return 0 if len(done) == len(pairs) else 1


if __name__ == "__main__":
    sys.exit(main())
