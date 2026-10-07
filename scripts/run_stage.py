#!/usr/bin/env python
"""Run one pipeline stage (resume/reuse by default).

    python scripts/run_stage.py parent
    python scripts/run_stage.py external          # parent..external_freeze in order
    python scripts/run_stage.py compose
    python scripts/run_stage.py yolo11n_ablation
    python scripts/run_stage.py synthetic --force
    python scripts/run_stage.py dedup --dry-run

Exit codes: 0 complete/reused/dry-run, 2 needs human review, 1 error.
"""
from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from yolov11sdi.pipeline import STAGES, run_external_pipeline, run_stage


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=[*STAGES, "external"])
    ap.add_argument("--config", default=None, help="config name in configs/ (default: per stage)")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--resume", action="store_true", default=True,
                      help="reuse complete stages and resume partial ones (default)")
    mode.add_argument("--force", action="store_true", help="rebuild even if a matching result exists")
    ap.add_argument("--dry-run", action="store_true", help="show what would run; change nothing")
    args = ap.parse_args()

    try:
        if args.stage == "external":
            outcomes = run_external_pipeline(args.config, force=args.force, dry_run=args.dry_run)
        else:
            outcomes = [run_stage(args.stage, args.config, force=args.force, dry_run=args.dry_run)]
    except KeyboardInterrupt:
        print("interrupted — state saved; rerun the same command to resume", file=sys.stderr)
        return 130
    except Exception as e:  # noqa: BLE001
        print(f"\nFAILED: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    for o in outcomes:
        print(o)
    return 2 if any(o.status == "needs_review" for o in outcomes) else 0


if __name__ == "__main__":
    sys.exit(main())
