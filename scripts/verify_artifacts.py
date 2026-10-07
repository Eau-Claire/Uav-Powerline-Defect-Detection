#!/usr/bin/env python
"""Verify local artifacts against what freezes and state files claim.

* every state output: exists, size, sha256 (deep);
* every freeze in artifacts/freezes: referenced local mirror files match sha256;
* V6.5 archives: image/label pairing, class ids 0..5, normalized finite boxes, no TEST.

    python scripts/verify_artifacts.py
"""
from __future__ import annotations

import json
import sys

import _bootstrap  # noqa: F401

from yolov11sdi.archive_utils import verify_yolo_tar
from yolov11sdi.config import load_config
from yolov11sdi.hashing import sha256_file
from yolov11sdi.paths import ProjectPaths
from yolov11sdi.state import StateStore


def main() -> int:
    cfg = load_config("yolo11n_ablation")
    paths = ProjectPaths.detect()
    prefix = cfg.get("camber.stash_prefix")
    bad = 0

    for name, st in StateStore(paths.state).all().items():
        for o in st.outputs:
            p = paths.resolve(o["path"])
            if not p.exists():
                print(f"MISSING  {name}:{o['name']} {o['path']}")
                bad += 1
            elif o.get("sha256") and p.is_file() and sha256_file(p) != o["sha256"]:
                print(f"CHANGED  {name}:{o['name']} {o['path']}")
                bad += 1
            else:
                print(f"ok       {name}:{o['name']}")

    for fz in sorted(paths.freezes.glob("*.json")):
        freeze = json.loads(fz.read_text())
        arts = freeze.get("artifacts") or {}
        flat = {}
        for k, v in arts.items():
            if isinstance(v, dict) and "remote" in v:
                flat[k] = v
            elif isinstance(v, dict):
                flat.update({f"{k}.{kk}": vv for kk, vv in v.items() if isinstance(vv, dict) and "remote" in vv})
        for k, v in flat.items():
            local = paths.mirror_for(v["remote"], prefix)
            if not local.exists():
                print(f"absent   {fz.name}:{k} (not mirrored locally)")
                continue
            if v.get("sha256") and sha256_file(local) != v["sha256"]:
                print(f"MISMATCH {fz.name}:{k} {paths.rel(local)}")
                bad += 1
                continue
            if local.name.endswith(".tar.gz") and ("synthetic" in k or "direct" in k):
                rep = verify_yolo_tar(local, len(cfg.names))
                if not rep.ok:
                    print(f"INVALID  {fz.name}:{k} {rep.label_problems[:3]} {rep.missing_labels[:3]}")
                    bad += 1
                    continue
            print(f"ok       {fz.name}:{k}")
    print(f"\n{bad} problem(s)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
