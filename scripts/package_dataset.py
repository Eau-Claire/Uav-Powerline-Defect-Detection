#!/usr/bin/env python
"""Pack a built dataset (datasets/<name>) into one self-contained tar for other machines and upload it to Camber.

    python scripts/package_dataset.py datasets/v6p5_v5 [--out packages] [--upload stash://.../datasets/v6p5_v5]

Images are stored as real files (symlinks followed); the ultralytics caches (*.npy, labels/*.cache) are left out.
data.yaml inside the tar keeps the build path; notebooks/train_local_v6p5.ipynb rewrites `path` after extracting.
Writes <name>.tar, <name>.tar.sha256 and <name>.package.json (file counts, labels_sha256 from the manifest).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("dataset", type=Path)
    p.add_argument("--out", type=Path, default=Path("packages"))
    p.add_argument("--upload", help="stash:// folder to upload the three files to (needs CAMBER_API_KEY)")
    a = p.parse_args()
    src, name = a.dataset.resolve(), a.dataset.name
    a.out.mkdir(parents=True, exist_ok=True)
    tar_path = a.out / f"{name}.tar"
    counts = {}
    with tarfile.open(tar_path, "w", dereference=True) as tar:
        for f in sorted(src.rglob("*")):
            rel = f.relative_to(src)
            if f.is_dir() or f.suffix in (".npy", ".cache"):
                continue
            tar.add(f, arcname=f"{name}/{rel.as_posix()}", recursive=False)
            key = "/".join(rel.parts[:2]) if rel.parts[0] in ("images", "labels") else rel.parts[0]
            counts[key] = counts.get(key, 0) + 1
    h = hashlib.sha256()
    with tar_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    (a.out / f"{name}.tar.sha256").write_text(f"{h.hexdigest()}  {tar_path.name}\n")
    manifest = json.loads((src / "manifest.json").read_text())
    info = {"dataset": name, "tar": tar_path.name, "bytes": tar_path.stat().st_size, "sha256": h.hexdigest(),
            "files": counts, "labels_sha256": manifest["labels_sha256"], "built_utc": manifest["created_utc"]}
    (a.out / f"{name}.package.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    if a.upload:
        camber = os.environ.get("CAMBER_BIN", str(Path.home() / ".camber/bin/camber"))
        for f in (tar_path, a.out / f"{name}.tar.sha256", a.out / f"{name}.package.json"):
            subprocess.run([camber, "stash", "cp", str(f), f"{a.upload.rstrip('/')}/{f.name}"], check=True,
                           stdout=subprocess.DEVNULL)
            print("uploaded", f.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
