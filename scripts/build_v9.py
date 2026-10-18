#!/usr/bin/env python
"""Build dataset v6p5_v9 from v6p5_v8: hold out a foreign_object test split (rule N, user decision 2026-10-16).

val_clean_v8 has only 57 FO boxes once the leaked images are removed (AP50 CI95 +-0.14), too few to see an FO change.
Train FO labels were fully reviewed for v6 (data/label_audit/fo_review_v6), so they are the cleanest FO labels we have.
Rule N  fo_test   train images are clustered by near-duplicates (dHash 64-bit, Hamming <= 6, the leak threshold);
                  whole clusters holding FO are drawn at random (seed) until >= TARGET_FO FO boxes; they move, with all
                  their labels, to `test_fo` - no near-duplicate of a test_fo image stays in train.
Only models trained on v9 (or later) may be scored on test_fo: v5/v6/v8-trained models have seen these images.
Splits: train (minus test_fo), test_fo, val_clean_v8 (+ val, val_clean, val_clean_v6 unchanged, leaked_val.txt copied).
Usage: python scripts/build_v9.py --v8 datasets/v6p5_v8 --out datasets/v6p5_v9
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand", "person"]
TARGET_FO, SEED, HAMMING = 150, 20261016, 6


def link(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def dhash(p: Path) -> np.ndarray:
    a = np.asarray(Image.open(p).convert("L").resize((9, 8), Image.LANCZOS), dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--v8", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    tr = sorted(f.name for f in (a.v8 / "images/train").glob("*.jpg"))
    lab = {n: [l for l in (a.v8 / "labels/train" / (Path(n).stem + ".txt")).read_text().splitlines() if l.strip()] for n in tr}
    n_fo = {n: sum(l.split()[0] == "4" for l in v) for n, v in lab.items()}
    H = np.packbits(np.array([dhash(a.v8 / "images/train" / n) for n in tr]), axis=1)   # N x 8 bytes
    pc = np.unpackbits(np.arange(256, dtype=np.uint8)[:, None], axis=1).sum(1)          # popcount table
    parent = list(range(len(tr)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i in range(len(tr)):   # O(N^2) in chunks of rows, ~17.5k images
        d = pc[np.bitwise_xor(H[i], H[i + 1:])].sum(1)
        for j in np.flatnonzero(d <= HAMMING) + i + 1: parent[find(i)] = find(j)
    cl = defaultdict(list)
    for i, n in enumerate(tr): cl[find(i)].append(n)
    fo_cl = sorted((c for c, v in cl.items() if any(n_fo[n] for n in v)), key=lambda c: min(cl[c]))
    rng = np.random.default_rng(SEED); test, got = set(), 0
    for c in rng.permutation(fo_cl):
        if got >= TARGET_FO: break
        test |= set(cl[c]); got += sum(n_fo[n] for n in cl[c])

    stats = defaultdict(Counter)
    a.out.mkdir(parents=True, exist_ok=False)
    for split, src, names in (("train", "train", [n for n in tr if n not in test]), ("test_fo", "train", sorted(test)),
                              *[(s, s, None) for s in ("val", "val_clean", "val_clean_v6", "val_clean_v8")]):
        (a.out / "images" / split).mkdir(parents=True); (a.out / "labels" / split).mkdir(parents=True)
        for n in names if names is not None else sorted(f.name for f in (a.v8 / "images" / src).glob("*.jpg")):
            stem = Path(n).stem
            shutil.copy2(a.v8 / "labels" / src / (stem + ".txt"), a.out / "labels" / split / (stem + ".txt"))
            link(a.v8 / "images" / src / n, a.out / "images" / split / n)
            npy = a.v8 / "images" / src / (stem + ".npy")
            if split == "train" and npy.exists(): link(npy, a.out / "images" / split / (stem + ".npy"))
            stats[split]["images"] += 1
            for l in (a.out / "labels" / split / (stem + ".txt")).read_text().splitlines():
                if l.strip(): stats[split][f"box:{NAMES[int(l.split()[0])]}"] += 1
    for f in ("leaked_val.txt", "val_clean_v8_clusters.json"):
        if (a.v8 / f).exists(): shutil.copy2(a.v8 / f, a.out / f)
    for f, v in (("data.yaml", "val_clean_v8"), ("data_test_fo.yaml", "test_fo"), ("data_val_clean_v8.yaml", "val_clean_v8")):
        (a.out / f).write_text(yaml.safe_dump({"path": str(a.out.resolve()), "train": "images/train", "val": f"images/{v}",
                                               "names": dict(enumerate(NAMES))}, sort_keys=False))
    h = hashlib.sha256()
    for f in sorted((a.out / "labels").rglob("*.txt")):
        h.update(f.relative_to(a.out).as_posix().encode() + f.read_bytes().replace(b"\r\n", b"\n"))
    manifest = {"created_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "source": "v6p5_v8",
                "source_labels_sha256": json.loads((a.v8 / "manifest.json").read_text())["labels_sha256"],
                "labels_sha256": h.hexdigest(), "test_fo": {"images": len(test), "fo_boxes": got, "seed": SEED,
                "clusters_with_fo": len(fo_cl), "hamming": HAMMING},
                "conventions_2026-10-16": {"insulator_broken": "1 box per damaged shed", "broken_strand": "tight around the visible fraying (+~5%)",
                                           "insulator_flashover": "the whole burnt shed", "drone": "not foreign_object"},
                "stats": {k: dict(sorted(v.items())) for k, v in stats.items()}}
    (a.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ("test_fo", "labels_sha256")}, indent=1))
    print(json.dumps({k: manifest["stats"][k] for k in ("train", "test_fo")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
