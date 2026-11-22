"""Audit train/val/test leakage with exact (SHA-256) and perceptual (dHash) hashes.

- exact: identical file bytes in two splits
- near:  64-bit difference hash within --max-distance bits (resized, re-encoded or lightly
         edited copies). Perceptual hashes can collide on visually similar but different
         scenes, so every near pair gets a side-by-side preview for manual confirmation.
Optionally reports how many leaked test images fall in a subset manifest (e.g. Finding Nemo V5,
whose file names are the SHA-256 of the image bytes).

  python audit_split_leakage.py --dataset data/finding_nemo_human_reviewed_6cls \
      --subset-manifest data/test_operational_min8_v5_full_consensus_assisted/manifest.csv \
      --out artifacts/dataset_audit/leakage_<name>
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
PAIRS = (('train', 'val'), ('train', 'test'), ('val', 'test'))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--subset-manifest', type=Path, help='CSV with an image column and optional kept column')
    p.add_argument('--max-distance', type=int, default=4, help='dHash Hamming distance for near duplicates')
    p.add_argument('--previews', type=int, default=40)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=8)
    return p.parse_args()


def dhash(image: Image.Image) -> int:
    g = np.asarray(image.convert('L').resize((9, 8), Image.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int(np.packbits(bits).view('>u8')[0])


def hash_file(path: Path):
    data = path.read_bytes()
    with Image.open(path) as im:
        return path.name, hashlib.sha256(data).hexdigest(), dhash(im)


def hash_split(folder: Path, workers: int):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    with ProcessPoolExecutor(workers) as pool:
        rows = list(pool.map(hash_file, files, chunksize=64))
    names = [r[0] for r in rows]
    return names, [r[1] for r in rows], np.array([r[2] for r in rows], dtype=np.uint64)


def near_pairs(ha, hb, max_d, chunk=1024):
    out = []
    for s in range(0, len(ha), chunk):
        d = np.bitwise_count(ha[s:s + chunk, None] ^ hb[None, :])
        i, j = np.nonzero(d <= max_d)
        out += [(s + a, b, int(d[a, b])) for a, b in zip(i, j)]
    return out


def preview(a: Path, b: Path, dst: Path, caption: str):
    ims = [Image.open(x).convert('RGB') for x in (a, b)]
    h = 320
    ims = [im.resize((max(1, im.width * h // im.height), h)) for im in ims]
    canvas = Image.new('RGB', (ims[0].width + ims[1].width + 10, h + 24), 'white')
    canvas.paste(ims[0], (0, 24)); canvas.paste(ims[1], (ims[0].width + 10, 24))
    from PIL import ImageDraw
    ImageDraw.Draw(canvas).text((4, 4), caption, fill='black')
    canvas.save(dst, quality=85)


def main():
    a = parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    splits = {}
    for s in ('train', 'val', 'test'):
        folder = a.dataset / 'images' / s
        print(f'hashing {s} ...', flush=True)
        splits[s] = hash_split(folder, a.workers)
    subset = None
    if a.subset_manifest:
        with a.subset_manifest.open(encoding='utf-8-sig') as f:
            subset = {r['image'] for r in csv.DictReader(f) if r.get('kept', '1') in ('1', 'True', 'true')}

    rows, summary = [], {'dataset': a.dataset.name, 'max_distance': a.max_distance,
                         'images': {s: len(v[0]) for s, v in splits.items()}, 'pairs': {}}
    for x, y in PAIRS:
        nx, sx, hx = splits[x]
        ny, sy, hy = splits[y]
        by_sha = {}
        for i, h in enumerate(sx):
            by_sha.setdefault(h, []).append(i)
        exact = [(i, j) for j, h in enumerate(sy) for i in by_sha.get(h, [])]
        near = near_pairs(hx, hy, a.max_distance)
        exact_set = set(exact)
        near_only = [(i, j, d) for i, j, d in near if (i, j) not in exact_set]
        for i, j in exact:
            rows.append(dict(split_a=x, image_a=nx[i], split_b=y, image_b=ny[j], kind='exact', dhash_distance=0))
        for i, j, d in near_only:
            rows.append(dict(split_a=x, image_a=nx[i], split_b=y, image_b=ny[j], kind='near', dhash_distance=d))
        info = dict(exact_pairs=len(exact), near_pairs=len(near_only),
                    images_b_affected=len({j for _, j in exact} | {j for _, j, _ in near_only}))
        if subset is not None and y == 'test':
            info['subset_images_affected'] = len(({ny[j] for _, j in exact} | {ny[j] for _, j, _ in near_only}) & subset)
            info['subset_size'] = len(subset)
        summary['pairs'][f'{x}-{y}'] = info
        print(x, y, info, flush=True)

    with (a.out / 'leakage_pairs.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['split_a', 'image_a', 'split_b', 'image_b', 'kind', 'dhash_distance'])
        w.writeheader(); w.writerows(rows)
    pv = a.out / 'previews'; pv.mkdir(exist_ok=True)
    for k, r in enumerate(sorted(rows, key=lambda r: r['dhash_distance'])[:a.previews]):
        preview(a.dataset / 'images' / r['split_a'] / r['image_a'], a.dataset / 'images' / r['split_b'] / r['image_b'],
                pv / f'{k:03d}_{r["split_a"]}-{r["split_b"]}_d{r["dhash_distance"]}.jpg',
                f'{r["split_a"]}:{r["image_a"]} | {r["split_b"]}:{r["image_b"]} | {r["kind"]} d={r["dhash_distance"]}')
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('Wrote', a.out)


if __name__ == '__main__':
    main()
