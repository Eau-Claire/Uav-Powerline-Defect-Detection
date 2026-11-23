"""Freeze Finding Nemo V5 after the human audit of the 215 model-proposed insulator boxes.

Output: data/test_main_v5r_human_verified/ (the old test_main_v5_* folders are left untouched)
  labels/test/*.txt   V5 parent labels (v1 + v3 log) + accepted / adjusted proposals
  images/test/*       hard links (or copies) of the test_main_v1 images
  dataset.yaml, manifest.csv (image + label SHA-256), review_decisions.csv (copy), freeze.json

Refuses to freeze unless every proposal has a decision in {accept, reject, adjust} with a reviewer,
notes for reject/adjust, and a valid adjusted box for adjust. `unsure` must be resolved first
(or pass --unsure-as-reject, which is recorded in freeze.json).

  .venv/bin/python tools/fn_v5_review/freeze_v5_reviewed.py --review artifacts/fn_v5_review/review.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_review import DATA, NAMES, ROOT, V1, V3_LOG, V5_ADD, parse_box, read_csv, sha256, v5_parent_labels  # noqa: E402

OUT_NAME = 'test_main_v5r_human_verified'
DECISIONS = {'accept', 'reject', 'adjust'}


def check(rows, unsure_as_reject: bool):
    errors = []
    expected = {f'P{n:03d}' for n in range(1, len(read_csv(V5_ADD)) + 1)}
    got = [r['box_id'] for r in rows]
    if set(got) != expected or len(got) != len(expected):
        errors.append(f'review has {len(got)} rows / {len(set(got))} unique ids, expected {len(expected)}')
    for r in rows:
        d = r['decision'].strip().lower()
        if d == 'unsure' and unsure_as_reject:
            continue
        if d not in DECISIONS:
            errors.append(f'{r["box_id"]}: decision {r["decision"]!r}')
        if not r['reviewer'].strip():
            errors.append(f'{r["box_id"]}: reviewer missing')
        if d in ('reject', 'adjust') and not r['notes'].strip():
            errors.append(f'{r["box_id"]}: notes required for {d}')
        if d == 'adjust':
            try:
                x1, y1, x2, y2 = (float(v) for v in r['adjusted_xyxy_px'].split(','))
                w, h = float(r['img_w']), float(r['img_h'])
                assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
            except Exception:
                errors.append(f'{r["box_id"]}: adjusted_xyxy_px must be x1,y1,x2,y2 inside the image')
    return errors


def link(src: Path, dst: Path):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--unsure-as-reject', action='store_true')
    ap.add_argument('--out', type=Path, default=DATA / OUT_NAME)
    a = ap.parse_args()

    rows = read_csv(a.review)
    errors = check(rows, a.unsure_as_reject)
    if errors:
        print('NOT FROZEN, fix review.csv first:\n  ' + '\n  '.join(errors[:50]))
        sys.exit(1)
    if a.out.exists():
        print(f'{a.out} already exists; a frozen set is never overwritten. Remove it deliberately or use --out.')
        sys.exit(1)

    labels = v5_parent_labels()
    decisions = Counter()
    for r in rows:
        d = r['decision'].strip().lower()
        d = 'reject' if d == 'unsure' else d
        decisions[d] += 1
        if d == 'accept':
            labels[r['label_file']].append(parse_box(f"0 {r['x_center']} {r['y_center']} {r['width']} {r['height']}"))
        elif d == 'adjust':
            x1, y1, x2, y2 = (float(v) for v in r['adjusted_xyxy_px'].split(','))
            w, h = float(r['img_w']), float(r['img_h'])
            labels[r['label_file']].append((0, (x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h))

    (a.out / 'labels' / 'test').mkdir(parents=True)
    (a.out / 'images' / 'test').mkdir(parents=True)
    manifest, dist, background = [], Counter(), 0
    for img in sorted((V1 / 'images' / 'test').iterdir()):
        name = img.stem + '.txt'
        boxes = labels[name]
        lp = a.out / 'labels' / 'test' / name
        lp.write_text(''.join(f'{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}\n' for c, x, y, w, h in boxes))
        link(img, a.out / 'images' / 'test' / img.name)
        background += not boxes
        dist.update(NAMES[b[0]] for b in boxes)
        manifest.append({'image': img.name, 'image_sha256': sha256(img), 'label_sha256': sha256(lp), 'boxes': len(boxes)})
    with (a.out / 'manifest.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0]))
        w.writeheader()
        w.writerows(manifest)
    shutil.copy2(a.review, a.out / 'review_decisions.csv')
    (a.out / 'dataset.yaml').write_text(
        'path: .\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n'
        + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES)))
    freeze = {
        'name': OUT_NAME,
        'role': 'primary test set (Finding Nemo V5, human-verified)',
        'created_at': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'parent': 'test_main_v1 + test_main_v3 correction log (= V5 without the 215 proposals)',
        'proposal_audit': {
            'proposals': len(rows), 'decisions': dict(decisions),
            'reviewers': sorted({r['reviewer'].strip() for r in rows if r['reviewer'].strip()}),
            'unsure_as_reject': a.unsure_as_reject,
            'note': 'Proposals came from 4/4 consensus of evaluated YOLO checkpoints; every box was accepted, '
                    'adjusted or rejected by a human reviewer who did not see model confidence.',
        },
        'images': len(manifest), 'boxes': sum(dist.values()), 'background_images': background,
        'distribution': dict(dist),
        'sources_sha256': {
            'insulator_additions.csv': sha256(V5_ADD), 'v3_correction_log.csv': sha256(V3_LOG),
            'review_decisions.csv': sha256(a.review),
        },
        'manifest_sha256': sha256(a.out / 'manifest.csv'),
    }
    (a.out / 'freeze.json').write_text(json.dumps(freeze, indent=2))
    print(json.dumps({k: freeze[k] for k in ('images', 'boxes', 'background_images', 'distribution', 'manifest_sha256')}
                     | {'decisions': dict(decisions)}, indent=1))


if __name__ == '__main__':
    main()
