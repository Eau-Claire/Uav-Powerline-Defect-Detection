"""Build the human audit kit for the 215 model-proposed insulator boxes of Finding Nemo V5.

Existing ground truth = test_main_v1 labels + the test_main_v3 correction log (the V5 parent).
Each proposal from test_main_v5_full_consensus_assisted/insulator_additions.csv gets:
  crops/<box_id>.jpg   full image (all existing GT + the proposal) | zoom on the proposal
  sheets/sheet_NN.jpg  contact sheet, 6 proposals per page
  review.csv           one row per proposal; decision/reviewer/notes left empty for the reviewer
  index.html           keyboard review tool (a/r/j/u, saves in the browser, exports review.csv)

Model confidence and voters are kept in review.csv for the record but are NOT shown on the crops,
sheets or review page, so the reviewer judges the image, not the model score.

  .venv/bin/python tools/fn_v5_review/build_review.py --out artifacts/fn_v5_review
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'benchmakring_transmission_line_defect_model' / 'data'
V1 = DATA / 'test_main_v1'
V3_LOG = DATA / 'test_main_v3_model_aligned_exploratory' / 'correction_log.csv'
V5_ADD = DATA / 'test_main_v5_full_consensus_assisted' / 'insulator_additions.csv'
NAMES = ['insulator', 'insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']
COLORS = ['#22c55e', '#ef4444', '#f97316', '#a855f7', '#3b82f6', '#eab308']   # existing GT per class
PROPOSAL = '#ff00ff'
FULL_W, ZOOM_W, PANEL_H = 900, 450, 600


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_csv(p: Path):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def parse_box(s: str):
    c, x, y, w, h = s.split()[:5]
    return int(c), float(x), float(y), float(w), float(h)


def v5_parent_labels() -> dict[str, list]:
    """test_main_v1 labels with the test_main_v3 corrections applied (by original line index)."""
    labels = {}
    for p in sorted((V1 / 'labels' / 'test').glob('*.txt')):
        labels[p.name] = [parse_box(l) for l in p.read_text().splitlines() if l.strip()]
    edits = {}
    for r in read_csv(V3_LOG):
        edits.setdefault(r['label'], []).append(r)
    for name, rows in edits.items():
        boxes = list(labels[name])
        for r in rows:
            i = int(r['line_index_zero_based'])
            if boxes[i] is None or parse_box(r['old_label'])[0] != boxes[i][0]:
                raise SystemExit(f'v3 log does not match v1 labels: {name} line {i}')
            if r['action'].startswith('replace'):
                boxes[i] = parse_box(r['new_label'])
            elif r['action'].startswith('remove'):
                boxes[i] = None
            else:
                raise SystemExit(f'unknown v3 action {r["action"]}')
        labels[name] = [b for b in boxes if b is not None]
    return labels


def xyxy(b, w, h):
    _, x, y, bw, bh = b
    return (x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def font(size):
    for f in ('DejaVuSans-Bold.ttf', '/usr/share/fonts/TTF/DejaVuSans-Bold.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(img: Image.Image, gt, prop, box_id: str) -> Image.Image:
    w, h = img.size
    full = img.copy()
    d = ImageDraw.Draw(full)
    lw = max(2, round(max(w, h) / 400))
    for b in gt:
        d.rectangle(xyxy(b, w, h), outline=COLORS[b[0]], width=lw)
    px = xyxy(prop, w, h)
    d.rectangle(px, outline=PROPOSAL, width=lw * 2)
    scale = min(FULL_W / w, PANEL_H / h)
    full = full.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    bw, bh = px[2] - px[0], px[3] - px[1]
    pad = 0.35 * max(bw, bh) + 20
    crop_box = (max(0, px[0] - pad), max(0, px[1] - pad), min(w, px[2] + pad), min(h, px[3] + pad))
    zoom = img.crop(tuple(round(v) for v in crop_box))
    zd = ImageDraw.Draw(zoom)
    for b in gt:
        x1, y1, x2, y2 = xyxy(b, w, h)
        zd.rectangle((x1 - crop_box[0], y1 - crop_box[1], x2 - crop_box[0], y2 - crop_box[1]),
                     outline=COLORS[b[0]], width=lw)
    zd.rectangle((px[0] - crop_box[0], px[1] - crop_box[1], px[2] - crop_box[0], px[3] - crop_box[1]),
                 outline=PROPOSAL, width=lw * 2)
    zs = min(ZOOM_W / zoom.width, PANEL_H / zoom.height)
    zoom = zoom.resize((max(1, round(zoom.width * zs)), max(1, round(zoom.height * zs))), Image.LANCZOS)

    out = Image.new('RGB', (FULL_W + ZOOM_W + 10, PANEL_H + 40), 'white')
    out.paste(full, (0, 40))
    out.paste(zoom, (FULL_W + 10, 40))
    ImageDraw.Draw(out).text((8, 8), f'{box_id}   magenta = proposed insulator   (left: full image, right: zoom)',
                             fill='black', font=font(20))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=Path, default=ROOT / 'artifacts' / 'fn_v5_review')
    a = ap.parse_args()
    (a.out / 'crops').mkdir(parents=True, exist_ok=True)
    (a.out / 'sheets').mkdir(parents=True, exist_ok=True)

    labels = v5_parent_labels()
    adds = read_csv(V5_ADD)
    rows, per_image = [], {}
    for n, r in enumerate(adds, 1):
        box_id = f'P{n:03d}'
        name = r['label']
        stem = name[:-4]
        img_path = V1 / 'images' / 'test' / f'{stem}.jpg'
        if not img_path.exists():
            img_path = next((V1 / 'images' / 'test').glob(stem + '.*'))
        prop = parse_box(r['new_label'])
        if prop[0] != 0:
            raise SystemExit(f'{box_id}: proposal is not insulator')
        with Image.open(img_path) as im:
            img = im.convert('RGB')
        w, h = img.size
        gt = labels[name]
        px = xyxy(prop, w, h)
        ious = [(iou(px, xyxy(b, w, h)), NAMES[b[0]]) for b in gt]
        best = max(ious, default=(0.0, ''))
        render(img, gt, prop, box_id).save(a.out / 'crops' / f'{box_id}.jpg', quality=88)
        per_image.setdefault(stem, []).append(box_id)
        rows.append({
            'box_id': box_id, 'image_id': stem, 'label_file': name, 'class': 'insulator',
            'x_center': f'{prop[1]:.8f}', 'y_center': f'{prop[2]:.8f}', 'width': f'{prop[3]:.8f}', 'height': f'{prop[4]:.8f}',
            'x1_px': round(px[0]), 'y1_px': round(px[1]), 'x2_px': round(px[2]), 'y2_px': round(px[3]),
            'img_w': w, 'img_h': h, 'existing_gt_boxes': len(gt),
            'existing_insulator_boxes': sum(b[0] == 0 for b in gt),
            'max_iou_existing_gt': f'{best[0]:.3f}', 'max_iou_class': best[1],
            'voters': r['voters'], 'mean_confidence': r['mean_confidence'],
            'sheet': '', 'decision': '', 'adjusted_xyxy_px': '', 'reviewer': '', 'notes': '',
        })

    per_sheet = 6
    for s in range(0, len(rows), per_sheet):
        chunk = rows[s:s + per_sheet]
        tiles = [Image.open(a.out / 'crops' / f'{r["box_id"]}.jpg') for r in chunk]
        tw, th = tiles[0].size
        sheet = Image.new('RGB', (tw * 2 + 30, th * 3 + 40), 'white')
        for i, t in enumerate(tiles):
            sheet.paste(t, (10 + (i % 2) * (tw + 10), 10 + (i // 2) * (th + 10)))
        sheet_name = f'sheet_{s // per_sheet + 1:02d}.jpg'
        sheet.save(a.out / 'sheets' / sheet_name, quality=82)
        for r in chunk:
            r['sheet'] = sheet_name

    fields = list(rows[0])
    with (a.out / 'review.csv').open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)

    public = [{k: r[k] for k in ('box_id', 'image_id', 'existing_insulator_boxes', 'max_iou_existing_gt', 'max_iou_class',
                         'img_w', 'img_h', 'x1_px', 'y1_px', 'x2_px', 'y2_px')}
              for r in rows]
    html = (Path(__file__).with_name('review_template.html').read_text(encoding='utf-8')
            .replace('__ITEMS__', json.dumps(public)).replace('__FIELDS__', json.dumps(fields))
            .replace('__ROWS__', json.dumps(rows)))
    (a.out / 'index.html').write_text(html, encoding='utf-8')

    manifest = {
        'proposals': len(rows), 'images': len(per_image),
        'images_with_multiple_proposals': {k: v for k, v in per_image.items() if len(v) > 1},
        'sources': {str(p.relative_to(ROOT)): sha256(p) for p in (V5_ADD, V3_LOG)},
        'v1_labels_used': len(labels),
        'note': 'Existing GT = test_main_v1 + test_main_v3 correction log; model confidence hidden from reviewers.',
    }
    (a.out / 'kit_manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: manifest[k] for k in ('proposals', 'images', 'images_with_multiple_proposals')}, indent=1))


if __name__ == '__main__':
    main()
