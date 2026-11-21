from pathlib import Path
from collections import Counter
import csv
import argparse


ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=Path, default=ROOT / "benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls_v2_171126")
parser.add_argument("--output", type=Path, default=ROOT / "artifacts/dataset_audit/v7p1_171126_imported_annotation_audit/cross_class_overlaps.csv")
args = parser.parse_args()
DATA, OUT = args.dataset.resolve(), args.output.resolve()


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    aa = (a[2]-a[0]) * (a[3]-a[1]); bb = (b[2]-b[0]) * (b[3]-b[1])
    return inter / max(aa + bb - inter, 1e-12)


rows = []
for label in sorted((DATA/"labels/train").glob("pp3_*.txt")):
    boxes = []
    for line_no, line in enumerate(label.read_text(encoding="utf-8-sig").splitlines(), 1):
        c, x, y, w, h = map(float, line.split())
        boxes.append((int(c), (x-w/2, y-h/2, x+w/2, y+h/2), line_no))
    for i, (ca, a, la) in enumerate(boxes):
        for cb, b, lb in boxes[i+1:]:
            if ca == cb:
                continue
            overlap = iou(a, b)
            if overlap >= .5:
                rows.append({"label": label.name, "class_a": int(ca), "line_a": la,
                             "class_b": int(cb), "line_b": lb, "iou": round(overlap, 8)})

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["label","class_a","line_a","class_b","line_b","iou"])
    writer.writeheader(); writer.writerows(rows)
print(f"overlaps_iou50={len(rows)} images={len({r['label'] for r in rows})}")
print(Counter((r["class_a"], r["class_b"]) for r in rows))
print(f"exact_iou999={sum(r['iou'] >= .999 for r in rows)}")
