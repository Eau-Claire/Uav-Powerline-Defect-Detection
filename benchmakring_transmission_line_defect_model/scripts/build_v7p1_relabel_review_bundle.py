from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / "benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls_v2_171126"
V3 = ROOT / "benchmakring_transmission_line_defect_model/data/finding_nemo_6cls_v3_seed85_repaired_181126"
AUDIT = ROOT / "artifacts/dataset_audit/v7p1_171126_imported_annotation_audit"
OUT = ROOT / "artifacts/review_v7p1_seed85_relabel_181126"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def link_or_copy(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


if OUT.exists():
    raise FileExistsError(f"Refusing to overwrite review bundle: {OUT}")

reasons = defaultdict(set)
with (V3/"repair_report/changes.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        reasons[row["image"]].add(f"{row['action']}:{row['class']}:{row['reason']}")
with (AUDIT/"class_conflicts.csv").open(encoding="utf-8-sig") as f:
    for row in csv.DictReader(f):
        reasons[row["image"]].add(f"judge_class_conflict:{row['gt_class']}->{row['pred_class']}:iou={row['iou']}")

for d in ("images", "labels_before", "labels_seed85", "reviewed_labels", "decisions"):
    (OUT/d).mkdir(parents=True, exist_ok=True)

rows = []
for index, image_name in enumerate(sorted(reasons), 1):
    image = V3/"images/train"/image_name
    if not image.is_file():
        raise FileNotFoundError(image)
    label_name = f"{image.stem}.txt"
    link_or_copy(image, OUT/"images"/image.name)
    shutil.copy2(V2/"labels/train"/label_name, OUT/"labels_before"/label_name)
    shutil.copy2(V3/"labels/train"/label_name, OUT/"labels_seed85"/label_name)
    rows.append({"index": index, "image": image.name, "label": label_name,
                 "reason_count": len(reasons[image_name]), "reasons": " | ".join(sorted(reasons[image_name]))})

with (OUT/"manifest.csv").open("w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
(OUT/"README.md").write_text(
    "# V7.1 seed85 relabel review bundle\n\n"
    f"Queue: {len(rows)} unique train images.\n\n"
    "- `labels_before`: labels from v2 before repair\n"
    "- `labels_seed85`: conservative repaired labels proposed for v3\n"
    "- `reviewed_labels`: reviewer-approved/edited labels\n"
    "- `decisions`: one JSON decision per reviewed image\n"
    "- `manifest.csv`: queue and reasons\n\n"
    "Run: `.venv-roboflow\\Scripts\\python.exe -m streamlit run tools/roboflow_relabel/review_v7p1_relabel.py`\n",
    encoding="utf-8",
)
print(json.dumps({"review_images": len(rows), "bundle": str(OUT)}, indent=2))
