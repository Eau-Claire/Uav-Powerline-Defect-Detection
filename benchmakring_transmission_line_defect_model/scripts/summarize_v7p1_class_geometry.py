from pathlib import Path
from collections import defaultdict
import csv


ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "artifacts/dataset_audit/v7p1_171126_imported_annotation_audit"
DATA = ROOT / "benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls_v2_171126"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]


def q(values, p):
    values = sorted(values)
    return values[round((len(values)-1)*p)] if values else None


inventory = {r["image"]: r for r in csv.DictReader((AUDIT/"image_inventory.csv").open(encoding="utf-8"))}
stats = defaultdict(lambda: defaultdict(list))
border = defaultdict(int)
for image, meta in inventory.items():
    group, source = meta["group"], meta["source"]
    width, height = int(meta["width"]), int(meta["height"])
    label = DATA/"labels/train"/(Path(image).stem + ".txt")
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        if len(line.split()) != 5:
            continue
        cls, x, y, w, h = map(float, line.split())
        key = (group, source, int(cls))
        stats[key]["w_px"].append(w*width); stats[key]["h_px"].append(h*height)
        stats[key]["area"].append(w*h); stats[key]["aspect"].append(w/h)
        border[key] += int(x-w/2 <= .002 or y-h/2 <= .002 or x+w/2 >= .998 or y+h/2 >= .998)

rows = []
for (group, source, cls), values in sorted(stats.items()):
    n = len(values["area"])
    rows.append({"group": group, "source": source, "class": NAMES[cls], "boxes": n,
                 "w_p05": q(values["w_px"], .05), "w_p50": q(values["w_px"], .5), "w_p95": q(values["w_px"], .95),
                 "h_p05": q(values["h_px"], .05), "h_p50": q(values["h_px"], .5), "h_p95": q(values["h_px"], .95),
                 "area_p05": q(values["area"], .05), "area_p50": q(values["area"], .5), "area_p95": q(values["area"], .95),
                 "aspect_p50": q(values["aspect"], .5), "border_boxes": border[(group, source, cls)],
                 "border_pct": border[(group, source, cls)]/n*100})

with (AUDIT/"class_geometry.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
for r in rows:
    print(f"{r['group']:8} {r['class']:20} n={r['boxes']:5} area50={r['area_p50']:.5f} "
          f"w50={r['w_p50']:.1f} h50={r['h_p50']:.1f} border={r['border_pct']:.1f}% source={r['source']}")
