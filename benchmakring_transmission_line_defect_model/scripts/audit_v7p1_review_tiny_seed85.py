from pathlib import Path
import csv

from PIL import Image
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "artifacts/review_v7p1_seed85_relabel_181126"
WEIGHTS = ROOT / "weights/v7p1_seed85_curated_yolo26m_best.pt"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]


def read_boxes(path, size):
    w, h = size; out = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        p = line.split()
        if len(p) != 5: continue
        c, x, y, bw, bh = int(float(p[0])), *map(float, p[1:])
        out.append((c, ((x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h)))
    return out


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0],b[0]), max(a[1],b[1]), min(a[2],b[2]), min(a[3],b[3])
    inter=max(0,ix2-ix1)*max(0,iy2-iy1); aa=(a[2]-a[0])*(a[3]-a[1]); bb=(b[2]-b[0])*(b[3]-b[1])
    return inter/max(aa+bb-inter,1e-12)


manifest = list(csv.DictReader((BUNDLE/"manifest.csv").open(encoding="utf-8-sig")))
items = [r for r in manifest if "Rule F" in r["reasons"]]
model = YOLO(str(WEIGHTS)); rows=[]
for start in range(0, len(items), 8):
    chunk=items[start:start+8]; paths=[BUNDLE/"images"/r["image"] for r in chunk]
    results=model.predict(source=[str(p) for p in paths], imgsz=640, conf=.25, iou=.7, batch=8, device=0, verbose=False)
    for meta,path,result in zip(chunk,paths,results):
        with Image.open(path) as im: size=im.size
        label=BUNDLE/"labels_seed85"/meta["label"]; gt=read_boxes(label,size)
        for xyxy,cls,conf in zip(result.boxes.xyxy.cpu().tolist(),result.boxes.cls.cpu().tolist(),result.boxes.conf.cpu().tolist()):
            max_any=max([iou(xyxy,b) for _,b in gt],default=0)
            if conf>=.60 and max_any<.30:
                rows.append({"image":meta["image"],"class":NAMES[int(cls)],"confidence":round(float(conf),6),
                             "max_iou":round(max_any,6),"x1":round(xyxy[0],2),"y1":round(xyxy[1],2),
                             "x2":round(xyxy[2],2),"y2":round(xyxy[3],2)})
out=BUNDLE/"tiny_seed85_missing_candidates.csv"
with out.open("w",newline="",encoding="utf-8") as f:
    fields=["image","class","confidence","max_iou","x1","y1","x2","y2"]
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"tiny_images={len(items)} candidates={len(rows)} candidate_images={len({r['image'] for r in rows})} output={out}")
