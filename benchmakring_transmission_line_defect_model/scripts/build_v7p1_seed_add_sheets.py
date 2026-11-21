from pathlib import Path
import csv, json, math
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2]; B=ROOT/"artifacts/review_v7p1_seed85_relabel_181126"
rows=list(csv.DictReader((B/"manifest.csv").open(encoding="utf-8-sig")))
candidate_rows=list(csv.DictReader((ROOT/"artifacts/dataset_audit/v7p1_171126_imported_annotation_audit/potential_missing_boxes.csv").open(encoding="utf-8-sig")))
candidates={}
for c in candidate_rows: candidates.setdefault(c["image"],[]).append(c)
items=[]
for r in rows:
    stem=Path(r["image"]).stem
    if "seed85 conf" in r["reasons"] and not (B/"decisions"/f"{stem}.json").is_file(): items.append(r)
colors=["cyan","red","magenta","lime","orange","yellow"]
out=B/"contact_sheets_seed_add_pending"; out.mkdir(exist_ok=True)
for page in range(math.ceil(len(items)/9)):
    batch=items[page*9:(page+1)*9]; sheet=Image.new("RGB",(1200,990),"white"); sd=ImageDraw.Draw(sheet)
    for i,r in enumerate(batch):
        im=Image.open(B/"images"/r["image"]).convert("RGB"); w,h=im.size; d=ImageDraw.Draw(im)
        for line in (B/"labels_seed85"/r["label"]).read_text(encoding="utf-8-sig").splitlines():
            p=line.split();
            if len(p)!=5: continue
            c=int(float(p[0])); x,y,bw,bh=map(float,p[1:]); xy=((x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h)
            d.rectangle(xy,outline=colors[c],width=max(2,round(min(w,h)/180))); d.text((xy[0],max(0,xy[1]-14)),str(c),fill=colors[c])
        for c in candidates.get(r["image"],[]):
            xy=tuple(float(c[k]) for k in ("x1","y1","x2","y2"))
            d.rectangle(xy,outline="white",width=max(5,round(min(w,h)/90)))
            d.text((xy[0]+3,xy[1]+3),f"ADD {c['pred_class']} {float(c['confidence']):.2f}",fill="white",stroke_width=2,stroke_fill="black")
        im.thumbnail((390,280)); x=(i%3)*400; y=(i//3)*330; sheet.paste(im,(x+(390-im.width)//2,y+(280-im.height)//2))
        sd.text((x+4,y+285),f"{r['index']} {Path(r['image']).stem[:28]}",fill="black")
        sd.text((x+4,y+303),r['reasons'][:58],fill="black")
    sheet.save(out/f"seed_add_{page+1:02d}.jpg",quality=94)
print(f"pending={len(items)} sheets={math.ceil(len(items)/9)} out={out}")
