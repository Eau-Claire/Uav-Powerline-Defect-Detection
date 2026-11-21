from pathlib import Path
import csv, json, math
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
B=ROOT/"artifacts/review_v7p1_seed85_relabel_181126"
rows=[r for r in csv.DictReader((B/"manifest.csv").open(encoding="utf-8-sig")) if "Rule F" in r["reasons"]]
pending=[r for r in rows if not (B/"decisions"/(Path(r["image"]).stem+".json")).is_file()]
out=B/"contact_sheets_tiny_pending"; out.mkdir(exist_ok=True)
thumb_w,thumb_h,caption=360,270,42
for page in range(math.ceil(len(pending)/9)):
    batch=pending[page*9:(page+1)*9]
    sheet=Image.new("RGB",(thumb_w*3,(thumb_h+caption)*3),"white")
    draw=ImageDraw.Draw(sheet)
    for i,r in enumerate(batch):
        im=Image.open(B/"images"/r["image"]).convert("RGB"); im.thumbnail((thumb_w,thumb_h))
        x=(i%3)*thumb_w; y=(i//3)*(thumb_h+caption)
        sheet.paste(im,(x+(thumb_w-im.width)//2,y+(thumb_h-im.height)//2))
        draw.text((x+4,y+thumb_h+2),f"{r['index']} {Path(r['image']).stem[:24]}",fill="black")
    sheet.save(out/f"tiny_pending_{page+1:02d}.jpg",quality=92)
print(f"pending={len(pending)} sheets={math.ceil(len(pending)/9)} out={out}")
