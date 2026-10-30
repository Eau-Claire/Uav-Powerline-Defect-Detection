from pathlib import Path
import json
from PIL import Image, ImageDraw, ImageOps
from core import Box, iou, read_yolo_boxes

root=Path(__file__).resolve().parents[2]
out=root/'artifacts/roboflow_relabel/cross_v3/preview'
out.mkdir(parents=True,exist_ok=True)
records=[]
with (out.parent/'yolo.jsonl').open(encoding='utf-8') as f:
    for line in f:
        try: records.append(json.loads(line))
        except json.JSONDecodeError: pass
selected=[]; seen={}
for r in records:
    path=Path(r['image']); label=path.parents[2]/'labels/train'/f'{path.stem}.txt'
    with Image.open(path) as im: size=im.size
    old=read_yolo_boxes(label,size)
    additions=[]
    for raw in r['predictions']:
        b=Box(**raw)
        if any(g.class_id==b.class_id and iou(b,g)>=0.3 for g in old): continue
        if any(g.class_id!=b.class_id and {g.class_id,b.class_id}!={0,1} and iou(b,g)>=0.5 for g in old): continue
        additions.append(b)
    if not additions: continue
    cls=additions[0].class_name
    if seen.get(cls,0)>=3: continue
    seen[cls]=seen.get(cls,0)+1
    selected.append((path,old,additions))
    if len(selected)>=12: break
canvas=Image.new('RGB',(1600,((len(selected)+1)//2)*350),'#202020')
for index,(path,old,added) in enumerate(selected):
    im=Image.open(path).convert('RGB'); d=ImageDraw.Draw(im)
    for b in old: d.rectangle((b.x1,b.y1,b.x2,b.y2),outline='deepskyblue',width=3)
    for b in added:
        d.rectangle((b.x1,b.y1,b.x2,b.y2),outline='lime',width=5)
        d.text((max(0,b.x1),max(0,b.y1)),f'ADD {b.class_name} {b.confidence:.2f}',fill='lime',stroke_width=1,stroke_fill='black')
    im.save(out/f'{index+1:02d}.jpg')
    tile=ImageOps.contain(im,(790,310)); x=(index%2)*800; y=(index//2)*350
    canvas.paste(tile,(x,y+30))
    ImageDraw.Draw(canvas).text((x+5,y+5),f'{index+1}: '+', '.join(f'{b.class_name} {b.confidence:.2f}' for b in added),fill='lime')
canvas.save(out/'contact_sheet.jpg')
(out/'samples.json').write_text(json.dumps([{'image':str(p),'added':[b.__dict__ for b in a]} for p,o,a in selected],indent=2),encoding='utf-8')
print('processed',len(records),'preview_images',len(selected),'classes',seen)
