import json
import random
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps
from ultralytics import YOLO
from core import Box, find_image, read_yolo_boxes, iou, load_jsonl, ROBOFLOW_CLASS_MAP

root = Path('data/cache/base_extract/curated_v6')
out = Path('artifacts/roboflow_relabel/cross_preview')
out.mkdir(parents=True, exist_ok=True)
groups = {i: [] for i in range(6)}
for p in sorted((root/'labels/train').glob('*.txt')):
    ids = {int(line.split()[0]) for line in p.read_text().splitlines() if line.strip()}
    for c in ids:
        if c in groups:
            groups[c].append(p)
rng = random.Random(42)
selected = list(dict.fromkeys(p for group in groups.values() for p in rng.sample(group, min(15,len(group)))))
rf = {Path(r['image_path']).stem:r for r in load_jsonl(Path('artifacts/roboflow_relabel/full_train/proposals.jsonl'))}
model = YOLO('weights/curated_yolo26m/best.pt')
records = []
for index,p in enumerate(selected):
    path = find_image(root/'images/train',p.stem)
    if path is None: continue
    im = Image.open(path).convert('RGB')
    old = read_yolo_boxes(p,im.size)
    result = model.predict(str(path),imgsz=640,conf=0.5,device='cpu',verbose=False)[0]
    added = []
    for coords,c,conf in zip(result.boxes.xyxy.tolist(),result.boxes.cls.tolist(),result.boxes.conf.tolist()):
        b = Box(*coords,int(c),conf,model.names[int(c)])
        if max((iou(b,g) for g in old if g.class_id==b.class_id),default=0)<0.3:
            added.append({'box':b.__dict__,'source':'YOLO26m','class_conflict':any(iou(b,g)>0.5 and g.class_id!=b.class_id for g in old)})
    for prediction in rf.get(p.stem,{}).get('predictions',[]):
        name = (prediction.get('class_name') or '').casefold()
        target = ROBOFLOW_CLASS_MAP.get(name)
        if target is None or (prediction.get('confidence') or 0) < 0.5:
            continue
        mapped = dict(prediction, class_id=target, class_name=model.names[target])
        b = Box(**mapped)
        if max((iou(b,g) for g in old if g.class_id==target),default=0) >= 0.3:
            continue
        if any(a['box']['class_id']==target and iou(b,Box(**a['box']))>=0.5 for a in added):
            continue
        added.append({'box':mapped,'source':'Roboflow','source_class':prediction['class_name'],'class_conflict':any(iou(b,g)>0.5 and g.class_id!=target for g in old)})
    records.append({'image':str(path),'existing_classes':sorted({b.class_id for b in old}),'additions':added})
    if added:
        draw=ImageDraw.Draw(im)
        for b in old: draw.rectangle((b.x1,b.y1,b.x2,b.y2),outline='deepskyblue',width=3)
        for a in added:
            b=Box(**a['box']); color='orange' if a['class_conflict'] else ('lime' if a['source']=='YOLO26m' else 'magenta')
            draw.rectangle((b.x1,b.y1,b.x2,b.y2),outline=color,width=4)
            draw.text((max(0,b.x1),max(0,b.y1)),f'{a["source"]} {b.class_name} {b.confidence:.2f}',fill=color,stroke_width=1,stroke_fill='black')
        im.save(out/f'{index:03d}.jpg')
    print(f'{index+1}/{len(selected)} additions={len(added)}',flush=True)
(out/'proposals.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
previews=list(out.glob('[0-9]*.jpg'))[:8]
canvas=Image.new('RGB',(1400,1000),'white')
for i,p in enumerate(previews):
    tile=ImageOps.contain(Image.open(p),(695,245)); canvas.paste(tile,((i%2)*700,(i//2)*250))
canvas.save(out/'contact_sheet.jpg')
print('DONE',out,flush=True)
