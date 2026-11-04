"""Render actual cached YOLO additions using the export overlap rules."""
import json
from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont, ImageOps
from core import Box, iou, load_jsonl, read_yolo_boxes

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'benchmakring_transmission_line_defect_model/data/curated_combined'
OUT = ROOT / 'artifacts/roboflow_relabel/curated_combined_serverless/yolo_preview'
NAMES = ['insulator','insulator_broken','insulator_flashover','bird_nest','foreign_object','broken_strand']

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    samples = {}
    for record in load_jsonl(OUT.parent / 'yolo.jsonl'):
        path = Path(record['image'])
        with Image.open(path) as im:
            w,h = im.size
        old = read_yolo_boxes(SOURCE / 'labels/train' / (path.stem + '.txt'), (w,h))
        accepted = []
        for raw in record['predictions']:
            b = Box(**raw)
            b = Box(max(0,b.x1),max(0,b.y1),min(w,b.x2),min(h,b.y2),b.class_id,b.confidence,b.class_name)
            if b.area <= 0: continue
            if any(g.class_id == b.class_id and iou(b,g) >= 0.3 for g in old+accepted): continue
            if any(g.class_id != b.class_id and {g.class_id,b.class_id} != {0,1} and iou(b,g) >= 0.5 for g in old+accepted): continue
            accepted.append(b)
            counts[b.class_name] += 1
        for b in accepted:
            if b.class_id not in samples or b.confidence > samples[b.class_id]['confidence']:
                samples[b.class_id] = {'image':str(path),'confidence':b.confidence,'old':old,'added':accepted}
    font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',20)
    small = ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    canvas = Image.new('RGB',(1600,1530),'#111827')
    draw = ImageDraw.Draw(canvas)
    draw.text((20,12),'YOLO26m additions | CYAN = existing labels | GREEN = new pseudo-labels',font=font,fill='white')
    records = []
    for index,(class_id,sample) in enumerate(sorted(samples.items())):
        with Image.open(sample['image']) as im: im = im.convert('RGB')
        original_size = im.size
        tile = ImageOps.contain(im,(780,425))
        sx,sy = tile.width/im.width,tile.height/im.height
        td = ImageDraw.Draw(tile)
        for group,color,prefix in [(sample['old'],'#00cfff','OLD'),(sample['added'],'#54ff53','NEW')]:
            for b in group:
                x1,y1,x2,y2 = b.x1*sx,b.y1*sy,b.x2*sx,b.y2*sy
                td.rectangle((x1,y1,x2,y2),outline=color,width=3 if prefix=='OLD' else 4)
                label = f'{prefix} {NAMES[b.class_id]}' + (f' {b.confidence:.2f}' if prefix=='NEW' else '')
                tx,ty = max(0,min(x1,tile.width-260)),max(0,y1-20)
                td.text((tx,ty),label,font=small,fill=color,stroke_width=2,stroke_fill='black')
        panel = Image.new('RGB',(800,490),'#111827')
        pd = ImageDraw.Draw(panel)
        pd.text((10,4),f'{index+1}. {NAMES[class_id]} | confidence {sample["confidence"]:.3f}',font=font,fill='white')
        pd.text((10,31),Path(sample['image']).name,font=small,fill='#cbd5e1')
        panel.paste(tile,((800-tile.width)//2,60))
        panel_path = OUT / f'{index+1:02d}_{NAMES[class_id]}.jpg'
        panel.save(panel_path,quality=95)
        canvas.paste(panel,((index%2)*800,55+(index//2)*490))
        records.append({'class':NAMES[class_id],'image':sample['image'],'preview':str(panel_path),'image_size':original_size,'additions':[b.__dict__ for b in sample['added']]})
    canvas.save(OUT/'contact_sheet.jpg',quality=95)
    (OUT/'samples.json').write_text(json.dumps({'selection':'Highest confidence accepted addition per class; not a random accuracy sample','yolo_only_added_boxes':dict(counts),'samples':records},indent=2),encoding='utf-8')
    print(json.dumps({'counts':dict(counts),'preview':str(OUT/'contact_sheet.jpg')},indent=2))

if __name__=='__main__': main()
