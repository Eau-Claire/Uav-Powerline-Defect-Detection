import json,sys
from pathlib import Path
from ultralytics import YOLO
m=YOLO('legacy/notebooks/v6p5/yolo11n.pt')
for s in ['val','train']:
    imgs=Path(f'data/label_audit/coco_person/{s}_fo_images.txt').read_text().split('\n')
    with open(f'data/label_audit/coco_person/pred_{s}.jsonl','w') as fh:
        for i in range(0,len(imgs),16):
            for p,r in zip(imgs[i:i+16],m.predict(imgs[i:i+16],conf=0.3,classes=[0],device='cpu',verbose=False)):
                fh.write(json.dumps({'image':Path(p).name,'xywhn':r.boxes.xywhn.tolist(),'conf':r.boxes.conf.tolist()})+'\n')
    print(s,'done',flush=True)
