"""Resumable cross-class annotation of TRAIN; original labels remain intact."""
import json
import subprocess
import os
import sys
from pathlib import Path
from collections import Counter
from PIL import Image
from ultralytics import YOLO
import yaml
from core import Box, ROBOFLOW_CLASS_MAP, find_image, iou, load_jsonl, read_yolo_boxes

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(os.environ.get('CROSS_DATASET_ROOT', str(ROOT/'data/cache/base_extract/curated_v6')))
OUT = Path(os.environ.get('CROSS_OUTPUT_ROOT', str(ROOT/'artifacts/roboflow_relabel/cross_full_v1')))
OUT.mkdir(parents=True, exist_ok=True)

def main():
    model = YOLO(str(ROOT/'weights/curated_yolo26m/best.pt'))
    expected = ['insulator','insulator_broken','insulator_flashover','bird_nest','foreign_object','broken_strand']
    assert [model.names[i] for i in range(6)] == expected
    labels = sorted((SOURCE/'labels/train').glob('*.txt'))
    cache = OUT/'yolo.jsonl'
    done = {Path(r['image'].replace('\\','/')).stem for r in load_jsonl(cache)}
    with cache.open('a',encoding='utf-8') as handle:
        for index,p in enumerate(labels):
            if p.stem in done: continue
            image = find_image(SOURCE/'images/train',p.stem)
            if image is None: raise RuntimeError(f'Missing image: {p}')
            result = model.predict(str(image),imgsz=640,conf=0.6,device=os.environ.get('CROSS_DEVICE','cpu'),verbose=False)[0]
            predictions = [Box(*xy,int(c),float(s),model.names[int(c)]).__dict__ for xy,c,s in zip(result.boxes.xyxy.tolist(),result.boxes.cls.tolist(),result.boxes.conf.tolist())]
            handle.write(json.dumps({'image':str(image),'predictions':predictions})+'\n'); handle.flush()
            if (index+1)%25==0: print(f'YOLO {index+1}/{len(labels)}',flush=True)
    sdk_python=ROOT/'.venv-roboflow'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    subprocess.run([str(sdk_python if sdk_python.exists() else sys.executable),str(ROOT/'tools/roboflow_relabel/full_rf.py')],check=True,cwd=ROOT)
    rf={Path(r['image_path'].replace('\\','/')).stem:r for r in load_jsonl(OUT/'roboflow/proposals.jsonl')}
    destination=Path(os.environ.get('CROSS_DATASET_OUTPUT',str(ROOT/'data/working/curated_cross_v1')))
    staged=destination/'labels/train'; staged.mkdir(parents=True,exist_ok=True)
    counts=Counter(); conflicts=[]; changes=[]
    for record in load_jsonl(cache):
        image=SOURCE/'images/train'/Path(record['image'].replace('\\','/')).name
        label=SOURCE/'labels/train'/f'{image.stem}.txt'
        with Image.open(image) as im: w,h=im.size
        original=label.read_text(encoding='utf-8'); old=read_yolo_boxes(label,(w,h)); accepted=[]
        candidates=[(Box(**b),'YOLO26m') for b in record['predictions']]
        for b in rf[image.stem]['predictions']:
            target=ROBOFLOW_CLASS_MAP.get((b.get('class_name') or '').casefold())
            if target is not None and (b.get('confidence') or 0)>=0.6:
                candidates.append((Box(**dict(b,class_id=target,class_name=expected[target])),'Roboflow'))
        for b,source in candidates:
            b=Box(max(0,b.x1),max(0,b.y1),min(w,b.x2),min(h,b.y2),b.class_id,b.confidence,b.class_name)
            if b.area<=0: continue
            if any(g.class_id==b.class_id and iou(b,g)>=0.3 for g in old+accepted): continue
            # Insulator context and broken boxes may overlap by design.
            conflict=any(g.class_id!=b.class_id and {g.class_id,b.class_id}!={0,1} and iou(b,g)>=0.5 for g in old+accepted)
            if conflict:
                conflicts.append({'image':str(image),'box':b.__dict__,'source':source}); continue
            accepted.append(b); counts[b.class_name]+=1
        lines=[f'{b.class_id} {(b.x1+b.x2)/2/w:.6f} {(b.y1+b.y2)/2/h:.6f} {b.width/w:.6f} {b.height/h:.6f}' for b in accepted]
        (staged/label.name).write_text(original.rstrip('\n')+'\n'+ ('\n'.join(lines)+'\n' if lines else ''),encoding='utf-8')
        changes.append({'image':str(image),'additions':[b.__dict__ for b in accepted]})
    # Share image paths with source; validation/test labels resolve in source too.
    data={'path':str(SOURCE),'train':str(SOURCE/'images/train'),'val':str(SOURCE/'images/val'),'names':dict(enumerate(expected))}
    if (SOURCE/'images/test').is_dir(): data['test']=str(SOURCE/'images/test')
    # A complete train view needs its own images directory so YOLO resolves staged labels.
    import shutil,os
    images=destination/'images/train'; images.mkdir(parents=True,exist_ok=True)
    for p in (SOURCE/'images/train').iterdir():
        target=images/p.name
        if target.exists(): continue
        try: os.link(p,target)
        except OSError: shutil.copy2(p,target)
    data['train']=str(images)
    (destination/'data.yaml').write_text(yaml.safe_dump(data,sort_keys=False),encoding='utf-8')
    if (SOURCE/'images/val_clean').is_dir():
        clean=dict(data,val=str(SOURCE/'images/val_clean'))
        (destination/'data_val_clean.yaml').write_text(yaml.safe_dump(clean,sort_keys=False),encoding='utf-8')
    (OUT/'changes.json').write_text(json.dumps(changes),encoding='utf-8')
    (OUT/'conflicts.json').write_text(json.dumps(conflicts),encoding='utf-8')
    summary={'images':len(labels),'added_boxes':dict(counts),'conflicts':len(conflicts),'dataset':str(destination),'status':'pseudo_labels_unreviewed'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
