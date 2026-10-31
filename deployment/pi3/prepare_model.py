"""Export a trusted six-class checkpoint on desktop."""
import argparse, hashlib, json, shutil
from pathlib import Path
NAMES=['insulator','insulator_broken','insulator_flashover','bird_nest','foreign_object','broken_strand']
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,default=Path(__file__).resolve().parent/'models/best.pt')
    p.add_argument('--family',choices=['yolo26n','yolo11n'],default='yolo11n')
    p.add_argument('--size',type=int,default=640)
    a=p.parse_args()
    if not a.checkpoint.is_file():p.error(f'Missing checkpoint: {a.checkpoint}')
    if a.size < 32 or a.size % 32:p.error('--size must be a positive multiple of 32')
    from ultralytics import YOLO
    model=YOLO(str(a.checkpoint))
    names=[model.names[i] for i in range(len(model.names))]
    if names!=NAMES:raise ValueError(f'Expected six benchmark classes, got {names}')
    root=Path(__file__).resolve().parent/'models';root.mkdir(exist_ok=True)
    dst=root/'pi.pt'
    if a.checkpoint.resolve()!=dst.resolve():shutil.copy2(a.checkpoint,dst)
    manifest={'source':str(a.checkpoint.resolve()),'sha256':hashlib.sha256(dst.read_bytes()).hexdigest(),'names':names}
    exported=Path(YOLO(str(dst)).export(format='ncnn',imgsz=a.size,batch=1,device='cpu',half=False))
    (exported/'deployment.json').write_text(json.dumps({'family':a.family,'imgsz':a.size,'names':names,'checkpoint_sha256':manifest['sha256']},indent=2))
    manifest['export']=str(exported)
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print('Prepared model in',root)
if __name__=='__main__':main()
