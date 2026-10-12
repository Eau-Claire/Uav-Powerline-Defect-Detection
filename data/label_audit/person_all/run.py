"""COCO person detections (yolo11x, conf>=0.25) on every train/val image of v6p5_6cls_v3 -> pred_{split}.jsonl."""
import json
from pathlib import Path
from ultralytics import YOLO
if __name__ == "__main__":
    R = Path('/teamspace/studios/this_studio/datasets/v6p5_6cls_v3/images')
    m = YOLO('yolo11x.pt')
    for s in ['val', 'train']:
        imgs = sorted(str(p) for p in (R/s).iterdir() if p.suffix != '.npy')
        with open(f'pred_{s}.jsonl', 'w') as fh:
            for i in range(0, len(imgs), 16):
                for p, r in zip(imgs[i:i+16], m.predict(imgs[i:i+16], conf=0.25, classes=[0], imgsz=960, half=True, verbose=False)):
                    fh.write(json.dumps({'image': Path(p).name, 'wh': r.orig_shape[::-1], 'xywhn': r.boxes.xywhn.tolist(), 'conf': r.boxes.conf.tolist()}) + '\n')
                if i % 2000 == 0: print(s, i, len(imgs), flush=True)
        print(s, 'done', flush=True)
