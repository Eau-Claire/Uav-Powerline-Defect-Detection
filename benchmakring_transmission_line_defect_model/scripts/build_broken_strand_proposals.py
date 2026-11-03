#!/usr/bin/env python
"""Generate human-reviewable GT-box replacement proposals for broken_strand."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results/finding_nemo/targeted_audit_test_main_v1_nms50"
PLAN = PACK / "broken_strand_correction_plan.csv"
MODEL = ROOT / "artifacts/finding_nemo_selected_checkpoints/yolo26m_epoch45.pt"
OUT = PACK / "broken_strand_proposals"


def read_labels(path: Path, width: int, height: int):
    rows = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, x, y, w, h = map(float, parts[:5])
        rows.append({"line_index": index, "class_id": int(cls), "line": line,
                     "xyxy": np.array([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height], dtype=np.float32)})
    return rows


def iou(a, b):
    tl=np.maximum(a[:2],b[:2]); br=np.minimum(a[2:],b[2:]); wh=np.maximum(0,br-tl); inter=wh[0]*wh[1]
    aa=max(0,a[2]-a[0])*max(0,a[3]-a[1]); bb=max(0,b[2]-b[0])*max(0,b[3]-b[1])
    return float(inter/(aa+bb-inter+1e-9))


def yolo_line(box, width, height):
    x1,y1,x2,y2=box; return f"5 {((x1+x2)/2/width):.8f} {((y1+y2)/2/height):.8f} {((x2-x1)/width):.8f} {((y2-y1)/height):.8f}"


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--device",default="auto"); parser.add_argument("--batch",type=int,default=4)
    parser.add_argument("--minimum-iou",type=float,default=.10)
    parser.add_argument("--suffix",default="")
    args=parser.parse_args(); device="0" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    plan=[r for r in csv.DictReader(PLAN.open(encoding="utf-8-sig")) if r["normalized_action"]=="adjust_gt_using_model_proposal"]
    model=YOLO(str(MODEL)); OUT.mkdir(parents=True,exist_ok=True); proposal_rows=[]
    for start in range(0,len(plan),args.batch):
        chunk=plan[start:start+args.batch]; paths=[r["image"] for r in chunk]
        results=model.predict(paths,imgsz=640,conf=.10,iou=.7,max_det=300,device=device,batch=args.batch,verbose=False)
        for item,result in zip(chunk,results):
            image_path=Path(item["image"]); image=cv2.imread(str(image_path)); h,w=image.shape[:2]
            label_path=Path(str(image_path).replace("images\\test","labels\\test")).with_suffix(".txt")
            labels=read_labels(label_path,w,h); gt=[r for r in labels if r["class_id"]==5]
            boxes=result.boxes.xyxy.cpu(); cls=result.boxes.cls.cpu().long(); scores=result.boxes.conf.cpu()
            ids=torch.where(cls==5)[0]; kept=[]
            if len(ids): kept=ids[nms(boxes[ids],scores[ids],.5)].tolist()
            preds=[boxes[i].numpy() for i in kept]; confs=[float(scores[i]) for i in kept]
            used=set(); matched=0
            for g in gt:
                candidates=sorted(((iou(g["xyxy"],p),j) for j,p in enumerate(preds) if j not in used),reverse=True)
                best_iou,j=candidates[0] if candidates else (0,-1)
                if j < 0 or best_iou < args.minimum_iou: continue
                used.add(j); matched+=1
                proposal_rows.append({"rank":item["rank"],"image":str(image_path),"label":str(label_path),"line_index":g["line_index"],
                                      "old_label":g["line"],"proposed_label":yolo_line(preds[j],w,h),"iou_old_vs_proposal":round(best_iou,4),
                                      "confidence":round(confs[j],4),"decision":"","review_note":item["review_note"]})
                x1,y1,x2,y2=map(int,preds[j]); cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),3); cv2.putText(image,f"PROPOSAL {confs[j]:.2f}",(x1,max(18,y1-5)),0,.55,(255,0,255),2)
            for g in gt:
                x1,y1,x2,y2=map(int,g["xyxy"]); cv2.rectangle(image,(x1,y1),(x2,y2),(0,220,0),2); cv2.putText(image,"OLD GT",(x1,min(h-5,y2+18)),0,.5,(0,220,0),1)
            cv2.rectangle(image,(0,0),(w,30),(20,20,20),-1); cv2.putText(image,f"rank {item['rank']} | matched {matched}/{len(gt)} | green=old GT magenta=proposal",(8,21),0,.55,(255,255,255),1)
            preview_dir=PACK/f"broken_strand_proposals{args.suffix}"; preview_dir.mkdir(parents=True,exist_ok=True)
            cv2.imwrite(str(preview_dir/f"{int(item['rank']):03d}_{image_path.stem}.jpg"),image)
    target=PACK/f"broken_strand_box_proposals{args.suffix}.csv"
    with target.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(proposal_rows[0])); writer.writeheader(); writer.writerows(proposal_rows)
    print(f"images={len(plan)} proposals={len(proposal_rows)} device={device} minimum_iou={args.minimum_iou}"); print(target.resolve()); print(preview_dir.resolve())


if __name__=="__main__": main()
