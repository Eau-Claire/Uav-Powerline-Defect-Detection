#!/usr/bin/env python
"""Find likely missing insulator GT boxes using agreement among four YOLO checkpoints."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import nms
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results/finding_nemo/targeted_audit_test_main_v1_nms50"
CHECKPOINTS = ROOT / "artifacts/finding_nemo_selected_checkpoints"
MODELS = {
    "yolo26m": CHECKPOINTS / "yolo26m_epoch45.pt",
    "yolo26n": CHECKPOINTS / "yolo26n_epoch45.pt",
    "yolo11m": CHECKPOINTS / "yolo11m_epoch45.pt",
    "yolo11n": CHECKPOINTS / "yolo11n_last.pt",
}
OUT = PACK / "insulator_consensus_proposals"


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b): return np.zeros((len(a),len(b)),dtype=np.float32)
    tl=np.maximum(a[:,None,:2],b[None,:,:2]); br=np.minimum(a[:,None,2:],b[None,:,2:]); wh=np.clip(br-tl,0,None)
    inter=wh[...,0]*wh[...,1]; aa=(a[:,2]-a[:,0])*(a[:,3]-a[:,1]); bb=(b[:,2]-b[:,0])*(b[:,3]-b[:,1])
    return inter/np.clip(aa[:,None]+bb[None,:]-inter,1e-9,None)


def gt_boxes(image: Path, width: int, height: int) -> np.ndarray:
    label=Path(str(image).replace("images\\test","labels\\test")).with_suffix(".txt"); boxes=[]
    for line in label.read_text(encoding="utf-8").splitlines():
        s=line.split();
        if len(s)>=5 and int(float(s[0]))==0:
            x,y,w,h=map(float,s[1:5]); boxes.append([(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height])
    return np.asarray(boxes,dtype=np.float32).reshape(-1,4)


def extract(result, conf: float):
    ids=torch.where((result.boxes.cls.cpu().long()==0)&(result.boxes.conf.cpu()>=conf))[0]
    if not len(ids): return np.empty((0,4),np.float32),np.empty(0,np.float32)
    boxes=result.boxes.xyxy.cpu()[ids]; scores=result.boxes.conf.cpu()[ids]; keep=nms(boxes,scores,.5)
    return boxes[keep].numpy(),scores[keep].numpy()


def yolo_line(box,w,h):
    x1,y1,x2,y2=box; return f"0 {((x1+x2)/2/w):.8f} {((y1+y2)/2/h):.8f} {((x2-x1)/w):.8f} {((y2-y1)/h):.8f}"


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--device",default="auto"); ap.add_argument("--batch",type=int,default=8)
    ap.add_argument("--conf",type=float,default=.20); ap.add_argument("--minimum-models",type=int,default=3)
    ap.add_argument("--all-images",action="store_true"); ap.add_argument("--suffix",default="")
    args=ap.parse_args(); device="0" if args.device=="auto" and torch.cuda.is_available() else ("cpu" if args.device=="auto" else args.device)
    selected=list(csv.DictReader((PACK/"selected_for_review.csv").open(encoding="utf-8-sig")))
    if args.all_images:
        images=sorted((ROOT/"data/test_main_v3_model_aligned_exploratory/images/test").glob("*.*"))
        images=[p for p in images if p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}]
    else:
        images=[Path(r["image"]) for r in selected if r["audit_class"]=="insulator" and r["selection"]=="error"]
    predictions=defaultdict(dict)
    for model_name,checkpoint in MODELS.items():
        print(f"[{model_name}]",flush=True); model=YOLO(str(checkpoint))
        for start in range(0,len(images),args.batch):
            chunk=images[start:start+args.batch]
            results=model.predict([str(p) for p in chunk],imgsz=640,conf=args.conf,iou=.7,max_det=300,device=device,batch=args.batch,verbose=False)
            for path,result in zip(chunk,results): predictions[str(path)][model_name]=extract(result,args.conf)
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_dir=PACK/f"insulator_consensus_proposals{args.suffix}"; out_dir.mkdir(parents=True,exist_ok=True); proposals=[]
    for image in images:
        im=cv2.imread(str(image)); h,w=im.shape[:2]; gt=gt_boxes(image,w,h)
        anchors,anchor_scores=predictions[str(image)]["yolo26m"]
        accepted=[]
        for anchor,anchor_score in zip(anchors,anchor_scores):
            if len(gt) and float(iou_matrix(anchor[None],gt).max())>=.10: continue
            matched=[anchor]; voters=["yolo26m"]; confidences=[float(anchor_score)]
            for name in ("yolo26n","yolo11m","yolo11n"):
                boxes,scores=predictions[str(image)][name]
                if not len(boxes): continue
                values=iou_matrix(anchor[None],boxes)[0]; index=int(values.argmax())
                if float(values[index])>=.50:
                    matched.append(boxes[index]); voters.append(name); confidences.append(float(scores[index]))
            if len(voters)<args.minimum_models: continue
            proposal=np.mean(np.stack(matched),axis=0); accepted.append(proposal)
            proposals.append({"image":str(image),"proposed_label":yolo_line(proposal,w,h),"models":len(voters),
                              "voters":"|".join(voters),"mean_confidence":round(float(np.mean(confidences)),4),
                              "max_iou_with_existing_gt":round(float(iou_matrix(proposal[None],gt).max()) if len(gt) else 0,4),
                              "decision":"","review_note":""})
        for box in gt:
            x1,y1,x2,y2=map(int,box); cv2.rectangle(im,(x1,y1),(x2,y2),(0,220,0),2); cv2.putText(im,"EXISTING GT",(x1,max(15,y1-4)),0,.45,(0,220,0),1)
        for box in accepted:
            x1,y1,x2,y2=map(int,box); cv2.rectangle(im,(x1,y1),(x2,y2),(255,0,255),3); cv2.putText(im,"CONSENSUS MISSING?",(x1,max(18,y1-5)),0,.5,(255,0,255),2)
        if accepted: cv2.imwrite(str(out_dir/f"{image.stem}.jpg"),im)
    target=PACK/f"insulator_consensus_proposals{args.suffix}.csv"
    if proposals:
        with target.open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,fieldnames=list(proposals[0])); writer.writeheader(); writer.writerows(proposals)
    print(f"images={len(images)} proposal_boxes={len(proposals)} proposal_images={len(set(r['image'] for r in proposals))} device={device}")
    print(target.resolve()); print(out_dir.resolve())


if __name__=="__main__": main()
