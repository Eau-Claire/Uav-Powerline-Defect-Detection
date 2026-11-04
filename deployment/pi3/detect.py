"""NCNN CPU inference on a still image or USB camera; no torch dependency."""
import argparse,json,time,os,urllib.request
from pathlib import Path
import cv2
import numpy as np

def decode(output,family,names,conf,iou):
    a=np.asarray(output)
    if a.ndim!=2:raise ValueError(f'Unexpected NCNN shape {a.shape}')
    if family=='yolo26n':
        if a.shape[1]!=6:raise ValueError(f'Expected end-to-end Nx6 output, got {a.shape}')
        a=a[a[:,4]>=conf];return [(row[:4].copy(),float(row[4]),int(row[5])) for row in a]
    if a.shape[0]==4+len(names):a=a.T
    if a.shape[1]!=4+len(names):raise ValueError(f'Unexpected YOLO11 output: {a.shape}')
    ids=a[:,4:].argmax(1);scores=a[:,4:].max(1);valid=scores>=conf
    a=a[valid];ids=ids[valid];scores=scores[valid];out=[]
    xywh=a[:,:4].copy();xywh[:,:2]-=xywh[:,2:]/2
    for cls in np.unique(ids):
        subset=np.flatnonzero(ids==cls)
        keep=cv2.dnn.NMSBoxes(xywh[subset].tolist(),scores[subset].tolist(),conf,iou)
        for j in np.asarray(keep,dtype=int).reshape(-1):
            k=subset[j];x,y,w,h=xywh[k];out.append((np.array([x,y,x+w,y+h]),float(scores[k]),int(cls)))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument('--image',type=Path);g.add_argument('--camera',type=int)
    p.add_argument('--conf',type=float,default=.25);p.add_argument('--iou',type=float,default=.45);p.add_argument('--threads',type=int,default=3)
    p.add_argument('--cloud-url',default=os.getenv('POWERLINE_CLOUD_URL'));p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--alerts-config',type=Path);p.add_argument('--outbox',type=Path,default=Path('alerts.sqlite3'))
    p.add_argument('--repeat',type=int,default=1,help='Still-image timing iterations');a=p.parse_args()
    if not 0<a.conf<=1 or not 0<a.iou<=1 or a.repeat<1 or a.threads<1 or a.warmup<0:p.error('Invalid inference options')
    from alerts import AlertOutbox
    outbox=AlertOutbox(a.alerts_config,a.outbox) if a.alerts_config else None
    import ncnn
    cfg=json.loads((a.model/'deployment.json').read_text());size=cfg['imgsz'];names=cfg['names']
    net=ncnn.Net();net.opt.use_vulkan_compute=False;net.opt.num_threads=a.threads
    if net.load_param(str(a.model/'model.ncnn.param'))!=0 or net.load_model(str(a.model/'model.ncnn.bin'))!=0:raise RuntimeError('NCNN load failed')
    def infer(frame):
        h,w=frame.shape[:2];r=min(size/w,size/h);nw,nh=round(w*r),round(h*r);left,top=(size-nw)//2,(size-nh)//2
        canvas=np.full((size,size,3),114,np.uint8);canvas[top:top+nh,left:left+nw]=cv2.resize(frame,(nw,nh))
        mat=ncnn.Mat.from_pixels(canvas,ncnn.Mat.PixelType.PIXEL_BGR2RGB,size,size);mat.substract_mean_normalize([],[1/255.]*3)
        ex=net.create_extractor()
        if ex.input('in0',mat)!=0:raise RuntimeError('NCNN input failed')
        code,out=ex.extract('out0')
        if code!=0:raise RuntimeError('NCNN output failed')
        result=[]
        for box,score,cls in decode(np.array(out),cfg['family'],names,a.conf,a.iou):
            if not 0<=cls<len(names):raise ValueError('Invalid class in model output')
            box[[0,2]]=np.clip((box[[0,2]]-left)/r,0,w);box[[1,3]]=np.clip((box[[1,3]]-top)/r,0,h)
            result.append({'class_id':cls,'class_name':names[cls],'confidence':score,'bbox_xyxy':box.tolist()})
        return result
    cap=cv2.VideoCapture(a.camera) if a.camera is not None else None
    try:
        count=0
        while True:
            if cap is None:
                if count>=a.repeat:break
                frame=cv2.imread(str(a.image))
            else:
                ok,frame=cap.read()
                if not ok:raise RuntimeError('Camera frame unavailable')
            if frame is None:raise ValueError('Cannot decode input image')
            if count==0:
                for _ in range(a.warmup):infer(frame)
            t=time.perf_counter();dets=infer(frame);record={'detections':dets,'local_ms':(time.perf_counter()-t)*1000}
            if outbox is not None:record['alerts_queued']=outbox.enqueue(dets)
            if a.cloud_url and any(d['class_id']!=0 for d in dets):
                # Synchronous, bounded queue: no accumulation of stale camera frames.
                try:
                    ok,jpg=cv2.imencode('.jpg',frame)
                    if not ok:raise ValueError('JPEG encoding failed')
                    headers={'Content-Type':'image/jpeg','Modal-Key':os.environ['MODAL_KEY'],'Modal-Secret':os.environ['MODAL_SECRET']}
                    req=urllib.request.Request(a.cloud_url,data=jpg.tobytes(),headers=headers,method='POST')
                    with urllib.request.urlopen(req,timeout=90) as response:record['cloud']=json.load(response)
                except Exception as exc:record['cloud_error']=type(exc).__name__
            print(json.dumps(record),flush=True);count+=1
    finally:
        if cap is not None:cap.release()
        net.clear()
        if outbox is not None:outbox.close()
if __name__=='__main__':main()
