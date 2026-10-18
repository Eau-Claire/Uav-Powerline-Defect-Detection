import shutil
from pathlib import Path
from ultralytics import YOLO
if __name__ == "__main__":
    A = Path('/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection/artifacts')
    for tag, w in [('y26n_v5', A/'v6p5_v5/runs/student_distill/weights/best.pt'), ('y26s_coco', 'yolo26s.pt')]:
        for sz in (640, 480):
            out = Path(f'{tag}_{sz}_ncnn')
            if out.is_dir(): continue
            src = Path(YOLO(str(w)).export(format='ncnn', imgsz=sz, half=False))
            shutil.move(str(src), out)
            print('exported', out, sum(f.stat().st_size for f in out.iterdir()) // 2**20, 'MB', flush=True)
