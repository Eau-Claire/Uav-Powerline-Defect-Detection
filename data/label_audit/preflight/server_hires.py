"""Server-side model without the Pi constraints: teacher v5 at larger input sizes, with/without TTA, on val_clean.
Usage: python server_hires.py IMGSZ [tta] [device]. Appends one row per run to server_hires.csv."""
import sys, csv, time
from pathlib import Path
from ultralytics import YOLO
HERE = Path(__file__).resolve().parent
T = '/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection/artifacts/v6p5_v5/runs/teacher/weights/best.pt'
D = '/teamspace/studios/this_studio/datasets/v6p5_v5/data_val_clean.yaml'
DEF = ['insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']
if __name__ == "__main__":
    sz = int(sys.argv[1]); tta = 'tta' in sys.argv[2:]; dev = '0' if 'gpu' in sys.argv else 'cpu'
    m = YOLO(T); t = time.time()
    b = m.val(data=D, imgsz=sz, batch=8, device=dev, augment=tta, plots=False, verbose=False, workers=2,
              project=str(HERE/'server_hires_runs'), name=f'{sz}{"_tta" if tta else ""}', exist_ok=True).box
    ap = {m.names[int(c)]: (float(a), float(a95)) for c, a, a95 in zip(b.ap_class_index, b.ap50, b.ap)}
    row = {'imgsz': sz, 'tta': tta, 'device': dev, 'minutes': round((time.time()-t)/60, 1),
           'MEAN_5_AP50': round(sum(ap[k][0] for k in DEF)/5, 4), 'MEAN_5_AP50_95': round(sum(ap[k][1] for k in DEF)/5, 4),
           **{f'{k}_AP50': round(v[0], 4) for k, v in ap.items()}, **{f'{k}_AP50_95': round(v[1], 4) for k, v in ap.items()}}
    f = HERE/'server_hires.csv'; new = not f.exists()
    with f.open('a', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
    print('RESULT', row, flush=True)
