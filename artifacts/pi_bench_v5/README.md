# Pi 3 speed test: YOLO26n vs YOLO26s (NCNN), 640 vs 480

Speed does not depend on training, so this can run before/while v6p5-20 trains.
- `y26n_v5_*_ncnn`: the v5 student (YOLO26n, MEAN_5 val_clean 0.779 at 640).
- `y26s_coco_*_ncnn`: YOLO26s with COCO weights (same size/speed as a trained 7-class YOLO26s).

On the Raspberry Pi (Python 3, ~5 min per model, runs continuously to include thermal throttling):
```bash
pip install ncnn numpy opencv-python-headless
unzip pi_bench_v5.zip && cd pi_bench_v5
python3 pi_bench.py --model y26n_v5_640_ncnn  --imgsz 640 --seconds 300
python3 pi_bench.py --model y26n_v5_480_ncnn  --imgsz 480 --seconds 300
python3 pi_bench.py --model y26s_coco_640_ncnn --imgsz 640 --seconds 300
python3 pi_bench.py --model y26s_coco_480_ncnn --imgsz 480 --seconds 300
```
Report the ms/image (median and p90), temperature and `throttled` lines. Download:
`camber stash cp stash://bunpmc/projects/yolov11sdi/checkpoints/pi_bench_v5/pi_bench_v5.zip .`
(or copy this folder from the studio). `export_pi.py` regenerates the models.
