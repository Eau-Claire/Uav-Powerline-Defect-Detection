# Pi 3: n_v5 — square 640 vs. 640×480

Both exports use the same `n_v5.pt` model (standalone Pi `MEAN_5=0.761` on leak-free `val_clean_v8`). On a 40-image set of 4:3 drone images, the 640×480 export produced the same outputs as the `.pt` model while using 4.05 GFLOPs instead of approximately 5.4. Use 640×480 only when the drone camera is 4:3.

Run on the Pi with Python 3:

```bash
pip install ncnn numpy opencv-python-headless
unzip pi_v8.zip && cd pi_v8
python3 pi_bench.py --model n_v5_640_ncnn_model     --imgsz 640     --seconds 300
python3 pi_bench.py --model n_v5_640x480_ncnn_model --imgsz 640x480 --seconds 300
```

Report the `TONG: ...` line from each command, including milliseconds per image, temperature, and throttling status. Run each command continuously for 300 seconds so the measurement includes thermal throttling. These measurements determine whether a Pi 3 can support a higher-resolution student, such as 800 px, for detecting small defects.
