#!/usr/bin/env python3
"""Do toc do YOLO26n NCNN tren Raspberry Pi (chi can: pip install ncnn numpy opencv-python-headless).

Chay lien tuc --seconds giay de thay ca luc CPU bi ha xung vi nong (throttling).
    python3 pi_bench.py --model y26n_v1_640_ncnn --imgsz 640 --seconds 300
    python3 pi_bench.py --model y26n_v1_480_ncnn --imgsz 480 --seconds 300
"""
import argparse, glob, os, statistics, subprocess, time

import cv2
import ncnn
import numpy as np


def cpu_temp():
    try:
        return int(open("/sys/class/thermal/thermal_zone0/temp").read()) / 1000
    except OSError:
        return float("nan")


def throttled():
    try:  # 0x0 = OK; bit 2 = dang ha xung, bit 18 = da tung ha xung
        return subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return "n/a"


def letterbox(img, size):
    h, w = img.shape[:2]
    r = size / max(h, w)
    nh, nw = round(h * r), round(w * r)
    out = np.full((size, size, 3), 114, np.uint8)
    out[(size - nh) // 2:(size - nh) // 2 + nh, (size - nw) // 2:(size - nw) // 2 + nw] = cv2.resize(img, (nw, nh))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--images", default="images")
    p.add_argument("--seconds", type=int, default=300)
    p.add_argument("--threads", type=int, default=os.cpu_count())
    a = p.parse_args()

    net = ncnn.Net()
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = a.threads
    net.load_param(os.path.join(a.model, "model.ncnn.param"))
    net.load_model(os.path.join(a.model, "model.ncnn.bin"))
    imgs = [cv2.imread(f) for f in sorted(glob.glob(os.path.join(a.images, "*.jpg")))]
    print(f"model={a.model} imgsz={a.imgsz} threads={a.threads} images={len(imgs)} temp={cpu_temp():.1f}C")

    def infer(img):  # gom ca tien xu ly, giong khi chay that
        x = letterbox(img, a.imgsz)
        mat = ncnn.Mat.from_pixels(x, ncnn.Mat.PixelType.PIXEL_BGR2RGB, a.imgsz, a.imgsz)
        mat.substract_mean_normalize([], [1 / 255.0] * 3)
        ex = net.create_extractor()
        ex.input("in0", mat)
        _, out = ex.extract("out0")
        return np.array(out)

    for img in imgs[:3]:  # warmup
        infer(img)
    t_end, times, i, t_report = time.time() + a.seconds, [], 0, time.time() + 30
    while time.time() < t_end:
        t = time.perf_counter()
        infer(imgs[i % len(imgs)])
        times.append(time.perf_counter() - t)
        i += 1
        if time.time() >= t_report:
            last = times[-10:]
            print(f"{len(times):5d} anh | 10 anh gan nhat {1000 * statistics.mean(last):7.0f} ms/anh "
                  f"| temp {cpu_temp():.1f}C | throttled {throttled()}", flush=True)
            t_report += 30
    n = len(times)
    print(f"\nTONG: {n} anh trong {a.seconds}s | trung binh {1000 * statistics.mean(times):.0f} ms | "
          f"1/4 dau {1000 * statistics.mean(times[:n // 4]):.0f} ms | 1/4 cuoi {1000 * statistics.mean(times[-n // 4:]):.0f} ms "
          f"| temp cuoi {cpu_temp():.1f}C | throttled {throttled()}")


if __name__ == "__main__":
    main()
