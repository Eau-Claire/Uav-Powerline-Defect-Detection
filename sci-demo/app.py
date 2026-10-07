from __future__ import annotations

import json
import os
import time
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO

from sci_defs_v3 import register_sci

NAMES = ["blq_jyjc", "czjyz_lw", "czjyz_ps", "czjyz_sl", "dx_dg_sg", "dx_jypps", "nw"]
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "sci_best.pt"
torch.set_num_threads(min(4, os.cpu_count() or 1))


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Missing {MODEL_PATH}. Run: python download_model.py")
    register_sci()  # Must happen before YOLO unpickles the checkpoint.
    return YOLO(str(MODEL_PATH))


model = load_model()


def predict(image: Image.Image, conf: float, iou: float):
    if image is None:
        return None, json.dumps({"error": "Please upload an image."}, ensure_ascii=False, indent=2)
    started = time.perf_counter()
    results = model.predict(source=image, imgsz=704, conf=conf, iou=iou, device="cpu", verbose=False)
    inference_ms = sum(float(getattr(r.speed, "inference", 0.0)) for r in results)
    detections = []
    for result in results:
        boxes = result.boxes
        for xyxy, score, cls in zip(boxes.xyxy.tolist(), boxes.conf.tolist(), boxes.cls.tolist()):
            index = int(cls)
            detections.append({"class": NAMES[index] if index < len(NAMES) else str(index), "confidence": round(float(score), 6), "xyxy": [round(float(v), 2) for v in xyxy]})
    payload = {"detections": detections, "inference_ms": round(inference_ms, 2), "total_wall_time_ms": round((time.perf_counter() - started) * 1000, 2)}
    return results[0].plot(), json.dumps(payload, ensure_ascii=False, indent=2)


with gr.Blocks(title="YOLO11s SCI demo") as demo:
    gr.Markdown("# YOLO11s + SCI CPU inference")
    with gr.Row():
        image = gr.Image(type="pil", label="Input image")
        output = gr.Image(label="Detections")
    with gr.Row():
        conf = gr.Slider(0.01, 0.99, value=0.25, step=0.01, label="Confidence")
        iou = gr.Slider(0.01, 0.99, value=0.70, step=0.01, label="IoU")
    run = gr.Button("Run inference", variant="primary")
    details = gr.Code(label="JSON", language="json")
    run.click(predict, inputs=[image, conf, iou], outputs=[output, details])

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
