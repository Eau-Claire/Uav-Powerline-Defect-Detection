#!/usr/bin/env python
"""
Benchmark Evaluation Script for Selected Finding Nemo Checkpoints.

Evaluates 6 user-selected checkpoints on the raw benchmark test set:
1. yolo26n (epoch 45 - best mAP)
2. yolo26m (epoch 45)
3. yolo11n (last epoch)
4. yolo11m (epoch 45)
5. rtdetr50 (epoch 15)
6. rtdetr101 (epoch 15)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmakring_transmission_line_defect_model"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from ultralytics import YOLO
from ultralytics.utils.metrics import ap_per_class, box_iou

try:
    from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor
except ImportError:
    RTDetrConfig = None
    RTDetrForObjectDetection = None
    RTDetrImageProcessor = None


# Default taxonomy for 6-class defect detection
CLASS_NAMES: Dict[int, str] = {
    0: "insulator",
    1: "insulator_broken",
    2: "insulator_flashover",
    3: "bird_nest",
    4: "foreign_object",
    5: "broken_strand",
}

DEFECT_CLASSES: List[str] = [
    "insulator_broken",
    "insulator_flashover",
    "bird_nest",
    "foreign_object",
    "broken_strand",
]

# Checkpoint Registry
CHECKPOINTS_DIR = BENCHMARK_ROOT / "artifacts" / "finding_nemo_selected_checkpoints"

CHECKPOINT_CONFIGS: Dict[str, Dict[str, Any]] = {
    "yolo26n": {
        "model_key": "yolo26n",
        "checkpoint_file": "yolo26n_epoch45.pt",
        "family": "yolo26",
        "type": "yolo",
        "description": "YOLO26n (epoch 45 - Best validation mAP)",
        "source_stash": "stash://bunpmc/projects/finding_nemo/yolo26n/weights/epoch45.pt",
    },
    "yolo26m": {
        "model_key": "yolo26m",
        "checkpoint_file": "yolo26m_epoch45.pt",
        "family": "yolo26",
        "type": "yolo",
        "description": "YOLO26m (epoch 45)",
        "source_stash": "stash://bunpmc/projects/finding_nemo/yolo26m/weights/epoch45.pt",
    },
    "yolo11n": {
        "model_key": "yolo11n",
        "checkpoint_file": "yolo11n_last.pt",
        "family": "yolo11",
        "type": "yolo",
        "description": "YOLO11n (last epoch)",
        "source_stash": "stash://bunpmc/projects/finding_nemo/yolo11n/weights/last.pt",
    },
    "yolo11m": {
        "model_key": "yolo11m",
        "checkpoint_file": "yolo11m_epoch45.pt",
        "family": "yolo11",
        "type": "yolo",
        "description": "YOLO11m (epoch 45)",
        "source_stash": "stash://bunpmc/projects/finding_nemo/yolo11m/weights/epoch45.pt",
    },
    "rtdetr50": {
        "model_key": "rtdetr50",
        "checkpoint_file": "rtdetr50_epoch15.pt",
        "family": "rtdetr",
        "type": "rtdetr",
        "model_id": "PekingU/rtdetr_r50vd",
        "description": "RT-DETR-R50vd (epoch 15)",
        "source_stash": "stash://bunpmc/projects/finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/r50vd/checkpoints/epoch15.pt",
    },
    "rtdetr101": {
        "model_key": "rtdetr101",
        "checkpoint_file": "rtdetr101_epoch15.pt",
        "family": "rtdetr",
        "type": "rtdetr",
        "model_id": "PekingU/rtdetr_r101vd",
        "description": "RT-DETR-R101vd (epoch 15)",
        "source_stash": "stash://bunpmc/projects/finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/r101vd/checkpoints/epoch15.pt",
    },
}

DATASET_PRESETS: Dict[str, Path] = {
    "test_main_v1": BENCHMARK_ROOT / "data" / "test_main_v1" / "dataset.yaml",
    "test_operational_min8_v1": BENCHMARK_ROOT / "data" / "test_operational_min8_v1" / "dataset.yaml",
    "test_main_v2_model_assisted_exploratory": BENCHMARK_ROOT / "data" / "test_main_v2_model_assisted_exploratory" / "dataset.yaml",
    "test_main_v3_model_aligned_exploratory": BENCHMARK_ROOT / "data" / "test_main_v3_model_aligned_exploratory" / "dataset.yaml",
    "test_operational_min8_v3_model_aligned": BENCHMARK_ROOT / "data" / "test_operational_min8_v3_model_aligned" / "dataset.yaml",
    "test_operational_min8_v4_consensus_assisted": BENCHMARK_ROOT / "data" / "test_operational_min8_v4_consensus_assisted" / "dataset.yaml",
    "test_operational_min8_v5_full_consensus_assisted": BENCHMARK_ROOT / "data" / "test_operational_min8_v5_full_consensus_assisted" / "dataset.yaml",
    "v6p2e_frozen_test": BENCHMARK_ROOT / "data" / "v6p2e_frozen_test" / "dataset.yaml",
    "finding_nemo": BENCHMARK_ROOT / "data" / "finding_nemo_human_reviewed_6cls" / "data.yaml",
}


def resolve_device(device_arg: str) -> str:
    """Resolve device argument into a valid torch/ultralytics device string."""
    if device_arg.lower() in ("auto", "default"):
        return "0" if torch.cuda.is_available() else "cpu"
    if device_arg.isdigit():
        return device_arg if torch.cuda.is_available() else "cpu"
    if "cuda" in device_arg.lower() and not torch.cuda.is_available():
        print(f"[WARN] CUDA requested ('{device_arg}') but not available. Falling back to CPU.", flush=True)
        return "cpu"
    return device_arg


def resolve_dataset_yaml(dataset_input: str) -> Path:
    """Resolve preset name or path to absolute Path."""
    if dataset_input in DATASET_PRESETS:
        p = DATASET_PRESETS[dataset_input]
    else:
        p = Path(dataset_input)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Dataset YAML not found at: {p}")
    return p.resolve()


def profile_yolo_hardware(model: YOLO, imgsz: int = 640, device_str: str = "cpu", warmup: int = 15, benchmark_iters: int = 50) -> Dict[str, Any]:
    """Profiles YOLO parameters, latency, FPS, peak VRAM."""
    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() or device_str.startswith("cuda") else "cpu")
    n_params = sum(p.numel() for p in model.model.parameters()) / 1e6

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    dummy_input = torch.zeros((1, 3, imgsz, imgsz), device=device)
    for _ in range(warmup):
        _ = model.predict(dummy_input, verbose=False)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    start_time = time.perf_counter()
    for _ in range(benchmark_iters):
        _ = model.predict(dummy_input, verbose=False)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    total_time = time.perf_counter() - start_time
    avg_latency_ms = (total_time / benchmark_iters) * 1000.0
    fps = benchmark_iters / total_time if total_time > 0 else 0.0
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0

    gflops = 0.0
    try:
        from ultralytics.utils.torch_utils import get_flops
        gflops = get_flops(model.model, imgsz)
    except Exception:
        pass

    return {
        "parameters_m": round(n_params, 2),
        "gflops": round(gflops, 2),
        "latency_ms": round(avg_latency_ms, 2),
        "fps": round(fps, 1),
        "inference_peak_vram_mb": round(peak_vram_mb, 1),
    }


def load_huggingface_rtdetr(model_id: str, checkpoint_path: Path, device: torch.device) -> Tuple[Any, Any]:
    """Loads a PyTorch state-dict checkpoint into Hugging Face RT-DETR."""
    if RTDetrConfig is None:
        raise ImportError("transformers is required for RT-DETR evaluation. Install with 'pip install transformers'")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("model_state", "state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break

    state_dict: Dict[str, torch.Tensor] = {}
    for key, value in checkpoint.items():
        curr = key
        if curr.startswith(("class_embed.", "bbox_embed.")):
            curr = f"model.decoder.{curr}"
        curr = curr.replace("model.encoder.encoder.", "model.encoder.aifi.").replace(".self_attn.out_proj.", ".self_attn.o_proj.")
        if curr.startswith(("model.encoder.aifi.", "model.decoder.layers.")):
            curr = re.sub(r"(?<!\.mlp)\.fc1\.", ".mlp.fc1.", curr)
            curr = re.sub(r"(?<!\.mlp)\.fc2\.", ".mlp.fc2.", curr)
        if curr in state_dict:
            continue
        state_dict[curr] = value

    config = RTDetrConfig.from_pretrained(model_id)
    config.num_labels = len(CLASS_NAMES)
    config.id2label = CLASS_NAMES
    config.label2id = {name: idx for idx, name in CLASS_NAMES.items()}
    processor = RTDetrImageProcessor.from_pretrained(model_id)
    model = RTDetrForObjectDetection(config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    del checkpoint, state_dict
    return model, processor


def evaluate_rtdetr_checkpoint(
    model_key: str,
    checkpoint_path: Path,
    dataset_yaml: Path,
    split: str = "test",
    device_str: str = "cpu",
    batch_size: int = 4,
    limit: Optional[int] = None,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """Evaluates an RT-DETR checkpoint against a YOLO formatted dataset split."""
    info = CHECKPOINT_CONFIGS[model_key]
    model_id = info["model_id"]
    device = torch.device(f"cuda:{device_str}" if device_str.isdigit() or device_str.startswith("cuda") else "cpu")

    model, processor = load_huggingface_rtdetr(model_id, checkpoint_path, device)

    # Profile GFLOPS
    gflops = 0.0
    try:
        from thop import profile
        sample = torch.zeros((1, 3, imgsz, imgsz), device=device, dtype=next(model.parameters()).dtype)
        with torch.inference_mode():
            macs, _ = profile(model, inputs=(sample,), verbose=False)
        gflops = round(float(macs) * 2 / 1e9, 2)
    except Exception:
        pass

    # Read dataset paths
    dataset_cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    dataset_root = Path(dataset_cfg.get("path", dataset_yaml.parent))
    if not dataset_root.is_absolute():
        dataset_root = (dataset_yaml.parent / dataset_root).resolve()

    image_split = Path(dataset_cfg[split])
    if not image_split.is_absolute():
        image_split = dataset_root / image_split

    image_dir_parts = list(image_split.parts)
    idx_img = next((i for i in range(len(image_dir_parts) - 1, -1, -1) if image_dir_parts[i].lower() == "images"), None)
    if idx_img is None:
        raise ValueError(f"Expected an 'images' directory in evaluation path: {image_split}")
    image_dir_parts[idx_img] = "labels"
    label_split = Path(*image_dir_parts)

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_paths = sorted(p for p in image_split.iterdir() if p.is_file() and p.suffix.lower() in image_exts)
    if limit and limit > 0:
        image_paths = image_paths[:limit]

    print(f"  [RT-DETR] Evaluating {len(image_paths)} images with batch_size={batch_size} on {device}...", flush=True)

    iou_thresholds = torch.linspace(0.5, 0.95, 10).tolist()
    batch_paths: List[Path] = []
    batch_images: List[Any] = []
    batch_sizes: List[Tuple[int, int]] = []
    batch_targets: List[Tuple[np.ndarray, np.ndarray]] = []
    stats: List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    inference_seconds = 0.0
    seen_images = 0
    total_targets = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    def process_batch():
        nonlocal inference_seconds, seen_images, total_targets
        inputs = processor(images=batch_images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        target_sizes = torch.tensor([(h, w) for w, h in batch_sizes], device=device)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model(pixel_values=pixel_values)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - t0

        predictions = processor.post_process_object_detection(outputs, threshold=0.0, target_sizes=target_sizes)

        for path, (w, h), (target_boxes, target_classes), prediction in zip(batch_paths, batch_sizes, batch_targets, predictions):
            pred_boxes = prediction["boxes"].detach().cpu().numpy().astype(np.float32)
            pred_classes = prediction["labels"].detach().cpu().numpy().astype(np.float32)
            confidences = prediction["scores"].detach().cpu().numpy().astype(np.float32)
            total_targets += len(target_classes)

            if len(target_boxes) and len(pred_boxes):
                overlaps = box_iou(torch.from_numpy(target_boxes), torch.from_numpy(pred_boxes)).numpy()
                correct = np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool)
                same_class = target_classes[:, None] == pred_classes[None, :]
                overlaps *= same_class
                for t_idx, thr in enumerate(iou_thresholds):
                    matches = np.array(np.nonzero(overlaps >= thr)).T
                    if len(matches):
                        if len(matches) > 1:
                            matches = matches[np.argsort(overlaps[matches[:, 0], matches[:, 1]])[::-1]]
                            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                        correct[matches[:, 1].astype(int), t_idx] = True
            else:
                correct = np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool)

            stats.append((correct, confidences, pred_classes, target_classes))
            seen_images += 1

    last_print = time.time()
    for image_path in image_paths:
        label_path = label_split / f"{image_path.stem}.txt"
        target_boxes = []
        target_classes = []
        if label_path.is_file():
            for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
                fields = line.split()
                if len(fields) == 5:
                    cid, xc, yc, bw, bh = map(float, fields)
                    cid_int = int(cid)
                    if cid_int in CLASS_NAMES:
                        target_classes.append(cid_int)
                        # We will compute absolute pixel coordinates when image size is known
                        target_boxes.append([xc, yc, bw, bh])

        with Image.open(image_path) as src_img:
            img = src_img.convert("RGB")
        w, h = img.size

        # Convert normalized boxes to pixel coordinates [x1, y1, x2, y2]
        converted_boxes = []
        for xc, yc, bw, bh in target_boxes:
            converted_boxes.append([(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h])

        batch_paths.append(image_path)
        batch_images.append(img)
        batch_sizes.append((w, h))
        batch_targets.append((
            np.asarray(converted_boxes, dtype=np.float32).reshape(-1, 4),
            np.asarray(target_classes, dtype=np.float32),
        ))

        if len(batch_paths) == batch_size:
            process_batch()
            batch_paths.clear()
            batch_images.clear()
            batch_sizes.clear()
            batch_targets.clear()

            if time.time() - last_print > 5.0 or seen_images == len(image_paths):
                pct = (seen_images / len(image_paths)) * 100
                print(f"    Progress: {seen_images}/{len(image_paths)} images ({pct:.1f}%)", flush=True)
                last_print = time.time()

    if batch_paths:
        process_batch()

    tp = np.concatenate([item[0] for item in stats], axis=0) if stats else np.zeros((0, 10), dtype=bool)
    conf = np.concatenate([item[1] for item in stats], axis=0) if stats else np.zeros(0, dtype=np.float32)
    pred_cls = np.concatenate([item[2] for item in stats], axis=0) if stats else np.zeros(0, dtype=np.float32)
    target_cls = np.concatenate([item[3] for item in stats], axis=0) if stats else np.zeros(0, dtype=np.float32)

    _, _, precision, recall, _, ap, ap_classes, *_ = ap_per_class(tp, conf, pred_cls, target_cls, names=CLASS_NAMES)

    per_class = {}
    for cid, cname in CLASS_NAMES.items():
        matches = np.flatnonzero(ap_classes == cid)
        c_idx = int(matches[0]) if len(matches) else None
        per_class[cname] = {
            "ap50": round(float(ap[c_idx, 0]), 4) if c_idx is not None else 0.0,
            "ap50_95": round(float(ap[c_idx].mean()), 4) if c_idx is not None else 0.0,
        }

    mean_5 = float(np.mean([per_class[name]["ap50"] for name in DEFECT_CLASSES]))
    precision_mean = float(precision.mean()) if len(precision) else 0.0
    recall_mean = float(recall.mean()) if len(recall) else 0.0
    f1_mean = 2 * precision_mean * recall_mean / (precision_mean + recall_mean + 1e-8)

    parameter_count = sum(p.numel() for p in model.parameters()) / 1e6
    latency_ms = (inference_seconds / seen_images * 1000) if seen_images else 0.0
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0

    return {
        "run_key": model_key,
        "model_key": model_key,
        "checkpoint_path": str(checkpoint_path),
        "evaluation_split": split,
        "evaluation_dataset_yaml": str(dataset_yaml),
        "images_evaluated": seen_images,
        "accuracy": {
            "precision": round(precision_mean, 4),
            "recall": round(recall_mean, 4),
            "f1_score": round(f1_mean, 4),
            "mAP50": round(float(ap[:, 0].mean()), 4) if len(ap) else 0.0,
            "mAP50_95": round(float(ap.mean()), 4) if len(ap) else 0.0,
            "MEAN_5_defect_ap50": round(mean_5, 4),
            "per_class": per_class,
        },
        "hardware": {
            "parameters_m": round(parameter_count, 2),
            "gflops": gflops,
            "latency_ms": round(latency_ms, 2),
            "fps": round(1000 / latency_ms, 1) if latency_ms > 0 else 0.0,
            "inference_peak_vram_mb": round(peak_vram_mb, 1),
        },
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def evaluate_yolo_checkpoint(
    model_key: str,
    checkpoint_path: Path,
    dataset_yaml: Path,
    split: str = "test",
    device_str: str = "cpu",
    batch_size: int = 8,
    limit: Optional[int] = None,
    imgsz: int = 640,
) -> Dict[str, Any]:
    """Evaluates a YOLO checkpoint using Ultralytics validation engine."""
    model = YOLO(str(checkpoint_path))

    # If limit is specified, generate temporary YAML pointing to the limited subset of image paths
    temp_yaml_file = None
    temp_list_file = None
    val_yaml_path = str(dataset_yaml.resolve())

    if limit and limit > 0:
        dataset_cfg = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
        dataset_root = Path(dataset_cfg.get("path", dataset_yaml.parent))
        if not dataset_root.is_absolute():
            dataset_root = (dataset_yaml.parent / dataset_root).resolve()

        image_split = Path(dataset_cfg[split])
        if not image_split.is_absolute():
            image_split = dataset_root / image_split

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(p.resolve().as_posix() for p in image_split.iterdir() if p.is_file() and p.suffix.lower() in image_exts)[:limit]

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf_list:
            tf_list.write("\n".join(image_paths))
            temp_list_file = tf_list.name

        sub_cfg = {
            "path": "",
            "train": temp_list_file,
            "val": temp_list_file,
            "test": temp_list_file,
            "names": CLASS_NAMES,
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tf_yaml:
            yaml.dump(sub_cfg, tf_yaml)
            temp_yaml_file = tf_yaml.name

        val_yaml_path = temp_yaml_file

    print(f"  [YOLO] Running model.val on {val_yaml_path} (batch={batch_size}, device={device_str})...", flush=True)

    try:
        val_res = model.val(
            data=val_yaml_path,
            split=split,
            imgsz=imgsz,
            batch=batch_size,
            device=device_str,
            plots=False,
            verbose=False,
        )
    finally:
        if temp_list_file and os.path.exists(temp_list_file):
            try:
                os.remove(temp_list_file)
            except Exception:
                pass
        if temp_yaml_file and os.path.exists(temp_yaml_file):
            try:
                os.remove(temp_yaml_file)
            except Exception:
                pass

    p_all = float(val_res.results_dict.get("metrics/precision(B)", 0.0))
    r_all = float(val_res.results_dict.get("metrics/recall(B)", 0.0))
    f1_all = 2 * (p_all * r_all) / (p_all + r_all + 1e-8)
    map50_all = float(val_res.results_dict.get("metrics/mAP50(B)", 0.0))
    map50_95_all = float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0))

    per_class = {}
    ap50_per_class = val_res.box.ap50 if hasattr(val_res, "box") else []
    ap50_95_per_class = val_res.box.ap if hasattr(val_res, "box") else []

    for idx, name in CLASS_NAMES.items():
        c_ap50 = float(ap50_per_class[idx]) if idx < len(ap50_per_class) else 0.0
        c_ap50_95 = float(ap50_95_per_class[idx]) if idx < len(ap50_95_per_class) else 0.0
        per_class[name] = {
            "ap50": round(c_ap50, 4),
            "ap50_95": round(c_ap50_95, 4),
        }

    mean_5 = float(np.mean([per_class[c]["ap50"] for c in DEFECT_CLASSES if c in per_class]))

    # Profile hardware
    hw_profile = profile_yolo_hardware(model, imgsz=imgsz, device_str=device_str)

    return {
        "run_key": model_key,
        "model_key": model_key,
        "checkpoint_path": str(checkpoint_path),
        "evaluation_split": split,
        "evaluation_dataset_yaml": str(dataset_yaml),
        "images_evaluated": limit if limit and limit > 0 else "all",
        "accuracy": {
            "precision": round(p_all, 4),
            "recall": round(r_all, 4),
            "f1_score": round(f1_all, 4),
            "mAP50": round(map50_all, 4),
            "mAP50_95": round(map50_95_all, 4),
            "MEAN_5_defect_ap50": round(mean_5, 4),
            "per_class": per_class,
        },
        "hardware": hw_profile,
        "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Selected Finding Nemo Checkpoints on Raw Test Set")
    parser.add_argument(
        "--dataset",
        type=str,
        default="test_main_v1",
        help="Dataset preset ('test_main_v1', 'v6p2e_frozen_test') or path to dataset.yaml (default: test_main_v1)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(CHECKPOINT_CONFIGS.keys()),
        choices=list(CHECKPOINT_CONFIGS.keys()),
        help=f"Models to evaluate. Choices: {list(CHECKPOINT_CONFIGS.keys())} (default: all 6)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on: 'auto' (default: cuda if available, else cpu), '0', 'cuda:0', 'cpu'",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8 for YOLO, clamped to 4 for RT-DETR)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: evaluate on first N images only (for smoke testing / quick validation)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split name in dataset.yaml (default: 'test')",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(BENCHMARK_ROOT / "results" / "finding_nemo" / "selected_checkpoints_eval"),
        help="Output directory to save evaluation metrics and report",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-evaluation even if results already exist",
    )

    args = parser.parse_args()

    device_str = resolve_device(args.device)
    dataset_yaml = resolve_dataset_yaml(args.dataset)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print(" FINDING NEMO - SELECTED CHECKPOINTS BENCHMARK EVALUATION")
    print("=" * 80)
    print(f"  Dataset YAML : {dataset_yaml}")
    print(f"  Split        : {args.split}")
    print(f"  Device       : {device_str} (CUDA available: {torch.cuda.is_available()})")
    print(f"  Models       : {', '.join(args.models)}")
    print(f"  Batch Size   : {args.batch_size}")
    print(f"  Image Limit  : {args.limit if args.limit else 'ALL'}")
    print(f"  Output Dir   : {output_dir}")
    print("=" * 80)

    # Load existing results if not force
    results_json_path = output_dir / "benchmark_test_results.json"
    results_data: Dict[str, Any] = {}
    if results_json_path.is_file() and not args.force:
        try:
            results_data = json.loads(results_json_path.read_text(encoding="utf-8"))
        except Exception:
            results_data = {}

    summary_records = []
    per_class_records = []

    for model_key in args.models:
        cfg = CHECKPOINT_CONFIGS[model_key]
        ckpt_path = CHECKPOINTS_DIR / cfg["checkpoint_file"]

        print(f"\n>>> [{model_key.upper()}] {cfg['description']}")
        print(f"    Checkpoint : {ckpt_path.name} ({ckpt_path.stat().st_size:,} bytes)")

        if not ckpt_path.is_file():
            print(f"    [ERROR] Checkpoint file missing: {ckpt_path}")
            continue

        cached = results_data.get(model_key)
        if (
            not args.force
            and cached
            and cached.get("evaluation_split") == args.split
            and cached.get("evaluation_dataset_yaml") == str(dataset_yaml)
            and cached.get("checkpoint_path") == str(ckpt_path)
            and (
                (args.limit is None and cached.get("images_evaluated") in ("all", 1931, 2012))
                or (args.limit is not None and cached.get("images_evaluated") == args.limit)
            )
        ):
            print(f"    [REUSE] Cached results found for '{model_key}'.")
            res = cached
        else:
            if cfg["type"] == "yolo":
                res = evaluate_yolo_checkpoint(
                    model_key=model_key,
                    checkpoint_path=ckpt_path,
                    dataset_yaml=dataset_yaml,
                    split=args.split,
                    device_str=device_str,
                    batch_size=args.batch_size,
                    limit=args.limit,
                )
            elif cfg["type"] == "rtdetr":
                rtdetr_batch = min(args.batch_size, 4)
                res = evaluate_rtdetr_checkpoint(
                    model_key=model_key,
                    checkpoint_path=ckpt_path,
                    dataset_yaml=dataset_yaml,
                    split=args.split,
                    device_str=device_str,
                    batch_size=rtdetr_batch,
                    limit=args.limit,
                )
            else:
                raise ValueError(f"Unknown model type: {cfg['type']}")

            results_data[model_key] = res
            results_json_path.write_text(json.dumps(results_data, indent=2), encoding="utf-8")

        acc = res["accuracy"]
        hw = res["hardware"]
        print(
            f"    -> mAP50: {acc['mAP50']:.4f} | mAP50-95: {acc['mAP50_95']:.4f} | "
            f"MEAN_5: {acc['MEAN_5_defect_ap50']:.4f} | F1: {acc['f1_score']:.4f} | "
            f"Latency: {hw['latency_ms']:.1f}ms | FPS: {hw['fps']}"
        )

        summary_records.append({
            "Model": model_key,
            "Description": cfg["description"],
            "Checkpoint": cfg["checkpoint_file"],
            "Precision": acc["precision"],
            "Recall": acc["recall"],
            "F1-Score": acc["f1_score"],
            "mAP@0.5": acc["mAP50"],
            "mAP@0.5:0.95": acc["mAP50_95"],
            "MEAN_5": acc["MEAN_5_defect_ap50"],
            "Latency (ms)": hw["latency_ms"],
            "FPS": hw["fps"],
            "GFLOPS": hw["gflops"],
            "Params (M)": hw["parameters_m"],
            "VRAM (MB)": hw["inference_peak_vram_mb"],
            "Source Stash": cfg["source_stash"],
        })

        per_cls_row = {
            "Model": model_key,
            "Description": cfg["description"],
        }
        for cname in CLASS_NAMES.values():
            if cname in acc["per_class"]:
                per_cls_row[f"{cname}_AP50"] = acc["per_class"][cname]["ap50"]
                per_cls_row[f"{cname}_AP50_95"] = acc["per_class"][cname]["ap50_95"]
        per_class_records.append(per_cls_row)

    # Save CSVs
    summary_df = pd.DataFrame(summary_records)
    summary_csv = output_dir / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    per_class_df = pd.DataFrame(per_class_records)
    per_class_csv = output_dir / "per_class_summary.csv"
    per_class_df.to_csv(per_class_csv, index=False, encoding="utf-8-sig")

    # Generate Markdown Summary
    report_lines = [
        "# Finding Nemo: Selected Checkpoints Test Benchmark",
        "",
        f"- **Dataset Evaluated**: `{dataset_yaml.name}` ({dataset_yaml})",
        f"- **Split**: `{args.split}`",
        f"- **Images Evaluated**: `{args.limit if args.limit else 'Full Dataset'}`",
        f"- **Device**: `{device_str}`",
        f"- **Evaluated At**: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## Overall Benchmark Summary",
        "",
    ]
    report_lines.append(summary_df[["Model", "Description", "Precision", "Recall", "F1-Score", "mAP@0.5", "mAP@0.5:0.95", "MEAN_5", "Latency (ms)", "FPS"]].to_markdown(index=False))
    report_lines.append("\n## Per-Class AP@0.5 Breakdown\n")

    ap50_cols = ["Model"] + [f"{c}_AP50" for c in CLASS_NAMES.values() if f"{c}_AP50" in per_class_df.columns]
    report_lines.append(per_class_df[ap50_cols].to_markdown(index=False))

    report_lines.append("\n## Hardware Profile\n")
    report_lines.append(summary_df[["Model", "Params (M)", "GFLOPS", "Latency (ms)", "FPS", "VRAM (MB)"]].to_markdown(index=False))

    (output_dir / "README.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print(" EVALUATION FINISHED SUCCESSFULLY!")
    print(f" Summary CSV      : {summary_csv}")
    print(f" Per-class CSV    : {per_class_csv}")
    print(f" JSON results     : {results_json_path}")
    print(f" Markdown report  : {output_dir / 'README.md'}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()

