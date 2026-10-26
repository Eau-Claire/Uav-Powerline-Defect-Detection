from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import torch
import yaml

from .config import ROOT_DIR, get_benchmark_config, get_class_names

class BenchmarkEvaluator:
    def __init__(self, config_path: str | Path = "configs/benchmark_config.yaml"):
        self.cfg = get_benchmark_config(config_path)
        self.class_names = get_class_names()
        self.results_dir = ROOT_DIR / "results"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.eval_file = self.results_dir / "benchmark_test_results.json"
        self.eval_data = self._load_results()

    def _load_results(self) -> Dict[str, Any]:
        if self.eval_file.exists():
            try:
                with open(self.eval_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def _save_results(self) -> None:
        with open(self.eval_file, "w", encoding="utf-8") as f:
            json.dump(self.eval_data, f, indent=2)

    def _load_huggingface_rtdetr(
        self,
        run_key: str,
        checkpoint_path: Path
    ) -> tuple[Any, Any]:
        try:
            from transformers import RTDetrConfig, RTDetrForObjectDetection, RTDetrImageProcessor
        except ImportError as exc:
            raise ImportError(
                "Evaluating these RT-DETR state-dict checkpoints requires the optional "
                "transformers dependency. Install it with "
                "`python -m pip install -e \".[rtdetr-eval]\"` from Uav-Powerline-Defect-Detection."
            ) from exc

        match = re.search(r"rtdetr(50|100)$", run_key.lower())
        if match is None:
            raise ValueError(f"Unsupported Hugging Face RT-DETR run key: {run_key}")
        model_id = "PekingU/rtdetr_r50vd" if match.group(1) == "50" else "PekingU/rtdetr_r101vd"

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict):
            for key in ("model_state", "state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict) or not all(torch.is_tensor(value) for value in checkpoint.values()):
            raise TypeError(
                f"Expected a tensor state dict in RT-DETR checkpoint {checkpoint_path}; "
                f"found {type(checkpoint).__name__}."
            )

        config = RTDetrConfig.from_pretrained(model_id)
        config.num_labels = len(self.class_names)
        config.id2label = self.class_names
        config.label2id = {name: idx for idx, name in self.class_names.items()}
        processor = RTDetrImageProcessor.from_pretrained(model_id)
        model = RTDetrForObjectDetection(config)
        model.load_state_dict(checkpoint, strict=True)
        device = torch.device(
            f"cuda:{self.cfg['training']['device']}" if torch.cuda.is_available() else "cpu"
        )
        model.to(device).eval()
        del checkpoint
        return model, processor

    @staticmethod
    def _profile_huggingface_gflops(model: torch.nn.Module, imgsz: int | tuple[int, int]) -> float:
        from thop import profile

        height, width = (imgsz, imgsz) if isinstance(imgsz, int) else imgsz
        parameter = next(model.parameters())
        sample = torch.zeros(
            (1, 3, height, width),
            device=parameter.device,
            dtype=parameter.dtype
        )
        with torch.inference_mode():
            macs, _ = profile(model, inputs=(sample,), verbose=False)
        gflops = float(macs) * 2 / 1e9
        if not np.isfinite(gflops) or gflops <= 0:
            raise RuntimeError(f"THOP did not produce a valid GFLOPs estimate: {gflops}")
        return round(gflops, 2)

    def _evaluate_huggingface_rtdetr(
        self,
        run_key: str,
        checkpoint_path: Path,
        test_dataset_yaml: Path,
        split: str
    ) -> Dict[str, Any]:
        """Evaluate a Hugging Face RT-DETR state-dict checkpoint on a YOLO-format split."""
        try:
            from PIL import Image
            from ultralytics.utils.metrics import ap_per_class, box_iou
        except ImportError as exc:
            raise ImportError("RT-DETR evaluation requires Pillow and Ultralytics metrics.") from exc

        model, processor = self._load_huggingface_rtdetr(run_key, checkpoint_path)
        device = next(model.parameters()).device
        gflops = self._profile_huggingface_gflops(model, self.cfg["training"]["imgsz"])

        yaml_path = Path(test_dataset_yaml).resolve()
        dataset_cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        dataset_root = Path(dataset_cfg.get("path", yaml_path.parent))
        if not dataset_root.is_absolute():
            dataset_root = (yaml_path.parent / dataset_root).resolve()
        image_split = Path(dataset_cfg[split])
        if not image_split.is_absolute():
            image_split = dataset_root / image_split
        if not image_split.is_dir():
            raise FileNotFoundError(f"RT-DETR evaluation split directory not found: {image_split}")

        image_dir_parts = list(image_split.parts)
        image_dir_index = next(
            (i for i in range(len(image_dir_parts) - 1, -1, -1) if image_dir_parts[i].lower() == "images"),
            None
        )
        if image_dir_index is None:
            raise ValueError(f"Expected an 'images' directory in evaluation path: {image_split}")
        image_dir_parts[image_dir_index] = "labels"
        label_split = Path(*image_dir_parts)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(
            path for path in image_split.iterdir()
            if path.is_file() and path.suffix.lower() in image_extensions
        )
        if not image_paths:
            raise FileNotFoundError(f"No images found in RT-DETR evaluation split: {image_split}")

        iou_thresholds = torch.linspace(0.5, 0.95, 10).tolist()
        batch_size = 2
        batch_paths: list[Path] = []
        batch_images = []
        batch_sizes: list[tuple[int, int]] = []
        batch_targets: list[tuple[np.ndarray, np.ndarray]] = []
        stats: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        inference_seconds = 0.0
        seen_images = 0
        total_targets = 0

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        def evaluate_batch() -> None:
            nonlocal inference_seconds, seen_images, total_targets
            inputs = processor(images=batch_images, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            target_sizes = torch.tensor(
                [(height, width) for width, height in batch_sizes],
                device=device
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            with torch.inference_mode():
                outputs = model(pixel_values=pixel_values)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            predictions = processor.post_process_object_detection(
                outputs,
                threshold=0.0,
                target_sizes=target_sizes
            )

            for path, size, (target_boxes, target_classes), prediction in zip(
                batch_paths, batch_sizes, batch_targets, predictions
            ):
                width, height = size
                pred_boxes = prediction["boxes"].detach().cpu().numpy().astype(np.float32)
                pred_classes = prediction["labels"].detach().cpu().numpy().astype(np.float32)
                confidences = prediction["scores"].detach().cpu().numpy().astype(np.float32)
                total_targets += len(target_classes)
                if len(target_boxes) and len(pred_boxes):
                    overlaps = box_iou(
                        torch.from_numpy(target_boxes),
                        torch.from_numpy(pred_boxes)
                    ).numpy()
                    correct = np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool)
                    same_class = target_classes[:, None] == pred_classes[None, :]
                    overlaps *= same_class
                    for threshold_index, threshold in enumerate(iou_thresholds):
                        matches = np.array(np.nonzero(overlaps >= threshold)).T
                        if len(matches):
                            if len(matches) > 1:
                                matches = matches[np.argsort(overlaps[matches[:, 0], matches[:, 1]])[::-1]]
                                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
                            correct[matches[:, 1].astype(int), threshold_index] = True
                else:
                    correct = np.zeros((len(pred_boxes), len(iou_thresholds)), dtype=bool)
                stats.append((correct, confidences, pred_classes, target_classes))
                seen_images += 1

        for image_path in image_paths:
            label_path = label_split / f"{image_path.stem}.txt"
            if not label_path.is_file():
                raise FileNotFoundError(f"Missing test label for image: {image_path}")
            with Image.open(image_path) as source_image:
                image = source_image.convert("RGB")
            width, height = image.size
            target_boxes = []
            target_classes = []
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1
            ):
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
                class_id, x_center, y_center, box_width, box_height = map(float, fields)
                if not class_id.is_integer() or int(class_id) not in self.class_names:
                    raise ValueError(f"Invalid class id at {label_path}:{line_number}: {class_id}")
                target_classes.append(int(class_id))
                target_boxes.append([
                    (x_center - box_width / 2) * width,
                    (y_center - box_height / 2) * height,
                    (x_center + box_width / 2) * width,
                    (y_center + box_height / 2) * height,
                ])
            batch_paths.append(image_path)
            batch_images.append(image)
            batch_sizes.append((width, height))
            batch_targets.append((
                np.asarray(target_boxes, dtype=np.float32).reshape(-1, 4),
                np.asarray(target_classes, dtype=np.float32)
            ))
            if len(batch_paths) == batch_size:
                evaluate_batch()
                batch_paths.clear()
                batch_images.clear()
                batch_sizes.clear()
                batch_targets.clear()
                if seen_images % 200 == 0:
                    print(f"  Evaluated {seen_images}/{len(image_paths)} images")
        if batch_paths:
            evaluate_batch()

        if seen_images != len(image_paths):
            raise RuntimeError(f"Evaluated {seen_images} images, expected {len(image_paths)}")
        if total_targets == 0:
            raise ValueError(f"No labeled objects found in evaluation split: {image_split}")

        tp = np.concatenate([item[0] for item in stats], axis=0)
        conf = np.concatenate([item[1] for item in stats], axis=0)
        pred_cls = np.concatenate([item[2] for item in stats], axis=0)
        target_cls = np.concatenate([item[3] for item in stats], axis=0)
        _, _, precision, recall, f1, ap, ap_classes, *_ = ap_per_class(
            tp, conf, pred_cls, target_cls, names=self.class_names
        )
        per_class = {}
        for class_id, class_name in self.class_names.items():
            class_matches = np.flatnonzero(ap_classes == class_id)
            class_index = int(class_matches[0]) if len(class_matches) else None
            per_class[class_name] = {
                "ap50": round(float(ap[class_index, 0]), 4) if class_index is not None else 0.0,
                "ap50_95": round(float(ap[class_index].mean()), 4) if class_index is not None else 0.0
            }
        mean_5 = float(np.mean([
            per_class[name]["ap50"]
            for name in (
                "insulator_broken", "insulator_flashover", "bird_nest",
                "foreign_object", "broken_strand"
            )
        ]))
        accuracy = {
            "precision": round(float(precision.mean()), 4) if len(precision) else 0.0,
            "recall": round(float(recall.mean()), 4) if len(recall) else 0.0,
            "f1_score": round(float(f1.mean()), 4) if len(f1) else 0.0,
            "mAP50": round(float(ap[:, 0].mean()), 4) if len(ap) else 0.0,
            "mAP50_95": round(float(ap.mean()), 4) if len(ap) else 0.0,
            "MEAN_5_defect_ap50": round(mean_5, 4),
            "per_class": per_class
        }
        parameter_count = sum(parameter.numel() for parameter in model.parameters()) / 1e6
        latency_ms = inference_seconds / seen_images * 1000
        hardware = {
            "parameters_m": round(parameter_count, 2),
            "gflops": gflops,
            "latency_ms": round(latency_ms, 2),
            "fps": round(1000 / latency_ms, 1) if latency_ms else 0.0,
            "inference_peak_vram_mb": round(
                torch.cuda.max_memory_allocated(device) / (1024 ** 2), 1
            ) if device.type == "cuda" else 0.0
        }
        record = {
            "run_key": run_key,
            "checkpoint_path": str(checkpoint_path),
            "evaluation_split": split,
            "evaluation_dataset_yaml": str(yaml_path),
            "accuracy": accuracy,
            "hardware": hardware,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.eval_data[run_key] = record
        self._save_results()
        print(
            f"  Result -> mAP50: {accuracy['mAP50']:.3f} | "
            f"mAP50-95: {accuracy['mAP50_95']:.3f} | "
            f"Latency: {hardware['latency_ms']:.1f}ms | FPS: {hardware['fps']}"
        )
        return record

    def profile_hardware(self, model: Any, imgsz: int = 640) -> Dict[str, Any]:
        """Profiles parameters, GFLOPS, latency (ms), and FPS on GPU."""
        device = torch.device(f"cuda:{self.cfg['training']['device']}" if torch.cuda.is_available() else "cpu")
        
        # Count Parameters
        n_params = sum(p.numel() for p in model.model.parameters()) / 1e6
        
        # Reset Peak VRAM
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.empty_cache()

        # Warmup
        dummy_input = torch.zeros((1, 3, imgsz, imgsz), device=device)
        for _ in range(self.cfg["profiling"]["warmup_runs"]):
            _ = model.predict(dummy_input, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Measure Latency
        n_iters = self.cfg["profiling"]["benchmark_images"]
        start_time = time.time()
        for _ in range(n_iters):
            _ = model.predict(dummy_input, verbose=False)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        total_time = time.time() - start_time
        avg_latency_ms = (total_time / n_iters) * 1000.0
        fps = n_iters / total_time
        peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024**2)) if torch.cuda.is_available() else 0.0

        # Estimate GFLOPS if available from ultralytics thop / info
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
            "inference_peak_vram_mb": round(peak_vram_mb, 1)
        }

    def evaluate_checkpoint(
        self,
        run_key: str,
        checkpoint_path: str | Path,
        test_dataset_yaml: str | Path,
        force: bool = False,
        split: str = "test"
    ) -> Dict[str, Any]:
        """Evaluates a single model checkpoint on the requested dataset split."""
        test_dataset_yaml = Path(test_dataset_yaml).resolve()
        checkpoint_path = Path(checkpoint_path).resolve()
        cached_result = self.eval_data.get(run_key)
        if (
            not force
            and cached_result
            and cached_result.get("evaluation_split") == split
            and cached_result.get("evaluation_dataset_yaml") == str(test_dataset_yaml)
            and cached_result.get("checkpoint_path") == str(checkpoint_path)
        ):
            if (
                re.search(r"rtdetr(50|100)$", run_key.lower())
                and cached_result.get("hardware", {}).get("gflops", 0) <= 0
            ):
                model, _ = self._load_huggingface_rtdetr(run_key, checkpoint_path)
                cached_result["hardware"]["gflops"] = self._profile_huggingface_gflops(
                    model, self.cfg["training"]["imgsz"]
                )
                self._save_results()
                del model
            print(f"[REUSE] Evaluation for '{run_key}' already cached.")
            return cached_result

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"\n--- [EVALUATE] Evaluating '{run_key}' on split '{split}' ---")

        if re.search(r"rtdetr(50|100)$", run_key.lower()):
            return self._evaluate_huggingface_rtdetr(
                run_key, checkpoint_path, test_dataset_yaml, split
            )
        
        from ultralytics import YOLO, RTDETR
        is_rtdetr = "rtdetr" in run_key.lower()
        model_cls = RTDETR if is_rtdetr else YOLO
        model = model_cls(str(checkpoint_path))

        # Run validation on the requested split.
        val_res = model.val(
            data=str(test_dataset_yaml.resolve()),
            split=split,
            imgsz=self.cfg["training"]["imgsz"],
            device=self.cfg["training"]["device"],
            plots=False,
            verbose=False
        )

        # Extract accuracy metrics
        p_all = float(val_res.results_dict.get("metrics/precision(B)", 0.0))
        r_all = float(val_res.results_dict.get("metrics/recall(B)", 0.0))
        f1_all = 2 * (p_all * r_all) / (p_all + r_all + 1e-8)
        map50_all = float(val_res.results_dict.get("metrics/mAP50(B)", 0.0))
        map50_95_all = float(val_res.results_dict.get("metrics/mAP50-95(B)", 0.0))

        # Per-class metrics
        per_class = {}
        ap50_per_class = val_res.box.ap50 if hasattr(val_res, "box") else []
        ap50_95_per_class = val_res.box.ap if hasattr(val_res, "box") else []
        
        for idx, name in self.class_names.items():
            c_ap50 = float(ap50_per_class[idx]) if idx < len(ap50_per_class) else 0.0
            c_ap50_95 = float(ap50_95_per_class[idx]) if idx < len(ap50_95_per_class) else 0.0
            per_class[name] = {
                "ap50": round(c_ap50, 4),
                "ap50_95": round(c_ap50_95, 4)
            }

        # Calculate MEAN_5 (5 defect classes, excluding healthy insulator)
        defect_classes = ["insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
        mean_5_ap50 = float(np.mean([per_class[c]["ap50"] for c in defect_classes if c in per_class]))

        # Hardware profiling
        hw_profile = self.profile_hardware(model, self.cfg["training"]["imgsz"])

        # Compile record
        record = {
            "run_key": run_key,
            "checkpoint_path": str(checkpoint_path),
            "evaluation_split": split,
            "evaluation_dataset_yaml": str(test_dataset_yaml),
            "accuracy": {
                "precision": round(p_all, 4),
                "recall": round(r_all, 4),
                "f1_score": round(f1_all, 4),
                "mAP50": round(map50_all, 4),
                "mAP50_95": round(map50_95_all, 4),
                "MEAN_5_defect_ap50": round(mean_5_ap50, 4),
                "per_class": per_class
            },
            "hardware": hw_profile,
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.eval_data[run_key] = record
        self._save_results()
        print(f"  Result -> mAP50: {map50_all:.3f} | mAP50-95: {map50_95_all:.3f} | Latency: {hw_profile['latency_ms']:.1f}ms | FPS: {hw_profile['fps']}")
        return record

    def evaluate_all(
        self,
        state_runs: Dict[str, Any],
        test_dataset_yaml: str | Path,
        split_by_dataset: Optional[Dict[str, str]] = None,
        checkpoint_field: str = "best_weights"
    ) -> pd.DataFrame:
        """Evaluates all runs using the selected checkpoint field."""
        records = []
        for run_key, run_info in state_runs.items():
            checkpoint_path = run_info.get(checkpoint_field)
            if checkpoint_path and Path(checkpoint_path).exists():
                dataset, model = run_key.split("_", 1)
                split = (split_by_dataset or {}).get(dataset, "test")
                res = self.evaluate_checkpoint(run_key, checkpoint_path, test_dataset_yaml, split=split)
                
                row = {
                    "Run": run_key,
                    "Dataset": dataset.upper(),
                    "Model": model,
                    "Evaluation Split": split,
                    "Precision": res["accuracy"]["precision"],
                    "Recall": res["accuracy"]["recall"],
                    "F1-Score": res["accuracy"]["f1_score"],
                    "mAP@0.5": res["accuracy"]["mAP50"],
                    "mAP@0.5:0.95": res["accuracy"]["mAP50_95"],
                    "MEAN_5": res["accuracy"]["MEAN_5_defect_ap50"],
                    "Latency (ms)": res["hardware"]["latency_ms"],
                    "FPS": res["hardware"]["fps"],
                    "GFLOPS": res["hardware"]["gflops"],
                    "Params (M)": res["hardware"]["parameters_m"],
                    "VRAM (MB)": res["hardware"]["inference_peak_vram_mb"]
                }
                records.append(row)

        df = pd.DataFrame(records)
        df.to_csv(self.results_dir / "benchmark_summary.csv", index=False)
        return df
