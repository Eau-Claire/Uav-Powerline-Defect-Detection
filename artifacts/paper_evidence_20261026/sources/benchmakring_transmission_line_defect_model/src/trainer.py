from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch

from .config import ROOT_DIR, get_benchmark_config

class BenchmarkTrainer:
    def __init__(self, config_path: str | Path = "configs/benchmark_config.yaml"):
        self.cfg = get_benchmark_config(config_path)
        self.runs_dir = ROOT_DIR / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.runs_dir / "benchmark_state.json"
        self.state = self._load_state()

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {"runs": {}, "updated_at": time.time()}

    def _save_state(self) -> None:
        self.state["updated_at"] = time.time()
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def is_run_completed(self, run_key: str) -> bool:
        run_info = self.state.get("runs", {}).get(run_key, {})
        if run_info.get("status") == "completed":
            best_pt = Path(run_info.get("best_weights", ""))
            if best_pt.exists() and best_pt.stat().st_size > 0:
                return True
        return False

    def get_run_status(self) -> Dict[str, str]:
        """Returns the status of all 12 experimental variants."""
        statuses = {}
        for d_key in ["raw", "curated"]:
            for m_key in self.cfg["models"].keys():
                run_key = f"{d_key}_{m_key}"
                if self.is_run_completed(run_key):
                    statuses[run_key] = "COMPLETED"
                else:
                    run_dir = self.runs_dir / run_key
                    last_pt = run_dir / "weights" / "last.pt"
                    if last_pt.exists():
                        statuses[run_key] = "IN_PROGRESS (Resumable)"
                    else:
                        statuses[run_key] = "PENDING"
        return statuses

    def train_single_run(
        self,
        dataset_type: str,
        model_key: str,
        dataset_yaml: Optional[Path] = None,
        force_restart: bool = False
    ) -> Dict[str, Any]:
        """Trains or resumes a single model variant with automatic VRAM management."""
        run_key = f"{dataset_type}_{model_key}"
        run_dir = self.runs_dir / run_key
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if already completed
        if not force_restart and self.is_run_completed(run_key):
            print(f"[SKIP] Run '{run_key}' already completed. Using saved checkpoint.")
            return self.state["runs"][run_key]

        model_cfg = self.cfg["models"][model_key]
        train_cfg = self.cfg["training"]
        
        # Determine dataset yaml
        if dataset_yaml is None:
            dataset_yaml = ROOT_DIR / self.cfg["datasets"][dataset_type]["yaml_path"]
        if not dataset_yaml.exists():
            raise FileNotFoundError(f"Dataset YAML not found: {dataset_yaml}")

        last_pt = run_dir / "weights" / "last.pt"
        is_resuming = last_pt.exists() and not force_restart

        print(f"\n{'='*70}")
        if is_resuming:
            print(f"[RESUME] Resuming training for '{run_key}' from: {last_pt}")
        else:
            print(f"[START] Starting new training run for '{run_key}'")
        print(f"  Architecture: {model_key} | Family: {model_cfg['family']}")
        print(f"  Dataset: {dataset_type} ({dataset_yaml.name})")
        print(f"  Batch: {model_cfg['batch']} | Accumulate: {model_cfg['accumulate']}")
        print(f"  Image Size: {train_cfg['imgsz']} | Optimizer: {train_cfg['optimizer']}")
        print(f"{'='*70}\n")

        # Import Ultralytics
        from ultralytics import YOLO, RTDETR
        
        # Model instantiation
        if model_cfg["type"] == "rtdetr":
            model_cls = RTDETR
        else:
            model_cls = YOLO

        if is_resuming:
            model = model_cls(str(last_pt))
        else:
            weights_src = model_cfg["weights"]
            # Check local weights folder first if available
            local_weights = ROOT_DIR.parent / "Uav-Powerline-Defect-Detection" / "legacy/notebooks/v6p5/weights" / weights_src
            if local_weights.exists():
                weights_src = str(local_weights)
            model = model_cls(weights_src)

        # Build train kwargs
        train_args = dict(
            data=str(dataset_yaml.resolve()),
            epochs=train_cfg["epochs"],
            imgsz=train_cfg["imgsz"],
            batch=model_cfg["batch"],
            device=train_cfg["device"],
            optimizer=train_cfg["optimizer"],
            lr0=train_cfg["lr0"],
            lrf=train_cfg["lrf"],
            cos_lr=train_cfg["cos_lr"],
            warmup_epochs=train_cfg["warmup_epochs"],
            weight_decay=train_cfg["weight_decay"],
            close_mosaic=train_cfg["close_mosaic"],
            amp=train_cfg["amp"],
            seed=train_cfg["seed"],
            deterministic=train_cfg["deterministic"],
            workers=train_cfg["workers"],
            project=str(self.runs_dir),
            name=run_key,
            exist_ok=True,
            save=True,
            save_period=train_cfg["save_period"],
            plots=train_cfg["plots"],
            resume=is_resuming
        )

        start_time = time.time()
        start_vram = torch.cuda.memory_allocated(0) if torch.cuda.is_available() else 0
        peak_vram_start = torch.cuda.max_memory_allocated(0) if torch.cuda.is_available() else 0

        # Execute Training
        results = model.train(**train_args)
        
        elapsed_sec = time.time() - start_time
        peak_vram_mb = (torch.cuda.max_memory_allocated(0) / (1024**2)) if torch.cuda.is_available() else 0

        best_pt = run_dir / "weights" / "best.pt"
        run_record = {
            "run_key": run_key,
            "dataset": dataset_type,
            "model": model_key,
            "family": model_cfg["family"],
            "status": "completed",
            "best_weights": str(best_pt.resolve()),
            "last_weights": str(last_pt.resolve()),
            "run_dir": str(run_dir.resolve()),
            "training_time_seconds": elapsed_sec,
            "peak_vram_mb": peak_vram_mb,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        self.state["runs"][run_key] = run_record
        self._save_state()
        print(f"\n[DONE] Run '{run_key}' finished successfully in {elapsed_sec/60:.2f} mins.")
        return run_record

    def run_all_benchmarks(
        self,
        datasets: Optional[List[str]] = None,
        models: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Runs the complete 2x6 Factorial Benchmark suite."""
        if datasets is None:
            datasets = ["raw", "curated"]
        if models is None:
            models = list(self.cfg["models"].keys())

        print(f"\n>>> Launching Factorial Benchmark: {len(datasets)} datasets x {len(models)} models = {len(datasets)*len(models)} runs <<<")
        for d in datasets:
            for m in models:
                self.train_single_run(d, m)
                
        return self.state["runs"]

