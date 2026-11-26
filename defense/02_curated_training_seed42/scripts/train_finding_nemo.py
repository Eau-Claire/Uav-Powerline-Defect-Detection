#!/usr/bin/env python
"""Train four YOLO variants sequentially on the human-reviewed dataset.

Each run is resumable from local ``weights/last.pt``. Checkpoints and run
metadata are mirrored to ``stash://bunpmc/projects/finding_nemo/<model>/`` every
five epochs and the complete run folder is uploaded after training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import YOLO


REPO = Path(__file__).resolve().parents[2]
BENCHMARK = REPO / "benchmakring_transmission_line_defect_model"
DATASET_ROOT = BENCHMARK / "data/finding_nemo_human_reviewed_6cls"
DATA = DATASET_ROOT / "data.yaml"
REVIEW_STATUS = DATASET_ROOT / "DATASET_STATUS.json"
RUNS = BENCHMARK / "runs/finding_nemo"
STATE = RUNS / "training_state.json"
REMOTE_ROOT = "stash://bunpmc/projects/finding_nemo"
CAMBER_BIN = "/home/dotru/.camber/bin/camber"
ORDER = ("yolo11n", "yolo11m", "yolo26n", "yolo26m")
MODEL_CONFIG = {
    "yolo11n": {"weights": BENCHMARK / "yolo11n.pt", "batch": 16},
    "yolo11m": {"weights": BENCHMARK / "yolo11m.pt", "batch": 8},
    "yolo26n": {"weights": BENCHMARK / "weights/yolo26n.pt", "batch": 16},
    "yolo26m": {"weights": BENCHMARK / "yolo26m.pt", "batch": 8},
}


def load_api_key() -> None:
    if os.environ.get("CAMBER_API_KEY"):
        return
    env_file = REPO / ".env"
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "CAMBER_API_KEY" and value.strip():
            os.environ["CAMBER_API_KEY"] = value.strip().strip("'\"")
            return
    raise RuntimeError(f"CAMBER_API_KEY missing from {env_file}")


def wsl_path(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    if not drive:
        raise ValueError(f"Expected Windows drive path: {resolved}")
    tail = resolved.as_posix().split(":", 1)[1]
    return f"/mnt/{drive}{tail}"


def camber_environment() -> dict[str, str]:
    env = os.environ.copy()
    entries = [entry for entry in env.get("WSLENV", "").split(":") if entry]
    if "CAMBER_API_KEY" not in entries:
        entries.append("CAMBER_API_KEY")
    env["WSLENV"] = ":".join(entries)
    return env


def camber_cp(source: Path | str, destination: Path | str, attempts: int = 3) -> None:
    source_value = wsl_path(source) if isinstance(source, Path) else source
    destination_value = wsl_path(destination) if isinstance(destination, Path) else destination
    command = ["wsl", "--", CAMBER_BIN, "stash", "cp", source_value, destination_value]
    last_error = ""
    for attempt in range(1, attempts + 1):
        result = subprocess.run(command, env=camber_environment(), text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        if result.returncode == 0:
            return
        last_error = result.stdout[-2000:]
        if attempt < attempts:
            time.sleep(5 * attempt)
    raise RuntimeError(f"Camber copy failed after {attempts} attempts:\n{last_error}")


def upload_file(local: Path, model_name: str, run_dir: Path) -> None:
    if not local.is_file():
        return
    relative = local.relative_to(run_dir).as_posix()
    camber_cp(local, f"{REMOTE_ROOT}/{model_name}/{relative}")


def checkpoint_files(trainer: Any, run_dir: Path) -> list[Path]:
    files = [run_dir / "weights/last.pt", run_dir / "weights/best.pt",
             run_dir / "args.yaml", run_dir / "results.csv"]
    epoch_number = trainer.epoch + 1
    epoch_checkpoint = run_dir / f"weights/epoch{epoch_number}.pt"
    if epoch_checkpoint.is_file():
        files.append(epoch_checkpoint)
    return files


def full_upload(model_name: str, run_dir: Path) -> None:
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.suffix != ".cache":
            upload_file(path, model_name, run_dir)


def read_state() -> dict[str, Any]:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"dataset": str(DATA), "remote_root": REMOTE_ROOT, "runs": {}}


def write_state(state: dict[str, Any]) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2), encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to start a ~20-hour CPU training sequence")
    if not DATA.is_file() or not REVIEW_STATUS.is_file():
        raise FileNotFoundError("Human-reviewed dataset or audit status is missing")
    review = json.loads(REVIEW_STATUS.read_text(encoding="utf-8"))
    if review.get("status") != "human_review_applied":
        raise RuntimeError(f"Dataset review status is not final: {review.get('status')}")
    metadata = yaml.safe_load(DATA.read_text(encoding="utf-8"))
    expected = ["insulator", "insulator_broken", "insulator_flashover",
                "bird_nest", "foreign_object", "broken_strand"]
    names = metadata["names"]
    if isinstance(names, dict):
        names = [names[index] for index in range(len(names))]
    if names != expected:
        raise RuntimeError(f"Unexpected taxonomy: {names}")
    missing = [str(config["weights"]) for config in MODEL_CONFIG.values()
               if not config["weights"].is_file()]
    if missing:
        raise FileNotFoundError(f"Missing pretrained weights: {missing}")


def prepare_state(models: list[str] | tuple[str, ...] = ORDER) -> dict[str, Any]:
    """Validate the environment and initialize local/remote run state."""
    load_api_key()
    validate_inputs()
    RUNS.mkdir(parents=True, exist_ok=True)
    state = read_state()
    state.update(dataset=str(DATA.resolve()), dataset_yaml_sha256=sha256(DATA),
                 review_status_sha256=sha256(REVIEW_STATUS), remote_root=REMOTE_ROOT,
                 order=list(models), device=torch.cuda.get_device_name(0))
    write_state(state)
    camber_cp(STATE, f"{REMOTE_ROOT}/training_state.json")
    return state


def train_one(model_name: str, state: dict[str, Any]) -> None:
    config = MODEL_CONFIG[model_name]
    run_dir = RUNS / model_name
    last = run_dir / "weights/last.pt"
    best = run_dir / "weights/best.pt"
    record = state["runs"].setdefault(model_name, {})
    if record.get("status") == "completed" and best.is_file():
        print(f"[SKIP] {model_name} already completed; refreshing Camber mirror", flush=True)
        full_upload(model_name, run_dir)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    is_resuming = last.is_file()
    model = YOLO(str(last if is_resuming else config["weights"]))
    print(f"[{model_name}] {'RESUME' if is_resuming else 'START'} | batch={config['batch']} | {run_dir}", flush=True)

    def on_model_save(trainer: Any) -> None:
        epoch_number = trainer.epoch + 1
        if epoch_number % 5 != 0 and epoch_number != trainer.epochs:
            return
        print(f"[{model_name}] syncing epoch {epoch_number} checkpoint to Camber", flush=True)
        try:
            for path in checkpoint_files(trainer, run_dir):
                upload_file(path, model_name, run_dir)
            record.update(status="training", last_synced_epoch=epoch_number,
                          updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            write_state(state)
        except Exception as exc:
            # Local checkpoint remains valid; next sync or launcher rerun retries it.
            record.update(status="training", sync_error=str(exc),
                          updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
            write_state(state)
            print(f"[{model_name}] checkpoint sync warning: {exc}", file=sys.stderr, flush=True)

    model.add_callback("on_model_save", on_model_save)
    record.update(status="training", resumed=is_resuming, run_dir=str(run_dir),
                  remote=f"{REMOTE_ROOT}/{model_name}",
                  started_at=record.get("started_at") or time.strftime("%Y-%m-%d %H:%M:%S"))
    write_state(state)
    try:
        model.train(
            data=str(DATA.resolve()), epochs=50, imgsz=640, batch=config["batch"], device=0,
            optimizer="AdamW", lr0=0.001, lrf=0.01, cos_lr=True, warmup_epochs=3.0,
            weight_decay=0.0005, close_mosaic=10, amp=True, seed=42, deterministic=True,
            workers=2, project=str(RUNS), name=model_name, exist_ok=True, save=True,
            save_period=5, plots=True, cache=False, pretrained=True, resume=is_resuming,
        )
    except BaseException as exc:
        # Preserve the most recent local checkpoint remotely when a notebook cell
        # is interrupted or training fails between scheduled five-epoch syncs.
        record.update(status="interrupted", error=f"{type(exc).__name__}: {exc}",
                      updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        write_state(state)
        try:
            for path in (last, best, run_dir / "args.yaml", run_dir / "results.csv"):
                upload_file(path, model_name, run_dir)
            camber_cp(STATE, f"{REMOTE_ROOT}/training_state.json")
        except Exception as sync_exc:
            print(f"[{model_name}] interruption sync warning: {sync_exc}", file=sys.stderr, flush=True)
        raise
    if not best.is_file() or not last.is_file():
        raise RuntimeError(f"Training returned without final checkpoints: {run_dir}")
    record.update(status="uploading", updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    write_state(state)
    full_upload(model_name, run_dir)
    record.update(status="completed", best_sha256=sha256(best), last_sha256=sha256(last),
                  completed_at=time.strftime("%Y-%m-%d %H:%M:%S"), sync_error=None)
    write_state(state)
    camber_cp(STATE, f"{REMOTE_ROOT}/training_state.json")
    print(f"[{model_name}] COMPLETE and uploaded", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=ORDER, default=list(ORDER))
    args = parser.parse_args()
    os.chdir(REPO)
    state = prepare_state(args.models)
    for model_name in args.models:
        train_one(model_name, state)
    state["status"] = "complete"
    state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_state(state)
    camber_cp(STATE, f"{REMOTE_ROOT}/training_state.json")


if __name__ == "__main__":
    main()
