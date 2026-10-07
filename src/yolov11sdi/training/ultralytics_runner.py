"""Ultralytics training with true resume, run_state tracking and optional Camber sync.

Resume rules:
* weights/last.pt + matching train hash  -> YOLO(last.pt).train(resume=True)
* finished run (best.pt, status trained)  -> skip training
* changed material config / dataset / compose, or unloadable checkpoint
                                          -> old run moved aside (never deleted), fresh start
* no local checkpoint but Camber has one  -> restore it, then resume
Camber sync is best-effort; local training never depends on it.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ..hashing import stable_json_hash
from ..io_utils import atomic_write_json
from ..state import utc_now
from .checkpoint import (ResumeAction, RunState, decide_resume, last_epoch_from_results, quarantine_run,
                         train_hash)

log = logging.getLogger("yolov11sdi.train")

SYNC_FILES = ("weights/last.pt", "weights/best.pt", "run_state.json", "results.csv", "args.yaml",
              "experiment_manifest.json")


def remote_run_tag(t: dict, compose_tag: str, train_count: int) -> str:
    """Legacy tag formula from the Kaggle train worker."""
    raw = json.dumps({"compose_tag": compose_tag, "model": t["model_label"], "imgsz": t["imgsz"],
                      "epochs": t["epochs"], "train_count": train_count}, sort_keys=True).encode()
    import hashlib

    return t["remote_tag_prefix"] + hashlib.sha256(raw).hexdigest()[:12]


def sync_run(camber, run_dir: Path, remote_root: str, names=SYNC_FILES) -> list[str]:
    if camber is None or not camber.available():
        return []
    pairs = [(run_dir / n, f"{remote_root}/{Path(n).name}") for n in names if (run_dir / n).exists()]
    return camber.put_many(pairs, required=False)


def try_restore_remote(camber, run_dir: Path, remote_root: str, th: str) -> bool:
    """Pull last.pt + run_state.json from Camber when the local run is missing."""
    if camber is None or not camber.available() or (run_dir / "weights/last.pt").exists():
        return False
    try:
        rs_local = camber.fetch(f"{remote_root}/run_state.json", run_dir / "run_state.json", reuse=False)
        rs = RunState.load(rs_local)
        if rs is None or rs.train_hash != th:
            log.warning("remote run_state does not match this run; not restoring")
            rs_local.unlink(missing_ok=True)
            return False
        camber.fetch(f"{remote_root}/last.pt", run_dir / "weights/last.pt", reuse=False)
        for n in ("best.pt",):
            try:
                camber.fetch(f"{remote_root}/{n}", run_dir / "weights" / n, reuse=False)
            except Exception:  # noqa: BLE001
                pass
        for n in ("results.csv", "args.yaml"):
            try:
                camber.fetch(f"{remote_root}/{n}", run_dir / n, reuse=False)
            except Exception:  # noqa: BLE001
                pass
        log.info("restored checkpoint from Camber at epoch %s", rs.last_completed_epoch)
        return True
    except Exception as e:  # noqa: BLE001
        log.info("no remote checkpoint restored (%s)", e)
        return False


def _check_device(device) -> None:
    if str(device).lower() == "cpu":
        return
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"training.device={device!r} but no CUDA GPU is visible. Run this stage on Kaggle/Camber "
            "GPU (scripts/export_kaggle.py) or set training.device=cpu for a smoke test."
        )


def train(cfg, data_yaml: Path, run_dir: Path, dataset_tag: str, compose_tag: str, remote_root: str,
          camber=None, force: bool = False, session: dict | None = None) -> RunState:
    from ultralytics import YOLO

    t = cfg.get("training")
    th = train_hash(t, dataset_tag, compose_tag)
    run_dir.mkdir(parents=True, exist_ok=True)
    try_restore_remote(camber, run_dir, remote_root, th)

    action, reason = decide_resume(run_dir, th, force)
    log.info("resume decision: %s (%s)", action.value, reason)
    if action is ResumeAction.STALE:
        moved = quarantine_run(run_dir, stable_json_hash([th, reason]))
        log.warning("previous run moved aside to %s", moved)
        run_dir.mkdir(parents=True, exist_ok=True)
        action = ResumeAction.FRESH

    state_path = run_dir / "run_state.json"
    state = RunState.load(state_path) or RunState(
        experiment=t["experiment"], model=t["model"], dataset_tag=dataset_tag, compose_tag=compose_tag,
        imgsz=int(t["imgsz"]), epochs_target=int(t["epochs"]), train_hash=th, remote_root=remote_root,
    )
    if action is ResumeAction.ALREADY_TRAINED:
        return state

    _check_device(t["device"])
    sync_every = int(t.get("sync_every_epochs") or 0)

    def on_fit_epoch_end(trainer) -> None:
        epoch = int(trainer.epoch) + 1
        state.last_completed_epoch = epoch
        bf = getattr(trainer, "best_fitness", None)
        if bf is not None and (state.best_metric is None or float(bf) >= state.best_metric):
            state.best_metric, state.best_epoch = float(bf), epoch
        state.checkpoint = str((run_dir / "weights/last.pt").relative_to(run_dir))
        state.status = "running"
        state.save(state_path)
        if sync_every and (epoch % sync_every == 0 or epoch == state.epochs_target):
            if sync_run(camber, run_dir, remote_root):
                state.last_synced_epoch = epoch
                state.save(state_path)

    state.status = "running"
    state.sessions.append({**(session or {}), "action": action.value, "start_epoch": state.last_completed_epoch,
                           "started_at": utc_now()})
    state.save(state_path)
    if action is ResumeAction.RESUME:
        model = YOLO(str(run_dir / "weights/last.pt"))
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(resume=True)
    else:
        model = YOLO(t["model"])
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.train(
            data=str(data_yaml), imgsz=int(t["imgsz"]), epochs=int(t["epochs"]), batch=t["batch"],
            device=t["device"], workers=int(t["workers"]), pretrained=bool(t["pretrained"]),
            seed=int(t["seed"]), project=str(run_dir.parent), name=run_dir.name, exist_ok=True,
            plots=bool(t["plots"]), verbose=True,
        )
    if not (run_dir / "weights/best.pt").exists():
        raise RuntimeError(f"Training finished without best.pt in {run_dir}")
    state.last_completed_epoch = max(state.last_completed_epoch, last_epoch_from_results(run_dir / "results.csv"))
    state.status = "trained"
    if state.sessions:
        state.sessions[-1].update(end_epoch=state.last_completed_epoch, ended_at=utc_now())
    state.save(state_path)
    if sync_run(camber, run_dir, remote_root):
        state.last_synced_epoch = state.last_completed_epoch
        state.save(state_path)
    return state


def write_report(run_dir: Path, report: dict) -> Path:
    import pandas as pd

    p = run_dir / "v6p5_yolo11n_data_ablation_report.json"
    atomic_write_json(p, report)
    pd.DataFrame(report["per_class"]).to_csv(run_dir / "v6p5_yolo11n_strict_iou_per_class.csv", index=False)
    return p
