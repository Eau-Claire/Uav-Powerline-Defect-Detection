"""Run directory + run_state.json + resume decision for Ultralytics training.

runs/<family>/<run_name>/
    weights/best.pt, weights/last.pt     (written by Ultralytics)
    results.csv, args.yaml               (written by Ultralytics)
    experiment_manifest.json, run_state.json
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

from ..hashing import stable_json_hash
from ..io_utils import atomic_write_json, read_json
from ..state import utc_now

# Training keys that materially change a run. Others (device, workers, sync
# cadence, plots) may change between sessions without invalidating last.pt.
MATERIAL_TRAINING_KEYS = ("model", "imgsz", "epochs", "batch", "seed", "pretrained", "ultralytics_version")


def train_hash(training_cfg: dict, dataset_tag: str, compose_tag: str) -> str:
    return stable_json_hash({
        "training": {k: training_cfg.get(k) for k in MATERIAL_TRAINING_KEYS},
        "dataset_tag": dataset_tag,
        "compose_tag": compose_tag,
    })


@dataclass
class RunState:
    experiment: str
    model: str
    dataset_tag: str
    compose_tag: str
    imgsz: int
    epochs_target: int
    train_hash: str
    last_completed_epoch: int = 0
    best_metric: float | None = None
    best_epoch: int | None = None
    checkpoint: str | None = None
    status: str = "pending"            # pending|running|trained|evaluated|failed
    remote_root: str | None = None
    last_synced_epoch: int | None = None
    updated_at: str | None = None
    history: list[dict] = field(default_factory=list)
    # One entry per train() call: which code (git commit / package version),
    # host and environment trained which epochs. Lets a local run be traced
    # back to `git checkout <commit>`.
    sessions: list[dict] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "RunState | None":
        d = read_json(path)
        if not d:
            return None
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})

    def save(self, path: Path) -> None:
        self.updated_at = utc_now()
        atomic_write_json(path, asdict(self))


class ResumeAction(str, Enum):
    FRESH = "fresh"
    RESUME = "resume"
    ALREADY_TRAINED = "already_trained"
    STALE = "stale"          # previous run exists but is incompatible


def decide_resume(run_dir: Path, current_hash: str, force: bool) -> tuple[ResumeAction, str]:
    state = RunState.load(run_dir / "run_state.json")
    last = run_dir / "weights" / "last.pt"
    if force:
        return (ResumeAction.STALE if run_dir.exists() and any(run_dir.iterdir()) else ResumeAction.FRESH,
                "force rebuild requested")
    if state is None and not last.exists():
        return ResumeAction.FRESH, "no previous run"
    if state is not None and state.train_hash != current_hash:
        return ResumeAction.STALE, "training config / dataset / compose changed materially"
    if state is not None and state.status in ("trained", "evaluated") and (run_dir / "weights/best.pt").exists():
        return ResumeAction.ALREADY_TRAINED, f"run finished at epoch {state.last_completed_epoch}"
    if last.exists():
        if not checkpoint_loadable(last):
            return ResumeAction.STALE, "last.pt is not loadable"
        return ResumeAction.RESUME, f"resume from last.pt (epoch {state.last_completed_epoch if state else '?'})"
    return ResumeAction.FRESH, "state without checkpoint"


def checkpoint_loadable(path: Path) -> bool:
    try:
        import torch

        ck = torch.load(path, map_location="cpu", weights_only=False)
        return isinstance(ck, dict) and ("model" in ck or "ema" in ck)
    except Exception:  # noqa: BLE001
        return False


def quarantine_run(run_dir: Path, reason_hash: str) -> Path | None:
    """Never delete prior work: move an incompatible run aside."""
    if not run_dir.exists():
        return None
    target = run_dir.with_name(f"{run_dir.name}__stale_{reason_hash[:8]}")
    n = 1
    while target.exists():
        target = run_dir.with_name(f"{run_dir.name}__stale_{reason_hash[:8]}_{n}")
        n += 1
    run_dir.rename(target)
    return target


def last_epoch_from_results(results_csv: Path) -> int:
    if not results_csv.exists():
        return 0
    with open(results_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return 0
    key = next((k for k in rows[-1] if k.strip() == "epoch"), None)
    try:
        return int(float(rows[-1][key])) if key else len(rows)
    except (TypeError, ValueError):
        return len(rows)
