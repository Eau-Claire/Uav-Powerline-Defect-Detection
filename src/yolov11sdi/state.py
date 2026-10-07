"""Resumable stage state: state/<stage>.json.

Rules implemented here:
* a complete stage is reused when its config hash, input hashes and outputs
  still match (FORCE_REBUILD=False by default);
* a config change marks the old record `stale` (kept in `history`), never
  deletes outputs;
* every write is atomic; progress writes are throttled.
"""
from __future__ import annotations

import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .hashing import sha256_file
from .io_utils import atomic_write_json, read_json

STATUSES = ("pending", "running", "complete", "failed", "needs_review", "stale")


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class FileRecord:
    name: str
    path: str
    sha256: str | None = None
    size: int | None = None
    remote: str | None = None

    @classmethod
    def of(cls, name: str, path: Path, stored_path: str, remote: str | None = None,
           with_sha: bool = True) -> "FileRecord":
        return cls(
            name=name,
            path=stored_path,
            sha256=sha256_file(path) if with_sha and path.is_file() else None,
            size=path.stat().st_size if path.is_file() else None,
            remote=remote,
        )


@dataclass
class StageState:
    stage: str
    status: str = "pending"
    started_at: str | None = None
    updated_at: str | None = None
    completed_at: str | None = None
    config_hash: str | None = None
    git_commit: str | None = None
    host: str | None = None
    environment: str | None = None
    inputs: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    error: str | None = None
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StageState":
        known = {k: d[k] for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)

    def output(self, name: str) -> dict | None:
        return next((o for o in self.outputs if o.get("name") == name), None)


class StateStore:
    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        return self.dir / f"{stage}.json"

    def load(self, stage: str) -> StageState | None:
        d = read_json(self.path(stage))
        return StageState.from_dict(d) if d else None

    def save(self, state: StageState) -> None:
        state.updated_at = utc_now()
        atomic_write_json(self.path(state.stage), state.to_dict())

    def all(self) -> dict[str, StageState]:
        out = {}
        for p in sorted(self.dir.glob("*.json")):
            d = read_json(p)
            if d and "stage" in d:
                out[d["stage"]] = StageState.from_dict(d)
        return out


def _inputs_signature(records: list[dict]) -> dict:
    return {r["name"]: r.get("sha256") for r in records}


def check_reusable(
    state: StageState | None,
    config_hash: str,
    inputs: list[dict],
    resolve: Callable[[str], Path],
    deep: bool = False,
) -> tuple[bool, str]:
    """Can a previous complete result be reused as-is?"""
    if state is None:
        return False, "no previous state"
    if state.status != "complete":
        return False, f"previous status={state.status}"
    if state.config_hash != config_hash:
        return False, "config hash changed"
    if _inputs_signature(state.inputs) != _inputs_signature(inputs):
        return False, "input hashes changed"
    for o in state.outputs:
        if o.get("path") is None:
            continue
        p = resolve(o["path"])
        if not p.exists():
            return False, f"output missing: {o['path']}"
        if o.get("size") is not None and p.is_file() and p.stat().st_size != o["size"]:
            return False, f"output size changed: {o['path']}"
        if deep and o.get("sha256") and p.is_file() and sha256_file(p) != o["sha256"]:
            return False, f"output sha256 changed: {o['path']}"
    return True, "complete and matching"


class StageRun:
    """Handle for one execution of a stage; owns its state record."""

    def __init__(self, store: StateStore, stage: str, config_hash: str, inputs: list[dict],
                 meta: dict, progress_interval_s: float = 15.0):
        self.store = store
        self.interval = progress_interval_s
        self._last_write = 0.0
        prev = store.load(stage)
        history = list(prev.history) if prev else []
        if prev and prev.status in ("complete", "needs_review") and prev.config_hash != config_hash:
            snapshot = prev.to_dict()
            snapshot.pop("history", None)
            snapshot["status"] = "stale"
            history.append(snapshot)
        # Keep partial progress from a previous attempt with the same config,
        # so incremental checkpoints can report where they resume from.
        progress = prev.progress if prev and prev.config_hash == config_hash else {}
        self.state = StageState(
            stage=stage, status="running", started_at=utc_now(), config_hash=config_hash,
            inputs=inputs, progress=progress, history=history[-10:], **meta,
        )
        store.save(self.state)

    def progress(self, current: int, total: int | None = None, force: bool = False, **extra) -> None:
        self.state.progress = {"current": int(current), "total": total, **extra}
        now = time.monotonic()
        if force or now - self._last_write >= self.interval:
            self._last_write = now
            self.store.save(self.state)

    def add_summary(self, **kv) -> None:
        self.state.summary.update(kv)

    def complete(self, outputs: list[FileRecord | dict], **summary) -> StageState:
        self.state.outputs = [asdict(o) if isinstance(o, FileRecord) else o for o in outputs]
        self.state.summary.update(summary)
        self.state.status = "complete"
        self.state.completed_at = utc_now()
        self.state.error = None
        self.store.save(self.state)
        return self.state

    def needs_review(self, outputs: list[FileRecord | dict] | None = None, **summary) -> StageState:
        if outputs is not None:
            self.state.outputs = [asdict(o) if isinstance(o, FileRecord) else o for o in outputs]
        self.state.summary.update(summary)
        self.state.status = "needs_review"
        self.store.save(self.state)
        return self.state

    def fail(self, exc: BaseException) -> StageState:
        self.state.status = "failed"
        self.state.error = "".join(traceback.format_exception_only(type(exc), exc)).strip()[:4000]
        self.store.save(self.state)
        return self.state
