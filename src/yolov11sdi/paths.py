"""Centralized filesystem layout. The ONLY place that knows about /kaggle/working.

Layout (local):          <project>/data, artifacts, state, logs, runs
Layout (Kaggle):         /kaggle/working/yolov11sdi/{data,artifacts,state,logs,runs}
Layout (Camber worker):  /tmp/yolov11sdi/...

Overrides:
  YOLOV11SDI_HOME       root for data/artifacts/state/logs/runs
  YOLOV11SDI_DATA_ROOT  put only bulky data/ on another disk
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .environment import RuntimeEnv, detect_environment

KAGGLE_HOME = Path("/kaggle/working/yolov11sdi")
CAMBER_HOME = Path("/tmp/yolov11sdi")


def find_project_root(start: Path | None = None) -> Path:
    env_root = os.environ.get("YOLOV11SDI_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()
    candidates = [start or Path.cwd(), Path(__file__).resolve()]
    for c in candidates:
        for p in [c, *c.parents]:
            if (p / "pyproject.toml").exists() and (p / "configs").is_dir():
                return p
    raise RuntimeError(
        "Project root not found (needs pyproject.toml + configs/). "
        "Set YOLOV11SDI_PROJECT_ROOT."
    )


@dataclass(frozen=True)
class ProjectPaths:
    project_root: Path
    home: Path
    data_root: Path
    env: RuntimeEnv

    @classmethod
    def detect(cls, project_root: Path | None = None, env: RuntimeEnv | None = None) -> "ProjectPaths":
        env = env or detect_environment()
        root = project_root or find_project_root()
        if os.environ.get("YOLOV11SDI_HOME"):
            home = Path(os.environ["YOLOV11SDI_HOME"])
        elif env is RuntimeEnv.KAGGLE:
            home = KAGGLE_HOME
        elif env is RuntimeEnv.CAMBER:
            home = CAMBER_HOME
        else:
            home = root
        data_root = Path(os.environ.get("YOLOV11SDI_DATA_ROOT", home / "data"))
        return cls(project_root=root.resolve(), home=home.resolve(), data_root=data_root.resolve(), env=env)

    # ------------------------------------------------------------ data/
    @property
    def cache(self) -> Path:
        return self.data_root / "cache"

    @property
    def external(self) -> Path:
        return self.data_root / "external"

    @property
    def parent(self) -> Path:
        return self.data_root / "parent"

    @property
    def working(self) -> Path:
        return self.data_root / "working"

    @property
    def bulk_mirror(self) -> Path:
        """Large remote archives mirrored with the Camber hierarchy."""
        return self.cache / "remote"

    # ------------------------------------------------------- artifacts/
    @property
    def artifacts(self) -> Path:
        return self.home / "artifacts"

    @property
    def remote_mirror(self) -> Path:
        return self.artifacts / "remote_mirror"

    @property
    def freezes(self) -> Path:
        return self.artifacts / "freezes"

    @property
    def manifests(self) -> Path:
        return self.artifacts / "manifests"

    @property
    def qa(self) -> Path:
        return self.artifacts / "qa"

    @property
    def compose(self) -> Path:
        return self.artifacts / "compose"

    @property
    def exports(self) -> Path:
        return self.artifacts / "exports"

    # ------------------------------------------------------------- other
    @property
    def state(self) -> Path:
        return self.home / "state"

    @property
    def logs(self) -> Path:
        return self.home / "logs"

    @property
    def runs(self) -> Path:
        return self.home / "runs"

    @property
    def configs(self) -> Path:
        return self.project_root / "configs"

    def ensure(self) -> "ProjectPaths":
        for p in [self.cache, self.external, self.parent, self.working, self.artifacts,
                  self.remote_mirror, self.freezes, self.manifests, self.qa, self.compose,
                  self.exports, self.state, self.logs, self.runs]:
            p.mkdir(parents=True, exist_ok=True)
        return self

    # ------------------------------------------------------- manifests
    def rel(self, path: str | Path) -> str:
        """Portable path for manifests: relative to home or data_root when possible."""
        p = Path(path).resolve()
        for base, prefix in ((self.data_root, "data"), (self.home, "")):
            try:
                r = p.relative_to(base).as_posix()
            except ValueError:
                continue
            return f"{prefix}/{r}" if prefix else r
        return p.as_posix()

    def resolve(self, stored: str | Path) -> Path:
        """Inverse of rel()."""
        s = str(stored)
        p = Path(s)
        if p.is_absolute():
            return p
        if s == "data" or s.startswith("data/"):
            return self.data_root / s[len("data/"):] if s != "data" else self.data_root
        return self.home / s

    def mirror_for(self, remote: str, stash_prefix: str, bulk: bool = False) -> Path:
        """Local mirror path for a Camber stash path (same hierarchy)."""
        prefix = stash_prefix.rstrip("/") + "/"
        if remote.startswith(prefix):
            rel = remote[len(prefix):]
        elif remote.startswith("stash://"):
            rel = "_other/" + remote[len("stash://"):]
        else:
            raise ValueError(f"Not a stash path: {remote}")
        return (self.bulk_mirror if bulk else self.remote_mirror) / rel
