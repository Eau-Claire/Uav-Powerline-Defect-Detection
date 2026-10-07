"""Runtime environment detection (LOCAL / KAGGLE / CAMBER) and host facts."""
from __future__ import annotations

import os
import shutil
import socket
import subprocess
from enum import Enum
from pathlib import Path

ENV_OVERRIDE = "YOLOV11SDI_ENV"
CAMBER_KEY_ENV = "CAMBER_API_KEY"


class RuntimeEnv(str, Enum):
    LOCAL = "local"
    KAGGLE = "kaggle"
    CAMBER = "camber"


def detect_environment() -> RuntimeEnv:
    """Explicit override first, then Kaggle markers, then Camber worker markers."""
    override = os.environ.get(ENV_OVERRIDE)
    if override:
        return RuntimeEnv(override.strip().lower())
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") or os.environ.get("KAGGLE_URL_BASE"):
        return RuntimeEnv.KAGGLE
    if Path("/kaggle/working").is_dir():
        return RuntimeEnv.KAGGLE
    # Camber job workers: we cannot rely on an official marker, so launchers
    # set YOLOV11SDI_ENV=camber; these are best-effort fallbacks.
    if any(os.environ.get(k) for k in ("CAMBER_JOB_ID", "CAMBER_WORKER", "CAMBER_ENGINE_ID")):
        return RuntimeEnv.CAMBER
    return RuntimeEnv.LOCAL


def hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def git_commit(root: Path) -> str | None:
    if shutil.which("git") is None:
        return None
    try:
        r = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if r.returncode != 0:
        return None
    sha = r.stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True
    ).stdout.strip()
    return f"{sha}+dirty" if dirty else sha


def load_camber_api_key(env: RuntimeEnv | None = None) -> bool:
    """Ensure CAMBER_API_KEY is in os.environ. Never logs or returns the value.

    Sources: existing environment, then Kaggle Secrets (on Kaggle only).
    """
    if os.environ.get(CAMBER_KEY_ENV):
        return True
    env = env or detect_environment()
    if env is RuntimeEnv.KAGGLE:
        try:
            from kaggle_secrets import UserSecretsClient  # type: ignore

            key = UserSecretsClient().get_secret(CAMBER_KEY_ENV)
        except Exception:
            key = None
        if key:
            os.environ[CAMBER_KEY_ENV] = key
            return True
    return False


def disk_usage(path: Path) -> dict:
    path = Path(path)
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    total, used, free = shutil.disk_usage(probe)
    gb = 1024**3
    return {"path": str(path), "total_gb": total / gb, "used_gb": used / gb, "free_gb": free / gb}
