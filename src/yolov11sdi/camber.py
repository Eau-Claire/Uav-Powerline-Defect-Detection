"""All Camber CLI interaction lives here. No other module runs `camber`.

* Every subprocess gets env=os.environ.copy() (CAMBER_API_KEY passthrough).
* Downloads land in a local mirror of the stash hierarchy and are reused when
  present (and digest-verified when an expected sha256 is known).
* Camber is optional for local work: `available()` False => callers degrade.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .hashing import extract_tags, sha256_file
from .paths import ProjectPaths

log = logging.getLogger("yolov11sdi.camber")


class CamberError(RuntimeError):
    pass


@dataclass
class CamberResult:
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def resolve_camber_bin(explicit: str | None = None) -> str | None:
    candidates = [
        explicit,
        os.environ.get("CAMBER_BIN"),
        shutil.which("camber"),
        str(Path.home() / ".camber/bin/camber"),
    ]
    for c in candidates:
        if c and Path(c).exists() and os.access(c, os.X_OK):
            return c
    return None


class CamberClient:
    def __init__(self, paths: ProjectPaths, stash_prefix: str, binary: str | None = None,
                 dry_run: bool = False, timeout_s: int = 3600):
        self.paths = paths
        self.stash_prefix = stash_prefix.rstrip("/")
        self.binary = resolve_camber_bin(binary)
        self.dry_run = dry_run
        self.timeout_s = timeout_s

    # ------------------------------------------------------------ basics
    def available(self) -> bool:
        return self.binary is not None and bool(os.environ.get("CAMBER_API_KEY") or self._logged_in())

    def _logged_in(self) -> bool:
        if self.binary is None:
            return False
        r = self._run(["me"], check=False, quiet=True)
        return r.ok and "authentication error" not in (r.stdout + r.stderr).lower()

    def _run(self, args: list[str], check: bool = True, quiet: bool = False) -> CamberResult:
        if self.binary is None:
            raise CamberError("Camber CLI not found (PATH, ~/.camber/bin, CAMBER_BIN).")
        cmd = [self.binary, *args]
        if not quiet:
            log.debug("camber %s", " ".join(args))
        try:
            r = subprocess.run(cmd, text=True, capture_output=True, env=os.environ.copy(),
                               timeout=self.timeout_s)
        except subprocess.TimeoutExpired as e:
            raise CamberError(f"camber timed out: {' '.join(args)}") from e
        res = CamberResult(r.returncode, r.stdout or "", r.stderr or "")
        if check and not res.ok:
            raise CamberError(
                f"camber {' '.join(args)} failed ({r.returncode})\n"
                f"STDOUT:\n{res.stdout[:4000]}\nSTDERR:\n{res.stderr[:4000]}"
            )
        return res

    # ------------------------------------------------------------- stash
    def ls(self, remote: str, recursive: bool = False, check: bool = False) -> CamberResult:
        args = ["stash", "ls"] + (["-r"] if recursive else []) + [remote]
        return self._run(args, check=check)

    def exists(self, remote: str) -> bool:
        r = self.ls(remote, recursive=False, check=False)
        return r.ok and "not found" not in (r.stdout + r.stderr).lower()

    def discover_tags(self, remote_root: str, prefix: str) -> list[str]:
        texts = []
        for recursive in (False, True):
            r = self.ls(remote_root, recursive=recursive, check=False)
            if r.ok:
                texts += [r.stdout, r.stderr]
        return extract_tags("\n".join(texts), prefix)

    def mirror_path(self, remote: str, bulk: bool = False) -> Path:
        return self.paths.mirror_for(remote, self.stash_prefix, bulk=bulk)

    def fetch(self, remote: str, local: Path | None = None, bulk: bool = False,
              expected_sha256: str | None = None, reuse: bool = True) -> Path:
        """Download once into the local mirror; reuse verified local copies."""
        local = Path(local) if local else self.mirror_path(remote, bulk=bulk)
        if reuse and local.exists() and local.stat().st_size > 0:
            if expected_sha256 is None or sha256_file(local) == expected_sha256:
                log.info("reuse cached %s", self.paths.rel(local))
                return local
            log.warning("cached copy sha mismatch, re-downloading: %s", local)
        if self.dry_run:
            log.info("[dry-run] would download %s -> %s", remote, local)
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        tmp = local.with_name(f".{local.name}.part")
        tmp.unlink(missing_ok=True)
        self._run(["stash", "cp", remote, str(tmp)])
        if not tmp.exists():
            raise CamberError(f"Camber download produced no file: {remote}")
        if expected_sha256 and sha256_file(tmp) != expected_sha256:
            tmp.unlink(missing_ok=True)
            raise CamberError(f"Downloaded file sha256 mismatch: {remote}")
        os.replace(tmp, local)
        log.info("downloaded %s (%.1f MB)", remote, local.stat().st_size / 1024**2)
        return local

    def put(self, local: Path, remote: str) -> None:
        local = Path(local)
        if not local.exists():
            raise FileNotFoundError(local)
        if self.dry_run:
            log.info("[dry-run] would upload %s -> %s", local, remote)
            return
        self._run(["stash", "cp", str(local), remote])
        log.info("uploaded %s -> %s", self.paths.rel(local), remote)

    def put_many(self, pairs: list[tuple[Path, str]], required: bool = False) -> list[str]:
        """Upload several files. Non-required failures are logged, not raised."""
        done = []
        for local, remote in pairs:
            try:
                self.put(local, remote)
                done.append(remote)
            except Exception as e:  # noqa: BLE001 - sync must not kill local work
                if required:
                    raise
                log.warning("upload failed (non-fatal) %s -> %s: %s", local, remote, e)
        return done
