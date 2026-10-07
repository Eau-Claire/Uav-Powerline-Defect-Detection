"""Per-stage logging: logs/<stage>/<timestamp>.log + notebook-friendly console."""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

FMT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"


def setup_stage_logger(stage: str, logs_dir: Path, level: str = "INFO", debug: bool = False) -> logging.Logger:
    logger = logging.getLogger(f"yolov11sdi.{stage}")
    logger.setLevel(logging.DEBUG if debug else getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    for h in list(logger.handlers):
        logger.removeHandler(h)
        h.close()

    log_dir = Path(logs_dir) / stage
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fh = logging.FileHandler(log_dir / f"{ts}.log", encoding="utf-8")
    fh.setFormatter(logging.Formatter(FMT))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(levelname)s [%(name)s] %(message)s"))
    logger.addHandler(ch)
    logger.debug("log file: %s", fh.baseFilename)
    return logger


class Progress:
    """Periodic progress logging without one line per file."""

    def __init__(self, logger: logging.Logger, label: str, total: int | None, every: int = 500,
                 on_tick=None):
        self.logger, self.label, self.total, self.every = logger, label, total, max(1, every)
        self.n = 0
        self.on_tick = on_tick

    def update(self, k: int = 1) -> None:
        self.n += k
        if self.n % self.every == 0 or (self.total and self.n >= self.total):
            tot = f"/{self.total}" if self.total else ""
            self.logger.info("%s: %d%s", self.label, self.n, tot)
        if self.on_tick:
            self.on_tick(self.n, self.total)
