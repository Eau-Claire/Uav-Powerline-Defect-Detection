"""Atomic file IO, YOLO label IO, link-or-copy, CSV helpers."""
from __future__ import annotations

import contextlib
import json
import math
import os
import shutil
from pathlib import Path
from typing import Iterator, Sequence

YoloRow = tuple[int, float, float, float, float]
IMG_EXTS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


@contextlib.contextmanager
def atomic_path(final: str | Path) -> Iterator[Path]:
    """Yield a temp path next to `final`; on success fsync + rename into place."""
    final = Path(final)
    final.parent.mkdir(parents=True, exist_ok=True)
    tmp = final.with_name(f".{final.name}.tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        yield tmp
        if not tmp.exists():
            raise FileNotFoundError(f"atomic_path: nothing written to {tmp}")
        with open(tmp, "rb+") as f:
            os.fsync(f.fileno())
        os.replace(tmp, final)
    finally:
        if tmp.exists():
            tmp.unlink()


def atomic_write_bytes(path: str | Path, raw: bytes) -> Path:
    with atomic_path(path) as tmp:
        tmp.write_bytes(raw)
    return Path(path)


def atomic_write_text(path: str | Path, text: str) -> Path:
    return atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, default=str) + "\n")


def read_json(path: str | Path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def atomic_write_csv(path: str | Path, df, **kwargs) -> Path:
    with atomic_path(path) as tmp:
        compression = "gzip" if str(path).endswith(".gz") else None
        df.to_csv(tmp, index=False, compression=compression, **kwargs)
    return Path(path)


def read_csv_safe(path: str | Path):
    import pandas as pd

    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_link_or_copy(src: str | Path, dst: str | Path) -> Path:
    """Hard-link first; copy only when linking is impossible. Never overwrites."""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


# ----------------------------------------------------------- YOLO labels

def parse_yolo_text(text: str) -> list[YoloRow]:
    rows: list[YoloRow] = []
    for line in str(text).splitlines():
        p = line.strip().split()
        if len(p) < 5:
            continue
        try:
            rows.append((int(float(p[0])), *map(float, p[1:5])))
        except ValueError:
            continue
    return rows


def read_yolo_label(path: str | Path) -> list[YoloRow]:
    p = Path(path)
    if not p.exists():
        return []
    return parse_yolo_text(p.read_text(errors="ignore"))


def format_yolo_rows(rows: Sequence[YoloRow]) -> str:
    txt = "\n".join(
        f"{int(c)} {x:.8f} {y:.8f} {w:.8f} {h:.8f}" for c, x, y, w, h in rows if w > 0 and h > 0
    )
    return txt + ("\n" if txt else "")


def write_yolo_label(path: str | Path, rows: Sequence[YoloRow]) -> Path:
    return atomic_write_text(path, format_yolo_rows(rows))


def yolo_to_xyxy(xc: float, yc: float, w: float, h: float) -> list[float]:
    return [xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2]


def xyxy_to_yolo(box: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (max(0.0, min(1.0, float(v))) for v in box)
    return [(x1 + x2) / 2, (y1 + y2) / 2, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def validate_yolo_text(text: str, n_classes: int, where: str = "") -> list[str]:
    """Return a list of problems (empty == valid). Class ids 0..n-1, finite, normalized."""
    problems = []
    for ln, line in enumerate(str(text).splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            problems.append(f"{where}:{ln} expected 5 fields: {line!r}")
            continue
        try:
            cid_f = float(parts[0])
            vals = [float(v) for v in parts[1:]]
        except ValueError:
            problems.append(f"{where}:{ln} non-numeric: {line!r}")
            continue
        if not cid_f.is_integer() or not 0 <= int(cid_f) < n_classes:
            problems.append(f"{where}:{ln} bad class id {parts[0]}")
        if not all(math.isfinite(v) for v in vals):
            problems.append(f"{where}:{ln} non-finite bbox {vals}")
        elif not all(0.0 <= v <= 1.0 for v in vals):
            problems.append(f"{where}:{ln} bbox out of [0,1] {vals}")
        elif vals[2] <= 0 or vals[3] <= 0:
            problems.append(f"{where}:{ln} non-positive size {vals}")
    return problems
