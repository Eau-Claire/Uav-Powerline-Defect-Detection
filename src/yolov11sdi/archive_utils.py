"""Streaming archive helpers: suffix indexes, deterministic tar/zip writers, integrity."""
from __future__ import annotations

import gzip
import io
import tarfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .io_utils import IMG_EXTS, atomic_path, validate_yolo_text

FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------- reading

def normalize_member(name: str) -> str:
    return name.replace("\\", "/").lstrip("./")


def build_tar_suffix_index(tf: tarfile.TarFile) -> tuple[dict[str, tarfile.TarInfo], int]:
    """Index members by their 'images/...' or 'labels/...' suffix (legacy rule).

    Iterates headers only; file payloads are never extracted.
    Returns (index, number_of_duplicate_keys).
    """
    idx: dict[str, tarfile.TarInfo] = {}
    dups = 0
    for m in tf:
        if not m.isfile():
            continue
        n = normalize_member(m.name)
        key = None
        pi, pl = n.find("images/"), n.find("labels/")
        if pi >= 0:
            key = n[pi:]
        elif pl >= 0:
            key = n[pl:]
        if key is None:
            continue
        if key in idx:
            dups += 1
        else:
            idx[key] = m
    return idx, dups


def tar_find_member(member_names: Iterable[str], suffix: str) -> str | None:
    suffix = str(suffix).replace("\\", "/").lstrip("/")
    hits = [n for n in member_names if n.replace("\\", "/").endswith(suffix)]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        return None
    raise RuntimeError(f"Ambiguous TAR member suffix {suffix}: {hits[:10]}")


# ---------------------------------------------------------------- writing

@dataclass
class TarEntry:
    arcname: str
    source: Path | None = None
    data: bytes | None = None
    mode: int = 0o644

    def payload(self) -> bytes:
        if self.data is not None:
            return self.data
        assert self.source is not None
        return Path(self.source).read_bytes()


def write_deterministic_targz(out_path: Path, entries: Iterable[TarEntry]) -> Path:
    """Byte-reproducible .tar.gz: sorted members, mtime=0, uid/gid=0, gzip mtime=0.

    Same content => same SHA256, so digest-matched archives can be reused.
    """
    items = sorted(entries, key=lambda e: e.arcname)
    with atomic_path(out_path) as tmp:
        with open(tmp, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
                for e in items:
                    body = e.payload()
                    info = tarfile.TarInfo(name=e.arcname.replace("\\", "/"))
                    info.size = len(body)
                    info.mode = e.mode
                    info.mtime = 0
                    info.uid = info.gid = 0
                    info.uname = info.gname = ""
                    tf.addfile(info, io.BytesIO(body))
    return out_path


def dir_entries(root: Path, arc_prefix: str) -> list[TarEntry]:
    root = Path(root)
    return [
        TarEntry(arcname=f"{arc_prefix}/{p.relative_to(root).as_posix()}", source=p)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]


def write_deterministic_zip(out_path: Path, files: Iterable[tuple[str, Path]]) -> Path:
    with atomic_path(out_path) as tmp:
        with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for arcname, src in sorted(files, key=lambda x: x[0]):
                info = zipfile.ZipInfo(arcname, date_time=FIXED_ZIP_TIME)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, Path(src).read_bytes())
    return out_path


# -------------------------------------------------------------- integrity

@dataclass
class ArchiveReport:
    images: int = 0
    labels: int = 0
    missing_labels: list[str] = field(default_factory=list)
    orphan_labels: list[str] = field(default_factory=list)
    label_problems: list[str] = field(default_factory=list)
    splits: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not (self.missing_labels or self.orphan_labels or self.label_problems)

    def raise_if_bad(self, what: str) -> None:
        if not self.ok:
            raise RuntimeError(
                f"Archive integrity failed for {what}: missing_labels={self.missing_labels[:10]} "
                f"orphan_labels={self.orphan_labels[:10]} problems={self.label_problems[:10]}"
            )


def _pair_key(name: str, kind: str) -> tuple[str, str] | None:
    """('<prefix>/<split>/<stem>' key, split) for '.../images/<split>/x.jpg'."""
    n = normalize_member(name)
    marker = f"/{kind}/" if f"/{kind}/" in n else (f"{kind}/" if n.startswith(f"{kind}/") else None)
    if marker is None:
        return None
    head, tail = n.split(marker, 1)
    if "/" not in tail:
        return None
    split, fname = tail.split("/", 1)
    return f"{head}|{split}|{Path(fname).with_suffix('').as_posix()}", split


def verify_yolo_tar(path: Path, n_classes: int, forbidden_splits: Iterable[str] = ("test",)) -> ArchiveReport:
    """Image/label pairing + label validity for a YOLO-layout tar(.gz), streamed."""
    rep = ArchiveReport()
    images: dict[str, str] = {}
    labels: dict[str, str] = {}
    forbidden = set(forbidden_splits)
    with tarfile.open(path, "r:*") as tf:
        for m in tf:
            if not m.isfile():
                continue
            suffix = Path(m.name).suffix.lower()
            if suffix in IMG_EXTS:
                k = _pair_key(m.name, "images")
                if k:
                    images[k[0]] = m.name
                    rep.splits[k[1]] = rep.splits.get(k[1], 0) + 1
                    if k[1] in forbidden:
                        rep.label_problems.append(f"forbidden split '{k[1]}' in {m.name}")
            elif suffix == ".txt":
                k = _pair_key(m.name, "labels")
                if k:
                    labels[k[0]] = m.name
                    f = tf.extractfile(m)
                    text = f.read().decode("utf-8", errors="replace") if f else ""
                    rep.label_problems.extend(validate_yolo_text(text, n_classes, m.name))
    rep.images, rep.labels = len(images), len(labels)
    rep.missing_labels = sorted(images[k] for k in set(images) - set(labels))
    rep.orphan_labels = sorted(labels[k] for k in set(labels) - set(images))
    return rep
