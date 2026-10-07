"""Deterministic hashing helpers: file SHA256, stable JSON digests, tags, pHash LSH."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable, Sequence

CHUNK = 1024 * 1024


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def stable_json_hash(obj) -> str:
    """SHA256 of canonical JSON (sorted keys, compact separators)."""
    raw = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def deterministic_fraction(key: str) -> float:
    """Map a string to [0, 1) deterministically (legacy split rule)."""
    h = hashlib.sha256(str(key).encode()).hexdigest()
    return int(h[:12], 16) / float(16**12)


def seeded_int(*parts) -> int:
    """Stable 63-bit integer from arbitrary parts (for per-unit RNG seeds)."""
    return int(stable_json_hash([str(p) for p in parts])[:15], 16)


def dir_digest(root: str | Path) -> str:
    """Legacy overlay digest: sha256 over sorted '<relpath>:<file sha256>' lines."""
    root = Path(root)
    items = [
        f"{p.relative_to(root)}:{sha256_file(p)}"
        for p in sorted(root.rglob("*"))
        if p.is_file()
    ]
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


def make_tag(prefix: str, digest: str, length: int = 12) -> str:
    """`v6p5_ext_` + first 12 hex chars. Never timestamp based."""
    if not re.fullmatch(r"[0-9a-f]{%d,}" % length, digest):
        raise ValueError(f"Not a hex digest: {digest!r}")
    return f"{prefix}{digest[:length]}"


def extract_tags(text: str, prefix: str, length: int = 12) -> list[str]:
    pat = rf"{re.escape(prefix)}[0-9a-fA-F]{{{length}}}(?![0-9a-fA-F])"
    return sorted(set(re.findall(pat, text or "")))


# ---------------------------------------------------------------- pHash LSH

def hamming(a: int, b: int) -> int:
    return (int(a) ^ int(b)).bit_count()


def _band_shifts(widths: Sequence[int]) -> list[int]:
    shifts, cur = [], 0
    for w in widths:
        shifts.append(cur)
        cur += w
    return shifts


def band_keys(value: int, widths: Sequence[int]) -> list[tuple[int, int]]:
    """Split a 64-bit hash into bands. Pigeonhole: hamming<=4 with 5 bands => one band equal."""
    return [
        (band, (int(value) >> sh) & ((1 << w) - 1))
        for band, (w, sh) in enumerate(zip(widths, _band_shifts(widths)))
    ]


class PHashIndex:
    """Banded LSH over 64-bit perceptual hashes (legacy widths 13,13,13,13,12)."""

    def __init__(self, widths: Sequence[int] = (13, 13, 13, 13, 12)):
        self.widths = tuple(widths)
        self.values: list[int] = []
        self.payloads: list[dict] = []
        self._buckets: dict[tuple[int, int], list[int]] = {}

    def add(self, value: int, payload: dict) -> None:
        idx = len(self.values)
        self.values.append(int(value))
        self.payloads.append(payload)
        for key in band_keys(value, self.widths):
            self._buckets.setdefault(key, []).append(idx)

    def query(self, value: int, max_hamming: int) -> list[tuple[int, dict]]:
        cand: set[int] = set()
        for key in band_keys(value, self.widths):
            cand.update(self._buckets.get(key, []))
        out = []
        for idx in sorted(cand):
            hd = hamming(value, self.values[idx])
            if hd <= max_hamming:
                out.append((hd, self.payloads[idx]))
        return out


def phash_image(image, hash_size: int = 8) -> int:
    import imagehash

    return int(str(imagehash.phash(image.convert("RGB"), hash_size=hash_size)), 16)


def digest_records(records: Iterable[dict]) -> str:
    return stable_json_hash(list(records))
