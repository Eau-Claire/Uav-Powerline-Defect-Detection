"""Leakage QA for direct external images against canonical V6.2E.

* exact SHA256 vs ALL V6.2E rows -> HARD gate (compose refuses to proceed);
* pHash (Hamming <= 4) vs VAL/TEST only -> surfaced for human review.
Synthetic images are built on internal TRAIN backgrounds on purpose and are
NOT audited as external leakage.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ..hashing import PHashIndex, phash_image, sha256_file

log = logging.getLogger("yolov11sdi.dedup")

EXACT_COLUMNS = ["external_image", "base_split", "base_image_source", "sha256"]
PHASH_COLUMNS = ["external_image", "base_split", "base_image_source", "base_sha256", "phash_hamming"]


def exact_sha_audit(images: list[tuple[str, Path]], manifest: pd.DataFrame) -> pd.DataFrame:
    """images: [(stored_path, absolute_path)]."""
    by_sha: dict[str, list[dict]] = {}
    for _, r in manifest.iterrows():
        by_sha.setdefault(str(r["sha256"]), []).append(
            {"split": str(r["split"]), "image_path_source": str(r["image_path_source"])}
        )
    hits = []
    for stored, path in images:
        sha = sha256_file(path)
        for hit in by_sha.get(sha, []):
            hits.append({"external_image": stored, "base_split": hit["split"],
                         "base_image_source": hit["image_path_source"], "sha256": sha})
    return pd.DataFrame(hits, columns=EXACT_COLUMNS)


def phash_audit(images: list[tuple[str, Path]], phash_cache: pd.DataFrame, max_hamming: int,
                band_widths, hash_size: int = 8) -> pd.DataFrame:
    from PIL import Image

    index = PHashIndex(band_widths)
    for _, r in phash_cache.iterrows():
        index.add(int(r["phash_int"]), {"split": r["split"], "image_path_source": r["image_path_source"],
                                        "sha256": r["sha256"]})
    near = []
    for stored, path in images:
        try:
            with Image.open(path) as im:
                hv = phash_image(im, hash_size)
        except Exception:  # noqa: BLE001
            log.warning("cannot pHash %s", stored)
            continue
        for hd, p in index.query(hv, max_hamming):
            near.append({"external_image": stored, "base_split": p["split"],
                         "base_image_source": p["image_path_source"], "base_sha256": p["sha256"],
                         "phash_hamming": hd})
    return pd.DataFrame(near, columns=PHASH_COLUMNS)


def exact_gate(exact: pd.DataFrame, allow: bool) -> None:
    if len(exact) and not allow:
        raise RuntimeError(
            f"HARD QA gate: {len(exact)} exact SHA duplicate(s) between external candidates "
            "and canonical V6.2E. Remove them from the source stage; do not override."
        )
