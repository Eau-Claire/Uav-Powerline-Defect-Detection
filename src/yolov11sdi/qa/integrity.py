"""Split / taxonomy / label integrity checks used by stages, scripts and tests."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..config import CANONICAL_TAXONOMY
from ..io_utils import IMG_EXTS, validate_yolo_text


class IntegrityError(RuntimeError):
    pass


def check_taxonomy(names) -> None:
    if tuple(names) != CANONICAL_TAXONOMY:
        raise IntegrityError(f"Taxonomy mismatch: {list(names)} != {list(CANONICAL_TAXONOMY)}")


def training_rows(manifest: pd.DataFrame, splits) -> pd.DataFrame:
    """Rows of the parent manifest allowed into a training dataset. TEST is refused."""
    splits = [str(s) for s in splits]
    if "test" in splits:
        raise IntegrityError("TEST is immutable and may never be materialized for training.")
    return manifest[manifest["split"].astype(str).isin(splits)]


def check_no_test_in_training(dataset_manifest: pd.DataFrame, parent_manifest: pd.DataFrame) -> None:
    test_sha = set(parent_manifest.loc[parent_manifest["split"].astype(str) == "test", "sha256"].astype(str))
    used = set(dataset_manifest["sha256"].astype(str))
    leak = used & test_sha
    if leak:
        raise IntegrityError(f"{len(leak)} V6.2E TEST image(s) found in a training dataset.")
    if dataset_manifest["dataset_split"].astype(str).eq("test").any():
        raise IntegrityError("Training dataset contains a 'test' split.")


def check_val_frozen(dataset_manifest: pd.DataFrame, parent_manifest: pd.DataFrame) -> None:
    """VAL must be exactly the V6.2E VAL set: same SHA set, nothing added."""
    want = set(parent_manifest.loc[parent_manifest["split"].astype(str) == "val", "sha256"].astype(str))
    val = dataset_manifest[dataset_manifest["dataset_split"].astype(str) == "val"]
    got = set(val["sha256"].astype(str))
    if got != want or len(val) != len(want):
        raise IntegrityError(
            f"VAL is not frozen: expected {len(want)} V6.2E val images, got {len(val)} "
            f"(missing={len(want - got)}, extra={len(got - want)})"
        )
    if (val["origin"].astype(str) != "v6p2e").any():
        raise IntegrityError("Non-V6.2E images were added to VAL.")


def check_synthetic_train_only(dataset_manifest: pd.DataFrame) -> None:
    syn = dataset_manifest[dataset_manifest["origin"].astype(str) == "v6p5_synthetic"]
    if (syn["dataset_split"].astype(str) != "train").any():
        raise IntegrityError("Synthetic samples outside TRAIN.")


def check_yolo_dir(root: Path, n_classes: int) -> list[str]:
    """Pairing + label validity for an on-disk YOLO dataset (images/<split>, labels/<split>)."""
    problems = []
    for split_dir in sorted((root / "images").iterdir()) if (root / "images").exists() else []:
        split = split_dir.name
        imgs = {p.stem for p in split_dir.iterdir() if p.suffix.lower() in IMG_EXTS}
        lbl_dir = root / "labels" / split
        lbls = {p.stem: p for p in lbl_dir.glob("*.txt")} if lbl_dir.exists() else {}
        for s in sorted(imgs - set(lbls)):
            problems.append(f"{split}: missing label for {s}")
        for s in sorted(set(lbls) - imgs):
            problems.append(f"{split}: orphan label {s}")
        for s, p in lbls.items():
            problems.extend(validate_yolo_text(p.read_text(errors="replace"), n_classes, f"{split}/{p.name}"))
    return problems
