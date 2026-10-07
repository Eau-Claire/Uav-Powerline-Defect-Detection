"""Direct power-line overlay (MPCD / PowerEquipment) shared by both source stages.

Layout (legacy-compatible, under the V6.5 overlay root):
    direct/images/{train,external_eval}/<source>_<sha16><ext>
    direct/labels/{train,external_eval}/<source>_<sha16>.txt
Each source stage owns only files with its own prefix.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..hashing import sha256_file
from ..io_utils import safe_link_or_copy, write_yolo_label

DIRECT_SPLITS = ("train", "external_eval")

DIRECT_COLUMNS = [
    "overlay_image", "overlay_label", "target_split", "source", "source_split", "source_image",
    "source_sha256", "mapped_classes", "mapped_class_names", "n_mapped_boxes", "license_note",
]


def source_prefix(source: str) -> str:
    return source.lower()


def clear_source(overlay_root: Path, source: str) -> None:
    prefix = source_prefix(source) + "_"
    for kind in ("images", "labels"):
        for split in DIRECT_SPLITS:
            d = overlay_root / "direct" / kind / split
            if d.exists():
                for p in d.glob(prefix + "*"):
                    p.unlink()


def add_direct_records(records: list[dict], overlay_root: Path, names: list[str], rel) -> pd.DataFrame:
    """Copy (hard-link) images + write mapped labels; return the direct manifest rows."""
    out = []
    for rec in records:
        target = rec["target_split"]
        if target not in DIRECT_SPLITS:
            raise ValueError(f"direct target split must be train/external_eval, got {target!r}")
        ip = Path(rec["source_image_abs"])
        sha = sha256_file(ip)
        name = f"{source_prefix(rec['source'])}_{sha[:16]}{ip.suffix.lower()}"
        di = overlay_root / "direct/images" / target / name
        dl = overlay_root / "direct/labels" / target / Path(name).with_suffix(".txt")
        safe_link_or_copy(ip, di)
        write_yolo_label(dl, rec["mapped_rows"])
        classes = sorted({int(r[0]) for r in rec["mapped_rows"]})
        out.append({
            "overlay_image": rel(di),
            "overlay_label": rel(dl),
            "target_split": target,
            "source": rec["source"],
            "source_split": rec["source_split"],
            "source_image": rec["source_image"],
            "source_sha256": sha,
            "mapped_classes": ",".join(map(str, classes)),
            "mapped_class_names": ",".join(names[c] for c in classes),
            "n_mapped_boxes": len(rec["mapped_rows"]),
            "license_note": rec.get("license", ""),
        })
    return pd.DataFrame(out, columns=DIRECT_COLUMNS).sort_values(["target_split", "overlay_image"]).reset_index(drop=True)


def rows_to_json(rows) -> str:
    return json.dumps([[int(r[0]), *map(float, r[1:5])] for r in rows])
