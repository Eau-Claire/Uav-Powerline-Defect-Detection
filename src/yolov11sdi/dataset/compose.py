"""Compose QA: gate the external stage through human review -> v6p5_compose_<tag>.

Remote contract (the YOLO11n train worker reads `synthetic_archive`):
    <compose_root>/v6p5_compose_<tag>/
        v6p5_compose_<tag>.index.json
        v6p5_compose_<tag>.freeze.json
        v6p5_compose_<tag>.review.resolved.csv
        v6p5_compose_<tag>.synthetic_manifest.csv
        v6p5_compose_<tag>.direct_approved_manifest.csv
        v6p5_compose_<tag>.qa_summary.json
        v6p5_compose_<tag>_synthetic_approved.tar.gz
"""
from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..archive_utils import TarEntry, tar_find_member, write_deterministic_targz
from ..hashing import make_tag, sha256_bytes, sha256_file, stable_json_hash

COMPOSE_PREFIX = "v6p5_compose_"
INDEX_REQUIRED_FIELDS = ("compose_tag", "source_stage_tag", "synthetic_archive")


@dataclass
class ComposeRemote:
    compose_root: str
    tag: str

    @property
    def dir(self) -> str:
        return f"{self.compose_root}/{self.tag}"

    def file(self, suffix: str) -> str:
        return f"{self.dir}/{self.tag}{suffix}"

    @property
    def synthetic(self) -> str:
        return self.file("_synthetic_approved.tar.gz")

    @property
    def index(self) -> str:
        return self.file(".index.json")

    @property
    def freeze(self) -> str:
        return self.file(".freeze.json")

    @property
    def resolved(self) -> str:
        return self.file(".review.resolved.csv")

    @property
    def synthetic_manifest(self) -> str:
        return self.file(".synthetic_manifest.csv")

    @property
    def direct_manifest(self) -> str:
        return self.file(".direct_approved_manifest.csv")

    @property
    def qa_summary(self) -> str:
        return self.file(".qa_summary.json")


def hard_gates(stage_freeze: dict, names: list[str], exact: pd.DataFrame, near: pd.DataFrame,
               allow_exact: bool, allow_phash: bool) -> None:
    tax = stage_freeze.get("taxonomy", {})
    got = [tax[k] for k in sorted(tax, key=lambda x: int(x))] if isinstance(tax, dict) else list(tax)
    if got != list(names):
        raise RuntimeError(f"Taxonomy mismatch in stage freeze: {got}")
    pol = stage_freeze.get("policy", {})
    if pol.get("test_modified") not in (False, None):
        raise RuntimeError("Hard gate: stage freeze says test_modified=True")
    if pol.get("auto_promotion") not in (False, None):
        raise RuntimeError("Hard gate: staging artifact unexpectedly auto-promoted.")
    if len(exact) and not allow_exact:
        raise RuntimeError(f"Hard gate: {len(exact)} exact SHA duplicate(s) vs canonical V6.2E.")
    if len(near) and not allow_phash:
        raise PHashReviewRequired(len(near))


class PHashReviewRequired(RuntimeError):
    def __init__(self, n: int):
        super().__init__(
            f"{n} pHash near-duplicate candidate(s) vs V6.2E VAL/TEST need review. Inspect them, "
            "DROP affected candidates, then set compose.allow_phash_candidates=true."
        )
        self.n = n


def build_approved_synthetic(overlay_tar: Path, keep_synth: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Stream approved synthetic pairs from the stage overlay into a deterministic archive."""
    entries, included = [], []
    with tarfile.open(overlay_tar, "r:*") as src:
        members = {m.name: m for m in src.getmembers() if m.isfile()}
        names = list(members)
        for _, row in keep_synth.sort_values("review_id").iterrows():
            ib, lb = Path(str(row["overlay_image"])).name, Path(str(row["overlay_label"])).name
            im_name = tar_find_member(names, f"synthetic_foreign/images/train/{ib}")
            lb_name = tar_find_member(names, f"synthetic_foreign/labels/train/{lb}")
            if im_name is None or lb_name is None:
                raise RuntimeError(f"Missing synthetic pair in stage overlay: {ib}")
            im_raw = src.extractfile(members[im_name]).read()
            lb_raw = src.extractfile(members[lb_name]).read()
            out_im = f"v6p5_overlay/synthetic_foreign/images/train/{ib}"
            out_lb = f"v6p5_overlay/synthetic_foreign/labels/train/{lb}"
            entries += [TarEntry(out_im, data=im_raw), TarEntry(out_lb, data=lb_raw)]
            included.append({"review_id": row["review_id"], "image": out_im, "label": out_lb,
                             "image_sha256": sha256_bytes(im_raw), "label_sha256": sha256_bytes(lb_raw)})
    write_deterministic_targz(out_path, entries)
    df = pd.DataFrame(included)
    if len(df) != len(keep_synth):
        raise RuntimeError(f"Approved synthetic mismatch: {len(df)} != {len(keep_synth)}")
    return df


def compose_seed(stage_tag: str, stage_freeze: dict, resolved: pd.DataFrame, synthetic_archive: Path,
                 names: list[str]) -> dict:
    """Legacy compose seed. The tag is derived from this — never from time."""
    payload = [{
        "review_id": str(r["review_id"]), "candidate_type": str(r["candidate_type"]),
        "source": str(r["source"]), "mapped_class_names": str(r["mapped_class_names"]),
        "subtype": str(r.get("subtype", "")), "decision": str(r["decision"]),
    } for _, r in resolved.sort_values("review_id").iterrows()]
    return {
        "source_stage_tag": stage_tag,
        "source_overlay_sha256": stage_freeze.get("overlay_sha256"),
        "review_decisions_sha256": stable_json_hash(payload),
        "approved_synthetic_archive_sha256": sha256_file(synthetic_archive),
        "taxonomy": list(names),
    }


def compose_tag_from_seed(seed: dict) -> str:
    return make_tag(COMPOSE_PREFIX, stable_json_hash(seed))


def build_index(tag: str, stage_tag: str, stage_remote, remote: ComposeRemote, direct_archives: dict) -> dict:
    return {
        "compose_tag": tag,
        "source_stage_tag": stage_tag,
        "source_stage_remote": stage_remote.dir,
        # IMPORTANT: the YOLO11n train worker reads this field.
        "synthetic_archive": remote.synthetic,
        "resolved_review_manifest": remote.resolved,
        "approved_synthetic_manifest": remote.synthetic_manifest,
        "approved_direct_manifest": remote.direct_manifest,
        "qa_summary": remote.qa_summary,
        "source_stage_freeze": stage_remote.freeze,
        "source_stage_qa": stage_remote.qa,
        "source_stage_overlay": stage_remote.overlay,
        "direct_archives": direct_archives,
    }


def validate_index(index: dict) -> None:
    missing = [k for k in INDEX_REQUIRED_FIELDS if not index.get(k)]
    if missing:
        raise RuntimeError(f"Compose index missing required field(s): {missing}")
    if not str(index["synthetic_archive"]).startswith("stash://"):
        raise RuntimeError("Compose index synthetic_archive must be a Camber stash path.")
