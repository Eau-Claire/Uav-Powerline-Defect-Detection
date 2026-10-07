"""Freeze + package the V6.5 external stage -> v6p5_ext_<tag>.

Remote contract (unchanged from the Kaggle notebooks):
    <stage_root>/v6p5_ext_<tag>/v6p5_ext_<tag>.freeze.json
    <stage_root>/v6p5_ext_<tag>/v6p5_ext_<tag>_overlay.tar.gz
    <stage_root>/v6p5_ext_<tag>/v6p5_ext_<tag>_qa.zip
    <stage_root>/mpcd/mpcd_broken_strand_stage.tar.gz
    <stage_root>/power_equipment/power_equipment_stage.tar.gz

Tag = first 12 hex of the legacy overlay digest (sorted relpath:sha256 lines).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..archive_utils import TarEntry, dir_entries, write_deterministic_targz, write_deterministic_zip
from ..hashing import dir_digest, make_tag, sha256_file

EXT_PREFIX = "v6p5_ext_"
DIRECT_ARCHIVE_ROOTS = {"MPCD": "mpcd_broken_strand", "PowerEquipment": "power_equipment_stage"}


@dataclass
class StageRemote:
    stage_root: str
    tag: str

    @property
    def dir(self) -> str:
        return f"{self.stage_root}/{self.tag}"

    @property
    def freeze(self) -> str:
        return f"{self.dir}/{self.tag}.freeze.json"

    @property
    def overlay(self) -> str:
        return f"{self.dir}/{self.tag}_overlay.tar.gz"

    @property
    def qa(self) -> str:
        return f"{self.dir}/{self.tag}_qa.zip"


def overlay_tag(overlay_root: Path) -> tuple[str, str]:
    digest = dir_digest(overlay_root)
    return make_tag(EXT_PREFIX, digest), digest


def build_direct_archive(overlay_root: Path, direct: pd.DataFrame, source: str, out_path: Path, resolve) -> Path:
    """Train-compatible direct archive: <root>/{images,labels}/{train,external_eval}/..."""
    arc_root = DIRECT_ARCHIVE_ROOTS[source]
    entries = []
    for _, r in direct[direct["source"] == source].iterrows():
        split = r["target_split"]
        ip, lp = resolve(r["overlay_image"]), resolve(r["overlay_label"])
        entries.append(TarEntry(f"{arc_root}/images/{split}/{ip.name}", source=ip))
        entries.append(TarEntry(f"{arc_root}/labels/{split}/{lp.name}", source=lp))
    return write_deterministic_targz(out_path, entries)


def build_freeze(cfg, tag: str, digest: str, direct: pd.DataFrame, synth: pd.DataFrame,
                 review: pd.DataFrame, exact: pd.DataFrame, near: pd.DataFrame, provenance: dict,
                 config_hash: str, git_commit: str | None) -> dict:
    """Legacy freeze fields + extended lineage. Must never contain timestamps in identity."""
    oi = cfg.get("openimages")
    pe = cfg.get("power_equipment")
    return {
        "tag": tag,
        "type": "v6p5_external_stage",
        "taxonomy": {i: n for i, n in enumerate(cfg.names)},
        "base_reference": {
            "dataset": "V6.2E",
            "tag": cfg.get("v6p2e.tag"),
            "role": "dedup + internal TRAIN backgrounds only",
            "lineage": cfg.get("v6p2e.expected_lineage"),
        },
        "sources": {
            "OpenImagesV7": {
                "classes": oi["classes"],
                "split": oi["split"],
                "max_samples_per_class": oi["max_samples_per_class"],
                "seed": oi["seed"],
                "fiftyone_version": oi["fiftyone_version"],
                "role": "appearance source for relation-aware synthetic foreign objects",
                "annotation_download": oi["annotation_download"],
                "mask_strategy": oi["mask_strategy"],
                "openimages_segmentation_zip_downloaded": oi["openimages_segmentation_zip_downloaded"],
                "reason": "avoid OpenImages segmentation mask ZIP extraction on low disk",
            },
            "MPCD": {
                "role": "broken_strand direct power-line source",
                "gdrive_id": cfg.get("mpcd.gdrive_id"),
                "target_class": cfg.names[int(cfg.get("mpcd.target_class"))],
                "name_keywords": cfg.get("mpcd.name_keywords"),
                "archive_sha256": provenance.get("mpcd_zip_sha256"),
            },
            "PowerEquipment": {
                "hf_repo": pe["hf_repo"],
                "filename": pe["filename"],
                "map": pe["map"],
                "review_only": pe["review_only"],
                "train_frac": pe["train_frac"],
                "split_key_prefix": pe["split_key_prefix"],
                "archive_sha256": provenance.get("power_equipment_zip_sha256"),
            },
        },
        "split_policy": {
            "direct_source_train": "train",
            "direct_source_other": "external_eval (never V6.2E val/test)",
            "power_equipment": "deterministic_fraction(split_key) < train_frac",
            "synthetic": "train only",
        },
        "seed": cfg.get("seed"),
        "synthetic_policy": {
            "per_subtype": cfg.get("synthetic.per_subtype"),
            "require_mask": cfg.get("synthetic.require_mask"),
            "background": "internal V6.2E TRAIN, insulator anchor",
            "relation": "near electrical structure -> foreign_object",
            "hard_negatives": cfg.get("synthetic.hard_negatives"),
        },
        "dedup_policy": {
            "exact_sha256_vs_all_v6p2e": "hard gate",
            "phash_vs": cfg.get("parent_prepare.phash_splits"),
            "phash_max_hamming": cfg.get("dedup.phash_max_hamming"),
            "phash_candidates": "human review",
        },
        "counts": {
            "direct_train_images": int((direct["target_split"] == "train").sum()),
            "direct_external_eval_images": int((direct["target_split"] == "external_eval").sum()),
            "synthetic_train_images": int(len(synth)),
            "review_candidates": int(len(review)),
            "exact_external_internal_duplicates": int(len(exact)),
            "phash_candidates": int(len(near)),
            "direct_by_source_split": {
                f"{s}/{t}": int(n) for (s, t), n in direct.groupby(["source", "target_split"]).size().items()
            },
            "synthetic_by_subtype": {k: int(v) for k, v in synth["foreign_subtype"].value_counts().sort_index().items()},
        },
        "policy": {
            "generic_full_images_direct_merge": False,
            "synthetic_background_split": "internal train only",
            "dx_sg_direct_merge": False,
            "test_modified": False,
            "val_modified": False,
            "auto_promotion": False,
            "openimages_segmentation_zip_downloaded": False,
            "openimages_mask_strategy": "grabcut_from_detection",
        },
        "overlay_sha256": digest,
        "config_hash": config_hash,
        "code_commit": git_commit,
    }


def package_stage(overlay_root: Path, qa_root: Path, out_dir: Path, tag: str) -> dict[str, Path]:
    overlay_tar = write_deterministic_targz(out_dir / f"{tag}_overlay.tar.gz", dir_entries(overlay_root, "v6p5_overlay"))
    qa_files = [(p.relative_to(qa_root).as_posix(), p) for p in sorted(qa_root.rglob("*")) if p.is_file()]
    qa_zip = write_deterministic_zip(out_dir / f"{tag}_qa.zip", qa_files)
    return {"overlay": overlay_tar, "qa": qa_zip}


def artifact_hashes(paths: dict[str, Path]) -> dict[str, str]:
    return {k: sha256_file(p) for k, p in paths.items()}
