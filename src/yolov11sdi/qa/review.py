"""Human review manifest + decision resolution. No auto-promotion.

Allowed decisions: KEEP, DROP, blank (pending). REVIEW_THEN_KEEP /
VISUAL_REVIEW_THEN_KEEP are *recommendations*, never decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

REVIEW_COLUMNS = [
    "review_id", "candidate_type", "overlay_image", "overlay_label", "source", "target_split",
    "mapped_class_names", "subtype", "recommended_action", "decision", "reviewer", "comment",
]
DECISION_COLUMNS = ["review_id", "decision", "reviewer", "comment"]
ALLOWED = {"KEEP", "DROP", ""}


def normalize_decision(x) -> str:
    v = str(x if x is not None else "").strip().upper()
    if v in ("NAN", "NONE"):
        v = ""
    if v not in ALLOWED:
        raise ValueError(f"Invalid decision {x!r}; allowed: KEEP, DROP, blank")
    return v


def build_review_manifest(direct: pd.DataFrame, synth: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in direct.iterrows():
        rows.append({
            "candidate_type": "direct_powerline", "overlay_image": r["overlay_image"],
            "overlay_label": r["overlay_label"], "source": r["source"], "target_split": r["target_split"],
            "mapped_class_names": r["mapped_class_names"], "subtype": "",
            "recommended_action": "REVIEW_THEN_KEEP" if r["target_split"] == "train" else "KEEP_AS_EXTERNAL_EVAL",
            "decision": "", "reviewer": "", "comment": "",
        })
    for _, r in synth.iterrows():
        rows.append({
            "candidate_type": "synthetic_relation", "overlay_image": r["overlay_image"],
            "overlay_label": r["overlay_label"], "source": r["source"], "target_split": "train",
            "mapped_class_names": "foreign_object", "subtype": r["foreign_subtype"],
            "recommended_action": "VISUAL_REVIEW_THEN_KEEP", "decision": "", "reviewer": "", "comment": "",
        })
    df = pd.DataFrame(rows)
    df.insert(0, "review_id", [f"V65R{i:06d}" for i in range(len(df))])
    return df[REVIEW_COLUMNS]


def decisions_template(review: pd.DataFrame, existing: pd.DataFrame | None = None) -> pd.DataFrame:
    """Editable decisions CSV; preserves decisions already entered."""
    tpl = review[["review_id", "candidate_type", "source", "target_split", "mapped_class_names",
                  "subtype", "recommended_action", "overlay_image"]].copy()
    for c in ("decision", "reviewer", "comment"):
        tpl[c] = ""
    if existing is not None and len(existing):
        ex = existing.set_index("review_id")
        for c in ("decision", "reviewer", "comment"):
            if c in ex.columns:
                tpl[c] = tpl["review_id"].map(ex[c]).fillna("").astype(str)
    return tpl


@dataclass
class Resolution:
    resolved: pd.DataFrame
    pending_train: pd.DataFrame

    @property
    def complete(self) -> bool:
        return len(self.pending_train) == 0


def resolve_decisions(review: pd.DataFrame, decisions: pd.DataFrame | None,
                      auto_approve_recommended: bool = False) -> Resolution:
    out = review.copy()
    out["decision"] = out["decision"].map(normalize_decision)
    if decisions is not None and len(decisions):
        unknown = set(decisions["review_id"].astype(str)) - set(out["review_id"].astype(str))
        if unknown:
            raise KeyError(f"decisions reference unknown review_id(s): {sorted(unknown)[:10]}")
        dmap = decisions.set_index(decisions["review_id"].astype(str))
        for col in ("decision", "reviewer", "comment"):
            if col in dmap.columns:
                vals = out["review_id"].astype(str).map(dmap[col])
                if col == "decision":
                    vals = vals.map(lambda v: normalize_decision(v) if isinstance(v, str) else "")
                    out[col] = vals.where(vals != "", out[col])
                else:
                    out[col] = vals.where(vals.notna() & (vals.astype(str) != ""), out[col])
    is_train = out["target_split"].astype(str).str.lower().eq("train")
    if auto_approve_recommended:
        # Explicit, recorded human override (config + compose freeze), never default.
        blank = out["decision"].eq("")
        rec_keep = out["recommended_action"].astype(str).str.upper().str.contains("KEEP", na=False)
        out.loc[blank & is_train & rec_keep, "decision"] = "KEEP"
        out.loc[blank & ~(is_train & rec_keep), "decision"] = "DROP"
    pending = out[is_train & out["decision"].eq("")].copy()
    if len(pending) == 0:
        # Non-train rows (external_eval) never enter training.
        out.loc[~is_train, "decision"] = "DROP"
    return Resolution(resolved=out, pending_train=pending)


def check_review_manifest(review: pd.DataFrame) -> None:
    missing = set(REVIEW_COLUMNS) - set(review.columns)
    if missing:
        raise RuntimeError(f"Review manifest missing columns: {sorted(missing)}")
    if review["review_id"].duplicated().any():
        raise RuntimeError("review_id is duplicated.")
    if review["target_split"].astype(str).str.lower().isin(["test", "val"]).any():
        raise RuntimeError("Candidate targets V6.2E val/test. External data may only target train/external_eval.")
    bad_syn = review[(review["candidate_type"] == "synthetic_relation") & (review["target_split"] != "train")]
    if len(bad_syn):
        raise RuntimeError("Synthetic candidates must be TRAIN only.")


def load_decisions(path: Path) -> pd.DataFrame:
    from ..io_utils import read_csv_safe

    return read_csv_safe(path)
