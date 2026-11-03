#!/usr/bin/env python
"""Normalize free-text broken-strand review notes into an auditable correction plan."""
from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "results/finding_nemo/targeted_audit_test_main_v1_nms50"


def classify(note: str) -> str:
    note = note.lower().strip()
    if any(text in note for text in ("không broken", "có broken đâu", "sai class", "insulator broken sai broken strand")):
        return "remove_or_reclass_gt"
    if any(text in note for text in ("model và box sai", "model và box đều sai", "box và model đều sai", "khó nhìn")):
        return "manual_relabel_or_unsure"
    if ("box đúng model" in note or "model thiếu" in note or note == "model sai"
            or "model sai, đánh dư" in note or "model sai broken strand rồi" in note):
        return "model_error_keep_gt"
    if "model đúng" in note or "gt quá chặt" in note:
        return "adjust_gt_using_model_proposal"
    if "box sai" in note:
        return "manual_adjust_gt"
    return "manual_review"


def main() -> None:
    source = PACK / "review_progress.csv"
    rows = list(csv.DictReader(source.open(encoding="utf-8-sig")))
    output = []
    for row in rows:
        if row["audit_class"] != "broken_strand" or not row["review_verdict"]:
            continue
        output.append({
            "rank": row["rank"], "selection": row["selection"], "image": row["image"],
            "original_verdict": row["review_verdict"], "review_note": row["review_note"],
            "normalized_action": classify(row["review_note"]), "confirmed": "",
        })
    target = PACK / "broken_strand_correction_plan.csv"
    with target.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader(); writer.writerows(output)
    print(target.resolve())


if __name__ == "__main__":
    main()
