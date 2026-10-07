"""Compare a data-ablation result against the old V6.2E YOLO11n@640 baseline."""
from __future__ import annotations

import pandas as pd


def add_deltas(report: dict, baseline: dict) -> dict:
    report["reference_baseline"] = baseline
    report["delta_vs_baseline"] = {
        k: report[k] - baseline[k] for k in ("mAP50", "mAP50_95", "gap")
        if baseline.get(k) is not None and report.get(k) is not None
    }
    for row in report.get("per_class", []):
        b = (baseline.get("per_class") or {}).get(row["class"])
        if b:
            row["delta_AP50_vs_baseline"] = row["AP50"] - b["AP50"]
            row["delta_AP50_95_vs_baseline"] = row["AP50_95"] - b["AP50_95"]
    return report


def promotion_gate(report: dict, baseline: dict, gate: dict) -> dict:
    """Legacy gate. Passing it only allows REVIEW/PROMOTION, never auto-promotion."""
    pc = {r["class"]: r for r in report.get("per_class", [])}
    bpc = baseline["per_class"]
    res = {
        "global_map5095_not_regress_gt_1pt":
            report["mAP50_95"] >= baseline["mAP50_95"] - gate["global_map50_95_max_regression"],
        "foreign_AP50_improves_2pt":
            pc.get("foreign_object", {}).get("AP50", 0) >= bpc["foreign_object"]["AP50"] + gate["foreign_object_ap50_min_gain"],
        "foreign_AP5095_improves_1pt":
            pc.get("foreign_object", {}).get("AP50_95", 0) >= bpc["foreign_object"]["AP50_95"] + gate["foreign_object_ap50_95_min_gain"],
        "strand_AP5095_improves_1pt":
            pc.get("broken_strand", {}).get("AP50_95", 0) >= bpc["broken_strand"]["AP50_95"] + gate["broken_strand_ap50_95_min_gain"],
    }
    res["candidate_pass"] = all(res.values())
    return res


def comparison_table(report: dict, baseline: dict) -> pd.DataFrame:
    rows = [{
        "class": "ALL", "base_AP50": baseline.get("mAP50"), "AP50": report.get("mAP50"),
        "base_AP50_95": baseline.get("mAP50_95"), "AP50_95": report.get("mAP50_95"),
    }]
    for r in report.get("per_class", []):
        b = (baseline.get("per_class") or {}).get(r["class"], {})
        rows.append({"class": r["class"], "base_AP50": b.get("AP50"), "AP50": r["AP50"],
                     "base_AP50_95": b.get("AP50_95"), "AP50_95": r["AP50_95"]})
    df = pd.DataFrame(rows)
    df["dAP50"] = df["AP50"] - df["base_AP50"]
    df["dAP50_95"] = df["AP50_95"] - df["base_AP50_95"]
    return df.round(4)
