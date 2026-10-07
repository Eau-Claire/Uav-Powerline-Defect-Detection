"""Strict-IoU per-class metrics from an Ultralytics validator result."""
from __future__ import annotations

import numpy as np


def per_class_rows(box_metrics, names: list[str]) -> list[dict]:
    """AP50, AP50_95, gap, AP75/AP90/AP95 per class (legacy report format)."""
    all_ap = np.asarray(box_metrics.all_ap, dtype=float)
    ap50 = np.asarray(box_metrics.ap50, dtype=float)
    class_ids = [int(x) for x in np.asarray(box_metrics.ap_class_index).tolist()]
    if all_ap.ndim == 1:
        all_ap = all_ap[:, None]
    thresholds = np.arange(0.50, 0.50 + 0.05 * all_ap.shape[1], 0.05)

    def col(t: float) -> int:
        return int(np.argmin(np.abs(thresholds - t)))

    rows = []
    for ri, cid in enumerate(class_ids):
        vals = all_ap[ri]
        multi = all_ap.shape[1] > 1
        rows.append({
            "cid": cid, "class": names[cid],
            "AP50": float(ap50[ri]), "AP50_95": float(np.mean(vals)),
            "gap": float(ap50[ri] - np.mean(vals)),
            "AP75": float(vals[col(0.75)]) if multi else None,
            "AP90": float(vals[col(0.90)]) if multi else None,
            "AP95": float(vals[col(0.95)]) if multi else None,
        })
    return rows


def global_metrics(box_metrics) -> dict:
    return {
        "mAP50": float(box_metrics.map50),
        "mAP50_95": float(box_metrics.map),
        "gap": float(box_metrics.map50 - box_metrics.map),
        "precision": float(box_metrics.mp),
        "recall": float(box_metrics.mr),
    }


def evaluate_weights(weights, data_yaml, imgsz: int, batch: int, device, names: list[str]) -> dict:
    from ultralytics import YOLO

    res = YOLO(str(weights)).val(data=str(data_yaml), split="val", imgsz=imgsz, batch=batch,
                                 device=device, plots=False, verbose=False)
    return {**global_metrics(res.box), "per_class": per_class_rows(res.box, names)}
