#!/usr/bin/env python
"""Apply reviewed YOLO26m box proposals to a versioned exploratory test copy."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/test_main_v1"
PACK = ROOT / "results/finding_nemo/targeted_audit_test_main_v1_nms50"
PROPOSALS = PACK / "broken_strand_box_proposals.csv"
OUTPUT = ROOT / "data/test_main_v2_model_assisted_exploratory"
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", type=Path, default=PROPOSALS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--apply-reviewed-reclasses", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite: {output}")
    image_out, label_out = output / "images/test", output / "labels/test"
    image_out.mkdir(parents=True); label_out.mkdir(parents=True)
    for image in (SOURCE / "images/test").iterdir():
        target = image_out / image.name
        try:
            os.link(image, target)
        except OSError:
            shutil.copy2(image, target)
    for label in (SOURCE / "labels/test").glob("*.txt"):
        shutil.copy2(label, label_out / label.name)

    proposals = list(csv.DictReader(args.proposals.open(encoding="utf-8-sig")))
    grouped = defaultdict(list)
    for row in proposals:
        grouped[Path(row["label"]).name].append(row)
    log = []
    for filename, changes in grouped.items():
        target = label_out / filename
        lines = target.read_text(encoding="utf-8").splitlines()
        for row in sorted(changes, key=lambda value: int(value["line_index"])):
            index = int(row["line_index"])
            if lines[index].strip() != row["old_label"].strip():
                raise RuntimeError(f"Source label mismatch: {filename}:{index + 1}")
            lines[index] = row["proposed_label"]
            log.append({"label": filename, "line_index_zero_based": index, "action": "replace_box_with_yolo26m_proposal",
                        "old_label": row["old_label"], "new_label": row["proposed_label"],
                        "proposal_confidence": row["confidence"], "old_vs_new_iou": row["iou_old_vs_proposal"],
                        "review_rank": row["rank"]})
        target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    if args.apply_reviewed_reclasses:
        plan = list(csv.DictReader((PACK / "broken_strand_correction_plan.csv").open(encoding="utf-8-sig")))
        for row in plan:
            if row["normalized_action"] != "remove_or_reclass_gt":
                continue
            filename = Path(row["image"]).with_suffix(".txt").name
            target = label_out / filename
            lines = target.read_text(encoding="utf-8").splitlines()
            replacement = 4 if row["rank"] == "34" else (1 if row["rank"] == "42" else None)
            revised = []
            for index, line in enumerate(lines):
                parts = line.split()
                if parts and int(float(parts[0])) == 5:
                    if replacement is None:
                        log.append({"label": filename, "line_index_zero_based": index, "action": "remove_invalid_broken_strand_gt",
                                    "old_label": line, "new_label": "", "proposal_confidence": "", "old_vs_new_iou": "", "review_rank": row["rank"]})
                        continue
                    new_line = " ".join([str(replacement)] + parts[1:])
                    log.append({"label": filename, "line_index_zero_based": index, "action": f"reclass_broken_strand_to_{NAMES[replacement]}",
                                "old_label": line, "new_label": new_line, "proposal_confidence": "", "old_vs_new_iou": "", "review_rank": row["rank"]})
                    revised.append(new_line)
                else:
                    revised.append(line)
            target.write_text("\n".join(revised) + ("\n" if revised else ""), encoding="utf-8")
    correction_log = output / "correction_log.csv"
    with correction_log.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(log[0])); writer.writeheader(); writer.writerows(log)
    dataset = {"path": output.as_posix(), "train": "images/test", "val": "images/test", "test": "images/test",
               "names": {i: name for i, name in enumerate(NAMES)}}
    (output / "dataset.yaml").write_text(yaml.safe_dump(dataset, sort_keys=False), encoding="utf-8")
    freeze = {"name": output.name, "source": str(SOURCE), "role": "model-assisted exploratory analysis only",
              "warning": "GT boxes were proposed by YOLO26m epoch45; results for that checkpoint are circular and must not replace the primary benchmark.",
              "changed_boxes": len(log), "changed_images": len(grouped), "correction_log_sha256": digest(correction_log),
              "proposal_source": str(args.proposals.resolve()), "reviewed_reclasses_applied": args.apply_reviewed_reclasses}
    (output / "freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, indent=2))


if __name__ == "__main__":
    main()
