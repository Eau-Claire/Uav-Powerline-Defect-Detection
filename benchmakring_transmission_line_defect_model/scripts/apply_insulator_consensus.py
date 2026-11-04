#!/usr/bin/env python
"""Append high-consensus missing-insulator proposals to a versioned exploratory test copy."""
from __future__ import annotations

import csv
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import yaml


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"data/test_main_v3_model_aligned_exploratory"
PROPOSALS=ROOT/"results/finding_nemo/targeted_audit_test_main_v1_nms50/insulator_consensus_proposals.csv"
OUTPUT=ROOT/"data/test_main_v4_consensus_model_assisted"
NAMES=["insulator","insulator_broken","insulator_flashover","bird_nest","foreign_object","broken_strand"]


def sha(path):
    h=hashlib.sha256(); h.update(path.read_bytes()); return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source",type=Path,default=SOURCE)
    parser.add_argument("--proposals",type=Path,default=PROPOSALS); parser.add_argument("--output",type=Path,default=OUTPUT)
    parser.add_argument("--minimum-confidence",type=float,default=0.0); parser.add_argument("--minimum-models",type=int,default=4)
    args=parser.parse_args(); source=args.source.resolve(); output=args.output.resolve()
    if output.exists() and any(output.iterdir()): raise FileExistsError(f"Refusing to overwrite {output}")
    io,lo=output/"images/test",output/"labels/test"; io.mkdir(parents=True); lo.mkdir(parents=True)
    for image in (source/"images/test").iterdir():
        try: os.link(image,io/image.name)
        except OSError: shutil.copy2(image,io/image.name)
    for label in (source/"labels/test").glob("*.txt"): shutil.copy2(label,lo/label.name)
    rows=list(csv.DictReader(args.proposals.open(encoding="utf-8-sig")))
    accepted=[r for r in rows if int(r["models"])>=args.minimum_models and float(r["mean_confidence"])>=args.minimum_confidence]
    log=[]
    for row in accepted:
        label=lo/Path(row["image"]).with_suffix(".txt").name
        old=label.read_text(encoding="utf-8"); label.write_text(old+("" if not old or old.endswith("\n") else "\n")+row["proposed_label"]+"\n",encoding="utf-8")
        log.append({"label":label.name,"action":"add_missing_insulator_4of4_yolo_consensus","new_label":row["proposed_label"],
                    "voters":row["voters"],"mean_confidence":row["mean_confidence"],"source_image":row["image"]})
    correction=output/"insulator_additions.csv"
    with correction.open("w",newline="",encoding="utf-8-sig") as f:
        writer=csv.DictWriter(f,fieldnames=list(log[0])); writer.writeheader(); writer.writerows(log)
    cfg={"path":output.as_posix(),"train":"images/test","val":"images/test","test":"images/test","names":{i:n for i,n in enumerate(NAMES)}}
    (output/"dataset.yaml").write_text(yaml.safe_dump(cfg,sort_keys=False),encoding="utf-8")
    freeze={"name":output.name,"source":str(source),"role":"multi-model-consensus-assisted exploratory analysis only",
            "warning":"Missing GT boxes were proposed by four evaluated YOLO checkpoints; this is not an independent benchmark.",
            "added_insulator_boxes":len(log),"proposal_images":len(set(r["source_image"] for r in log)),
            "minimum_consensus_models":args.minimum_models,"minimum_mean_confidence":args.minimum_confidence,
            "proposal_source":str(args.proposals.resolve()),"correction_log_sha256":sha(correction)}
    (output/"freeze.json").write_text(json.dumps(freeze,indent=2),encoding="utf-8"); print(json.dumps(freeze,indent=2))


if __name__=="__main__": main()
