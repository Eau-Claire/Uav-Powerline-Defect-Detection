from pathlib import Path
import csv, json, shutil

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"benchmakring_transmission_line_defect_model/data/finding_nemo_human_reviewed_6cls_v2_171126"
BUNDLE=ROOT/"artifacts/review_v7p1_seed85_relabel_181126"

manifest=list(csv.DictReader((BUNDLE/"manifest.csv").open(encoding="utf-8-sig")))
applied=quarantined=0; status_counts={}
for r in manifest:
    stem=Path(r["image"]).stem
    decision_path=BUNDLE/"decisions"/f"{stem}.json"
    reviewed=BUNDLE/"reviewed_labels"/r["label"]
    if not decision_path.is_file(): raise RuntimeError(f"Missing decision: {stem}")
    decision=json.loads(decision_path.read_text(encoding="utf-8")); status=decision["status"]
    status_counts[status]=status_counts.get(status,0)+1
    image=DATA/"images/train"/r["image"]; label=DATA/"labels/train"/r["label"]
    if status=="quarantine_broken_no_local_proposal":
        if image.is_file(): image.unlink()
        if label.is_file(): label.unlink()
        cache=DATA/"images/train"/f"{stem}.npy"
        if cache.is_file(): cache.unlink()
        quarantined+=1
        continue
    if not reviewed.is_file(): raise RuntimeError(f"Missing reviewed label: {reviewed}")
    if not image.is_file() or not label.is_file(): raise RuntimeError(f"Missing v2 pair: {r['image']}")
    shutil.copy2(reviewed,label); applied+=1

cache=DATA/"labels/train.cache"
if cache.is_file(): cache.unlink()
summary={
    "review_bundle":str(BUNDLE),"dataset":str(DATA),"labels_applied":applied,
    "train_pairs_quarantined_and_removed":quarantined,"status_counts":status_counts,
    "policy":"reviewed labels applied in place to v2; quarantine images removed from train; val/test unchanged",
}
(DATA/"REVIEW_APPLIED_181126.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
print(json.dumps(summary,indent=2))
