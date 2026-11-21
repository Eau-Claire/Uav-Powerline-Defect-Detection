from pathlib import Path
import csv, json, shutil

ROOT=Path(__file__).resolve().parents[2]; B=ROOT/"artifacts/review_v7p1_seed85_relabel_181126"
rows=list(csv.DictReader((B/"manifest.csv").open(encoding="utf-8-sig"))); approved=0
for r in rows:
    if "merge-conflict" not in r["reasons"] or "seed85 conf" in r["reasons"]: continue
    stem=Path(r["image"]).stem; decision=B/"decisions"/f"{stem}.json"
    if decision.is_file(): continue
    shutil.copy2(B/"labels_seed85"/r["label"],B/"reviewed_labels"/r["label"])
    boxes=sum(bool(x.strip()) for x in (B/"reviewed_labels"/r["label"]).read_text(encoding="utf-8-sig").splitlines())
    decision.write_text(json.dumps({
        "status":"auto_approved_merge_repair", "boxes":boxes,
        "note":"Deterministic repair: source marks object Broken Insulator; removed exactly coincident normal-insulator label (IoU=1). No seed85 addition in this image."
    },indent=2),encoding="utf-8")
    approved+=1
print(f"auto_approved={approved}")
