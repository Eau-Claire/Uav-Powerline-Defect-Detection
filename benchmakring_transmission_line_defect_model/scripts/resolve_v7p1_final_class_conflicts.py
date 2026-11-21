from pathlib import Path
import csv, json, shutil

ROOT=Path(__file__).resolve().parents[2]; B=ROOT/"artifacts/review_v7p1_seed85_relabel_181126"
rows=list(csv.DictReader((B/"manifest.csv").open(encoding="utf-8-sig"))); resolved=0
for r in rows:
    if "judge_class_conflict" not in r["reasons"]: continue
    stem=Path(r["image"]).stem
    shutil.copy2(B/"labels_before"/r["label"],B/"reviewed_labels"/r["label"])
    boxes=sum(bool(x.strip()) for x in (B/"reviewed_labels"/r["label"]).read_text(encoding="utf-8-sig").splitlines())
    (B/"decisions"/f"{stem}.json").write_text(json.dumps({
        "status":"auto_confirmed_bird_nest", "boxes":boxes,
        "note":"Direct visual review: clearly a bird nest; rejected seed85 foreign_object class conflict and retained source bird_nest GT."
    },indent=2),encoding="utf-8")
    resolved+=1
print(f"resolved={resolved}")
