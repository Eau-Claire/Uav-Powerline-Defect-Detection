from pathlib import Path
import csv, json, shutil

ROOT=Path(__file__).resolve().parents[2]; B=ROOT/"artifacts/review_v7p1_seed85_relabel_181126"
rows=list(csv.DictReader((B/"manifest.csv").open(encoding="utf-8-sig"))); rejected=0
for r in rows:
    if "seed85 conf" not in r["reasons"] or "merge-conflict" in r["reasons"]: continue
    stem=Path(r["image"]).stem; decision=B/"decisions"/f"{stem}.json"
    if decision.is_file(): continue
    shutil.copy2(B/"labels_before"/r["label"],B/"reviewed_labels"/r["label"])
    boxes=sum(bool(x.strip()) for x in (B/"reviewed_labels"/r["label"]).read_text(encoding="utf-8-sig").splitlines())
    decision.write_text(json.dumps({
        "status":"auto_rejected_seed_add", "boxes":boxes,
        "note":"Visual review of ADD-only contact sheets: domain false positive, duplicate/misaligned existing object, person/tower/window/cable confusion. Restored pre-seed85 label."
    },indent=2),encoding="utf-8")
    rejected+=1
print(f"auto_rejected={rejected}")
