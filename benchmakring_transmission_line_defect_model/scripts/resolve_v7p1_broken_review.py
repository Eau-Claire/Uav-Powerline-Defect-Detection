from pathlib import Path
from collections import defaultdict
import csv, json, shutil
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]; B=ROOT/"artifacts/review_v7p1_seed85_relabel_181126"
AUDIT=ROOT/"artifacts/dataset_audit/v7p1_171126_imported_annotation_audit/potential_missing_boxes.csv"
manifest=list(csv.DictReader((B/"manifest.csv").open(encoding="utf-8-sig")))
proposals=defaultdict(list)
for r in csv.DictReader(AUDIT.open(encoding="utf-8-sig")):
    if r["pred_class"]=="insulator_broken": proposals[r["image"]].append(r)

def xyxy(box):
    c,x,y,w,h=box; return x-w/2,y-h/2,x+w/2,y+h/2
def area(a): return max(0,a[2]-a[0])*max(0,a[3]-a[1])
def iou(a,b):
    ix1,iy1,ix2,iy2=max(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),min(a[3],b[3]); inter=max(0,ix2-ix1)*max(0,iy2-iy1)
    return inter/max(area(a)+area(b)-inter,1e-12)

replaced_images=replacements=quarantined=0
for r in manifest:
    if "merge-conflict" not in r["reasons"]: continue
    stem=Path(r["image"]).stem; decision=B/"decisions"/f"{stem}.json"
    if "seed85 conf" not in r["reasons"]:
        decision.write_text(json.dumps({
            "status":"quarantine_broken_no_local_proposal", "boxes":None,
            "note":"Source broken box uses full-insulator extent, inconsistent with curated-base local-damage convention; seed85 supplied no reliable local replacement. Exclude until manual relabel."
        },indent=2),encoding="utf-8")
        quarantined+=1; continue

    image=B/"images"/r["image"]
    with Image.open(image) as im: W,H=im.size
    label=B/"labels_seed85"/r["label"]
    boxes=[]
    for line in label.read_text(encoding="utf-8-sig").splitlines():
        p=line.split()
        if len(p)==5: boxes.append((int(float(p[0])),*map(float,p[1:])))
    cand=[]
    for p in proposals[r["image"]]:
        a=(float(p["x1"])/W,float(p["y1"])/H,float(p["x2"])/W,float(p["y2"])/H)
        cand.append((float(p["confidence"]),a))
    broken=[(i,xyxy(b)) for i,b in enumerate(boxes) if b[0]==1]
    candidate_indices=set()
    for _,c in cand:
        matches=[(iou(c,b),i) for i,b in broken]
        if matches and max(matches)[0]>=.80: candidate_indices.add(max(matches)[1])
    full=[(i,b) for i,b in broken if i not in candidate_indices]
    assigned={}
    for conf,c in sorted(cand,reverse=True):
        cx,cy=(c[0]+c[2])/2,(c[1]+c[3])/2
        choices=[]
        for i,b in full:
            contains=b[0]<=cx<=b[2] and b[1]<=cy<=b[3]
            if (contains or iou(c,b)>0) and area(b)>=area(c)*1.5: choices.append((area(b),i,b))
        if choices:
            _,i,b=min(choices)
            if i not in assigned: assigned[i]=(conf,c)
    remove=set(candidate_indices)|set(assigned)
    output=[b for i,b in enumerate(boxes) if i not in remove]
    for conf,c in assigned.values():
        x1,y1,x2,y2=c; output.append((1,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1))
    out=B/"reviewed_labels"/r["label"]
    out.write_text("".join(f"{b[0]} {b[1]:.8f} {b[2]:.8f} {b[3]:.8f} {b[4]:.8f}\n" for b in output),encoding="utf-8")
    decision.write_text(json.dumps({
        "status":"auto_replaced_full_broken_with_seed85_damage", "boxes":len(output),
        "replacements":len(assigned), "seed_candidates_removed_or_collapsed":len(candidate_indices),
        "note":"Replaced inconsistent full-insulator broken GT with highest-confidence local seed85 damage box per containing object."
    },indent=2),encoding="utf-8")
    replaced_images+=1; replacements+=len(assigned)
print(json.dumps({"replaced_images":replaced_images,"replacements":replacements,"quarantined":quarantined},indent=2))
