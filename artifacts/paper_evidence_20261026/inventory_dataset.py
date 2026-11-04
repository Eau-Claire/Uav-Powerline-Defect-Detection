"""Audit a materialized YOLO dataset without modifying it (standard library only).
Usage: python inventory_dataset.py DATASET_ROOT --out OUTPUT_DIR
Counts class images once per image per class; distinguishes missing and empty labels.
"""
import argparse, csv, hashlib, json, math
from collections import Counter
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root',type=Path);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if not (a.root/'images').is_dir(): ap.error('Dataset root must contain images/')
    rows=[]; summaries=[]; classes=[]
    for split in sorted(p for p in (a.root/'images').iterdir() if p.is_dir()):
        paths=sorted(p for p in split.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'})
        ci=Counter();cb=Counter(); statuses=Counter(); hashes=set(); label_keys=Counter()
        for p in paths:
            rel=p.relative_to(split);lab=a.root/'labels'/split.name/rel.with_suffix('.txt')
            label_keys[str(lab)]+=1
            h=hashlib.sha256()
            with p.open('rb') as f:
                for block in iter(lambda:f.read(1048576),b''):h.update(block)
            digest=h.hexdigest();hashes.add(digest)
            counts=Counter(); errors=[]
            if not lab.exists(): status='missing'
            else:
                lines=[s for s in lab.read_text().splitlines() if s.strip()]
                status='empty' if not lines else 'annotated'
                for i,line in enumerate(lines,1):
                    try:
                        vals=list(map(float,line.split()))
                        if len(vals)!=5 or not all(math.isfinite(v) for v in vals) or not vals[0].is_integer() or vals[0]<0: raise ValueError()
                        if not all(0<=v<=1 for v in vals[1:]) or vals[3]<=0 or vals[4]<=0: raise ValueError()
                        counts[int(vals[0])]+=1
                    except ValueError: errors.append(i)
                if errors:status='invalid'
            statuses[status]+=1;ci.update(counts.keys());cb.update(counts)
            rows.append(dict(split=split.name,image=str(rel),sha256=digest,label_status=status,boxes=sum(counts.values()),class_ids=' '.join(map(str,sorted(counts))),invalid_lines=' '.join(map(str,errors))))
        summaries.append(dict(split=split.name,image_files=len(paths),unique_sha256_files=len(hashes),exact_duplicate_copies=len(paths)-len(hashes),missing_labels=statuses['missing'],empty_labels=statuses['empty'],invalid_label_files=statuses['invalid'],images_without_valid_boxes=sum(r['boxes']==0 for r in rows if r['split']==split.name),valid_boxes=sum(cb.values()),class_image_occurrences=sum(ci.values()),shared_label_paths=sum(v>1 for v in label_keys.values())))
        classes.extend(dict(split=split.name,class_id=c,images=ci[c],boxes=cb[c],image_percentage=100*ci[c]/len(paths) if paths else 0) for c in sorted(ci))
    a.out.mkdir(parents=True,exist_ok=True)
    for name,data in [('images',rows),('splits',summaries),('classes',classes)]:
        with (a.out/f'{name}.csv').open('w',newline='') as f:
            if data:
                w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    (a.out/'summary.json').write_text(json.dumps({'dataset_root':str(a.root.resolve()),'definition':'class images = distinct image paths containing >=1 valid box of that class; SHA256 uniqueness is byte-level, not perceptual uniqueness','splits':summaries,'classes':classes},indent=2))
if __name__=='__main__':main()
