from pathlib import Path
import os
from dotenv import load_dotenv
from core import find_image, run_workflow_batch, environment_defaults, load_jsonl

root=Path(__file__).resolve().parents[2]
load_dotenv(root/'.env')
source=Path(os.environ.get('CROSS_DATASET_ROOT',str(root/'data/cache/base_extract/curated_v6')))
out=Path(os.environ.get('CROSS_OUTPUT_ROOT',str(root/'artifacts/roboflow_relabel/cross_full_v1')))/'roboflow'
out.mkdir(parents=True,exist_ok=True)
cache=out/'proposals.jsonl'
if not cache.exists() and not os.environ.get('CROSS_DATASET_ROOT'):
    old=root/'artifacts/roboflow_relabel/full_train/proposals.jsonl'
    if old.exists():
        import shutil
        shutil.copy2(old,cache)
candidates=[]
for p in sorted((source/'labels/train').glob('*.txt')):
    image=find_image(source/'images/train',p.stem)
    candidates.append({'image_path':str(image),'label_path':str(p),'box_count':0})
# Normalize cached relative paths to match this run's absolute paths.
if cache.exists():
    import json
    rows=load_jsonl(cache)
    for r in rows:
        r['image_path']=str(source/'images/train'/Path(r['image_path'].replace('\\','/')).name)
        r['label_path']=str(source/'labels/train'/Path(r['label_path'].replace('\\','/')).name)
    cache.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
d=environment_defaults()
run_workflow_batch(candidates,out,d['api_url'],os.environ['ROBOFLOW_API_KEY'],d['workspace_name'],d['workflow_id'],{'insulator'},0.6,1.1,20,resume=True,workers=4)
