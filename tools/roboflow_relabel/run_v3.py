"""Extract the verified v3 archive and run the complete resumable cross audit."""
import os
import json
import subprocess
import sys
import zipfile
from pathlib import Path

root=Path(__file__).resolve().parents[2]
os.chdir(root)
archive=Path(os.environ.get('V3_ARCHIVE',str(root/'data/transfer/curated_v6p5_v3.zip')))
if not archive.exists() and os.name=='nt':
    archive=Path.home()/'Downloads/curated_v6p5_v3.zip'
extract=root/'data/working/v6p5_v3_source'
source=extract/'curated_combined'
marker=extract/'EXTRACT_COMPLETE.json'
if not marker.exists():
    extract.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        entries=z.infolist()
        for index,entry in enumerate(entries,1):
            target=(extract/entry.filename).resolve()
            if not target.is_relative_to(extract.resolve()): raise RuntimeError('Unsafe zip entry')
            z.extract(entry,extract)
            if index%2000==0: print(f'EXTRACT {index}/{len(entries)}',flush=True)
    marker.write_text(json.dumps({'archive':str(archive)}),encoding='utf-8')
for split,count in [('train',17903),('val',2272),('val_clean',2271)]:
    images=[p for p in (source/'images'/split).iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}]
    assert len(images)==count,(split,len(images))
print('V3 verified: train=17903 val=2272 val_clean=2271',flush=True)
os.environ['CROSS_DATASET_ROOT']=str(source)
os.environ['CROSS_OUTPUT_ROOT']=str(root/'artifacts/roboflow_relabel/cross_v3')
os.environ['CROSS_DATASET_OUTPUT']=str(root/'data/working/v6p5_v3_cross_labeled')
subprocess.run([sys.executable,'-u','tools/roboflow_relabel/full_cross_audit.py'],check=True)
