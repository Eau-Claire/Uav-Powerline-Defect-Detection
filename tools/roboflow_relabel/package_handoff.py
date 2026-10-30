"""Package complete checkpoint lines and exact required files, then upload to Camber."""
from pathlib import Path
import io
import json
import tarfile
import docker
from dotenv import dotenv_values

root=Path(__file__).resolve().parents[2]
out=root/'artifacts/roboflow_relabel/handoff'
out.mkdir(parents=True,exist_ok=True)
archive=out/'cross_v3_resume_v1.tar'
counts={}
with tarfile.open(archive,'w') as tar:
    for p in (root/'tools/roboflow_relabel').glob('*'):
        if p.suffix in {'.py','.md','.txt','.ps1'}:
            tar.add(p,arcname=str(p.relative_to(root)))
    tar.add(root/'weights/curated_yolo26m/best.pt',arcname='weights/curated_yolo26m/best.pt')
    tar.add(Path.home()/'Downloads/curated_v6p5_v3.zip',arcname='data/transfer/curated_v6p5_v3.zip')
    for relative in ['yolo.jsonl','roboflow/proposals.jsonl']:
        p=root/'artifacts/roboflow_relabel/cross_v3'/relative
        if not p.exists(): continue
        complete=[]
        with p.open('rb') as handle:
            for line in handle:
                try: json.loads(line)
                except (ValueError,UnicodeError): break
                complete.append(line)
        data=b''.join(complete); counts[relative]=len(complete)
        info=tarfile.TarInfo('artifacts/roboflow_relabel/cross_v3/'+relative)
        info.size=len(data); tar.addfile(info,io.BytesIO(data))
    data=json.dumps({'checkpoint_counts':counts},indent=2).encode()
    info=tarfile.TarInfo('HANDOFF_SNAPSHOT.json'); info.size=len(data); tar.addfile(info,io.BytesIO(data))
print('PACKAGED',archive,archive.stat().st_size,counts,flush=True)
key=dotenv_values(root/'.env')['CAMBER_API_KEY']
cli=root/'weights/curated_yolo26m'
client=docker.from_env()
c=client.containers.run('roboflow/roboflow-inference-server-cpu:latest',entrypoint='',command=['/cli/camber','stash','cp','/transfer/cross_v3_resume_v1.tar','stash://bunpmc/projects/yolov11sdi/transfers/cross_v3_resume_v1.tar'],environment={'CAMBER_API_KEY':key,'SSL_CERT_FILE':'/cli/windows-ca.pem'},volumes={str(cli):{'bind':'/cli','mode':'ro'},str(out):{'bind':'/transfer','mode':'ro'}},detach=True)
for chunk in c.logs(stream=True,follow=True):
    value=chunk.decode('utf-8','replace')
    if 'Uploading' not in value: print(value,end='',flush=True)
status=c.wait()['StatusCode']
print('UPLOAD_EXIT',status,flush=True)
raise SystemExit(status)
