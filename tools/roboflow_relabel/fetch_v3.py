from pathlib import Path
import docker
from dotenv import dotenv_values

root=Path(__file__).resolve().parents[2]
target=root/'data/working/v3_download'
target.mkdir(parents=True,exist_ok=True)
cli=root/'weights/curated_yolo26m'
key=dotenv_values(root/'.env')['CAMBER_API_KEY']
client=docker.from_env()
c=client.containers.run('roboflow/roboflow-inference-server-cpu:latest',entrypoint='',command=['/cli/camber','stash','cp','stash://bunpmc/projects/yolov11sdi/data/curated_v3_full.zip','/output/curated_v3_full.zip'],environment={'CAMBER_API_KEY':key,'SSL_CERT_FILE':'/cli/windows-ca.pem'},volumes={str(cli):{'bind':'/cli','mode':'ro'},str(target):{'bind':'/output','mode':'rw'}},detach=True)
for chunk in c.logs(stream=True,follow=True):
    value=chunk.decode('utf-8','replace')
    if 'Downloading' not in value: print(value,end='',flush=True)
status=c.wait()['StatusCode']
if status: raise SystemExit(status)
import zipfile
with zipfile.ZipFile(target/'curated_v3_full.zip') as archive:
    names=archive.namelist()
    print('entries',len(names),flush=True)
    for name in names:
        if name.endswith(('manifest.json','data.yaml')):
            print(name,archive.read(name).decode('utf-8','replace')[:6000],flush=True)
    print('top directories',sorted({n.split('/')[0] for n in names}),flush=True)
