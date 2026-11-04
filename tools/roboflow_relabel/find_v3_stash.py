from pathlib import Path
import docker
from dotenv import dotenv_values

root=Path(__file__).resolve().parents[2]
mount=root/'weights/curated_yolo26m'
key=dotenv_values(root/'.env')['CAMBER_API_KEY']
client=docker.from_env()
c=client.containers.run('roboflow/roboflow-inference-server-cpu:latest',entrypoint='',command=['/output/camber','stash','ls','-r','stash://bunpmc/projects'],environment={'CAMBER_API_KEY':key,'SSL_CERT_FILE':'/output/windows-ca.pem'},volumes={str(mount):{'bind':'/output','mode':'ro'}},detach=True)
lines=[]
for chunk in c.logs(stream=True,follow=True):
    value=chunk.decode('utf-8','replace'); lines.append(value)
    if any(term in value.lower() for term in ['v3','curated','error']): print(value,end='',flush=True)
(root/'artifacts/roboflow_relabel/stash_inventory.txt').write_text(''.join(lines),encoding='utf-8')
print('status',c.wait()['StatusCode'])
