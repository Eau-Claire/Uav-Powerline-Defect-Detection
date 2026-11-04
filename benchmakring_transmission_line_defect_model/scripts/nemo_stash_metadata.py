from pathlib import Path
import docker
import subprocess
import shlex
from dotenv import dotenv_values

root=Path(__file__).resolve().parents[2]
out=root/'artifacts/finding_nemo_stash'
out.mkdir(parents=True,exist_ok=True)
cli=root/'weights/curated_yolo26m'
key=dotenv_values(root/'.env')['CAMBER_API_KEY']
script='export CAMBER_API_KEY='+shlex.quote(key)+'\n~/.camber/bin/camber stash ls -r stash://bunpmc/projects/finding_nemo/\n'
proc=subprocess.run(['wsl.exe','-d','Ubuntu','--','bash','-s'],input=script.encode(),capture_output=True)
proc.stdout=proc.stdout.decode('utf-8','replace')
proc.stderr=proc.stderr.decode('utf-8','replace')
if proc.returncode==0:
    (out/'inventory.txt').write_text(proc.stdout,encoding='utf-8')
    print(proc.stdout)
    raise SystemExit(0)
print(proc.stderr)
raise SystemExit(proc.returncode)
c=docker.from_env().containers.run('roboflow/roboflow-inference-server-cpu:latest',entrypoint='',command=['/cli/camber','stash','ls','-r','stash://bunpmc/projects/finding_nemo/'],environment={'CAMBER_API_KEY':key,'SSL_CERT_FILE':'/cli/windows-ca.pem'},volumes={str(cli):{'bind':'/cli','mode':'ro'},str(out):{'bind':'/output','mode':'rw'}},detach=True)
result=c.wait()
value=c.logs().decode('utf-8','replace')
(out/'inventory.txt').write_text(value,encoding='utf-8')
print(value)
raise SystemExit(result['StatusCode'])
