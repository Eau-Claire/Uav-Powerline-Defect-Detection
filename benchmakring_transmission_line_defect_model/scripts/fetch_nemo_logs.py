from pathlib import Path
import subprocess,shlex
from dotenv import dotenv_values
root=Path(__file__).resolve().parents[2]
out=root/'artifacts/finding_nemo_stash'
key=dotenv_values(root/'.env')['CAMBER_API_KEY']
commands=['set -e','export CAMBER_API_KEY='+shlex.quote(key)]
for model in ['yolo11n','yolo11m','yolo26n','yolo26m','rtdetr50','rtdetr101']:
    remote=f'{model}/results.csv' if model.startswith('yolo') else f'rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/{"r50vd" if model=="rtdetr50" else "r101vd"}/state/result.csv'
    target=out/f'{model}_results.csv'
    linux='/mnt/'+target.drive[0].lower()+target.as_posix()[2:]
    commands.append(f'~/.camber/bin/camber stash cp stash://bunpmc/projects/finding_nemo/{remote} {shlex.quote(linux)}')
p=subprocess.run(['wsl.exe','-d','Ubuntu','--','bash','-s'],input=('\n'.join(commands)+'\n').encode(),capture_output=True)
print(p.stdout.decode('utf-8','replace')[-2500:]); print(p.stderr.decode('utf-8','replace')[-500:])
raise SystemExit(p.returncode)
