"""Create local secret file and print only the SHA-256 for BE registration."""
import argparse
import hashlib
import os
import secrets
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--out',type=Path,required=True)
args=parser.parse_args()
args.out.parent.mkdir(parents=True,exist_ok=True)
key=secrets.token_urlsafe(32)
fd=os.open(args.out,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
with os.fdopen(fd,'w') as target:target.write(key+'\n')
print('SHA256 (register in BE):',hashlib.sha256(key.encode()).hexdigest())
print('Secret stored in:',args.out)
