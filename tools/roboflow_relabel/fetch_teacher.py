from pathlib import Path
import ssl
import docker
from dotenv import dotenv_values

root = Path(__file__).resolve().parents[2]
target = root / 'weights' / 'curated_yolo26m'
target.mkdir(parents=True, exist_ok=True)
certificates = []
for store in ('ROOT', 'CA'):
    for certificate, encoding, trust in ssl.enum_certificates(store):
        if encoding == 'x509_asn':
            certificates.append(ssl.DER_cert_to_PEM_cert(certificate))
(target / 'windows-ca.pem').write_text(''.join(certificates), encoding='ascii')
key = dotenv_values(root / '.env').get('CAMBER_API_KEY')
if not key:
    raise SystemExit('Missing CAMBER_API_KEY')
command = '''set -e
chmod +x /output/camber
/output/camber stash cp stash://bunpmc/projects/curated_yolo26m/weights/best.pt /output/best.pt
'''
client = docker.from_env()
container = client.containers.run('roboflow/roboflow-inference-server-cpu:latest', command=['bash','-c',command], entrypoint='', environment={'CAMBER_API_KEY':key, 'SSL_CERT_FILE':'/output/windows-ca.pem'}, volumes={str(target):{'bind':'/output','mode':'rw'}}, detach=True)
for line in container.logs(stream=True, follow=True):
    print(line.decode('utf-8','replace'), end='', flush=True)
status = container.wait()['StatusCode']
if status:
    raise SystemExit(status)
print(target / 'best.pt')
