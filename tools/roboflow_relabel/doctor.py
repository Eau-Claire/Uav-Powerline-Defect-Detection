"""Safe preflight for a fresh or resumed v3 cross-label run."""
import os
import sys
import hashlib
import argparse
import shutil
import subprocess
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_MODEL_SHA256 = '65877bab3405bbec4a91ae888cad67c5f4a1bee8e43a56dd9331e229a12c4db0'
load_dotenv(ROOT/'.env')


def resolve(value: str, fallback: Path) -> Path:
    path = Path(value).expanduser() if value else fallback
    return path if path.is_absolute() else ROOT/path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-server', action='store_true',
                        help='also fail unless ROBOFLOW_API_URL accepts a connection')
    args = parser.parse_args()
    archive = resolve(os.getenv('V3_ARCHIVE', ''), ROOT/'data/transfer/curated_v6p5_v3.zip')
    model = resolve(os.getenv('CROSS_MODEL', ''), ROOT/'weights/curated_yolo26m/best.pt')
    model_hash_ok = False
    if model.is_file():
        digest = hashlib.sha256()
        with model.open('rb') as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(chunk)
        model_hash_ok = digest.hexdigest() == EXPECTED_MODEL_SHA256
    docker_cli = shutil.which('docker') is not None
    docker_daemon = False
    if docker_cli:
        try:
            docker_daemon = subprocess.run(
                ['docker', 'info'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                timeout=10, check=False,
            ).returncode == 0
        except (OSError, subprocess.SubprocessError):
            pass
    api_url = os.getenv('ROBOFLOW_API_URL', '').strip()
    server_reachable = False
    if api_url:
        try:
            with urllib.request.urlopen(api_url.rstrip('/')+'/', timeout=5) as response:
                server_reachable = response.status < 500
        except urllib.error.HTTPError as exc:
            server_reachable = exc.code < 500
        except (urllib.error.URLError, ValueError):
            pass
    checks = {
        'Python >= 3.10': sys.version_info >= (3, 10),
        f'Dataset ZIP: {archive}': archive.is_file(),
        'Dataset file is a ZIP': archive.is_file() and zipfile.is_zipfile(archive),
        f'Teacher model: {model}': model.is_file(),
        'Teacher model SHA-256': model_hash_ok,
        'ROBOFLOW_API_KEY configured': bool(os.getenv('ROBOFLOW_API_KEY', '').strip())
            and os.getenv('ROBOFLOW_API_KEY') != 'replace_with_a_new_key',
        'ROBOFLOW_API_URL configured': bool(api_url),
        'Docker CLI installed': docker_cli,
        'Docker daemon available': docker_daemon,
    }
    if args.require_server:
        checks['Roboflow inference server reachable'] = server_reachable
    for label, ok in checks.items():
        print(('OK   ' if ok else 'FAIL ')+label)
    progress = ROOT/'artifacts/roboflow_relabel/cross_v3'
    for relative in ('yolo.jsonl', 'roboflow/proposals.jsonl'):
        path = progress/relative
        count = sum(1 for line in path.open(encoding='utf-8') if line.strip()) if path.is_file() else 0
        print(f'INFO checkpoint {relative}: {count} records')
    print(f"INFO device: {os.getenv('CROSS_DEVICE', 'cpu')}")
    if not args.require_server:
        print('INFO Roboflow server reachable:', server_reachable,
              '(use --require-server before the run)')
    return 0 if all(checks.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
