"""Wait for both inference checkpoints, then export the merged pseudo-labels."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'artifacts/roboflow_relabel/curated_combined_serverless'

def main():
    manifest = json.loads((OUT / 'run_manifest.json').read_text(encoding='utf-8'))
    expected = len(manifest['train_images'])
    while True:
        counts = {}
        for name in ('yolo.jsonl', 'roboflow/proposals.jsonl'):
            path = OUT / name
            counts[name] = sum(1 for line in path.open(encoding='utf-8') if line.strip()) if path.exists() else 0
        status = {'expected': expected, 'completed': counts, 'status': 'inference_running', 'updated_at': time.strftime('%Y-%m-%d %H:%M:%S')}
        (OUT / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        print(json.dumps(status), flush=True)
        if all(count == expected for count in counts.values()):
            subprocess.run([sys.executable, '-u', str(ROOT / 'tools/roboflow_relabel/run_curated.py'), '--phase', 'export'], cwd=ROOT, check=True)
            status['status'] = 'export_complete'
            (OUT / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
            return
        time.sleep(30)

if __name__ == '__main__':
    main()
