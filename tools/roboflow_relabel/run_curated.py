"""Run the cross-label audit on an existing dataset, using this folder's .env."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from dotenv import load_dotenv
import yaml

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = ['insulator', 'insulator_broken', 'insulator_flashover',
            'bird_nest', 'foreign_object', 'broken_strand']


def main():
    os.chdir(ROOT)
    load_dotenv(ROOT / 'tools/roboflow_relabel/.env', override=True)
    load_dotenv(ROOT / '.env')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default=None)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'benchmakring_transmission_line_defect_model/data/curated_combined')
    parser.add_argument('--output-root', type=Path, default=ROOT / 'artifacts/roboflow_relabel/curated_combined_serverless')
    parser.add_argument('--destination', type=Path, default=ROOT / 'data/working/curated_combined_cross_labeled')
    parser.add_argument('--phase', choices=['all', 'yolo', 'roboflow', 'export'], default='all')
    args = parser.parse_args()
    if args.device is not None:
        os.environ['CROSS_DEVICE'] = args.device
    source = args.dataset.resolve()
    metadata = yaml.safe_load((source / 'data.yaml').read_text(encoding='utf-8'))
    names = metadata['names']
    if isinstance(names, dict):
        names = [names[i] for i in range(len(names))]
    if names != EXPECTED:
        raise SystemExit(f'Wrong dataset taxonomy: {names}')
    images = {p.stem for p in (source / 'images/train').iterdir()
              if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}}
    labels = {p.stem for p in (source / 'labels/train').glob('*.txt')}
    if not images or images != labels:
        raise SystemExit(f'TRAIN image/label mismatch: images={len(images)}, labels={len(labels)}')
    model = Path(os.environ['CROSS_MODEL']).resolve()
    if args.phase in {'all', 'yolo'} and not model.is_file():
        raise SystemExit(f'Missing checkpoint: {model}')
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity = {'dataset': str(source), 'model': str(model),
                'model_size': model.stat().st_size,
                'model_mtime_ns': model.stat().st_mtime_ns,
                'train_images': sorted(images),
                'api_url': os.environ.get('ROBOFLOW_API_URL'),
                'workspace': os.environ.get('ROBOFLOW_WORKSPACE'),
                'workflow': os.environ.get('ROBOFLOW_WORKFLOW_ID')}
    manifest = output / 'run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise SystemExit('Run inputs changed; choose a new --output-root.')
    manifest.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    os.environ.update(CROSS_DATASET_ROOT=str(source), CROSS_OUTPUT_ROOT=str(output),
                      CROSS_DATASET_OUTPUT=str(args.destination.resolve()),
                      CROSS_MODEL=str(model), CROSS_PHASE=args.phase)
    print(f'TRAIN={len(images)} phase={args.phase} device={os.environ.get("CROSS_DEVICE", "cpu")}', flush=True)
    subprocess.run([sys.executable, '-u', str(ROOT / 'tools/roboflow_relabel/full_cross_audit.py')], check=True)


if __name__ == '__main__':
    main()
