"""Extract v6p5 v3 and run the resumable TRAIN-only cross-label audit."""
import argparse
import os
import json
import subprocess
import sys
import zipfile
from pathlib import Path

from dotenv import load_dotenv

root=Path(__file__).resolve().parents[2]


def default_archive() -> Path:
    configured = os.environ.get("V3_ARCHIVE")
    candidates = [
        Path(configured).expanduser() if configured else root/'data/transfer/curated_v6p5_v3.zip',
        Path.home()/'Downloads/curated_v6p5_v3.zip',
    ]
    return next((path for path in candidates if path.is_file()), candidates[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=default_archive())
    parser.add_argument('--model', type=Path,
                        default=Path(os.environ.get('CROSS_MODEL', root/'weights/curated_yolo26m/best.pt')))
    parser.add_argument('--device', default=os.environ.get('CROSS_DEVICE', 'cpu'),
                        help="Ultralytics device: cpu or 0 for the first CUDA GPU")
    parser.add_argument('--phase', choices=('yolo', 'roboflow', 'export', 'all'),
                        default='all', help='run one phase or the full sequence')
    parser.add_argument('--output-root', type=Path,
                        default=root/'artifacts/roboflow_relabel/cross_v3')
    return parser.parse_args()


def main() -> None:
    os.chdir(root)
    load_dotenv(root/'.env')
    args = parse_args()
    archive = args.archive.expanduser().resolve()
    model = args.model.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f'Missing dataset ZIP: {archive}\nSet V3_ARCHIVE or pass --archive PATH.')
    if args.phase in {'yolo', 'all'} and not model.is_file():
        raise SystemExit(f'Missing YOLO teacher checkpoint: {model}\nSet CROSS_MODEL or pass --model PATH.')
    if args.phase in {'roboflow', 'all'} and not os.environ.get('ROBOFLOW_API_KEY', '').strip():
        raise SystemExit('ROBOFLOW_API_KEY is missing from .env. Use a newly issued key.')

    extract=root/'data/working/v6p5_v3_source'
    source=extract/'curated_combined'
    marker=extract/'EXTRACT_COMPLETE.json'
    archive_stat = archive.stat()
    identity = {'archive': str(archive), 'size': archive_stat.st_size,
                'mtime_ns': archive_stat.st_mtime_ns}
    marker_data = json.loads(marker.read_text(encoding='utf-8')) if marker.is_file() else None
    if marker_data != identity:
        if marker.exists():
            raise SystemExit(
                'The dataset archive changed after extraction. Remove only '
                f'{extract} after verifying the new ZIP, then rerun.'
            )
        extract.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(archive) as z:
            entries=z.infolist()
            for index,entry in enumerate(entries,1):
                target=(extract/entry.filename).resolve()
                if not target.is_relative_to(extract.resolve()):
                    raise RuntimeError('Unsafe zip entry')
                z.extract(entry,extract)
                if index%2000==0: print(f'EXTRACT {index}/{len(entries)}',flush=True)
        marker.write_text(json.dumps(identity, indent=2),encoding='utf-8')

    for split,count in [('train',17903),('val',2272),('val_clean',2271)]:
        images_dir = source/'images'/split
        labels_dir = source/'labels'/split
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise SystemExit(f'Missing images/{split} or labels/{split} under {source}')
        images=[p for p in images_dir.iterdir()
                if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'}]
        labels=list(labels_dir.glob('*.txt'))
        if len(images) != count or len(labels) != count:
            raise SystemExit(
                f'Unexpected {split} count: images={len(images)}, labels={len(labels)}, expected={count}'
            )
    print('V3 verified: train=17903 val=2272 val_clean=2271',flush=True)
    os.environ['CROSS_DATASET_ROOT']=str(source)
    os.environ['CROSS_OUTPUT_ROOT']=str(args.output_root.resolve())
    os.environ['CROSS_DATASET_OUTPUT']=str((root/'data/working/v6p5_v3_cross_labeled').resolve())
    os.environ['CROSS_MODEL']=str(model)
    os.environ['CROSS_DEVICE']=args.device
    os.environ['CROSS_PHASE']=args.phase
    subprocess.run([sys.executable,'-u','tools/roboflow_relabel/full_cross_audit.py'],check=True)


if __name__ == '__main__':
    main()
