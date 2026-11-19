#!/usr/bin/env python3
"""Repair checkpoint discovery in the Raw-vs-Curated evaluation notebook."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "benchmakring_transmission_line_defect_model" / "finding_nemo_evaluate_raw_curated_human_reviewed_test.ipynb"


nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

nb.setdefault("metadata", {})["kernelspec"] = {
    "display_name": "SEP490 CUDA (PyTorch 2.6 cu124)",
    "language": "python",
    "name": "sep490-cuda",
}

# This notebook now reproduces the latest model-assisted operational V5 audit.
title = "".join(nb["cells"][0]["source"])
title = title.replace("Raw vs Curated — human-reviewed test evaluation",
                      "Raw vs Curated — operational min-8 audit V5 evaluation")
title = title.replace("finding_nemo_human_reviewed_6cls/", "test_operational_min8_v5_full_consensus_assisted/")
title = title.replace("# 1,931 images", "# 1,871 images").replace("# 1,931 label files, 2,850 boxes", "# 1,871 label files, 2,467 boxes")
title = title.replace(
    "This is **not** `val_clean`, `test_main_v1`, or the exploratory `finding_nemo_test_audit_v5_bundle.tar`. The v5 bundle is intentionally not used.",
    "This is the **model-assisted exploratory V5 operational subset**, not the canonical independent `test_main_v1`. Report it explicitly as V5 audit results.",
)
nb["cells"][0]["source"] = title.splitlines(keepends=True)

cell1 = "".join(nb["cells"][1]["source"])
cell1 = cell1.replace(
    "MODEL_ROOT = REPO / 'camber' / 'artifacts' / 'benchmark'",
    "MODEL_ROOT = BENCH / 'artifacts' / 'raw_curated_fixed_checkpoints'",
)
cell1 = cell1.replace(
    "DATASET_DIR = BENCH / 'data' / 'finding_nemo_human_reviewed_6cls'\nDATASET_YAML = DATASET_DIR / 'data.yaml'",
    "DATASET_DIR = BENCH / 'data' / 'test_operational_min8_v5_full_consensus_assisted'\nDATASET_YAML = DATASET_DIR / 'dataset.yaml'",
)
cell1 = cell1.replace(
    "OUTPUT_DIR = BENCH / 'results_ablation' / 'finding_nemo' / 'raw_curated_human_reviewed_test'",
    "OUTPUT_DIR = BENCH / 'results_ablation' / 'finding_nemo' / 'raw_curated_operational_min8_v5'",
)
nb["cells"][1]["source"] = cell1.splitlines(keepends=True)

verify = "".join(nb["cells"][5]["source"])
verify = verify.replace("exact human-reviewed test", "operational min-8 audit V5 test")
verify = verify.replace("len(images) == 1931", "len(images) == 1871")
verify = verify.replace("Expected exactly 1,931 human-reviewed test images.", "Expected exactly 1,871 operational V5 test images.")
verify = verify.replace("len(labels) == 1931", "len(labels) == 1871")
verify = verify.replace("boxes == 2850", "boxes == 2467")
verify = verify.replace("Expected exactly 2,850 human-reviewed GT boxes.", "Expected exactly 2,467 operational V5 GT boxes.")
verify = verify.replace("HUMAN-REVIEWED TEST VERIFIED", "OPERATIONAL MIN-8 AUDIT V5 VERIFIED")
nb["cells"][5]["source"] = verify.splitlines(keepends=True)

cell6 = r'''# 5. Fixed Raw/Curated checkpoint manifest + robust local/Camber restore.
import shutil, subprocess, shlex

CHECKPOINTS = {
    ('Raw','yolo11n'): MODEL_ROOT/'raw/yolo11n/weights/best.pt',
    ('Raw','yolo11m'): MODEL_ROOT/'raw/yolo11m/weights/best.pt',
    ('Raw','yolo26n'): MODEL_ROOT/'raw/yolo26n/weights/best.pt',
    ('Raw','yolo26m'): MODEL_ROOT/'raw/yolo26m/weights/best.pt',
    ('Raw','rtdetr50'): MODEL_ROOT/'raw/rtdetr50/best.pt',
    ('Raw','rtdetr101'): MODEL_ROOT/'raw/rtdetr101/best.pt',
    ('Curated','yolo11n'): MODEL_ROOT/'curated/yolo11n/weights/best.pt',
    ('Curated','yolo11m'): MODEL_ROOT/'curated/yolo11m/weights/best.pt',
    ('Curated','yolo26n'): MODEL_ROOT/'curated/yolo26n/weights/best.pt',
    ('Curated','yolo26m'): MODEL_ROOT/'curated/yolo26m/weights/best.pt',
    ('Curated','rtdetr50'): MODEL_ROOT/'curated/rtdetr50/best.pt',
    ('Curated','rtdetr101'): MODEL_ROOT/'curated/rtdetr101/best.pt',
}

# Six curated checkpoints already downloaded by the selected-checkpoint benchmark.
LOCAL_ALIASES = {
    ('Curated','yolo11n'): BENCH/'artifacts/finding_nemo_selected_checkpoints/yolo11n_last.pt',
    ('Curated','yolo11m'): BENCH/'artifacts/finding_nemo_selected_checkpoints/yolo11m_epoch45.pt',
    ('Curated','yolo26n'): BENCH/'artifacts/finding_nemo_selected_checkpoints/yolo26n_epoch45.pt',
    ('Curated','yolo26m'): BENCH/'artifacts/finding_nemo_selected_checkpoints/yolo26m_epoch45.pt',
    ('Curated','rtdetr50'): BENCH/'artifacts/finding_nemo_selected_checkpoints/rtdetr50_epoch15.pt',
    ('Curated','rtdetr101'): BENCH/'artifacts/finding_nemo_selected_checkpoints/rtdetr101_epoch15.pt',
}

STASH = {
    ('Raw','yolo11n'): 'stash://bunpmc/projects/raw_yolo11n/weights/best.pt',
    ('Raw','yolo11m'): 'stash://bunpmc/projects/raw_yolo11m/weights/best.pt',
    ('Raw','yolo26n'): 'stash://bunpmc/projects/raw_yolo26n/weights/best.pt',
    ('Raw','yolo26m'): 'stash://bunpmc/projects/raw_yolo26m/weights/best.pt',
    ('Raw','rtdetr50'): 'stash://bunpmc/projects/raw_rtdetr50/checkpoints/best.pt',
    ('Raw','rtdetr101'): 'stash://bunpmc/projects/raw_rtdetr101/checkpoints/best.pt',
    ('Curated','yolo11n'): 'stash://bunpmc/projects/finding_nemo/yolo11n/weights/last.pt',
    ('Curated','yolo11m'): 'stash://bunpmc/projects/finding_nemo/yolo11m/weights/epoch45.pt',
    ('Curated','yolo26n'): 'stash://bunpmc/projects/finding_nemo/yolo26n/weights/epoch45.pt',
    ('Curated','yolo26m'): 'stash://bunpmc/projects/finding_nemo/yolo26m/weights/epoch45.pt',
    ('Curated','rtdetr50'): 'stash://bunpmc/projects/finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/r50vd/checkpoints/epoch15.pt',
    ('Curated','rtdetr101'): 'stash://bunpmc/projects/finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/r101vd/checkpoints/epoch15.pt',
}

# Mirror aliases into a stable manifest without duplicating hundreds of MB when hardlinks work.
for key, source in LOCAL_ALIASES.items():
    destination = CHECKPOINTS[key]
    if destination.is_file() or not source.is_file():
        continue
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copy2(source, destination)
    print(f'[LOCAL] {key} <- {source.relative_to(REPO)}')

missing_keys = [key for key, path in CHECKPOINTS.items() if not path.is_file()]

# API key may live in the repository .env. Never print it.
if not os.environ.get('CAMBER_API_KEY') and (REPO / '.env').is_file():
    for line in (REPO / '.env').read_text(encoding='utf-8-sig').splitlines():
        if line.strip().startswith('CAMBER_API_KEY='):
            os.environ['CAMBER_API_KEY'] = line.split('=', 1)[1].strip().strip('"').strip("'")
            break

CAMBER = shutil.which('camber') or shutil.which('camber.exe')
WSL = shutil.which('wsl') or shutil.which('wsl.exe')
WSL_CAMBER = False
if not CAMBER and WSL:
    probe = subprocess.run([WSL, 'sh', '-lc', '~/.camber/bin/camber version'],
                           capture_output=True, text=True)
    WSL_CAMBER = probe.returncode == 0

def camber_run(parts, destination=None):
    """Run native Camber, or the official Linux CLI through WSL on Windows."""
    if CAMBER:
        return subprocess.run([CAMBER, *parts], capture_output=True, text=True)
    if not WSL_CAMBER:
        raise RuntimeError('Không tìm thấy Camber CLI native hoặc ~/.camber/bin/camber trong WSL.')
    env = os.environ.copy()
    env['WSLENV'] = ':'.join(x for x in [env.get('WSLENV', ''), 'CAMBER_API_KEY'] if x)
    if destination is None:
        command = '~/.camber/bin/camber ' + ' '.join(parts)
        return subprocess.run([WSL, 'sh', '-lc', command], capture_output=True, text=True, env=env)
    # Convert C:\\path -> /mnt/c/path in Python. This avoids the multiple
    # PowerShell/wsl.exe/sh quoting layers that can eat every backslash.
    win_path = str(destination.resolve())
    if len(win_path) < 3 or win_path[1:3] != ':\\':
        raise RuntimeError(f'Không chuyển được đường dẫn Windows sang WSL: {win_path}')
    wsl_destination = '/mnt/' + win_path[0].lower() + win_path[2:].replace('\\', '/')
    # wsl.exe can drop positional args following `sh -lc` on some Windows
    # configurations, so quote both values into the Linux command itself.
    command = ('~/.camber/bin/camber stash cp '
               + shlex.quote(parts[-2]) + ' ' + shlex.quote(wsl_destination))
    return subprocess.run([WSL, 'sh', '-lc', command],
                          capture_output=True, text=True, env=env)

if missing_keys and (CAMBER or WSL_CAMBER):
    if not os.environ.get('CAMBER_API_KEY'):
        raise RuntimeError('Camber CLI đã có nhưng thiếu CAMBER_API_KEY trong environment hoặc file .env.')
    print('Camber CLI:', CAMBER or 'WSL ~/.camber/bin/camber')
    auth = camber_run(['me'])
    if auth.returncode:
        raise RuntimeError('Camber CLI có nhưng chưa xác thực: ' + (auth.stderr or auth.stdout)[-1000:])
    for key in missing_keys:
        destination = CHECKPOINTS[key]
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(f'[CAMBER] {key} -> {destination.relative_to(REPO)}')
        result = camber_run(['stash', 'cp', STASH[key], str(destination)], destination=destination)
        if result.returncode:
            raise RuntimeError(f'Camber restore lỗi cho {key}:\n{(result.stderr or result.stdout)[-1500:]}')

missing_keys = [key for key, path in CHECKPOINTS.items() if not path.is_file()]
if missing_keys:
    details = '\n'.join(f'  - {key}: {STASH[key]}' for key in missing_keys)
    raise RuntimeError(
        'Thiếu checkpoint và không tìm thấy Camber CLI native hoặc Camber CLI trong WSL.\n'
        'Cài/đăng nhập Camber CLI rồi chạy lại cell này, hoặc chép các file sau thủ công:\n' + details
    )

for (regime, model), path in CHECKPOINTS.items():
    print(f'{regime:7} {model:10} {path.stat().st_size/2**20:7.1f} MiB  {path.relative_to(REPO)}')
'''
nb["cells"][6]["source"] = cell6.splitlines(keepends=True)

evaluation = "".join(nb["cells"][9]["source"])
evaluation = evaluation.replace("human-reviewed test", "operational min-8 audit V5")
evaluation = evaluation.replace(
    "finding_nemo_human_reviewed_6cls/test (1931 images, 2850 boxes)",
    "test_operational_min8_v5_full_consensus_assisted/test (1871 images, 2467 boxes; model-assisted)",
)
nb["cells"][9]["source"] = evaluation.splitlines(keepends=True)

figure = "".join(nb["cells"][11]["source"])
figure = figure.replace("SAME human-reviewed test", "SAME operational min-8 audit V5 test")
figure = figure.replace(
    "Per-class outcomes on the human-reviewed test (1,931 images, 2,850 boxes)",
    "Per-class outcomes on operational min-8 audit V5 (1,871 images, 2,467 boxes)",
)
figure = figure.replace("per_class_raw_curated_human_reviewed_test", "per_class_raw_curated_operational_min8_v5")
nb["cells"][11]["source"] = figure.splitlines(keepends=True)

for cell in nb["cells"]:
    if cell.get("cell_type") == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

NOTEBOOK.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(NOTEBOOK)
