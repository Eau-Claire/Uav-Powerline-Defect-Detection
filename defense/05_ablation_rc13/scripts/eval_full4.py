"""Seed85 (deployed model) on the Finding Nemo V5 test, with the exact model.val call of src/evaluator.py (RC-9a);
only the device differs (CPU here, GPU for RC-9a)."""
import hashlib, json, sys, time
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'benchmakring_transmission_line_defect_model/data/test_main_v5_full_consensus_assisted'
W = Path(sys.argv[1])
NAMES = ['insulator', 'insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']
yaml = ROOT / 'v5full_local.yaml'
yaml.write_text(f'path: {DATA}\ntrain: images/test\nval: images/test\ntest: images/test\nnames:\n'
                + ''.join(f'  {i}: {n}\n' for i, n in enumerate(NAMES)))

t0 = time.time()
r = YOLO(str(W)).val(data=str(yaml), split='test', imgsz=640, device='cpu', plots=False, verbose=False)
p, rc = float(r.results_dict['metrics/precision(B)']), float(r.results_dict['metrics/recall(B)'])
per = {n: {'ap50': round(float(r.box.ap50[i]), 4), 'ap50_95': round(float(r.box.ap[i]), 4)} for i, n in enumerate(NAMES)}
out = {
    'model': 'curated YOLO26m seed 42 (all four modules)',
    'weights_sha256': hashlib.sha256(W.read_bytes()).hexdigest(),
    'dataset': 'test_operational_min8_v5_full_consensus_assisted (1,931 images, 2,769 boxes)',
    'freeze_json_sha256': hashlib.sha256((DATA / 'freeze.json').read_bytes()).hexdigest(),
    'call': "YOLO(w).val(data, split='test', imgsz=640, device='cpu', plots=False) - same as src/evaluator.py",
    'precision': round(p, 4), 'recall': round(rc, 4), 'f1': round(2 * p * rc / (p + rc + 1e-8), 4),
    'mAP50': round(float(r.results_dict['metrics/mAP50(B)']), 4),
    'mAP50_95': round(float(r.results_dict['metrics/mAP50-95(B)']), 4),
    'MEAN_5': round(sum(per[n]['ap50'] for n in NAMES[1:]) / 5, 4),
    'per_class': per,
    'seconds': round(time.time() - t0),
}
(ROOT / 'seed42_full4_v5full.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
