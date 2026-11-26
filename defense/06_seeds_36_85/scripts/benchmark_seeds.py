"""Evaluate every (seed, variant, architecture) checkpoint on several test sets and aggregate mean ± std.

Uses the project evaluator (benchmakring_transmission_line_defect_model/src/evaluator.py) so YOLO and
RT-DETR are scored the same way as the paper (MEAN_5 = mean AP50 of the 5 defect classes).

runs.csv (one row per checkpoint; checkpoint paths may be relative to the CSV):
    seed,variant,arch,checkpoint
    42,curated,yolo26m,weights/seed42/curated_yolo26m/best.pt
    85,raw,rtdetr50,weights/seed85/raw_rtdetr50/best.pt          (RT-DETR: HF checkpoint .pt)

  python tools/fn_v5_review/benchmark_seeds.py --runs runs.csv \
      --test v5r=benchmakring_transmission_line_defect_model/data/test_main_v5r_human_verified/dataset.yaml \
      --test v1=benchmakring_transmission_line_defect_model/data/test_main_v1/dataset.yaml \
      --out results/seed_benchmark

Outputs per test set: raw.json (every evaluation, cached: re-running skips finished ones),
runs.csv (one row per checkpoint), summary.csv (mean ± std over seeds), per_class.csv, table.tex.
A checkpoint missing on disk is reported and skipped, never silently counted.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / 'benchmakring_transmission_line_defect_model'
sys.path.insert(0, str(BENCH))
ARCHS = ['yolo11n', 'yolo11m', 'yolo26n', 'yolo26m', 'rtdetr50', 'rtdetr100']
LABEL = {'yolo11n': 'YOLO11n', 'yolo11m': 'YOLO11m', 'yolo26n': 'YOLO26n', 'yolo26m': 'YOLO26m',
         'rtdetr50': 'RT-DETR-R50', 'rtdetr100': 'RT-DETR-R101'}
METRICS = ['precision', 'recall', 'f1_score', 'mAP50', 'mAP50_95', 'MEAN_5_defect_ap50']
CLASSES = ['insulator', 'insulator_broken', 'insulator_flashover', 'bird_nest', 'foreign_object', 'broken_strand']


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def local_yaml(yaml_path: Path, out: Path) -> Path:
    """Copy of dataset.yaml whose `path` points at the yaml's folder (frozen yamls carry Windows paths)."""
    import yaml
    d = yaml.safe_load(yaml_path.read_text(encoding='utf-8'))
    d['path'] = str(yaml_path.resolve().parent)
    p = out / 'dataset_local.yaml'
    p.write_text(yaml.safe_dump(d, sort_keys=False), encoding='utf-8')
    return p


def fmt(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return ''
    return f'{v.mean():.3f} ± {v.std(ddof=1):.3f}' if len(v) > 1 else f'{v.mean():.3f}'


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs', type=Path, required=True)
    ap.add_argument('--test', action='append', required=True, help='name=dataset.yaml (repeatable)')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--force', action='store_true', help='re-evaluate even if cached')
    ap.add_argument('--device', default=None, help='0, 0,1 or cpu (default: GPU 0 if available, else cpu)')
    a = ap.parse_args()

    from src.evaluator import BenchmarkEvaluator   # project evaluator (YOLO + HF RT-DETR)

    runs = []
    with a.runs.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            ck = Path(r['checkpoint'])
            ck = ck if ck.is_absolute() else (a.runs.parent / ck)
            if r['arch'] not in ARCHS or r['variant'] not in ('raw', 'curated'):
                raise SystemExit(f'bad row {r}')
            runs.append(dict(seed=int(r['seed']), variant=r['variant'], arch=r['arch'], checkpoint=ck.resolve()))
    missing = [r for r in runs if not r['checkpoint'].exists()]
    for r in missing:
        print(f'[MISSING] seed {r["seed"]} {r["variant"]}_{r["arch"]}: {r["checkpoint"]}')
    runs = [r for r in runs if r['checkpoint'].exists()]
    for r in runs:
        r['checkpoint_sha256'] = sha256(r['checkpoint'])

    for spec in a.test:
        name, yaml_path = spec.split('=', 1)
        out = a.out / name
        out.mkdir(parents=True, exist_ok=True)
        data_yaml = local_yaml(Path(yaml_path), out)
        ev = BenchmarkEvaluator()
        if a.device is not None or not __import__('torch').cuda.is_available():
            ev.cfg['training']['device'] = a.device or 'cpu'
        ev.results_dir, ev.eval_file = out, out / 'raw.json'   # keep each test set separate
        ev.eval_data = ev._load_results()
        rows = []
        for r in runs:
            key = f's{r["seed"]}_{r["variant"]}_{r["arch"]}'        # suffix keeps the RT-DETR detection regex working
            res = ev.evaluate_checkpoint(key, r['checkpoint'], data_yaml, force=a.force, split='test')
            acc = res['accuracy']
            rows.append({'seed': r['seed'], 'variant': r['variant'], 'arch': r['arch'],
                         **{m: acc[m] for m in METRICS},
                         **{f'ap50_{c}': acc['per_class'][c]['ap50'] for c in CLASSES},
                         **{f'ap50_95_{c}': acc['per_class'][c]['ap50_95'] for c in CLASSES},
                         'checkpoint': str(r['checkpoint']), 'checkpoint_sha256': r['checkpoint_sha256']})
        if not rows:
            continue
        with (out / 'runs.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

        summary, per_class, tex = [], [], []
        for variant in ('raw', 'curated'):
            for arch in ARCHS:
                g = [x for x in rows if x['variant'] == variant and x['arch'] == arch]
                if not g:
                    continue
                seeds = sorted(x['seed'] for x in g)
                summary.append({'variant': variant, 'arch': arch, 'n_seeds': len(g), 'seeds': ' '.join(map(str, seeds)),
                                **{m: fmt([x[m] for x in g]) for m in METRICS}})
                per_class.append({'variant': variant, 'arch': arch, 'n_seeds': len(g),
                                  **{c: fmt([x[f'ap50_{c}'] for x in g]) for c in CLASSES}})
                cells = [fmt([x[m] for x in g]).replace('±', r'$\pm$') for m in ('mAP50', 'mAP50_95', 'MEAN_5_defect_ap50', 'f1_score')]
                tex.append(f'{LABEL[arch]} & {variant.capitalize()} & {len(g)} & ' + ' & '.join(cells) + r' \\')
        for fname, data in (('summary.csv', summary), ('per_class.csv', per_class)):
            with (out / fname).open('w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
        (out / 'table.tex').write_text(
            '% generated by tools/fn_v5_review/benchmark_seeds.py, test set: ' + name + '\n'
            r'\begin{tabular}{llcllll}' '\n' r'\toprule' '\n'
            r'Model & Train & Seeds & mAP$_{50}$ & mAP$_{50-95}$ & MEAN\_5 & F1 \\' '\n' r'\midrule' '\n'
            + '\n'.join(tex) + '\n' r'\bottomrule' '\n' r'\end{tabular}' '\n')
        print(f'\n[{name}] {len(rows)} evaluations -> {out}')
        for s in summary:
            print(f'  {s["variant"]:7s} {s["arch"]:9s} n={s["n_seeds"]} mAP50 {s["mAP50"]:>15s} mAP50-95 {s["mAP50_95"]:>15s} MEAN_5 {s["MEAN_5_defect_ap50"]}')
    if missing:
        print(f'\n{len(missing)} checkpoint(s) missing: results are incomplete.')


if __name__ == '__main__':
    main()
