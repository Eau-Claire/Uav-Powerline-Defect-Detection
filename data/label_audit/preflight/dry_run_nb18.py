"""Pre-flight: run every code cell of notebook 18 on a mini copy of the dataset (160/80/80 images, imgsz 320,
2 epochs) to catch code errors before the real GPU run. Writes only under ./dry/."""
import json, os, random, shutil, time, yaml
from pathlib import Path
HERE = Path(__file__).resolve().parent; D = HERE/'dry'
R = Path('/teamspace/studios/this_studio/datasets/v6p5_v4')
NB = Path('/teamspace/studios/this_studio/Uav-Powerline-Defect-Detection/legacy/notebooks/v6p5/v6p5-18_20261012-v4-person-teacher-distill-yolo26n.ipynb')
def mini():
    M = D/'mini'; random.seed(0)
    for split, n in [('train', 160), ('val', 80), ('val_clean', 80)]:
        (M/'images'/split).mkdir(parents=True); (M/'labels'/split).mkdir(parents=True)
        stems = sorted(p.stem for p in (R/'labels'/split).glob('*.txt'))
        pri = [s for s in stems if any(l[:2] in ('0 ', '6 ') for l in (R/'labels'/split/(s+'.txt')).read_text().splitlines())]
        for s in dict.fromkeys(random.sample(pri, n//4) + random.sample(stems, n - n//4)):
            os.symlink(os.path.realpath(R/'images'/split/(s+'.jpg')), M/'images'/split/(s+'.jpg'))
            shutil.copy(R/'labels'/split/(s+'.txt'), M/'labels'/split/(s+'.txt'))
    names = yaml.safe_load((R/'data.yaml').read_text())['names']
    for y, v in [('data.yaml', 'images/val'), ('data_val_clean.yaml', 'images/val_clean')]:
        (M/y).write_text(yaml.safe_dump({'path': str(M), 'train': 'images/train', 'val': v, 'names': names}, sort_keys=False))
    shutil.copy(R/'manifest.json', M/'manifest.json')
    return M
if __name__ == "__main__":
    shutil.rmtree(D, ignore_errors=True); M = mini()
    os.chdir(NB.parent)
    code = [''.join(c['source']) for c in json.load(open(NB))['cells'] if c['cell_type'] == 'code' and ''.join(c['source']).strip()]
    g = {'display': print}
    def run(prefix, patch=()):
        src = next(c for c in code if c.startswith(prefix))
        for a, b in patch: assert a in src, (prefix, a); src = src.replace(a, b)
        t = time.time(); print(f'\n===== {prefix} =====', flush=True); exec(src, g); print(f'----- ok {time.time()-t:.0f}s', flush=True)
    run('# 1. CONFIG')
    g.update(DATASET=M, DATA_YAML=M/'data.yaml', VAL_CLEAN_YAML=M/'data_val_clean.yaml', OUT=D/'out', RUNS=D/'out/runs', IMGSZ=320)
    g['COMMON'].update(data=str(M/'data.yaml'), imgsz=320, cache=False, workers=2, save_period=1)
    run('# 2. HÀM TRAIN'); run('# 2b.')
    run('# 3. TEACHER', [('epochs=25', 'epochs=2')])
    run('# 3. TEACHER', [('epochs=25', 'epochs=2')])  # second call must skip ("đã xong, bỏ qua")
    run('# 4. STUDENT', [('epochs=30', 'epochs=2')])
    run('# 5. EXPORT NCNN'); run('# 6. EVAL'); run('# 7. XEM TIẾN ĐỘ'); run('# 8. (TÙY CHỌN)', [('AUTO_STOP = True', 'AUTO_STOP = False')] if 'AUTO_STOP = True' in next(c for c in code if c.startswith('# 8.')) else [])  # never stop the studio from a dry run
    print('\nDRY RUN PASSED', flush=True)
