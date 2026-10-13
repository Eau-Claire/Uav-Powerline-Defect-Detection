"""Same as fo_verify.py but re-scores the cached candidates with CLIP ViT-L/14 (bigger verifier)."""
import json, numpy as np, torch, clip
from pathlib import Path
from PIL import Image
import fo_verify as F
if __name__ == "__main__":
    torch.set_num_threads(4)
    preds = json.load(open(F.HERE/'fo_verify_preds.json'))
    imgs = sorted(p for p in (F.R/'images/val_clean').iterdir() if p.suffix == '.jpg'); gts = {}
    for im in imgs:
        a = np.array([l.split() for l in (F.R/'labels/val_clean'/(im.stem+'.txt')).read_text().splitlines() if l.strip()], float).reshape(-1, 5)
        a = a[a[:, 0] == F.FO][:, 1:]; gts[im.name] = np.c_[a[:, :2]-a[:, 2:]/2, a[:, :2]+a[:, 2:]/2]
    model, prep = clip.load('ViT-L/14', device='cpu')
    with torch.no_grad():
        t = model.encode_text(clip.tokenize([f'a photo of {p}' for p in F.POS + F.NEG])); t /= t.norm(dim=-1, keepdim=True)
        for p in preds:
            im = Image.open(F.R/'images/val_clean'/p['image']).convert('RGB'); W, H = im.size
            x0, y0, x1, y1 = p['box']; pw, ph = 0.15*(x1-x0), 0.15*(y1-y0)
            f = model.encode_image(prep(im.crop((max(0, x0-pw)*W, max(0, y0-ph)*H, min(1, x1+pw)*W, min(1, y1+ph)*H)))[None])
            f /= f.norm(dim=-1, keepdim=True); pr = (100*f@t.T).softmax(-1)[0]
            p['p_fo_L'] = float(pr[:len(F.POS)].sum()); p['p_person_L'] = float(pr[len(F.POS):len(F.POS)+2].sum())
    json.dump(preds, open(F.HERE/'fo_verify_preds.json', 'w'))
    for name, score, keep in [('Pi only (conf)', lambda p: p['conf'], lambda p: True),
                              ('Pi conf x CLIP-L p_fo', lambda p: p['conf']*p['p_fo_L'], lambda p: True),
                              ('Pi conf x sqrt(CLIP-L)', lambda p: p['conf']*np.sqrt(p['p_fo_L']), lambda p: True),
                              ('Pi, drop CLIP-L person', lambda p: p['conf'], lambda p: p['p_person_L'] < 0.5),
                              ('Pi x sqrt(CLIP-L), drop person', lambda p: p['conf']*np.sqrt(p['p_fo_L']), lambda p: p['p_person_L'] < 0.5)]:
        ap, rmax, pr = F.ap_pr([(p['image'], score(p), p['box']) for p in preds if keep(p)], gts)
        print(f'{name:32s} AP50 {ap:.3f} | max recall {rmax:.3f} | precision @ recall 0.5/0.6/0.7/0.8: {pr}', flush=True)
