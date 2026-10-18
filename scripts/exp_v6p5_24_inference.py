"""v6p5-24: raise the two-tier MEAN_5 WITHOUT training — server-side inference tricks for the teacher.

Fine-tuning (nb22, v6p5-23) only holds v5_orig's level, so try what the server can afford instead:
  res      FO at 1024 (FO is small; 640 -> 800 already gave +0.03 on FO)
  tta      ultralytics augment=True (flips + scales)
  wbf      weighted box fusion of several teachers (v5_orig + fine-tuned ones), per resolution
Scored like nb22/v6p5-23: Pi n_v5 flags @0.05, val_clean_v6 (+ val_clean), MEAN_5 AP50, paired bootstrap vs the
current server (v5_orig: FO@800, rest @640). NOTE: configs are compared on the test set itself -> treat the best one as
a hypothesis, confirm on new data before shipping.

  .venv/bin/python scripts/exp_v6p5_24_inference.py      (waits for v6p5-23 to finish if it is still running)
Results: artifacts/v6p5_v7_gentle/infer/{eval_infer.csv, eval_infer_compare.csv, progress.log}
"""
import json, subprocess, sys, time
from pathlib import Path

import numpy as np, pandas as pd
from ensemble_boxes import weighted_boxes_fusion
from ultralytics import YOLO

REPO = Path(__file__).resolve().parents[1]
V7 = REPO / "artifacts/v6p5_v7_gentle"; OUT = V7 / "infer"; OUT.mkdir(parents=True, exist_ok=True)
NB22 = REPO / "artifacts/v6p5_v6_teachers"
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "data/label_audit/preflight"))
from system_two_tier import flagged, run as cached_predict   # noqa: E402
from system_two_tier_bootstrap import prep, wap               # noqa: E402

DS6 = REPO / "datasets/v6p5_v6"
DEFECTS = ["insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
TAU, B, SEED = 0.05, 2000, 20261015
log_f = open(OUT / "progress.log", "a")
def log(*a):
    s = f"[{time.strftime('%H:%M')}] " + " ".join(map(str, a)); print(s, flush=True); log_f.write(s + "\n"); log_f.flush()


def predict(w, imgs, tag, imgsz, augment=False):
    cache = OUT / f"pred_{tag}.json"
    if cache.is_file():
        d = json.loads(cache.read_text())
        if d["_weights"] == str(w): return d["dets"]
    m = YOLO(str(w), task="detect"); out = {}
    for k, im in enumerate(imgs):
        r = m.predict(str(im), imgsz=imgsz, conf=0.001, rect=False, augment=augment, verbose=False, device=0)[0]
        out[im.name] = [[int(c), float(s), *b] for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist())]
    cache.write_text(json.dumps({"_weights": str(w), "dets": out})); log(f"pred {tag} done"); return out


def wbf(dets_list, weights=None):
    out = {}
    for n in dets_list[0]:
        bl = [[np.clip(d[2:], 0, 1).tolist() for d in D[n]] for D in dets_list]
        sl = [[d[1] for d in D[n]] for D in dets_list]; ll = [[d[0] for d in D[n]] for D in dets_list]
        if not any(bl): out[n] = []; continue
        b, s, l = weighted_boxes_fusion(bl, sl, ll, weights=weights, iou_thr=0.55, skip_box_thr=0.001, conf_type="avg")
        out[n] = [[int(c), float(x), *bb] for c, x, bb in zip(l, s, b.tolist())]
    return out


if __name__ == "__main__":
    while subprocess.run(["pgrep", "-f", "exp_v6p5_23_gentle_ft.py"], capture_output=True).returncode == 0:
        time.sleep(60)                                    # GPU busy with v6p5-23
    log("===== v6p5-24 inference tricks")
    IMGS = sorted((DS6 / "images/val_clean_v6").glob("*.jpg")); IDX = {p.name: i for i, p in enumerate(IMGS)}
    def gt(split):
        G = {c: {} for c in range(1, 6)}
        for im in IMGS:
            x = np.array([l.split() for l in (DS6 / "labels" / split / (im.stem + ".txt")).read_text().splitlines() if l.strip()],
                         float).reshape(-1, 5)
            for c in range(1, 6):
                b = x[x[:, 0] == c][:, 1:]; G[c][im.name] = np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2]
        return G
    GTS = {s: gt(s) for s in ["val_clean_v6", "val_clean"]}
    rng = np.random.default_rng(SEED)
    W = [np.bincount(rng.integers(0, len(IMGS), len(IMGS)), minlength=len(IMGS)).astype(float) for _ in range(B)]
    flag = flagged(cached_predict(NB22 / "init/student_n_v5.pt", IMGS, "nb22_pi_n_v5_640", imgsz=640, device=0), TAU)

    T = {"v5_orig": NB22 / "init/teacher_v5.pt", "ft_v6": NB22 / "runs/teacher_v6/weights/best.pt"}
    for k in ["g_v6", "gf_v6"]:
        if (V7 / f"runs/{k}/weights/best.pt").is_file(): T[k] = V7 / f"runs/{k}/weights/best.pt"
    P = {}
    for k, w in T.items():
        for sz in (640, 800):
            P[(k, sz, False)] = predict(w, IMGS, f"{k}_{sz}", sz)
    P[("v5_orig", 1024, False)] = predict(T["v5_orig"], IMGS, "v5_orig_1024", 1024)
    for sz in (640, 800):
        try:
            P[("v5_orig", sz, True)] = predict(T["v5_orig"], IMGS, f"v5_orig_{sz}_tta", sz, augment=True)
        except Exception as e:
            log(f"TTA {sz} không chạy được: {type(e).__name__}: {e}")
    ens = [k for k in T]
    for sz in (640, 800):
        P[("wbf_all", sz, False)] = wbf([P[(k, sz, False)] for k in ens])
        P[("wbf_orig_ftv6", sz, False)] = wbf([P[("v5_orig", sz, False)], P[("ft_v6", sz, False)]])

    # server configs: {class: prediction key}; baseline = current server (FO@800, rest @640)
    def cfg(m, fo_sz=800, rest_sz=640, aug=False, fo_aug=None):
        fa = aug if fo_aug is None else fo_aug
        return {c: (m, fo_sz, fa) if c == 4 else (m, rest_sz, aug) for c in range(1, 6)}
    CFG = {"baseline v5_orig fo800": cfg("v5_orig"),
           "v5_orig fo1024": cfg("v5_orig", fo_sz=1024),
           "v5_orig all640 (FO@640)": cfg("v5_orig", fo_sz=640)}
    if ("v5_orig", 640, True) in P: CFG["v5_orig TTA640 + FO@800"] = cfg("v5_orig", aug=True, fo_aug=False)
    if ("v5_orig", 800, True) in P: CFG["v5_orig 640 + FO@800 TTA"] = cfg("v5_orig", aug=False, fo_aug=True)
    if ("v5_orig", 640, True) in P and ("v5_orig", 800, True) in P: CFG["v5_orig TTA all"] = cfg("v5_orig", aug=True)
    for k in T:
        if k != "v5_orig": CFG[f"{k} fo800"] = cfg(k)
    CFG["WBF all teachers fo800"] = cfg("wbf_all"); CFG["WBF orig+ft_v6 fo800"] = cfg("wbf_orig_ftv6")

    rows, S = [], {}
    for name, c2k in CFG.items():
        for s, G in GTS.items():
            pre = [prep(P[c2k[c]], G[c], c, flag, IDX) for c in range(1, 6)]
            ap = [wap(*x, np.ones(len(IMGS))) for x in pre]
            rows.append({"split": s, "server": name, "MEAN_5": np.mean(ap), **dict(zip(DEFECTS, ap))})
            S[(s, name)] = np.array([np.mean([wap(*x, w) for x in pre]) for w in W])
        log(f"{name:28s} MEAN_5 val_clean_v6 {rows[-2]['MEAN_5']:.4f}  val_clean {rows[-1]['MEAN_5']:.4f}")
    res = pd.DataFrame(rows); cmp = []
    for s in GTS:
        for name in CFG:
            if name.startswith("baseline"): continue
            d = S[(s, name)] - S[(s, "baseline v5_orig fo800")]; lo, hi = np.percentile(d, [2.5, 97.5])
            cmp.append({"split": s, "server": name, "diff_vs_baseline": d.mean(), "ci95_lo": lo, "ci95_hi": hi, "P(>0)": (d > 0).mean()})
    cmp = pd.DataFrame(cmp)
    res.to_csv(OUT / "eval_infer.csv", index=False); cmp.to_csv(OUT / "eval_infer_compare.csv", index=False)
    pd.set_option("display.width", 220)
    log("\n" + res[res.split == "val_clean_v6"].round(4).to_string(index=False) + "\n\n"
        + cmp[cmp.split == "val_clean_v6"].round(4).to_string(index=False))
    (OUT / "DONE").touch(); log("===== XONG v6p5-24")
