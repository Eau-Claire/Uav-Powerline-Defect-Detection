"""v6p5-23: why fine-tuning the converged v5 teacher loses ~0.01 MEAN_5, and recipes that don't.

nb22 fine-tuned teacher v5 (ended at lr 5e-5, no mosaic) with lr 1e-4 + mosaic + fresh AdamW for 20 ep -> ft_v5 and ft_v6
both below v5_orig. Here, same start weights, same seed, only the recipe changes:
  g_v6 / g_v5   lr0 1e-5, mosaic off, 8 ep, val each epoch (noisy `val` split = selection set)  -> v6 vs v5 control
  gf_v6         same on v6 with the backbone frozen (freeze=10), lr0 3e-5
  soup_*        WiSE-FT: 0.5 * v5_orig + 0.5 * fine-tuned (no training)
Student n (Pi): gentle distill from the best teacher on v6, plus soups with student v5.
Scored exactly like nb22 cell 6/7 (val_clean_v6 + val_clean, two-tier @TAU, FO@800, paired bootstrap).

Resumable: finished runs/evals are skipped.  Run:  nohup .venv/bin/python scripts/exp_v6p5_23_gentle_ft.py &
Results: artifacts/v6p5_v7_gentle/{progress.log, eval_*.csv, summary.md}
"""
import copy, gc, json, logging, sys, time
from pathlib import Path

import numpy as np, pandas as pd, torch
from ultralytics import YOLO
from ultralytics.utils import LOGGER

REPO = Path(__file__).resolve().parents[1]
DSD = REPO / "datasets"
OUT = REPO / "artifacts/v6p5_v7_gentle"; RUNS = OUT / "runs"; SOUPS = OUT / "soups"
NB22 = REPO / "artifacts/v6p5_v6_teachers"
TEACHER_V5, STUDENT_V5 = NB22 / "init/teacher_v5.pt", NB22 / "init/student_n_v5.pt"
DATA = {"v5": DSD / "v6p5_v5/data.yaml", "v6": DSD / "v6p5_v6/data.yaml"}
DS6 = DSD / "v6p5_v6"
DEFECTS = ["insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
TAU, B, SEED = 0.05, 2000, 20261015
COMMON = dict(device=0, workers=8, seed=SEED, deterministic=True, optimizer="AdamW", cos_lr=True, lrf=0.1,
              weight_decay=1e-3, cache="disk", amp=True, pretrained=True, plots=False, imgsz=640,
              warmup_epochs=0, mosaic=0.0, close_mosaic=0, val=True)
OUT.mkdir(parents=True, exist_ok=True); RUNS.mkdir(exist_ok=True); SOUPS.mkdir(exist_ok=True)


class _Tee:
    def __init__(self, f, o): self.f, self.o = f, o
    def write(self, x): self.o.write(x); self.f.write(x); self.f.flush(); return len(x)
    def flush(self): self.o.flush(); self.f.flush()
    def __getattr__(self, k): return getattr(self.o, k)
sys.stdout = _Tee(open(OUT / "progress.log", "a", encoding="utf-8"), sys.stdout)
log = lambda *a: print(f"[{time.strftime('%H:%M')}]", *a, flush=True)


def _epoch_line(tr):
    m = tr.metrics
    log(f"{tr.args.name} ep {tr.epoch + 1}/{tr.epochs}  val mAP50 {m.get('metrics/mAP50(B)', 0):.4f}  "
        f"mAP50-95 {m.get('metrics/mAP50-95(B)', 0):.4f}  lr {tr.optimizer.param_groups[0]['lr']:.2e}")


def train(name, model, **kw):
    run, best = RUNS / name, RUNS / name / "weights/best.pt"
    if (run / "DONE").exists(): log(f"[{name}] đã xong, bỏ qua"); return best
    last = run / "weights/last.pt"
    if last.is_file():
        log(f"[{name}] resume"); y, args = YOLO(str(last)), dict(resume=True)
    else:
        log(f"[{name}] bắt đầu từ {model.name} | {kw}")
        y, args = YOLO(str(model)), dict(project=str(RUNS), name=name, exist_ok=True, **{**COMMON, **kw})
    y.add_callback("on_fit_epoch_end", _epoch_line)
    lvl = LOGGER.level; LOGGER.setLevel(logging.WARNING)
    try:
        y.train(**args)
    finally:
        LOGGER.setLevel(lvl); del y; gc.collect(); torch.cuda.empty_cache()
    (run / "DONE").touch(); log(f"[{name}] xong"); return best


def soup(a, b, alpha, name):
    """WiSE-FT: (1-alpha)*a + alpha*b, same architecture; saved as a plain YOLO checkpoint."""
    dst = SOUPS / f"{name}.pt"
    if dst.is_file(): return dst
    ca, cb = (torch.load(p, map_location="cpu", weights_only=False) for p in (a, b))
    ma, mb = ((c.get("ema") or c["model"]).float() for c in (ca, cb))
    sb = mb.state_dict()
    sd = {k: v * (1 - alpha) + sb[k] * alpha if v.dtype.is_floating_point else v for k, v in ma.state_dict().items()}
    m = copy.deepcopy(ma); m.load_state_dict(sd)
    torch.save({"model": m.half(), "ema": None, "epoch": -1, "train_args": ca.get("train_args", {})}, dst)
    log(f"soup {name}: {1 - alpha:.1f}·{a} + {alpha:.1f}·{b}")
    return dst


# ---------------- eval (= nb22 cell 6/7) ----------------
sys.path.insert(0, str(REPO / "data/label_audit/preflight"))
from system_two_tier import run as predict, flagged          # noqa: E402
from system_two_tier_bootstrap import prep, wap              # noqa: E402

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
W = {s: [np.bincount(np.random.default_rng(SEED + i).integers(0, len(IMGS), len(IMGS)), minlength=len(IMGS)).astype(float)
         for i in range(B)] for s in GTS}
TAG = lambda name, sz: f"nb22_{name.replace('nb22:', '')}_{sz}" if name.startswith("nb22:") or name == "v5_orig" else f"nb23_{name}_{sz}"


def eval_teachers(teachers, pi):
    flag = flagged(pi, TAU); rows, S = [], {}
    for k, w in teachers.items():
        p = {sz: predict(w, IMGS, TAG(k, sz), imgsz=sz, device=0) for sz in (640, 800)}
        for s, G in GTS.items():
            for setting, keep in [("teacher only", set(IDX)), (f"2 tầng @{TAU}", flag)]:
                pre = [prep(p[800 if c == 4 else 640], G[c], c, keep, IDX) for c in range(1, 6)]
                ap = [wap(*x, np.ones(len(IMGS))) for x in pre]
                rows.append({"split": s, "model": k.replace("nb22:", ""), "setting": setting, "MEAN_5": np.mean(ap),
                             **dict(zip(DEFECTS, ap))})
                if setting != "teacher only":
                    S[(s, k.replace("nb22:", ""))] = np.array([np.mean([wap(*x, w_) for x in pre]) for w_ in W[s]])
        log(f"eval {k}: " + " ".join(f"{r['split']}={r['MEAN_5']:.4f}" for r in rows[-4:] if r["setting"] != "teacher only"))
    return pd.DataFrame(rows), S


def eval_students(students):
    rows, S = [], {}
    for k, w in students.items():
        p = predict(w, IMGS, TAG(k, 640), imgsz=640, device=0)
        for s, G in GTS.items():
            pre = [prep(p, G[c], c, set(IDX), IDX) for c in range(1, 6)]
            ap = [wap(*x, np.ones(len(IMGS))) for x in pre]
            rows.append({"split": s, "model": k.replace("nb22:", ""), "Pi-only MEAN_5": np.mean(ap), **dict(zip(DEFECTS, ap))})
            S[(s, k.replace("nb22:", ""))] = np.array([np.mean([wap(*x, w_) for x in pre]) for w_ in W[s]])
        log(f"eval {k}: Pi-only val_clean_v6={rows[-2]['Pi-only MEAN_5']:.4f}")
    return pd.DataFrame(rows), S


def compare(S, pairs):
    out = []
    for s in GTS:
        for a, b in pairs:
            if (s, a) in S and (s, b) in S:
                d = S[(s, a)] - S[(s, b)]; lo, hi = np.percentile(d, [2.5, 97.5])
                out.append({"split": s, "compare": f"{a} − {b}", "diff": d.mean(), "ci95_lo": lo, "ci95_hi": hi,
                            "P(>0)": (d > 0).mean()})
    return pd.DataFrame(out)


if __name__ == "__main__":
    pd.set_option("display.width", 220)
    log(f"===== v6p5-23 gentle fine-tune | {torch.cuda.get_device_name(0)}")

    # 1. teachers (YOLO26m from v5_orig), batch 16 like nb22
    T = {"v5_orig": TEACHER_V5,
         "nb22:ft_v5": NB22 / "runs/teacher_v5/weights/best.pt", "nb22:ft_v6": NB22 / "runs/teacher_v6/weights/best.pt"}
    T["g_v6"] = train("g_v6", TEACHER_V5, data=str(DATA["v6"]), epochs=8, batch=16, lr0=1e-5)
    T["g_v5"] = train("g_v5", TEACHER_V5, data=str(DATA["v5"]), epochs=8, batch=16, lr0=1e-5)
    T["gf_v6"] = train("gf_v6", TEACHER_V5, data=str(DATA["v6"]), epochs=8, batch=16, lr0=3e-5, freeze=10)
    for k in ["nb22:ft_v6", "g_v6", "gf_v6"]:
        T[f"soup_{k.replace('nb22:', '')}"] = soup(TEACHER_V5, T[k], 0.5, f"soup_{k.replace('nb22:', '')}")
    T["soup_g_v5"] = soup(TEACHER_V5, T["g_v5"], 0.5, "soup_g_v5")

    pi_v5 = predict(STUDENT_V5, IMGS, "nb22_pi_n_v5_640", imgsz=640, device=0)
    res, S = eval_teachers(T, pi_v5)
    cmp = compare(S, [(k.replace("nb22:", ""), "v5_orig") for k in T if k != "v5_orig"] + [("g_v6", "g_v5"), ("soup_g_v6", "soup_g_v5")])
    res.to_csv(OUT / "eval_teachers.csv", index=False); cmp.to_csv(OUT / "eval_teachers_compare.csv", index=False)
    two = res[(res.split == "val_clean_v6") & (res.setting != "teacher only")].set_index("model").MEAN_5
    log("\n" + two.sort_values(ascending=False).round(4).to_string() + "\n\n" + cmp.round(4).to_string(index=False))
    best_t = two.idxmax(); log(f">>> teacher tốt nhất (2 tầng, val_clean_v6): {best_t} {two.max():.4f}")
    (OUT / "best_teacher.txt").write_text(f"{best_t}\t{T['nb22:' + best_t] if 'nb22:' + best_t in T else T[best_t]}\n")

    # 2. student n (Pi): gentle distill on v6 from the best teacher, + soups
    distill_from = T.get(best_t) or T.get("nb22:" + best_t)
    ST = {"n_v5": STUDENT_V5, "nb22:n_v6": NB22 / "runs/student_n_v6/weights/best.pt"}
    try:
        ST["gn_v6"] = train("gn_v6", STUDENT_V5, data=str(DATA["v6"]), epochs=10, batch=32, lr0=1e-5,
                            distill_model=str(distill_from))
    except Exception as e:
        log(f"[gn_v6] lỗi với val=True ({type(e).__name__}: {e}) → thử lại val=False")
        import shutil; shutil.rmtree(RUNS / "gn_v6", ignore_errors=True)
        ST["gn_v6"] = train("gn_v6", STUDENT_V5, data=str(DATA["v6"]), epochs=10, batch=32, lr0=1e-5, val=False,
                            distill_model=str(distill_from))
    ST["soup_n_v6"] = soup(STUDENT_V5, ST["nb22:n_v6"], 0.5, "soup_n_v6")
    ST["soup_gn_v6"] = soup(STUDENT_V5, ST["gn_v6"], 0.5, "soup_gn_v6")
    sres, SS = eval_students(ST)
    scmp = compare(SS, [(k.replace("nb22:", ""), "n_v5") for k in ST if k != "n_v5"])
    sres.to_csv(OUT / "eval_students.csv", index=False); scmp.to_csv(OUT / "eval_students_compare.csv", index=False)
    log("\n" + sres.round(4).to_string(index=False) + "\n\n" + scmp.round(4).to_string(index=False))

    # 3. summary
    md = [f"# v6p5-23 gentle fine-tune — {time.strftime('%Y-%m-%d %H:%M')}", "",
          "## Teachers, MEAN_5 two-tier @0.05 (val_clean_v6)", two.sort_values(ascending=False).round(4).to_markdown(), "",
          "## Teacher paired bootstrap", cmp.round(4).to_markdown(index=False), "",
          f"Best teacher: **{best_t}** (student distilled from it)", "",
          "## Students, Pi-only MEAN_5", sres.round(4).to_markdown(index=False), "",
          "## Student paired bootstrap", scmp.round(4).to_markdown(index=False)]
    (OUT / "summary.md").write_text("\n".join(md)); (OUT / "ALL_DONE").touch()
    log(f"===== XONG — {OUT / 'summary.md'}")
