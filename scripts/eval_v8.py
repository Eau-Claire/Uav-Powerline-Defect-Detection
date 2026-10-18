#!/usr/bin/env python
"""Per-class evaluation on val_clean_v8 from cached predictions (CPU, no model run): data/label_audit/preflight/sys_<tag>.json
written by system_two_tier.run() (square letterbox, conf 0.001; *_640 and *_800 for the server, FO taken at 800).

Settings: Pi-only, server-only, two-tier (Pi flags an image when a defect box has conf >= TAU, server boxes on flagged
images). Subsets: `noleak` = val_clean_v8 minus leaked_val.txt (fair for models trained on v5/v6; the main number),
`all`. AP50 per class with image-bootstrap CI95, MEAN_5 with P(> target), paired differences between models.
Extras: broken_strand split by image domain (square 640x640 = phone / ground photos, other = drone frames);
insulator_broken with the dubious small boxes (review_v7 ids 57-138) ignored (their GT and the predictions on them).
Usage: python scripts/eval_v8.py [--b 1000]  ->  artifacts/eval_v8/{per_class.csv, compare.csv, report.md}
"""
from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "data/label_audit/preflight"))
from system_two_tier import flagged, iou  # noqa: E402
from system_two_tier_bootstrap import prep, wap  # noqa: E402

DS = REPO / "datasets/v6p5_v8"; PF = REPO / "data/label_audit/preflight"; OUT = REPO / "artifacts/eval_v8"
DEF = ["insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
SHORT = ["broken", "flashover", "bird_nest", "FO", "strand"]
TAU, TARGET, SEED = 0.05, 0.82, 20261015
PI = {"n_v5": "nb22_pi_n_v5_640", "gn_v6": "nb23_gn_v6_640"}                    # Pi students (current, gentle v6)
SERVER = {"v5_orig": "nb22_v5_orig", "soup_g_v6": "nb23_soup_g_v6"}            # server teachers (FO @800)


def dets(tag):
    return json.loads((PF / f"sys_{tag}.json").read_text())["dets"]


def gt(imgs, split="val_clean_v8"):
    G = {c: {} for c in range(1, 6)}
    for im in imgs:
        x = np.array([l.split() for l in (DS / "labels" / split / (im + ".txt").replace(".jpg.txt", ".txt")).read_text().splitlines()
                      if l.strip()], float).reshape(-1, 5)
        for c in range(1, 6):
            b = x[x[:, 0] == c][:, 1:]; G[c][im] = np.c_[b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2]
    return G


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--b", type=int, default=1000)
    p.add_argument("--cluster", action="store_true", help="bootstrap over near-duplicate clusters (val_clean_v8_clusters.json)")
    a = p.parse_args(); OUT.mkdir(parents=True, exist_ok=True)
    CL = json.loads((DS / "val_clean_v8_clusters.json").read_text()) if a.cluster else None
    out_tag = "_cluster" if a.cluster else ""
    ALL = sorted(f.name for f in (DS / "images/val_clean_v8").glob("*.jpg"))
    leak = set((DS / "leaked_val.txt").read_text().split())
    size = {n: Image.open(DS / "images/val_clean_v8" / n).size for n in ALL}
    G_all = gt(ALL)
    # dubious insulator_broken GT boxes (review ids 57-138, not the 2 already dropped in v8) -> "ignore" variant
    q = pd.read_csv(REPO / "artifacts/review_v7/review_queue.csv")
    q = q[(q.split == "val_clean_v6") & (q.cls == "insulator_broken") & (q.flag == "missed_gt") & ~q.id.isin([64, 77])]
    dubious = {}
    for r in q.itertuples(): dubious.setdefault(r.image, []).append(ast.literal_eval(r.box))
    P = {k: dets(t) for k, t in PI.items()}
    S = {k: {sz: dets(f"{t}_{sz}") for sz in (640, 800)} for k, t in SERVER.items()}

    def system(server, pi):   # final detections per image for a setting
        if server is None: return P[pi]
        s = S[server]; f = flagged(P[pi], TAU) if pi else None
        return {n: ([d for d in s[640][n] if d[0] != 4] + [d for d in s[800][n] if d[0] == 4]) if (f is None or n in f) else []
                for n in ALL}

    SETTINGS = {f"Pi only {k}": (None, k) for k in PI} | {f"server only {k}": (k, None) for k in SERVER} | \
               {f"2 tầng {pi}->{k}": (k, pi) for k in SERVER for pi in PI}
    D = {name: system(*cfg) for name, cfg in SETTINGS.items()}
    rng = np.random.default_rng(SEED); rows, samples = [], {}

    def score(name, subset, sub_name, G=G_all, drop_dubious=False):
        idx = {n: i for i, n in enumerate(subset)}; keep = set(subset); dd = D[name]
        if drop_dubious:   # ignore dubious broken GT: remove them and the predictions that hit them (IoU >= 0.5)
            dd = {n: [d for d in v if not (d[0] == 1 and n in dubious and iou(np.array([d[2:]]), np.array(dubious[n]))[0].max() >= 0.5)]
                  for n, v in dd.items()}
            G = {c: ({n: (np.array([b for b in g if not (n in dubious and iou(np.array([b]), np.array(dubious[n]))[0].max() >= 0.9)])
                          .reshape(-1, 4)) for n, g in G[c].items()} if c == 1 else G[c]) for c in G}
        if CL:   # cluster bootstrap: near-duplicate val images (dHash <= 6) are drawn together
            cid = np.unique([CL[n] for n in subset], return_inverse=True)[1]; k = cid.max() + 1
            W = [np.bincount(rng.integers(0, k, k), minlength=k)[cid].astype(float) for _ in range(a.b)]
        else:
            W = [np.bincount(rng.integers(0, len(subset), len(subset)), minlength=len(subset)).astype(float) for _ in range(a.b)]
        pre = [prep(dd, {n: G[c][n] for n in subset}, c, keep, idx) for c in range(1, 6)]
        full = [wap(*x, np.ones(len(subset))) for x in pre]
        boot = np.array([[wap(*x, w) for x in pre] for w in W])          # B x 5
        samples[(name, sub_name)] = boot
        r = {"setting": name, "subset": sub_name, "n_img": len(subset), "MEAN_5": np.mean(full),
             "MEAN_5_lo": np.percentile(boot.mean(1), 2.5), "MEAN_5_hi": np.percentile(boot.mean(1), 97.5),
             "P(>0.82)": (boot.mean(1) > TARGET).mean()}
        for c, s in enumerate(SHORT):
            r[s] = full[c]; r[f"{s}_lo"], r[f"{s}_hi"] = np.percentile(boot[:, c], [2.5, 97.5])
        rows.append(r); return r

    noleak = [n for n in ALL if n not in leak]
    for name in SETTINGS:
        score(name, noleak, "noleak"); score(name, ALL, "all")
    main_sys = "2 tầng n_v5->v5_orig"
    score(main_sys, noleak, "noleak, broken bỏ 80 box đáng ngờ", drop_dubious=True)
    sq = [n for n in noleak if size[n][0] == size[n][1]]; ns = [n for n in noleak if size[n][0] != size[n][1]]
    score(main_sys, sq, "noleak, ảnh vuông (điện thoại/mặt đất)"); score(main_sys, ns, "noleak, ảnh drone (không vuông)")
    res = pd.DataFrame(rows); res.to_csv(OUT / f"per_class{out_tag}.csv", index=False)

    cmp = []
    for a_, b_, sub in [("2 tầng n_v5->soup_g_v6", main_sys, "noleak"), ("Pi only gn_v6", "Pi only n_v5", "noleak"),
                        ("2 tầng gn_v6->v5_orig", main_sys, "noleak"), ("server only soup_g_v6", "server only v5_orig", "noleak")]:
        d = samples[(a_, sub)] - samples[(b_, sub)]
        r = {"compare": f"{a_} − {b_}", "subset": sub}
        for c, s in enumerate(["MEAN_5"] + SHORT):
            v = d.mean(1) if c == 0 else d[:, c - 1]
            r[s] = v.mean(); r[f"{s}_P>0"] = (v > 0).mean()
        cmp.append(r)
    cmp = pd.DataFrame(cmp); cmp.to_csv(OUT / f"compare{out_tag}.csv", index=False)

    n_cls = {s: sum(len(G_all[c + 1][n]) for n in noleak) for c, s in enumerate(SHORT)}
    n_dom = {k: sum(len(G_all[5][n]) for n in v) for k, v in [("vuông", sq), ("drone", ns)]}
    fmt = lambda r, s: f"{r[s]:.3f} [{r[s + '_lo']:.2f}–{r[s + '_hi']:.2f}]"
    L = [f"# Eval v8 ({len(noleak)} ảnh val_clean_v8 không rò / {len(ALL)} toàn bộ), AP50, CI95 bootstrap {a.b} vòng{' theo cụm ảnh gần trùng' if CL else ''}",
         "", f"Số box GT (không rò): " + ", ".join(f"{k} {v}" for k, v in n_cls.items()) + f" | strand theo miền: {n_dom}", "",
         "| setting | tập | MEAN_5 | P(>0.82) | " + " | ".join(SHORT) + " |", "|" + "---|" * (4 + len(SHORT))]
    for r in res.to_dict("records"):
        L.append(f"| {r['setting']} | {r['subset']} | {fmt(r, 'MEAN_5')} | {r['P(>0.82)']:.0%} | " + " | ".join(fmt(r, s) for s in SHORT) + " |")
    L += ["", "| so sánh (ghép cặp) | MEAN_5 | " + " | ".join(SHORT) + " |", "|" + "---|" * (2 + len(SHORT))]
    for r in cmp.to_dict("records"):
        L.append(f"| {r['compare']} | " + " | ".join(f"{r[s]:+.3f} (P>0 {r[s + '_P>0']:.0%})" for s in ["MEAN_5"] + SHORT) + " |")
    (OUT / f"report{out_tag}.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
