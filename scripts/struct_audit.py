"""RQ: Can structural consistency provide additional evidence for identifying annotation errors and unreliable
detections beyond conventional confidence- and loss-based methods?

Study set: val_clean_v6 minus the 148 images that near-duplicate train (artifacts/review_v7/val_train_near_dups.json),
so the detectors never saw it. Defect classes 1-5. Predictions: cached @640 (data/label_audit/preflight/sys_*.json).

Structural features (no ground truth needed at inference; parents come from PREDICTED insulators):
  host_ioa   share of the box inside the best insulator (classes 1/2: a defect sits on an insulator)
  rel_scale  log(area / host area)                (a chip is small relative to its insulator)
  wire       straight edge segment through the box, aligned (<15 deg) with its long side (classes 4/5: on a wire)
  conflict   max IoU with a box of a conflicting class (FO~person, FO~bird_nest, broken~flashover)
  closeup    box touches the border or covers > 25% of the image (parent may be out of frame -> host evidence weak)

Part A  unreliable detections (automatic, labels = GT match IoU>=0.5): teacher predictions conf >= 0.05, 5-fold
        image-grouped CV logistic regression  {conf}  vs  {conf + structure}; AUROC/AP for "is false positive",
        and per-class AP50 when the detections are re-scored with the CV probability.
Part B  annotation errors (needs human verification): rank GT boxes by
        conf      1 - conf of the best same-class teacher box (IoU >= 0.5)
        loss      1 - conf * IoU of the best same-class teacher box (box-quality-weighted loss proxy)
        disagree  std of matched conf across 6 models (4 teachers m, 2 students n)
        struct    -log p(features | class) under per-class histograms of TRAIN labels (structure "prior")
        -> blind verification sample (top-k per method + random), method hidden; precision@k + overlap once labelled.

  .venv/bin/python scripts/struct_audit.py      -> artifacts/struct_audit/
"""
import json, math
from pathlib import Path

import cv2, numpy as np, pandas as pd
from PIL import Image, ImageDraw
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

REPO = Path(__file__).resolve().parents[1]
DS = REPO / "datasets/v6p5_v6"; PF = REPO / "data/label_audit/preflight"; OUT = REPO / "artifacts/struct_audit"
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand", "person"]
CONFLICT = {4: [6, 3], 3: [4], 1: [2], 2: [1], 5: [4]}
MODELS = {"v5_orig": "nb22_v5_orig", "ft_v6": "nb22_ft_v6", "g_v6": "nb23_g_v6", "gf_v6": "nb23_gf_v6",
          "n_v5": "nb23_n_v5", "gn_v6": "nb23_gn_v6"}
LEAK = {v for v, _ in json.loads((REPO / "artifacts/review_v7/val_train_near_dups.json").read_text())}


def iou(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    i = wh[..., 0] * wh[..., 1]; ar = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return i / (ar(a)[:, None] + ar(b)[None] - i + 1e-9)


def ioa(a, b):
    if not len(a) or not len(b): return np.zeros((len(a), len(b)))
    wh = np.clip(np.minimum(a[:, None, 2:], b[None, :, 2:]) - np.maximum(a[:, None, :2], b[None, :, :2]), 0, None)
    return wh[..., 0] * wh[..., 1] / ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]) + 1e-9)[:, None]


def gt(split, name):
    g = np.array([l.split() for l in (DS / "labels" / split / (Path(name).stem + ".txt")).read_text().splitlines() if l.strip()],
                 float).reshape(-1, 5)
    return g[:, 0].astype(int), np.c_[g[:, 1:3] - g[:, 3:5] / 2, g[:, 1:3] + g[:, 3:5] / 2]


def wire_evidence(gray, box, W, H):
    """1 if a straight edge segment crosses the box aligned with its long side (wire through the object)."""
    x0, y0, x1, y1 = box[0] * W, box[1] * H, box[2] * W, box[3] * H; bw, bh = x1 - x0, y1 - y0
    cx0, cy0, cx1, cy1 = int(max(0, x0 - bw)), int(max(0, y0 - bh)), int(min(W, x1 + bw)), int(min(H, y1 + bh))
    c = gray[cy0:cy1, cx0:cx1]
    if c.size < 100: return 0.0
    L = cv2.HoughLinesP(cv2.Canny(c, 50, 150), 1, np.pi / 180, 30, minLineLength=max(15, int(0.6 * max(bw, bh))), maxLineGap=6)
    if L is None: return 0.0
    long_ang = 0.0 if bw >= bh else 90.0
    for xa, ya, xb, yb in L.reshape(-1, 4):
        ang = abs(math.degrees(math.atan2(yb - ya, xb - xa))) % 180; d = min(abs(ang - long_ang), 180 - abs(ang - long_ang))
        mx, my = (xa + xb) / 2 + cx0, (ya + yb) / 2 + cy0
        if d < 15 and x0 - 0.2 * bw <= mx <= x1 + 0.2 * bw and y0 - 0.2 * bh <= my <= y1 + 0.2 * bh: return 1.0
    return 0.0


def features(c, b, hosts, others, gray, W, H):
    """b: xyxyn box; hosts: insulator xyxyn (array); others: list of (cls, box) in the image"""
    f = {}
    if len(hosts):
        a = ioa(b[None], hosts)[0]; k = int(a.argmax()); f["host_ioa"] = float(a[k])
        hb = hosts[k]; f["rel_scale"] = float(np.log(((b[2] - b[0]) * (b[3] - b[1]) + 1e-9) / ((hb[2] - hb[0]) * (hb[3] - hb[1]) + 1e-9)))
    else:
        f["host_ioa"], f["rel_scale"] = 0.0, 0.0
    f["no_host"] = float(not len(hosts))
    f["wire"] = wire_evidence(gray, b, W, H) if c in (4, 5) else 0.0
    conf_boxes = np.array([ob for oc, ob in others if oc in CONFLICT.get(c, [])]).reshape(-1, 4)
    f["conflict"] = float(iou(b[None], conf_boxes).max()) if len(conf_boxes) else 0.0
    f["closeup"] = float(b[0] < 0.01 or b[1] < 0.01 or b[2] > 0.99 or b[3] > 0.99 or (b[2] - b[0]) * (b[3] - b[1]) > 0.25)
    f["log_area"] = float(np.log((b[2] - b[0]) * (b[3] - b[1]) + 1e-9))
    return f


FEATS = ["host_ioa", "rel_scale", "no_host", "wire", "conflict", "closeup", "log_area"]


def ap50(rows, npos):
    rows = sorted(rows, key=lambda r: -r[0]); tp = np.array([r[1] for r in rows], float)
    if not len(tp) or not npos: return 0.0
    c = np.cumsum(tp); rec, prec = c / npos, c / np.arange(1, len(tp) + 1)
    mrec, mpre = np.r_[0, rec, 1], np.flip(np.maximum.accumulate(np.flip(np.r_[1, prec, 0])))
    x = np.linspace(0, 1, 101); return float(np.trapezoid(np.interp(x, mrec, mpre), x))


if __name__ == "__main__":
    P = {k: json.loads((PF / f"sys_{t}_640.json").read_text())["dets"] for k, t in MODELS.items()}
    coco = json.loads((DS / "annotations/instances_val_clean_v6.json").read_text())
    size = {Path(i["file_name"]).name: (i["width"], i["height"]) for i in coco["images"]}
    imgs = sorted(n for n in size if n not in LEAK)
    print(f"study set: {len(imgs)} images (removed {len(LEAK)} leaked)", flush=True)

    det_rows, gt_rows, npos = [], [], {c: 0 for c in range(1, 6)}
    for n in imgs:
        W, H = size[n]; gc, gb = gt("val_clean_v6", n)
        gray = cv2.imread(str(DS / "images/val_clean_v6" / n), cv2.IMREAD_GRAYSCALE)
        d = np.array(P["v5_orig"][n], float).reshape(-1, 6); dc, ds, db = d[:, 0].astype(int), d[:, 1], d[:, 2:]
        pred_hosts = db[(dc == 0) & (ds >= 0.3)]
        pred_others = [(int(c_), b_) for c_, s_, b_ in zip(dc, ds, db) if s_ >= 0.3]
        for c in range(1, 6): npos[c] += int((gc == c).sum())
        # ---- Part A: every teacher detection of a defect class, TP/FP vs GT
        used = {c: np.zeros(int((gc == c).sum()), bool) for c in range(1, 6)}
        order = np.argsort(-ds)
        for k in order:
            c = dc[k]
            if not (1 <= c <= 5) or ds[k] < 0.05: continue
            g = gb[gc == c]; tp = 0
            if len(g):
                u = iou(db[k:k + 1], g)[0]; u[used[c]] = -1; j = int(u.argmax())
                if u[j] >= 0.5: used[c][j] = True; tp = 1
            others = [(oc, ob) for oc, ob in pred_others if not np.allclose(ob, db[k])]
            det_rows.append({"image": n, "cls": c, "conf": ds[k], "tp": tp, **features(c, db[k], pred_hosts, others, gray, W, H)})
        # ---- Part B: every GT defect box, four suspicion scores
        hosts_b = np.r_[gb[gc == 0], pred_hosts].reshape(-1, 4)
        for j in np.where((gc >= 1) & (gc <= 5))[0]:
            c = gc[j]; mc = []
            for m, PP in P.items():
                e = np.array(PP[n], float).reshape(-1, 6); e = e[e[:, 0] == c]
                if len(e):
                    u = iou(gb[j:j + 1], e[:, 2:])[0]; ok = u >= 0.5
                    mc.append((float(e[ok, 1].max()) if ok.any() else 0.0, float(u.max())))
                else: mc.append((0.0, 0.0))
            conf0, iou0 = mc[0]
            e0 = np.array(P["v5_orig"][n], float).reshape(-1, 6); e0 = e0[e0[:, 0] == c]
            qual = float((e0[:, 1] * iou(gb[j:j + 1], e0[:, 2:])[0]).max()) if len(e0) else 0.0
            others = [(int(oc), ob) for oc, ob in zip(gc, gb) if not np.allclose(ob, gb[j])]
            gt_rows.append({"image": n, "cls": c, "box": gb[j].round(4).tolist(), "s_conf": 1 - conf0, "s_loss": 1 - qual,
                            "s_disagree": float(np.std([x for x, _ in mc])), **features(c, gb[j], hosts_b, others, gray, W, H)})
    D, G = pd.DataFrame(det_rows), pd.DataFrame(gt_rows)

    # ---------- Part A
    res = []
    for c in list(range(1, 6)) + ["all"]:
        d = D if c == "all" else D[D.cls == c]
        if d.tp.nunique() < 2: continue
        Xc = np.c_[np.log(d.conf / (1 - d.conf + 1e-6))]
        Xs = np.c_[Xc, d[FEATS].values, *(np.c_[(d.cls == k).astype(float)] for k in range(1, 6))] if c == "all" else np.c_[Xc, d[FEATS].values]
        if c == "all": Xc = np.c_[Xc, *(np.c_[(d.cls == k).astype(float)] for k in range(1, 6))]
        pc, ps = np.zeros(len(d)), np.zeros(len(d))
        for tr, te in GroupKFold(5).split(Xc, d.tp, d.image):
            pc[te] = LogisticRegression(max_iter=2000).fit(Xc[tr], d.tp.values[tr]).predict_proba(Xc[te])[:, 1]
            ps[te] = LogisticRegression(max_iter=2000).fit(Xs[tr], d.tp.values[tr]).predict_proba(Xs[te])[:, 1]
        fp = 1 - d.tp.values
        row = {"cls": c if c == "all" else NAMES[c], "n_det": len(d), "fp_rate": fp.mean(),
               "AUROC_fp conf": roc_auc_score(fp, -pc), "AUROC_fp conf+struct": roc_auc_score(fp, -ps),
               "AP_fp conf": average_precision_score(fp, 1 - pc), "AP_fp conf+struct": average_precision_score(fp, 1 - ps)}
        if c != "all":
            row["AP50 raw conf"] = ap50(list(zip(d.conf, d.tp)), npos[c])
            row["AP50 rescored conf"] = ap50(list(zip(pc, d.tp)), npos[c])
            row["AP50 rescored conf+struct"] = ap50(list(zip(ps, d.tp)), npos[c])
        res.append(row)
    A = pd.DataFrame(res); A.to_csv(OUT / "partA_unreliable_detections.csv", index=False)
    pd.set_option("display.width", 250); print("\n== Part A: detections\n" + A.round(4).to_string(index=False), flush=True)

    # ---------- Part B: structural prior from TRAIN labels (histograms per class), NLL as suspicion
    tr_rows = []
    trP = json.loads((REPO / "artifacts/audit_v6_boxes/pred_train.json").read_text())
    tcoco = json.loads((DS / "annotations/instances_train.json").read_text())
    tsize = {Path(i["file_name"]).name: (i["width"], i["height"]) for i in tcoco["images"]}
    rng = np.random.default_rng(0)
    for n in rng.choice(sorted(tsize), 6000, replace=False):
        W, H = tsize[n]; gc, gb = gt("train", n)
        if not ((gc >= 1) & (gc <= 5)).any(): continue
        gray = cv2.imread(str(DS / "images/train" / n), cv2.IMREAD_GRAYSCALE)
        d = np.array(trP[n], float).reshape(-1, 6); hosts_b = np.r_[gb[gc == 0], d[(d[:, 0] == 0) & (d[:, 1] >= 0.3)][:, 2:]].reshape(-1, 4)
        for j in np.where((gc >= 1) & (gc <= 5))[0]:
            others = [(int(oc), ob) for oc, ob in zip(gc, gb) if not np.allclose(ob, gb[j])]
            tr_rows.append({"cls": gc[j], **features(gc[j], gb[j], hosts_b, others, gray, W, H)})
    T = pd.DataFrame(tr_rows)
    BINS = {"host_ioa": [-.01, .2, .6, .9, 1.01], "rel_scale": [-99, -4, -2.5, -1, 99], "no_host": [-.5, .5, 1.5],
            "wire": [-.5, .5, 1.5], "conflict": [-.01, .3, .6, 1.01], "closeup": [-.5, .5, 1.5], "log_area": [-99, -7, -5, -3, 99]}
    def nll(row):
        t = T[T.cls == row.cls]; s = 0.0
        for f, bins in BINS.items():
            h, _ = np.histogram(t[f], bins=bins); p = (h + 1) / (h.sum() + len(h))
            s -= math.log(p[min(np.searchsorted(bins, row[f], side="right") - 1, len(h) - 1)])
        return s
    G["s_struct"] = G.apply(nll, axis=1)
    G.to_csv(OUT / "partB_gt_scores.csv", index=False)
    S = ["s_conf", "s_loss", "s_disagree", "s_struct"]
    print("\n== Part B: Spearman correlation between suspicion scores (low = complementary)\n" + G[S].corr("spearman").round(2).to_string())
    K = 50; top = {s: set(G.sort_values(s, ascending=False).head(K).index) for s in S}
    conv = top["s_conf"] | top["s_loss"] | top["s_disagree"]
    print(f"top-{K} overlap: struct ∩ (conf ∪ loss ∪ disagree) = {len(top['s_struct'] & conv)} / {K}")

    # blind verification sample: top-K of each method + 40 random, deduplicated and shuffled; method columns hidden
    pick = sorted(set().union(*top.values()) | set(G.sample(40, random_state=1).index))
    V = G.loc[pick].sample(frac=1, random_state=2).reset_index().rename(columns={"index": "gt_row"})
    V.insert(0, "vid", range(1, len(V) + 1))
    V[["vid", "gt_row"] + S].to_csv(OUT / "verify_key.csv", index=False)           # keep hidden while labelling
    V["verdict"] = ""   # correct | wrong_class:<cls> | bad_box | not_defect | duplicate | unsure
    V[["vid", "image", "cls", "box", "verdict"]].assign(cls=V.cls.map(lambda c: NAMES[c])).to_csv(OUT / "verify_blind.csv", index=False)
    COL = [(0, 200, 255), (255, 128, 0), (255, 0, 255), (0, 200, 0), (255, 255, 0), (255, 0, 0), (128, 0, 255)]
    for p in range(0, len(V), 24):
        sh = Image.new("RGB", (6 * 330, 330 * 4), "white")
        for i, r in enumerate(V.iloc[p:p + 24].itertuples()):
            im = Image.open(DS / "images/val_clean_v6" / r.image).convert("RGB"); W, H = im.size; dr = ImageDraw.Draw(im)
            gc, gb = gt("val_clean_v6", r.image)
            for c, b in zip(gc, gb): dr.rectangle([b[0] * W, b[1] * H, b[2] * W, b[3] * H], outline=COL[c], width=max(1, W // 400))
            b = r.box; cx, cy = (b[0] + b[2]) / 2 * W, (b[1] + b[3]) / 2 * H; s = max((b[2] - b[0]) * W, (b[3] - b[1]) * H) * 2.5 + 80
            dr.rectangle([b[0] * W - 3, b[1] * H - 3, b[2] * W + 3, b[3] * H + 3], outline=(255, 255, 255), width=max(3, W // 250))
            t = im.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((330, 330))
            ImageDraw.Draw(t).rectangle([0, 0, 330, 14], fill=(0, 0, 0)); ImageDraw.Draw(t).text((3, 1), f"#{r.vid} {NAMES[r.cls]}", fill=(255, 255, 255))
            sh.paste(t, ((i % 6) * 330, (i // 6) * 330))
        sh.save(OUT / f"verify_{p // 24 + 1:02d}.jpg", quality=85)
    print(f"verification sample: {len(V)} boxes -> {OUT}/verify_*.jpg + verify_blind.csv (method key hidden in verify_key.csv)")
