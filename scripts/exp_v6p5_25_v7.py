"""v6p5-25: does the automatic v7 box clean-up (scripts/build_v7.py) help?  A/B against g_v6 — same start weights
(v5_orig), same gentle recipe (lr 1e-5, no mosaic, 8 ep), same seed; only the train labels differ. Scored on the
UNCHANGED val_clean_v6 / val_clean (human labels), two-tier @0.05, paired bootstrap.
Waits for v6p5-23 and v6p5-24 to free the GPU.  Results: artifacts/v6p5_v7_gentle/eval_v7*.csv (+ progress.log)
"""
import subprocess, time
from pathlib import Path

while any(subprocess.run(["pgrep", "-f", s], capture_output=True).returncode == 0
          for s in ("exp_v6p5_23_gentle_ft.py", "exp_v6p5_24_inference.py")):
    time.sleep(60)

import exp_v6p5_23_gentle_ft as E   # noqa: E402  (shares train/soup/eval + progress.log)

V7 = E.REPO / "datasets/v6p5_v7/data.yaml"
T = {"v5_orig": E.TEACHER_V5, "g_v6": E.RUNS / "g_v6/weights/best.pt", "soup_g_v6": E.SOUPS / "soup_g_v6.pt"}
T["g_v7"] = E.train("g_v7", E.TEACHER_V5, data=str(V7), epochs=8, batch=16, lr0=1e-5)
T["soup_g_v7"] = E.soup(E.TEACHER_V5, T["g_v7"], 0.5, "soup_g_v7")
pi_v5 = E.predict(E.STUDENT_V5, E.IMGS, "nb22_pi_n_v5_640", imgsz=640, device=0)
res, S = E.eval_teachers(T, pi_v5)
cmp = E.compare(S, [("g_v7", "g_v6"), ("soup_g_v7", "soup_g_v6"), ("g_v7", "v5_orig"), ("soup_g_v7", "v5_orig")])
res.to_csv(E.OUT / "eval_v7.csv", index=False); cmp.to_csv(E.OUT / "eval_v7_compare.csv", index=False)
E.log("\n" + res[res.setting != "teacher only"].round(4).to_string(index=False) + "\n\n" + cmp.round(4).to_string(index=False))
(E.OUT / "V7_DONE").touch(); E.log("===== XONG v6p5-25 (v7 A/B)")
