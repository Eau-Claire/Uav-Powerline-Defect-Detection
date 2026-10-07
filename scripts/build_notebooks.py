#!/usr/bin/env python
"""(Re)generate the thin local notebooks in notebooks/ from one definition.

    python scripts/build_notebooks.py
Notebooks hold no business logic; edit src/yolov11sdi instead.
"""
from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401

from yolov11sdi.nbgen import LOCAL_BOOTSTRAP, check_compiles, code, md, notebook, write_notebook

ROOT = Path(__file__).resolve().parents[1]

SHOW_OUTCOME = '''\
print(outcome)
display(pd.Series(outcome.summary, dtype=object).to_frame("value"))'''

STATUS_CELL = '''\
from yolov11sdi.reporting import status
cfg = load_config("yolo11n_ablation")
display(status.stage_table(cfg, paths))
ov = status.overview(cfg, paths)
display(pd.Series({k: v for k, v in ov.items() if k != "disk"}, dtype=object).to_frame("value"))
d = ov["disk"]; print(f"disk free {d['free_gb']:.1f} / {d['total_gb']:.1f} GB at {d['path']}")'''


def nb00():
    return [
        md("""
        # 00 — Environment and status
        Checks the runtime (LOCAL / KAGGLE / CAMBER), paths, optional dependencies,
        Camber CLI/auth and shows what state files prove so far.
        Nothing here downloads or changes data.
        """),
        code(LOCAL_BOOTSTRAP),
        code('''\
        import importlib
        mods = {"yaml": "core", "pandas": "core", "PIL": "core", "imagehash": "stage", "cv2": "stage",
                "fiftyone": "stage", "gdown": "stage", "huggingface_hub": "stage", "ultralytics": "train", "torch": "train"}
        rows = []
        for m, extra in mods.items():
            try:
                mod = importlib.import_module(m); rows.append((m, extra, getattr(mod, "__version__", "ok")))
            except Exception as e:
                rows.append((m, extra, f"MISSING ({type(e).__name__})"))
        display(pd.DataFrame(rows, columns=["module", "extra", "version"]))'''),
        code('''\
        from yolov11sdi.camber import CamberClient
        from yolov11sdi.environment import load_camber_api_key
        cfg = load_config("yolo11n_ablation")
        load_camber_api_key(paths.env)
        cam = CamberClient(paths, cfg.get("camber.stash_prefix"), cfg.runtime("camber_bin"))
        print("camber binary:", cam.binary)
        print("camber usable:", cam.available(), "(key from env/Kaggle secret or `camber login`)")
        print("config hash:", cfg.config_hash()[:12])'''),
        code(STATUS_CELL),
    ]


def nb10():
    return [
        md("""
        # 10 — V6.2E parent prepare (low disk)
        **V6.2E is immutable** (`v6p2e_23562a45b343`, 17,483 / 2,272 / 2,012).
        1. download freeze + split manifest, validate against config (conflicts stop the run);
        2. index base/overlay archives without extraction;
        3. stream ≤1200 TRAIN backgrounds with an insulator anchor;
        4. pHash VAL+TEST once → `parent_val_test_phash.csv.gz` (incremental checkpoint);
        5. close handles and delete the large archives (`runtime.clean_parent_archives_after_qa`).
        Re-running resumes; a complete matching result is reused.
        """),
        code(LOCAL_BOOTSTRAP),
        code('pipeline.run_stage("parent", dry_run=True)'),
        code('outcome = pipeline.run_stage("parent", force=FORCE_REBUILD, dry_run=DRY_RUN)\n' + SHOW_OUTCOME),
        code('''\
        pool = pd.read_csv(paths.resolve(store.load("parent").output("bg_pool_csv")["path"]))
        display(pool.head())
        for p in pool["image"].head(3):
            display(IPImage(filename=str(paths.resolve(p)), width=480))'''),
    ]


def nb20():
    return [
        md("""
        # 20 — V6.5 external expansion staging → `v6p5_ext_<hash>`
        Sources (settings unchanged from the last Kaggle notebook):
        * **Open Images V7 TRAIN** Balloon / Kite / Plastic bag / Bird / Person — appearance only,
          `detections_only`, local GrabCut mask, 250/class, seed 20261007;
        * **MPCD** broken/fracture → `broken_strand`;
        * **Power Equipment** `dx_dg→broken_strand`, `yw→foreign_object`, `nw→bird_nest`, `dx_sg` review-only;
        * **synthetic** relation-aware foreign objects on internal TRAIN backgrounds, 100/subtype, TRAIN only.
        Then exact-SHA + pHash QA, review manifest, contact sheets, deterministic packaging, freeze, Camber upload.
        Each stage checkpoints (per class / per subtype) and resumes after a restart.
        """),
        code(LOCAL_BOOTSTRAP),
        code('''\
        outcomes = {}
        for s in ["parent", "openimages", "mpcd", "power_equipment", "synthetic", "dedup"]:
            outcomes[s] = pipeline.run_stage(s, force=FORCE_REBUILD, dry_run=DRY_RUN)
            print(outcomes[s])'''),
        code('''\
        d = store.load("dedup").summary
        print("exact SHA duplicates vs V6.2E (HARD gate):", d["exact_duplicates"])
        print("pHash candidates vs VAL/TEST (review):   ", d["phash_candidates"])'''),
        code('outcome = pipeline.run_stage("external_freeze", force=FORCE_REBUILD, dry_run=DRY_RUN)\n' + SHOW_OUTCOME),
        code('''\
        contacts = sorted((paths.qa / "v6p5_external" / "contacts").glob("*.jpg"))
        print(len(contacts), "contact sheets")
        for p in contacts[:12]:
            print(p.name); display(IPImage(filename=str(p), width=640))'''),
    ]


def nb30():
    return [
        md("""
        # 30 — Compose QA review gate → `v6p5_compose_<hash>`
        **No auto-promotion.** Every TRAIN candidate needs `KEEP` or `DROP` in
        `artifacts/qa/v6p5_review_decisions.csv` (blank = pending). `REVIEW_THEN_KEEP` is a
        recommendation, never a decision. The stage stops with `needs_review` until all TRAIN
        candidates are resolved. Exact SHA duplicates are a hard gate; pHash candidates must be reviewed.

        Workflow: run → open the decisions CSV (spreadsheet) → fill decision/reviewer/comment → run again.
        """),
        code(LOCAL_BOOTSTRAP),
        code('outcome = pipeline.run_stage("compose", force=FORCE_REBUILD, dry_run=DRY_RUN)\n' + SHOW_OUTCOME),
        code('''\
        cfg = load_config("v6p5_compose")
        dec_path = paths.resolve(cfg.get("compose.decisions_file"))
        dec = pd.read_csv(dec_path, keep_default_na=False) if dec_path.exists() else pd.DataFrame()
        print("decisions file:", dec_path)
        if len(dec):
            display(dec.assign(decision=dec["decision"].replace("", "PENDING"))
                    .groupby(["candidate_type", "source", "target_split", "decision"]).size().rename("n").reset_index())'''),
        code('''\
        # Pending TRAIN candidates + their contact sheets (stage QA extract).
        pending = paths.compose / "pending_review.csv"
        if pending.exists():
            pend = pd.read_csv(pending, keep_default_na=False); display(pend.head(50))
        ph = paths.compose / "phash_candidates_to_review.csv"
        if ph.exists():
            print("pHash candidates to review:"); display(pd.read_csv(ph))
        extract = sorted((paths.working / "compose_qa_extract").glob("*/contacts/*.jpg"))
        for p in extract[:load_config("v6p5_compose").get("compose.max_contact_sheets")]:
            print(p.name); display(IPImage(filename=str(p), width=640))'''),
        md("""
        ### Review gate
        If the status above is `needs_review`, **stop here**, fill the decisions CSV, then re-run this notebook.
        After a `complete` compose, notebook 40 auto-selects the compose tag.
        """),
    ]


def nb40():
    return [
        md("""
        # 40 — YOLO11n@640 data ablation (50 epochs)
        **This is a data ablation, NOT the final V6.5-vs-V6.4 benchmark.**
        V6.2E TRAIN + reviewed MPCD/PowerEquipment TRAIN + approved synthetic, evaluated on the
        **frozen V6.2E VAL**. TEST is never materialized.
        `yolo11n.pt`, imgsz 640, epochs 50, batch -1, device 0, seed 20261007.

        Resume: if `runs/yolo11n_640/<run>/weights/last.pt` exists with a matching train hash,
        training resumes from it (also restorable from Camber). Needs a CUDA GPU.
        """),
        code(LOCAL_BOOTSTRAP),
        code('pipeline.run_stage("yolo11n_ablation", dry_run=True)'),
        code('outcome = pipeline.run_stage("yolo11n_ablation", force=FORCE_REBUILD, dry_run=DRY_RUN)\n' + SHOW_OUTCOME),
        code('''\
        from yolov11sdi.reporting.status import latest_checkpoint
        print(latest_checkpoint(paths))
        runs = sorted(paths.runs.glob("yolo11n_640/*/results.csv"))
        if runs:
            r = pd.read_csv(runs[-1]); r.columns = [c.strip() for c in r.columns]
            display(r.tail())
            r.plot(x="epoch", y=[c for c in r.columns if "mAP50" in c], figsize=(8, 4))'''),
    ]


def nb50():
    return [
        md("""
        # 50 — Evaluate and compare vs old V6.2E YOLO11n@640
        Baseline: mAP50 0.768 / mAP50:95 0.413 / gap 0.355.
        Gate (legacy): global mAP50:95 regression ≤ 1 pt; foreign_object AP50 +2 pt and AP50:95 +1 pt;
        broken_strand AP50:95 +1 pt. A pass only opens human review/promotion.
        """),
        code(LOCAL_BOOTSTRAP),
        code('outcome = pipeline.run_stage("evaluate", force=FORCE_REBUILD, dry_run=DRY_RUN)\n' + SHOW_OUTCOME),
        code('''\
        cfg = load_config("yolo11n_ablation")
        cmp_csv = paths.exports / f"{cfg.get('training.experiment')}_vs_baseline.csv"
        if cmp_csv.exists():
            display(pd.read_csv(cmp_csv))'''),
        md("""
        If foreign_object improves but broken_strand regresses, split the ablation by source
        (foreign-only vs strand-only) before freezing anything.
        """),
    ]


def nb60():
    return [
        md("""
        # 60 — Freeze canonical V6.5 candidate (HUMAN DECISION)
        Nothing is promoted automatically. This notebook only shows whether the data-ablation
        gate passed and which artifacts a canonical V6.5 freeze would be built from.
        Canonical V6.5 creation is intentionally not automated yet: it needs an explicit decision
        on which sources to keep (see notebook 50) and a new config `configs/v6p5_canonical.yaml`.
        """),
        code(LOCAL_BOOTSTRAP),
        code('''\
        ev = store.load("evaluate")
        if ev is None or ev.status != "complete":
            print("BLOCKED: evaluation not complete — run notebook 50 first.")
        else:
            print("gate:", ev.summary.get("promotion_gate"))
            print("candidate_pass:", ev.summary.get("candidate_pass"))
            comp = store.load("compose")
            print("compose tag:", comp.summary.get("tag") if comp else None)
            print("external stage tag:", comp.summary.get("stage_tag") if comp else None)'''),
    ]


def nb_future(title: str, cfg_name: str):
    return [
        md(f"""
        # {title}
        **Blocked until canonical V6.5 is frozen** (after the YOLO11n data ablation passes and a human
        promotes it). Config: `configs/{cfg_name}.yaml`. The same resumable runner is used once unblocked.
        """),
        code(LOCAL_BOOTSTRAP),
        code(f'''\
        cfg = load_config("{cfg_name}")
        print("blocked_until:", cfg.get("future.blocked_until", None))
        print("canonical dataset:", cfg.get("future.canonical_dataset_tag", None))
        display(pd.Series(cfg.get("training"), dtype=object).to_frame("value"))'''),
    ]


def nb99():
    return [
        md("""
        # 99 — Pipeline status
        Shows only what `state/*.json` and local artifacts prove. Set `CHECK_REMOTE=True` for a
        read-only Camber listing of v6p5_ext / v6p5_compose / training tags.
        """),
        code(LOCAL_BOOTSTRAP),
        code(STATUS_CELL),
        code('''\
        CHECK_REMOTE = False
        if CHECK_REMOTE:
            from yolov11sdi.camber import CamberClient
            from yolov11sdi.environment import load_camber_api_key
            load_camber_api_key(paths.env)
            print(status.remote_tags(cfg, CamberClient(paths, cfg.get("camber.stash_prefix"), cfg.runtime("camber_bin"))))'''),
        code('''\
        for name, st in store.all().items():
            if st.status in ("failed", "needs_review"):
                print(f"--- {name}: {st.status}"); print(st.error or st.summary)'''),
    ]


NOTEBOOKS = {
    "00_environment_and_status.ipynb": nb00,
    "10_v6p2e_parent_prepare.ipynb": nb10,
    "20_v6p5_external_stage.ipynb": nb20,
    "30_v6p5_compose_qa.ipynb": nb30,
    "40_yolo11n_640_data_ablation.ipynb": nb40,
    "50_evaluate_and_compare.ipynb": nb50,
    "60_freeze_v6p5_candidate.ipynb": nb60,
    "70_yolo11m_1024.ipynb": lambda: nb_future("70 — YOLO11m@1024 (canonical V6.5)", "yolo11m_1024"),
    "80_rtdetr_r50_b0.ipynb": lambda: nb_future("80 — RT-DETR-R50 B0 (canonical V6.5)", "rtdetr_b0"),
    "99_pipeline_status.ipynb": nb99,
}


def main() -> None:
    for name, fn in NOTEBOOKS.items():
        nb = notebook(fn())
        check_compiles(nb, name)
        print(write_notebook(ROOT / "notebooks" / name, nb).relative_to(ROOT))


if __name__ == "__main__":
    main()
