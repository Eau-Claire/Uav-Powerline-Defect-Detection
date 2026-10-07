"""Pipeline status table for notebooks/99 and scripts/status.py.

Only reports what state files / local artifacts prove. A stage with no
state file is shown as `not run (unverified)`, never as complete.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..environment import disk_usage, git_commit
from ..io_utils import read_csv_safe, read_json
from ..paths import ProjectPaths
from ..state import StateStore


def _row(label: str, st, artifact="") -> dict:
    if st is None:
        return {"Stage": label, "Status": "not run (unverified)", "Artifact/Tag": artifact, "Updated": ""}
    prog = st.progress or {}
    status = st.status
    if status == "running" and prog.get("total"):
        status = f"running {prog.get('current')}/{prog.get('total')}"
    return {"Stage": label, "Status": status, "Artifact/Tag": artifact, "Updated": st.updated_at or ""}


def stage_table(cfg, paths: ProjectPaths) -> pd.DataFrame:
    s = StateStore(paths.state).all()
    g = lambda k: s.get(k)  # noqa: E731
    summ = lambda k, f, d="": (g(k).summary.get(f, d) if g(k) else d)  # noqa: E731
    exp = cfg.get("training.experiment")
    compose = g("compose")
    rows = [
        _row("V6.2E parent", g("parent"), cfg.get("v6p2e.tag")),
        _row("TRAIN background pool", g("parent"), summ("parent", "bg_pool")),
        _row("VAL/TEST pHash", g("parent"), summ("parent", "phash_rows")),
        _row("OpenImages crops", g("openimages"), summ("openimages", "crops")),
        _row("MPCD stage", g("mpcd"), summ("mpcd", "images")),
        _row("PowerEquipment stage", g("power_equipment"), summ("power_equipment", "images")),
        _row("Synthetic foreign objects", g("synthetic"), summ("synthetic", "synthetic")),
        _row("Dedup (exact / pHash)", g("dedup"),
             f"{summ('dedup', 'exact_duplicates', '?')} / {summ('dedup', 'phash_candidates', '?')}" if g("dedup") else ""),
        _row("V6.5 external freeze", g("external_freeze"), summ("external_freeze", "tag")),
    ]
    if compose is None:
        rows += [_row("Compose review", None), _row("Compose freeze", None)]
    else:
        review_status = "complete" if compose.status == "complete" else compose.status
        rows.append({**_row("Compose review", compose, f"pending={compose.summary.get('pending_train', 0)}"),
                     "Status": review_status})
        rows.append({**_row("Compose freeze", compose, compose.summary.get("tag", "")),
                     "Status": "complete" if compose.status == "complete" else "pending"})
    rows.append(_row("YOLO11n@640", g(exp), summ(exp, "run_tag")))
    rows.append(_row("Evaluation", g("evaluate"),
                     "gate PASS" if summ("evaluate", "candidate_pass", None) is True
                     else ("gate FAIL" if summ("evaluate", "candidate_pass", None) is False else "")))
    return pd.DataFrame(rows)


def latest_checkpoint(paths: ProjectPaths) -> dict | None:
    cks = sorted(paths.runs.glob("*/*/weights/last.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cks:
        return None
    run_dir = cks[0].parent.parent
    rs = read_json(run_dir / "run_state.json", {}) or {}
    return {"checkpoint": paths.rel(cks[0]), "epoch": rs.get("last_completed_epoch"),
            "epochs_target": rs.get("epochs_target"), "status": rs.get("status"),
            "last_synced_epoch": rs.get("last_synced_epoch"), "remote_root": rs.get("remote_root")}


def latest_synced(paths: ProjectPaths) -> dict | None:
    best = None
    for st in StateStore(paths.state).all().values():
        if st.status == "complete" and st.summary.get("uploaded"):
            remote = st.summary.get("remote") or next((o.get("remote") for o in st.outputs if o.get("remote")), None)
            cand = {"stage": st.stage, "remote": remote, "at": st.completed_at}
            if best is None or (cand["at"] or "") > (best["at"] or ""):
                best = cand
    return best


def pending_review_count(cfg, paths: ProjectPaths) -> int | None:
    st = StateStore(paths.state).load("compose")
    if st and "pending_train" in st.summary and st.status == "needs_review":
        return int(st.summary["pending_train"])
    p = paths.resolve(cfg.get("compose.decisions_file"))
    df = read_csv_safe(p)
    if not len(df):
        return None
    train = df[df["target_split"].astype(str) == "train"]
    return int((train["decision"].astype(str).str.strip() == "").sum())


def overview(cfg, paths: ProjectPaths) -> dict:
    s = StateStore(paths.state)
    dd = s.load("dedup")
    return {
        "environment": paths.env.value,
        "home": str(paths.home),
        "data_root": str(paths.data_root),
        "disk": disk_usage(paths.data_root),
        "config": cfg.name,
        "config_hash": cfg.config_hash()[:12],
        "git_commit": git_commit(paths.project_root) or "n/a (not a git repo)",
        "latest_checkpoint": latest_checkpoint(paths),
        "latest_camber_synced": latest_synced(paths),
        "pending_review": pending_review_count(cfg, paths),
        "exact_duplicates": dd.summary.get("exact_duplicates") if dd else None,
        "phash_candidates": dd.summary.get("phash_candidates") if dd else None,
    }


def remote_tags(cfg, camber) -> dict:
    """Read-only Camber listing (only when the CLI is authenticated)."""
    from ..dataset.compose import COMPOSE_PREFIX
    from ..dataset.external_stage import EXT_PREFIX

    if not camber.available():
        return {"available": False}
    return {
        "available": True,
        "v6p5_ext": camber.discover_tags(cfg.get("v6p5.remote.stage_root"), EXT_PREFIX),
        "v6p5_compose": camber.discover_tags(cfg.get("v6p5.remote.compose_root"), COMPOSE_PREFIX),
        "train_runs": camber.discover_tags(cfg.get("v6p5.remote.train_root"), cfg.get("training.remote_tag_prefix")),
        "v6p2e_freeze_exists": camber.exists(
            f"{cfg.get('v6p2e.root')}/v6p2e/{cfg.get('v6p2e.tag')}/{cfg.get('v6p2e.files.freeze')}"),
    }
