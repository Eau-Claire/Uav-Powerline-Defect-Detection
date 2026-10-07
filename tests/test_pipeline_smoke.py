"""Offline end-to-end smoke test of the stage wiring on tiny fake data.

Network downloads are replaced by pre-populated local mirrors / source dirs;
Ultralytics training + validation are stubbed. Everything else is real:
streaming parent reader, background pool, pHash, direct overlays, synthetic
composition, dedup, freeze/packaging, review gate, compose, dataset build,
integrity gates, report + evaluate, and resume/reuse.
"""
import io
import json
import tarfile
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("imagehash")

from yolov11sdi import pipeline  # noqa: E402
from yolov11sdi.config import Config, deep_merge, load_config  # noqa: E402
from yolov11sdi.state import StateStore  # noqa: E402

from conftest import ROOT  # noqa: E402

TAG = "v6p2e_23562a45b343"


def noise_jpg(seed: int, size=(96, 72)) -> bytes:
    from PIL import Image

    arr = np.random.default_rng(seed).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def add(tf, name, raw):
    info = tarfile.TarInfo(name)
    info.size = len(raw)
    tf.addfile(info, io.BytesIO(raw))


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("YOLOV11SDI_HOME", str(tmp_path))
    monkeypatch.setenv("YOLOV11SDI_PROJECT_ROOT", str(ROOT))
    monkeypatch.setenv("YOLOV11SDI_ENV", "local")
    monkeypatch.delenv("YOLOV11SDI_DATA_ROOT", raising=False)
    monkeypatch.setenv("CAMBER_BIN", str(tmp_path / "no-camber"))
    monkeypatch.setattr("yolov11sdi.camber.resolve_camber_bin", lambda explicit=None: None)
    import hashlib

    mirror = tmp_path / "artifacts/remote_mirror/datasets"
    bulk = tmp_path / "data/cache/remote/datasets"
    rows, base_members = [], []
    plan = [("train", i) for i in range(8)] + [("val", 100), ("val", 101), ("test", 200), ("test", 201)]
    for split, i in plan:
        raw = noise_jpg(i)
        lbl = f"0 0.5 0.5 0.3 0.4\n{1 if i % 2 else 3} 0.3 0.3 0.1 0.1\n".encode()
        rows.append({"split": split, "source": "base", "sha256": hashlib.sha256(raw).hexdigest(),
                     "image_path_source": f"/old/images/{split}/base/img{i}.jpg",
                     "label_path_source": f"/old/labels/{split}/base/img{i}.txt",
                     "resolution": "unique", "previous_v6p1_split": split})
        base_members += [(f"curated/images/{split}/img{i}.jpg", raw), (f"curated/labels/{split}/img{i}.txt", lbl)]
    (mirror / f"v6p2e/{TAG}").mkdir(parents=True)
    pd.DataFrame(rows).to_csv(mirror / f"v6p2e/{TAG}/v6p2e_split_manifest.csv", index=False)
    (mirror / f"v6p2e/{TAG}/v6p2e.freeze.json").write_text(json.dumps({
        "final_tag": TAG, "base_tag": "56b7ea194ee5", "overlay_tag": "ee2d3c4497fd",
        "parent_dataset_tag": "v6p1_5d0b1149891c", "split_counts": {"train": 8, "val": 2, "test": 2},
        "taxonomy": ["insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object",
                     "broken_strand"]}))
    bulk.mkdir(parents=True)
    with tarfile.open(bulk / "curated_v6_rs1280_56b7ea194ee5.tar", "w") as tf:
        for n, raw in base_members:
            add(tf, n, raw)
    (bulk / "v6p1/v6p1_5d0b1149891c").mkdir(parents=True)
    with tarfile.open(bulk / "v6p1/v6p1_5d0b1149891c/v6p1_fotl_overlay_ee2d3c4497fd.tar.gz", "w:gz") as tf:
        add(tf, "fotl/images/train/placeholder.jpg", noise_jpg(999))
    (bulk / f"v6p2e/{TAG}").mkdir(parents=True)
    with zipfile.ZipFile(bulk / f"v6p2e/{TAG}/v6p2e_reconciled_labels.zip", "w") as z:
        z.writestr("dummy.txt", "")

    # MPCD source (already "extracted")
    m = tmp_path / "data/external/mpcd/MPCD/ds"
    for split, i in [("train", 300), ("train", 301), ("valid", 302)]:
        (m / f"{split}/images").mkdir(parents=True, exist_ok=True)
        (m / f"{split}/labels").mkdir(parents=True, exist_ok=True)
        (m / f"{split}/images/m{i}.jpg").write_bytes(noise_jpg(i))
        (m / f"{split}/labels/m{i}.txt").write_text("1 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n")
    (m / "data.yaml").write_text("path: .\ntrain: train/images\nval: valid/images\nnames: ['cable', 'broken']\n")

    # PowerEquipment source (already "extracted")
    pe = tmp_path / "data/external/power_equipment/power_equipment_object_detection/tlod/yw"
    pe.mkdir(parents=True)
    for i, cls in [(400, "yw"), (401, "nw"), (402, "dx_dg"), (403, "dx_sg"), (404, "yw")]:
        (pe / f"p{i}.jpg").write_bytes(noise_jpg(i))
        (pe / f"p{i}.xml").write_text(
            f"<annotation><filename>p{i}.jpg</filename><size><width>96</width><height>72</height></size>"
            f"<object><name>{cls}</name><bndbox><xmin>10</xmin><ymin>10</ymin><xmax>50</xmax><ymax>40</ymax>"
            f"</bndbox></object></annotation>")

    def fake_crops(cfg, work_root, crops_root, rel, on_class_done=None):
        from PIL import Image

        out = []
        for c in cfg.get("openimages.classes"):
            st = c.lower().replace(" ", "_")
            (crops_root / st).mkdir(parents=True, exist_ok=True)
            im = Image.new("RGBA", (20, 20), (255, 0, 0, 255))
            p = crops_root / st / "c0.png"
            im.save(p)
            out.append({"source": "open_images_v7", "source_split": "train", "oi_image_id": f"id_{st}",
                        "source_sha256": "0" * 64, "foreign_subtype": st, "detection_index": 0,
                        "crop_path": rel(p), "mask_available": True,
                        "mask_source": "grabcut_from_openimages_detection", "foreground_ratio": 0.5})
            if on_class_done:
                on_class_done(c)
        return pd.DataFrame(out)

    monkeypatch.setattr("yolov11sdi.dataset.openimages.build_openimages_crops", fake_crops)
    return tmp_path


def small_cfg() -> Config:
    base = load_config("yolo11n_ablation", ROOT / "configs")
    return Config(base.name, deep_merge(base.data, {
        "v6p2e": {"expected_counts": {"train": 8, "val": 2, "test": 2, "total": 12}},
        "parent_prepare": {"bg_pool_max": 5, "min_bg_pool": 2},
        "synthetic": {"per_subtype": 2},
        "runtime": {"clean_parent_archives_after_qa": False, "upload_camber": False, "progress_interval_s": 0},
    }), base.config_dir)


def test_full_pipeline_offline(env, monkeypatch):
    cfg = small_cfg()
    outs = pipeline.run_external_pipeline(cfg)
    assert [o.status for o in outs] == ["complete"] * 7
    store = StateStore(env / "state")
    ext = store.load("external_freeze").summary
    assert ext["tag"].startswith("v6p5_ext_")
    assert ext["counts"]["synthetic_train_images"] == 10
    assert store.load("parent").summary["phash_rows"] == 4  # VAL + TEST only
    assert store.load("power_equipment").summary["dx_sg_review_only_boxes"] == 1
    assert store.load("dedup").summary["exact_duplicates"] == 0
    freeze = json.loads((env / "artifacts/freezes" / f"{ext['tag']}.freeze.json").read_text())
    assert freeze["policy"]["test_modified"] is False
    assert freeze["sources"]["OpenImagesV7"]["annotation_download"] == "detections_only"

    # Resume/reuse: second run reuses every stage.
    assert [o.status for o in pipeline.run_external_pipeline(cfg)] == ["reused"] * 7

    # Review gate: nothing decided -> needs_review (no silent KEEP).
    o = pipeline.run_stage("compose", cfg)
    assert o.status == "needs_review" and o.summary["pending_train"] > 0
    dec_path = env / "artifacts/qa/v6p5_review_decisions.csv"
    dec = pd.read_csv(dec_path, keep_default_na=False)
    dec.loc[dec["target_split"] == "train", "decision"] = "KEEP"
    first_direct = dec[(dec["candidate_type"] == "direct_powerline") & (dec["target_split"] == "train")].index[0]
    dec.loc[first_direct, "decision"] = "DROP"
    dec["reviewer"] = "test"
    dec.to_csv(dec_path, index=False)
    o = pipeline.run_stage("compose", cfg)
    assert o.status == "complete" and o.summary["tag"].startswith("v6p5_compose_")
    compose_tag = o.summary["tag"]
    idx_path = next((env / "artifacts/remote_mirror").rglob(f"{compose_tag}.index.json"))
    assert json.loads(idx_path.read_text())["synthetic_archive"].endswith("_synthetic_approved.tar.gz")

    # Training stage with Ultralytics stubbed.
    from yolov11sdi.training import checkpoint as ck

    def fake_train(cfg, data_yaml, run_dir, dataset_tag, compose_tag, remote_root, camber=None, force=False,
                   session=None):
        assert session and "code_commit" in session and "code_version" in session
        (run_dir / "weights").mkdir(parents=True, exist_ok=True)
        (run_dir / "weights/best.pt").write_bytes(b"x")
        t = cfg.get("training")
        st = ck.RunState(t["experiment"], t["model"], dataset_tag, compose_tag, 640, 50,
                         ck.train_hash(t, dataset_tag, compose_tag), last_completed_epoch=50, status="trained")
        st.save(run_dir / "run_state.json")
        return st

    names = cfg.names
    fake_metrics = {"mAP50": 0.78, "mAP50_95": 0.42, "gap": 0.36, "precision": 0.8, "recall": 0.7,
                    "per_class": [{"cid": i, "class": n, "AP50": 0.8, "AP50_95": 0.4, "gap": 0.4,
                                   "AP75": 0.3, "AP90": 0.1, "AP95": 0.05} for i, n in enumerate(names)]}
    monkeypatch.setattr("yolov11sdi.training.ultralytics_runner.train", fake_train)
    monkeypatch.setattr("yolov11sdi.training.metrics.evaluate_weights", lambda *a, **k: fake_metrics)
    o = pipeline.run_stage("yolo11n_ablation", cfg)
    assert o.status == "complete"
    ds_manifest = next((env / "data/working/datasets").rglob("dataset_manifest.csv"))
    ds = pd.read_csv(ds_manifest)
    assert set(ds["dataset_split"]) == {"train", "val"}           # TEST never materialized
    assert (ds[ds["dataset_split"] == "val"]["origin"] == "v6p2e").all() and (ds["dataset_split"] == "val").sum() == 2
    # The DROP'ed direct row is excluded; MPCD/PE external_eval never enters training.
    assert (ds["origin"] == "v6p5_direct").sum() == store.load("compose").summary["approved_direct"]
    assert o.summary["promotion_gate"]["foreign_AP50_improves_2pt"] is True

    ev = pipeline.run_stage("evaluate", cfg)
    assert ev.status == "complete" and "candidate_pass" in ev.summary
