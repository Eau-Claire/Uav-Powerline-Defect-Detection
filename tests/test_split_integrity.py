import pandas as pd
import pytest

from yolov11sdi.config import CANONICAL_TAXONOMY, ConfigError, validate_taxonomy
from yolov11sdi.dataset import dedup
from yolov11sdi.dataset.power_equipment import map_object, split_for
from yolov11sdi.qa import integrity
from yolov11sdi.qa.review import build_review_manifest, check_review_manifest
from yolov11sdi.state import FileRecord, StageRun, StateStore, check_reusable


# ---------------------------------------------------------- 1. taxonomy

def test_taxonomy_order_unchanged(cfg):
    assert cfg.names == [
        "insulator", "insulator_broken", "insulator_flashover", "bird_nest", "foreign_object", "broken_strand"]
    assert tuple(cfg.names) == CANONICAL_TAXONOMY
    with pytest.raises(ConfigError):
        validate_taxonomy(list(reversed(cfg.names)))


def test_v6p5_settings_preserved(cfg):
    assert cfg.get("v6p2e.tag") == "v6p2e_23562a45b343"
    assert cfg.get("v6p2e.root") == "stash://bunpmc/projects/yolov11sdi/datasets"
    assert cfg.get("v6p2e.expected_counts") == {"train": 17483, "val": 2272, "test": 2012, "total": 21767}
    assert cfg.get("openimages.classes") == ["Balloon", "Kite", "Plastic bag", "Bird", "Person"]
    assert cfg.get("openimages.max_samples_per_class") == 250
    assert cfg.get("openimages.seed") == 20261007 == cfg.get("seed")
    assert cfg.get("openimages.annotation_download") == "detections_only"
    assert cfg.get("openimages.openimages_segmentation_zip_downloaded") is False
    assert cfg.get("synthetic.per_subtype") == 100 and cfg.get("synthetic.require_mask") is True
    assert cfg.get("power_equipment.hf_repo") == "sxiong/Power-equipment-image-dataset"
    assert cfg.get("power_equipment.filename") == "transmission line object detection.zip"
    assert cfg.get("power_equipment.train_frac") == 0.80
    assert cfg.get("mpcd.gdrive_id") == "1KyciMmwL2_p_-mwckHFqG5fkBA1jzATv"
    assert cfg.get("dedup.run_phash") is True and cfg.get("dedup.phash_max_hamming") == 4
    assert cfg.get("parent_prepare.bg_pool_max") == 1200
    assert cfg.get("parent_prepare.phash_splits") == ["val", "test"]
    assert cfg.get("qa.max_contacts_per_source") == 80 and cfg.get("qa.max_synth_contacts") == 100
    assert cfg.runtime("clean_parent_archives_after_qa") is True and cfg.runtime("min_free_gb_warn") == 3.0
    assert cfg.runtime("upload_camber") is True and cfg.get("run_sanity_train") is False
    t = cfg.get("training")
    assert (t["model"], t["imgsz"], t["epochs"], t["batch"], t["device"], t["seed"]) == \
        ("yolo11n.pt", 640, 50, -1, 0, 20261007)
    r = cfg.get("v6p5.remote")
    assert r["stage_root"] == "stash://bunpmc/projects/yolov11sdi/datasets/v6p5_external_stage"
    assert r["compose_root"] == r["stage_root"] + "/compose_qa"
    assert r["job_root"] == r["stage_root"] + "/jobs"
    assert r["train_root"] == "stash://bunpmc/projects/yolov11sdi/checkpoints/v6p5_data_ablation"


def test_runtime_section_does_not_change_config_hash(cfg):
    from yolov11sdi.config import Config, deep_merge

    other = Config(cfg.name, deep_merge(cfg.data, {"runtime": {"upload_camber": False}}), cfg.config_dir)
    assert other.config_hash() == cfg.config_hash()


# ----------------------------------------------- 2/3/4 split guarantees

def _parent_manifest():
    return pd.DataFrame({"split": ["train", "train", "val", "val", "test"],
                         "sha256": ["t1", "t2", "v1", "v2", "x1"]})


def _ds(rows):
    return pd.DataFrame(rows, columns=["dataset_split", "sha256", "origin"])


def test_test_split_never_used_for_training():
    m = _parent_manifest()
    with pytest.raises(integrity.IntegrityError):
        integrity.training_rows(m, ["train", "val", "test"])
    assert set(integrity.training_rows(m, ["train", "val"])["split"]) == {"train", "val"}
    leaked = _ds([("train", "x1", "v6p2e"), ("val", "v1", "v6p2e"), ("val", "v2", "v6p2e")])
    with pytest.raises(integrity.IntegrityError):
        integrity.check_no_test_in_training(leaked, m)


def test_val_frozen_for_ablation():
    m = _parent_manifest()
    good = _ds([("train", "t1", "v6p2e"), ("val", "v1", "v6p2e"), ("val", "v2", "v6p2e")])
    integrity.check_val_frozen(good, m)
    with pytest.raises(integrity.IntegrityError):  # external image added to VAL
        integrity.check_val_frozen(_ds([("val", "v1", "v6p2e"), ("val", "v2", "v6p2e"),
                                        ("val", "e9", "v6p5_direct")]), m)
    with pytest.raises(integrity.IntegrityError):  # VAL image missing
        integrity.check_val_frozen(_ds([("val", "v1", "v6p2e")]), m)


def test_synthetic_samples_train_only():
    integrity.check_synthetic_train_only(_ds([("train", "s1", "v6p5_synthetic")]))
    with pytest.raises(integrity.IntegrityError):
        integrity.check_synthetic_train_only(_ds([("val", "s1", "v6p5_synthetic")]))
    direct = pd.DataFrame(columns=["overlay_image", "overlay_label", "source", "target_split", "mapped_class_names"])
    synth = pd.DataFrame([{"overlay_image": "a.jpg", "overlay_label": "a.txt", "source": "syn", "foreign_subtype": "kite"}])
    review = build_review_manifest(direct, synth)
    assert set(review["target_split"]) == {"train"}
    review.loc[0, "target_split"] = "val"
    with pytest.raises(RuntimeError):
        check_review_manifest(review)


# ----------------------------------------------- 5/6 Power Equipment

def test_dx_sg_is_review_only(cfg):
    pm = cfg.get("power_equipment.map")
    ro = cfg.get("power_equipment.review_only")
    assert map_object("dx_sg", pm, ro) == (None, "strand_loosening_review_only")
    assert "dx_sg" not in pm


def test_power_equipment_mapping(cfg):
    pm, ro = cfg.get("power_equipment.map"), cfg.get("power_equipment.review_only")
    names = cfg.names
    assert names[map_object("dx_dg", pm, ro)[0]] == "broken_strand"
    assert names[map_object("yw", pm, ro)[0]] == "foreign_object"
    assert names[map_object("nw", pm, ro)[0]] == "bird_nest"
    assert map_object("czjyz_ps", pm, ro) == (None, None)


def test_power_equipment_split_reproduces_legacy_kaggle_key(cfg):
    from yolov11sdi.hashing import deterministic_fraction

    rel = "transmission line object detection/yw_hg20230103/JPEGImages/a.jpg"
    legacy = "/kaggle/working/v6p5_external_expansion/sources/power_equipment_object_detection/" + rel
    expected = "train" if deterministic_fraction(legacy) < 0.8 else "external_eval"
    assert split_for(rel, cfg) == expected


# ------------------------------------------------ 12/13 state reuse

def test_state_reuse_and_config_invalidation(tmp_paths):
    store = StateStore(tmp_paths.state)
    out = tmp_paths.artifacts / "x.csv"
    out.write_text("a\n1\n")
    inputs = [{"name": "in", "sha256": "abc"}]
    run = StageRun(store, "demo", "hash1", inputs, {"git_commit": None, "host": "h", "environment": "local"})
    run.complete([FileRecord.of("out", out, tmp_paths.rel(out))])
    st = store.load("demo")
    assert check_reusable(st, "hash1", inputs, tmp_paths.resolve)[0] is True
    ok, reason = check_reusable(st, "hash2", inputs, tmp_paths.resolve)
    assert not ok and "config" in reason
    assert not check_reusable(st, "hash1", [{"name": "in", "sha256": "zzz"}], tmp_paths.resolve)[0]
    out.write_text("changed content\n")
    assert not check_reusable(st, "hash1", inputs, tmp_paths.resolve)[0]
    # New run with a different config keeps the old record as stale history.
    StageRun(store, "demo", "hash2", inputs, {"git_commit": None, "host": "h", "environment": "local"})
    st2 = store.load("demo")
    assert st2.status == "running" and st2.history[-1]["status"] == "stale"
    assert st2.history[-1]["config_hash"] == "hash1"


# ------------------------------------------------- 14 exact SHA gate

def test_exact_sha_leakage_gate(tmp_path):
    from yolov11sdi.hashing import sha256_file

    img = tmp_path / "ext.jpg"
    img.write_bytes(b"same-bytes")
    manifest = pd.DataFrame({"split": ["test"], "sha256": [sha256_file(img)], "image_path_source": ["p"]})
    exact = dedup.exact_sha_audit([("ext.jpg", img)], manifest)
    assert len(exact) == 1 and exact.iloc[0]["base_split"] == "test"
    with pytest.raises(RuntimeError):
        dedup.exact_gate(exact, allow=False)
    dedup.exact_gate(exact.iloc[0:0], allow=False)
