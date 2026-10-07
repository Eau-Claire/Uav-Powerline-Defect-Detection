"""Data-pipeline checklist, one test per item (run: pytest -v tests/test_data_checklist.py).

raw sources -> canonicalize taxonomy -> canonicalize bbox -> QA per source ->
provenance preserved -> dedup / near-dup grouping -> split -> freeze
Scope: the V6.5 external sources handled by this repo (Open Images, MPCD,
PowerEquipment, synthetic). V6.2E base sources were checked by the legacy
v6p1/v6p2 notebooks and are consumed here as an immutable freeze.
"""
import pandas as pd
import pytest

from yolov11sdi.dataset import dedup, mpcd, power_equipment
from yolov11sdi.dataset.direct_overlay import DIRECT_COLUMNS, add_direct_records
from yolov11sdi.dataset.external_stage import build_freeze, overlay_tag
from yolov11sdi.dataset.synthetic import SYN_COLUMNS
from yolov11sdi.dataset.v6p2e import validate_parent
from yolov11sdi.hashing import PHashIndex
from yolov11sdi.io_utils import format_yolo_rows, validate_yolo_text, xyxy_to_yolo
from yolov11sdi.qa.review import build_review_manifest
from yolov11sdi.reporting.manifests import register_freeze

from conftest import write_img


def test_checklist_1_raw_sources_are_pinned(cfg):
    """Every external source is identified by an immutable origin (version / id / repo+file)."""
    assert cfg.get("openimages.version") == "v7" and cfg.get("openimages.split") == "train"
    assert cfg.get("openimages.fiftyone_version")
    assert cfg.get("mpcd.gdrive_id")
    assert cfg.get("power_equipment.hf_repo") and cfg.get("power_equipment.filename")
    assert cfg.get("v6p2e.tag").startswith("v6p2e_") and cfg.get("v6p2e.expected_lineage")


def test_checklist_2_canonicalize_taxonomy(cfg):
    """All source labels map into the fixed 0..5 taxonomy; nothing maps outside it."""
    n = len(cfg.names)
    targets = list(cfg.get("power_equipment.map").values()) + [
        cfg.get("mpcd.target_class"), cfg.get("synthetic.foreign_class"), cfg.get("synthetic.anchor_class")]
    assert all(0 <= int(t) < n for t in targets)
    assert cfg.names[cfg.get("mpcd.target_class")] == "broken_strand"
    assert cfg.names[cfg.get("synthetic.foreign_class")] == "foreign_object"
    ids = mpcd.broken_class_ids(["cable", "Broken-Strand", "fractured", "tower", "discontinuity"],
                                cfg.get("mpcd.name_keywords"))
    assert ids == [1, 2, 4]
    assert power_equipment.map_object("dx_sg", cfg.get("power_equipment.map"),
                                      cfg.get("power_equipment.review_only"))[0] is None


def test_checklist_3_canonicalize_bbox():
    """VOC pixel xyxy -> normalized YOLO cx,cy,w,h, clipped to the image; degenerate boxes dropped."""
    W, H = 200, 100
    cx, cy, w, h = xyxy_to_yolo([-10 / W, 10 / H, 250 / W, 60 / H])   # spills outside the image
    assert (round(cx, 6), round(cy, 6), round(w, 6), round(h, 6)) == (0.5, 0.35, 1.0, 0.5)
    txt = format_yolo_rows([(5, cx, cy, w, h), (4, 0.5, 0.5, 0.0, 0.2)])
    assert txt.count("\n") == 1 and validate_yolo_text(txt, 6) == []


def test_checklist_4_qa_per_source(tmp_paths, cfg):
    """Every candidate is reviewable per source; non-train source splits never become TRAIN."""
    recs = []
    for i, (src, split) in enumerate([("MPCD", "train"), ("MPCD", "external_eval"), ("PowerEquipment", "train")]):
        img = write_img(tmp_paths.external / f"s{i}.jpg", color=(i * 40, 10, 10))
        recs.append({"source": src, "source_split": split, "target_split": split, "source_image": f"s{i}.jpg",
                     "source_image_abs": str(img), "mapped_rows": [(5, .5, .5, .1, .1)]})
    direct = add_direct_records(recs, tmp_paths.working / "ov", cfg.names, tmp_paths.rel)
    review = build_review_manifest(direct, pd.DataFrame(columns=SYN_COLUMNS))
    by = review.groupby(["source", "target_split", "recommended_action"]).size().to_dict()
    assert by == {("MPCD", "external_eval", "KEEP_AS_EXTERNAL_EVAL"): 1, ("MPCD", "train", "REVIEW_THEN_KEEP"): 1,
                  ("PowerEquipment", "train", "REVIEW_THEN_KEEP"): 1}
    assert (review["decision"] == "").all()   # nothing pre-approved


def test_checklist_5_provenance_preserved(tmp_paths, cfg):
    """Each staged image keeps source, source split, original path, content hash and license note."""
    img = write_img(tmp_paths.external / "x.jpg")
    df = add_direct_records([{"source": "MPCD", "source_split": "val", "target_split": "external_eval",
                              "source_image": "valid/images/x.jpg", "source_image_abs": str(img),
                              "mapped_rows": [(5, .5, .5, .1, .1)], "license": "AGPL"}],
                            tmp_paths.working / "ov", cfg.names, tmp_paths.rel)
    row = df.iloc[0]
    assert list(df.columns) == DIRECT_COLUMNS
    assert (row["source"], row["source_split"], row["source_image"], row["license_note"]) == \
        ("MPCD", "val", "valid/images/x.jpg", "AGPL")
    assert len(row["source_sha256"]) == 64 and row["overlay_image"].endswith(row["source_sha256"][:16] + ".jpg")
    for col in ("oi_image_id", "oi_source_sha256", "background_sha256", "foreign_subtype"):
        assert col in SYN_COLUMNS   # synthetic keeps both appearance + background lineage


def test_checklist_6_dedup_and_near_dup_grouping(tmp_path):
    """Exact SHA hits are found against every split; pHash groups near-duplicates (Hamming <= 4)."""
    img = tmp_path / "a.jpg"
    img.write_bytes(b"identical")
    from yolov11sdi.hashing import sha256_file

    man = pd.DataFrame({"split": ["train", "val"], "sha256": [sha256_file(img)] * 2, "image_path_source": ["p", "q"]})
    assert sorted(dedup.exact_sha_audit([("a.jpg", img)], man)["base_split"]) == ["train", "val"]
    idx = PHashIndex()
    for v, g in [(0b0, "g1"), (0b111, "g1"), (0xFFFF << 32, "g2")]:  # g2 is >= 16 bits away
        idx.add(v, {"group": g})
    assert {p["group"] for _, p in idx.query(0b1, 4)} == {"g1"}


def test_checklist_7_split(cfg):
    """Parent split counts are verified against the frozen manifest; external splits are deterministic."""
    good = pd.DataFrame({"split": ["train"] * 17483 + ["val"] * 2272 + ["test"] * 2012, "sha256": "x",
                         "image_path_source": "p", "previous_v6p1_split": "train", "resolution": "unique"})
    freeze = {"final_tag": cfg.get("v6p2e.tag"), **cfg.get("v6p2e.expected_lineage")}
    assert validate_parent(cfg, freeze, good) == []
    bad = good.iloc[:-1]
    assert any("test=2011" in c for c in validate_parent(cfg, freeze, bad))
    keys = [f"tlod/img{i}.jpg" for i in range(2000)]
    splits = [power_equipment.split_for(k, cfg) for k in keys]
    assert splits == [power_equipment.split_for(k, cfg) for k in keys]
    assert 0.75 < splits.count("train") / len(splits) < 0.85
    assert set(splits) == {"train", "external_eval"}


def test_checklist_8_freeze(tmp_paths, cfg):
    """Content-derived tag, policy flags, and an immutable freeze registry."""
    img = write_img(tmp_paths.external / "f.jpg")
    ov = tmp_paths.working / "ov"
    direct = add_direct_records([{"source": "MPCD", "source_split": "train", "target_split": "train",
                                  "source_image": "f.jpg", "source_image_abs": str(img),
                                  "mapped_rows": [(5, .5, .5, .1, .1)]}], ov, cfg.names, tmp_paths.rel)
    tag, digest = overlay_tag(ov)
    empty = pd.DataFrame()
    syn = pd.DataFrame(columns=SYN_COLUMNS)
    fz = build_freeze(cfg, tag, digest, direct, syn, build_review_manifest(direct, syn), empty, empty, {}, "h", None)
    assert fz["tag"] == tag and fz["overlay_sha256"] == digest
    assert fz["policy"]["test_modified"] is False and fz["policy"]["val_modified"] is False
    assert fz["policy"]["auto_promotion"] is False and fz["policy"]["dx_sg_direct_merge"] is False
    assert "created" not in str(fz.keys())        # no timestamp in identity
    p = tmp_paths.artifacts / f"{tag}.freeze.json"
    p.write_text("{}")
    register_freeze(p, tmp_paths.freezes)
    p.write_text('{"changed": 1}')
    with pytest.raises(RuntimeError):
        register_freeze(p, tmp_paths.freezes)
