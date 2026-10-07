import pandas as pd

from yolov11sdi.archive_utils import TarEntry, build_tar_suffix_index, verify_yolo_tar, write_deterministic_targz
from yolov11sdi.dataset.direct_overlay import add_direct_records
from yolov11sdi.dataset.external_stage import build_direct_archive, overlay_tag
from yolov11sdi.io_utils import validate_yolo_text

from conftest import write_img


def test_yolo_class_ids_and_bbox_validation():
    assert validate_yolo_text("0 0.5 0.5 0.2 0.2\n5 0.1 0.1 0.05 0.05\n", 6) == []
    assert validate_yolo_text("6 0.5 0.5 0.2 0.2", 6)            # class id out of 0..5
    assert validate_yolo_text("-1 0.5 0.5 0.2 0.2", 6)
    assert validate_yolo_text("1.5 0.5 0.5 0.2 0.2", 6)          # non-integer class
    assert validate_yolo_text("0 nan 0.5 0.2 0.2", 6)            # non-finite
    assert validate_yolo_text("0 1.2 0.5 0.2 0.2", 6)            # not normalized
    assert validate_yolo_text("0 0.5 0.5 0.0 0.2", 6)            # zero width
    assert validate_yolo_text("0 0.5 0.5 0.2", 6)                # wrong field count


def test_archive_image_label_pairs_match(tmp_path):
    good = write_deterministic_targz(tmp_path / "g.tar.gz", [
        TarEntry("root/images/train/a.jpg", data=b"i"), TarEntry("root/labels/train/a.txt", data=b"5 .5 .5 .1 .1\n"),
    ])
    rep = verify_yolo_tar(good, 6)
    assert rep.ok and rep.images == rep.labels == 1
    bad = write_deterministic_targz(tmp_path / "b.tar.gz", [
        TarEntry("root/images/train/a.jpg", data=b"i"),
        TarEntry("root/labels/train/b.txt", data=b"9 .5 .5 .1 .1\n"),
        TarEntry("root/images/test/c.jpg", data=b"i"), TarEntry("root/labels/test/c.txt", data=b"0 .5 .5 .1 .1\n"),
    ])
    rep = verify_yolo_tar(bad, 6)
    assert not rep.ok
    assert rep.missing_labels and rep.orphan_labels
    assert any("bad class id" in p for p in rep.label_problems)
    assert any("forbidden split 'test'" in p for p in rep.label_problems)


def test_direct_overlay_and_archive_layout(tmp_paths, cfg):
    src = write_img(tmp_paths.external / "pe" / "x.JPG")
    recs = [{"source": "PowerEquipment", "source_split": "train", "target_split": "train", "source_image": "x.JPG",
             "source_image_abs": str(src), "mapped_rows": [(4, 0.5, 0.5, 0.2, 0.2), (5, 0.3, 0.3, 0.1, 0.1)]}]
    overlay = tmp_paths.working / "v6p5_overlay"
    df = add_direct_records(recs, overlay, cfg.names, tmp_paths.rel)
    assert df.iloc[0]["mapped_class_names"] == "foreign_object,broken_strand"
    assert df.iloc[0]["overlay_image"].startswith("data/working/v6p5_overlay/direct/images/train/powerequipment_")
    tag1, _ = overlay_tag(overlay)
    assert tag1.startswith("v6p5_ext_") and len(tag1) == len("v6p5_ext_") + 12
    arch = build_direct_archive(overlay, df, "PowerEquipment", tmp_paths.artifacts / "pe.tar.gz", tmp_paths.resolve)
    rep = verify_yolo_tar(arch, 6)
    assert rep.ok and rep.splits == {"train": 1}
    import tarfile

    with tarfile.open(arch) as t:
        idx, dups = build_tar_suffix_index(t)
    assert dups == 0 and any(k.startswith("images/train/powerequipment_") for k in idx)
    assert overlay_tag(overlay)[0] == tag1  # stable
