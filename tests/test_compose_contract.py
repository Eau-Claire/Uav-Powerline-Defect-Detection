import pandas as pd
import pytest

from yolov11sdi.archive_utils import TarEntry, write_deterministic_targz
from yolov11sdi.dataset import compose as cp
from yolov11sdi.dataset.external_stage import StageRemote
from yolov11sdi.qa.review import build_review_manifest, resolve_decisions

STAGE_ROOT = "stash://bunpmc/projects/yolov11sdi/datasets/v6p5_external_stage"


def _review():
    direct = pd.DataFrame([{"overlay_image": "data/x/mpcd_aa.jpg", "overlay_label": "data/x/mpcd_aa.txt",
                            "source": "MPCD", "target_split": "train", "mapped_class_names": "broken_strand"},
                           {"overlay_image": "data/x/mpcd_bb.jpg", "overlay_label": "data/x/mpcd_bb.txt",
                            "source": "MPCD", "target_split": "external_eval", "mapped_class_names": "broken_strand"}])
    synth = pd.DataFrame([{"overlay_image": "data/s/syn_kite_1.jpg", "overlay_label": "data/s/syn_kite_1.txt",
                           "source": "OpenImagesV7_synthetic_relation", "foreign_subtype": "kite"}])
    return build_review_manifest(direct, synth)


def test_compose_index_contains_synthetic_archive():
    crem = cp.ComposeRemote(f"{STAGE_ROOT}/compose_qa", "v6p5_compose_0123456789ab")
    idx = cp.build_index(crem.tag, "v6p5_ext_aaaaaaaaaaaa", StageRemote(STAGE_ROOT, "v6p5_ext_aaaaaaaaaaaa"), crem,
                         {"mpcd": f"{STAGE_ROOT}/mpcd/mpcd_broken_strand_stage.tar.gz",
                          "power_equipment": f"{STAGE_ROOT}/power_equipment/power_equipment_stage.tar.gz"})
    assert idx["synthetic_archive"] == (f"{STAGE_ROOT}/compose_qa/v6p5_compose_0123456789ab/"
                                        "v6p5_compose_0123456789ab_synthetic_approved.tar.gz")
    cp.validate_index(idx)
    with pytest.raises(RuntimeError):
        cp.validate_index({k: v for k, v in idx.items() if k != "synthetic_archive"})


def test_review_then_keep_is_never_silently_kept():
    review = _review()
    assert set(review["recommended_action"]) == {"REVIEW_THEN_KEEP", "KEEP_AS_EXTERNAL_EVAL", "VISUAL_REVIEW_THEN_KEEP"}
    res = resolve_decisions(review, None, auto_approve_recommended=False)
    assert not res.complete and len(res.pending_train) == 2
    assert (res.resolved["decision"] == "").all()


def test_decisions_resolve_and_external_eval_dropped():
    review = _review()
    dec = pd.DataFrame({"review_id": ["V65R000000", "V65R000002"], "decision": ["keep", "DROP"],
                        "reviewer": ["me", "me"], "comment": ["", "blurry"]})
    res = resolve_decisions(review, dec)
    assert res.complete
    d = dict(zip(res.resolved["review_id"], res.resolved["decision"]))
    assert d == {"V65R000000": "KEEP", "V65R000001": "DROP", "V65R000002": "DROP"}
    with pytest.raises(ValueError):
        resolve_decisions(review, pd.DataFrame({"review_id": ["V65R000000"], "decision": ["MAYBE"]}))


def test_compose_tag_deterministic_and_synthetic_archive(tmp_path):
    overlay = write_deterministic_targz(tmp_path / "ov.tar.gz", [
        TarEntry("v6p5_overlay/synthetic_foreign/images/train/syn_kite_1.jpg", data=b"img"),
        TarEntry("v6p5_overlay/synthetic_foreign/labels/train/syn_kite_1.txt", data=b"4 .5 .5 .1 .1\n"),
    ])
    review = _review()
    dec = pd.DataFrame({"review_id": ["V65R000000", "V65R000002"], "decision": ["KEEP", "KEEP"]})
    resolved = resolve_decisions(review, dec).resolved
    keep_synth = resolved[(resolved["candidate_type"] == "synthetic_relation") & (resolved["decision"] == "KEEP")]
    tags = []
    for i in range(2):
        arch = tmp_path / f"syn{i}.tar.gz"
        inc = cp.build_approved_synthetic(overlay, keep_synth, arch)
        assert len(inc) == 1
        seed = cp.compose_seed("v6p5_ext_aaaaaaaaaaaa", {"overlay_sha256": "f" * 64}, resolved, arch,
                               ["insulator", "insulator_broken", "insulator_flashover", "bird_nest",
                                "foreign_object", "broken_strand"])
        tags.append(cp.compose_tag_from_seed(seed))
    assert tags[0] == tags[1] and tags[0].startswith("v6p5_compose_")


def test_hard_gates(cfg):
    freeze = {"taxonomy": {str(i): n for i, n in enumerate(cfg.names)}, "policy": {"test_modified": False}}
    empty = pd.DataFrame()
    cp.hard_gates(freeze, cfg.names, empty, empty, False, False)
    with pytest.raises(RuntimeError):
        cp.hard_gates(freeze, cfg.names, pd.DataFrame([{"x": 1}]), empty, False, False)
    with pytest.raises(cp.PHashReviewRequired):
        cp.hard_gates(freeze, cfg.names, empty, pd.DataFrame([{"x": 1}]), False, False)
    with pytest.raises(RuntimeError):
        cp.hard_gates({**freeze, "policy": {"test_modified": True}}, cfg.names, empty, empty, False, False)
