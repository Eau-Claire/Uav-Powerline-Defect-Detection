import random

from yolov11sdi.archive_utils import TarEntry, write_deterministic_targz
from yolov11sdi.hashing import (PHashIndex, deterministic_fraction, dir_digest, extract_tags, make_tag,
                                sha256_file, stable_json_hash)


def test_stable_json_hash_is_key_order_independent():
    assert stable_json_hash({"a": 1, "b": [1, 2]}) == stable_json_hash({"b": [1, 2], "a": 1})
    assert stable_json_hash({"a": 1}) != stable_json_hash({"a": 2})


def test_make_tag_and_extract():
    d = "0123456789abcdef" * 4
    tag = make_tag("v6p5_ext_", d)
    assert tag == "v6p5_ext_0123456789ab"
    text = f"stash://x/{tag}/{tag}.freeze.json\nstash://x/v6p5_ext_zzz\n"
    assert extract_tags(text, "v6p5_ext_") == [tag]


def test_deterministic_tag_stable_for_same_inputs(tmp_path):
    for root in (tmp_path / "a", tmp_path / "b"):
        (root / "direct/images/train").mkdir(parents=True)
        (root / "direct/images/train/x.jpg").write_bytes(b"img")
        (root / "direct/labels/train").mkdir(parents=True)
        (root / "direct/labels/train/x.txt").write_text("5 0.5 0.5 0.1 0.1\n")
    assert dir_digest(tmp_path / "a") == dir_digest(tmp_path / "b")
    (tmp_path / "b/direct/labels/train/x.txt").write_text("4 0.5 0.5 0.1 0.1\n")
    assert dir_digest(tmp_path / "a") != dir_digest(tmp_path / "b")


def test_deterministic_targz_is_byte_reproducible(tmp_path):
    entries = [TarEntry("r/b.txt", data=b"2"), TarEntry("r/a.txt", data=b"1")]
    a = write_deterministic_targz(tmp_path / "a.tar.gz", entries)
    b = write_deterministic_targz(tmp_path / "b.tar.gz", list(reversed(entries)))
    assert sha256_file(a) == sha256_file(b)


def test_deterministic_fraction_range_and_stability():
    v = deterministic_fraction("some/key.jpg")
    assert 0 <= v < 1 and v == deterministic_fraction("some/key.jpg")


def test_phash_index_finds_near_duplicates():
    rng = random.Random(0)
    idx = PHashIndex()
    base = rng.getrandbits(64)
    idx.add(base, {"id": "base"})
    for _ in range(50):
        idx.add(rng.getrandbits(64), {"id": "noise"})
    near = base ^ 0b1011  # hamming 3
    hits = idx.query(near, 4)
    assert ("base" in [p["id"] for _, p in hits]) and all(hd <= 4 for hd, _ in hits)
    assert not [p for hd, p in idx.query(base ^ 0b11111, 4) if p["id"] == "base"]  # hamming 5
