#!/usr/bin/env python3
"""Download a compact, reusable model-only mirror from bunpmc Camber Stash."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEST = ROOT / "artifacts"
INVENTORY = ROOT / "inventory.csv"
STASH_ROOT = "stash://bunpmc/"
DOWNLOAD_WORKERS = 4

SMALL_METADATA = {
    "args.yaml",
    "data.yaml",
    "dataset_manifest.json",
    "done.json",
    "exit_status.json",
    "metrics.json",
    "result.csv",
    "results.csv",
    "status.csv",
    "summary.csv",
    "summary.json",
    "test_result.csv",
    "val_map.csv",
}


def run(*args: str, capture: bool = False) -> str:
    result = subprocess.run(
        args,
        check=True,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout if capture else ""


def remote_inventory() -> list[dict[str, str]]:
    return camber_json("stash", "ls", "-r", "--output", "json", STASH_ROOT)


def camber_json(*args: str, attempts: int = 5) -> object:
    last_error: subprocess.CalledProcessError | None = None
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            ["camber", *args],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        last_error = subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
        if attempt < attempts:
            time.sleep(2**attempt)
    assert last_error is not None
    raise last_error


def normalized_tail(name: str) -> str:
    return re.sub(r"/+", "/", name).strip("/")


def is_selected_file(name: str) -> bool:
    base = Path(name).name
    return base in SMALL_METADATA or base in {"best.pt", "best_map.pt"}


def classify(name: str) -> Path | None:
    n = normalized_tail(name)

    # Latest curated benchmark only. The 20261031 RT-DETR tree is intentionally legacy.
    m = re.fullmatch(r"projects/finding_nemo/(yolo11n|yolo11m|yolo26n|yolo26m)/(.+)", n)
    if m and is_selected_file(n):
        return Path("benchmark/curated_latest") / m.group(1) / m.group(2)
    m = re.fullmatch(
        r"projects/finding_nemo/rtdetr_training/finding_nemo_human_reviewed_6cls_20261101/"
        r"(r50vd|r101vd)/(checkpoints/best\.pt|state/result\.csv)",
        n,
    )
    if m:
        return Path("benchmark/curated_latest") / m.group(1) / Path(m.group(2)).name

    # Matching raw benchmark baselines, including the canonical non-double-slash RT-DETR objects.
    m = re.fullmatch(r"projects/raw_(yolo11n|yolo11m|yolo26n|yolo26m)/(.+)", n)
    if m and is_selected_file(n):
        return Path("benchmark/raw") / m.group(1) / m.group(2)
    m = re.fullmatch(
        r"projects/raw_(rtdetr50|rtdetr101)/(checkpoints/best\.pt|state/result\.csv|test_result\.csv)",
        n,
    )
    if m:
        return Path("benchmark/raw") / m.group(1) / Path(m.group(2)).name

    # Canonical ablation copies. finding_dori/finding_destiny duplicate these runs and are omitted.
    m = re.fullmatch(r"projects/finding_ablation/(add_one_in|leave_one_out)/(.+)", n)
    if m and is_selected_file(n):
        return Path("ablations") / m.group(1) / m.group(2)
    if n in {
        "projects/finding_ablation/summary.csv",
        "projects/finding_ablation/summary.json",
    }:
        return Path("ablations") / Path(n).name

    # Keep raw seed artifacts only; old curated seed runs are intentionally excluded.
    m = re.fullmatch(r"projects/finding_seed/(seed36|seed85)/(raw_[^/]+)/(.+)", n)
    if m and is_selected_file(n):
        return Path("seeds/raw") / m.group(1) / m.group(2) / m.group(3)

    # Current BN-fixed RT-DETR work is not legacy. Preserve both sides for ongoing comparison.
    m = re.fullmatch(r"projects/finding_seed/(seed\d+_rtdetr_bnfix)/([^/]+)/(.+)", n)
    if m and is_selected_file(n):
        # This checkpoint is overwritten while the active raw run is training, so its
        # presigned object can disappear mid-download. best.pt is the stable snapshot.
        if m.group(2).startswith("raw_") and Path(m.group(3)).name == "best_map.pt":
            return None
        return Path("active_runs") / m.group(1) / m.group(2) / m.group(3)
    return None


def reported_size_range(value: str) -> tuple[float, float] | None:
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(B|KB|MB|GB|TB)", value or "")
    if not match:
        return None
    number, unit = match.groups()
    # Camber labels binary sizes as KB/MB/GB (e.g. 42.0MB is about 44.0 million bytes).
    multiplier = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
    displayed = float(number) * multiplier
    if unit == "B":
        return displayed, displayed
    decimals = len(number.partition(".")[2])
    tolerance = 0.51 * (10 ** -decimals) * multiplier
    return max(0, displayed - tolerance), displayed + tolerance


def already_complete(path: Path, reported_size: str) -> bool:
    expected = reported_size_range(reported_size)
    return path.is_file() and expected is not None and expected[0] <= path.stat().st_size <= expected[1]


def download(item: tuple[str, Path, str]) -> dict[str, str | int]:
    remote_name, local_rel, reported_size = item
    local = DEST / local_rel
    local.parent.mkdir(parents=True, exist_ok=True)
    if already_complete(local, reported_size):
        action = "kept"
    else:
        presigned = camber_json(
            "stash", "presign-url", "-f", f"stash://bunpmc/{remote_name}", "--output", "json"
        )
        url = str(presigned["url"])
        temporary = local.with_suffix(local.suffix + ".part")
        subprocess.run(
            ["curl", "--fail", "--location", "--silent", "--show-error", "--retry", "3",
             "--output", str(temporary), url],
            check=True,
        )
        os.replace(temporary, local)
        action = "downloaded"
    return {
        "remote": f"stash://bunpmc/{remote_name}",
        "local": str(Path("artifacts") / local_rel),
        "reported_size": reported_size,
        "downloaded_bytes": local.stat().st_size,
        "action": action,
    }


def main() -> None:
    entries = remote_inventory()
    selected: list[tuple[str, Path, str]] = []
    for entry in entries:
        if entry.get("type") != "file":
            continue
        remote_name = entry["name"]
        local_rel = classify(remote_name)
        if local_rel is not None:
            selected.append((remote_name, local_rel, entry.get("file_size", "")))

    # A normalized Stash path can be listed twice when legacy keys contain //.
    unique: dict[Path, tuple[str, Path, str]] = {}
    for item in selected:
        unique.setdefault(item[1], item)
    selected = sorted(unique.values(), key=lambda item: item[1].as_posix())

    DEST.mkdir(parents=True, exist_ok=True)
    for temporary in DEST.rglob("*.part"):
        temporary.unlink()
    rows = []
    with ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        futures = {pool.submit(download, item): item for item in selected}
        for index, future in enumerate(as_completed(futures), 1):
            row = future.result()
            rows.append(row)
            print(
                f"[{index:03d}/{len(selected):03d}] {row['action']:10} "
                f"{int(row['downloaded_bytes']) / 1e6:8.1f} MB  {row['local']}",
                flush=True,
            )
    rows.sort(key=lambda row: str(row["local"]))

    with INVENTORY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys() if rows else ["remote", "local", "reported_size", "downloaded_bytes"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Downloaded {len(rows)} files ({sum(int(r['downloaded_bytes']) for r in rows) / 1e9:.3f} GB)")
    print(f"Inventory: {INVENTORY}")


if __name__ == "__main__":
    main()
